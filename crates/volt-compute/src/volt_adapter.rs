//! Validated projection of exact Volt contracts into floating-point pricing rows.
//!
//! Only option positions in a canonical portfolio are valued here. Cash balances,
//! physical underlying balances, execution, and lifecycle transitions remain in
//! Volt and the application ledger. Exact source contracts are retained alongside
//! the numeric projection so that a price never replaces contract meaning.

use crate::{ComputeError, OptionRow, OptionSide};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use volt::{
    CANONICAL_PORTFOLIO_SCHEMA_VERSION, CanonicalPortfolio, ContractId, Currency, ExerciseSchedule,
    OptionContract, OptionRight, Position, PriceMicros, TimestampNs, UnderlyingId,
};

pub const VALUATION_REQUEST_SCHEMA: &str = "volt.compute.portfolio_request.v1";
pub const MARKET_INPUTS_SCHEMA: &str = "volt.compute.market_inputs.v1";
pub const OPTION_BATCH_SCHEMA: &str = "volt.compute.option_batch.v1";

/// Time to expiry is actual elapsed nanoseconds divided by 365 days.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum YearBasis {
    #[serde(rename = "actual_365_fixed")]
    Actual365Fixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnderlyingMarketInput {
    pub underlying: UnderlyingId,
    pub observed_at: TimestampNs,
    pub spot: PriceMicros,
}

/// Flat Black–Scholes inputs for one maturity and exact option contract.
/// Rates and dividend yields are annual continuously compounded fractions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractPricingInput {
    pub contract_id: ContractId,
    pub observed_at: TimestampNs,
    pub volatility: f64,
    pub rate: f64,
    pub dividend_yield: f64,
}

/// Caller-declared point-in-time inputs. All observations must be no later than
/// as_of and no older than the explicit maximum age. A zero age requires an
/// observation exactly at as_of; the adapter does not infer a staleness policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortfolioMarketInputs {
    pub schema_version: String,
    pub snapshot_id: String,
    pub as_of: TimestampNs,
    pub max_observation_age_ns: i64,
    pub year_basis: YearBasis,
    pub underlyings: Vec<UnderlyingMarketInput>,
    pub contracts: Vec<ContractPricingInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoltValuationRequest {
    pub schema_version: String,
    pub portfolio: CanonicalPortfolio,
    pub market: PortfolioMarketInputs,
}

impl VoltValuationRequest {
    pub fn project(&self) -> Result<VoltOptionBatch, ComputeError> {
        if self.schema_version != VALUATION_REQUEST_SCHEMA {
            return Err(invalid(format!(
                "unsupported valuation request schema {}",
                self.schema_version
            )));
        }
        project_portfolio(&self.portfolio, &self.market)
    }
}

/// Exact source metadata, in the same order as the numeric rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectedContract {
    pub row_id: String,
    pub contract: OptionContract,
    pub quantity: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoltOptionBatch {
    pub schema_version: String,
    pub portfolio_hash: String,
    pub market_inputs_hash: String,
    pub snapshot_id: String,
    pub as_of: TimestampNs,
    pub currency: Currency,
    pub year_basis: YearBasis,
    pub source_contracts: Vec<ProjectedContract>,
    pub rows: Vec<OptionRow>,
}

/// Project a validated canonical European portfolio without changing its exact
/// identity. Mixed currencies require an explicit FX valuation layer and are
/// rejected here. Floating-point marks retain exact source money and time values
/// in source_contracts, even when their numeric conversion requires rounding.
pub fn project_portfolio(
    portfolio: &CanonicalPortfolio,
    market: &PortfolioMarketInputs,
) -> Result<VoltOptionBatch, ComputeError> {
    validate_canonical_portfolio(portfolio)?;
    if market.schema_version != MARKET_INPUTS_SCHEMA {
        return Err(invalid(format!(
            "unsupported market input schema {}",
            market.schema_version
        )));
    }
    if market.snapshot_id.trim().is_empty()
        || market.as_of.0 < 0
        || market.max_observation_age_ns < 0
    {
        return Err(invalid(
            "snapshot identity must be nonempty and timestamps and maximum age nonnegative",
        ));
    }
    let first = &portfolio.positions[0].contract;
    let currency = first.currency.clone();
    let mut required_underlyings = BTreeSet::new();
    let mut required_contracts = BTreeSet::new();
    for position in &portfolio.positions {
        let contract = &position.contract;
        if contract.currency != currency {
            return Err(ComputeError::Unsupported(
                "mixed-currency portfolios require an explicit FX valuation layer".into(),
            ));
        }
        if !matches!(contract.exercise, ExerciseSchedule::European { .. }) {
            return Err(ComputeError::Unsupported(format!(
                "{}: only European option valuation is implemented",
                contract.contract_id.0
            )));
        }
        if contract.expiry() <= market.as_of {
            return Err(ComputeError::Unsupported(format!(
                "{}: option is at or past expiry",
                contract.contract_id.0
            )));
        }
        required_underlyings.insert(contract.underlying.clone());
        required_contracts.insert(contract.contract_id.clone());
    }

    let mut underlyings = BTreeMap::new();
    for input in &market.underlyings {
        validate_observation_time(input.observed_at, market)?;
        if input.spot.0 <= 0 {
            return Err(invalid(format!(
                "{}: spot must be positive",
                input.underlying.0
            )));
        }
        if !required_underlyings.contains(&input.underlying) {
            return Err(invalid(format!(
                "unexpected underlying market input {}",
                input.underlying.0
            )));
        }
        if underlyings
            .insert(input.underlying.clone(), input)
            .is_some()
        {
            return Err(invalid(format!(
                "duplicate underlying market input {}",
                input.underlying.0
            )));
        }
    }
    let mut contracts = BTreeMap::new();
    for input in &market.contracts {
        validate_observation_time(input.observed_at, market)?;
        if !required_contracts.contains(&input.contract_id) {
            return Err(invalid(format!(
                "unexpected contract pricing input {}",
                input.contract_id.0
            )));
        }
        if !input.volatility.is_finite()
            || input.volatility <= 0.0
            || !input.rate.is_finite()
            || !input.dividend_yield.is_finite()
        {
            return Err(invalid(format!(
                "{}: volatility must be finite and positive and rates finite",
                input.contract_id.0
            )));
        }
        if contracts.insert(input.contract_id.clone(), input).is_some() {
            return Err(invalid(format!(
                "duplicate contract pricing input {}",
                input.contract_id.0
            )));
        }
    }
    for underlying in &required_underlyings {
        if !underlyings.contains_key(underlying) {
            return Err(invalid(format!(
                "missing underlying market input {}",
                underlying.0
            )));
        }
    }
    for contract in &required_contracts {
        if !contracts.contains_key(contract) {
            return Err(invalid(format!(
                "missing contract pricing input {}",
                contract.0
            )));
        }
    }

    // Canonicalize input ordering for deterministic provenance. The timestamp,
    // staleness policy, year basis, and snapshot ID remain part of this identity.
    let mut canonical_market = market.clone();
    canonical_market
        .underlyings
        .sort_by(|left, right| left.underlying.cmp(&right.underlying));
    canonical_market
        .contracts
        .sort_by(|left, right| left.contract_id.cmp(&right.contract_id));
    let bytes = serde_json::to_vec(&canonical_market).map_err(|e| invalid(e.to_string()))?;
    let mut digest = Sha256::new();
    digest.update(MARKET_INPUTS_SCHEMA);
    digest.update([0]);
    digest.update(bytes);
    let market_inputs_hash = format!("{:x}", digest.finalize());
    let portfolio_hash = portfolio.semantic_hash().map_err(volt_error)?;
    let mut rows = Vec::with_capacity(portfolio.positions.len());
    let mut source_contracts = Vec::with_capacity(portfolio.positions.len());
    let mut rates_by_expiry = BTreeMap::new();
    let mut dividends_by_underlying_expiry = BTreeMap::new();
    for position in &portfolio.positions {
        let contract = &position.contract;
        let input = contracts[&contract.contract_id];
        check_consistent(
            &mut rates_by_expiry,
            contract.expiry(),
            input.rate,
            "risk-free rates for the same currency and expiry",
        )?;
        check_consistent(
            &mut dividends_by_underlying_expiry,
            (contract.underlying.clone(), contract.expiry()),
            input.dividend_yield,
            "dividend yields for the same underlying and expiry",
        )?;
        let elapsed_ns = contract.expiry().0 - market.as_of.0;
        let row_id = format!(
            "{portfolio_hash}:{market_inputs_hash}:{}",
            contract.contract_id.0
        );
        let row = OptionRow {
            row_id: row_id.clone(),
            contract_id: contract.contract_id.0.clone(),
            side: match contract.right {
                OptionRight::Call => OptionSide::Call,
                OptionRight::Put => OptionSide::Put,
            },
            spot: underlyings[&contract.underlying].spot.0 as f64 / 1_000_000.0,
            strike: contract.strike.0 as f64 / 1_000_000.0,
            maturity_years: elapsed_ns as f64 / (365.0 * 86_400.0 * 1_000_000_000.0),
            volatility: input.volatility,
            rate: input.rate,
            dividend_yield: input.dividend_yield,
            quantity: position.quantity,
            multiplier: contract.multiplier,
        };
        row.validate()?;
        rows.push(row);
        source_contracts.push(ProjectedContract {
            row_id,
            contract: contract.clone(),
            quantity: position.quantity,
        });
    }
    Ok(VoltOptionBatch {
        schema_version: OPTION_BATCH_SCHEMA.into(),
        portfolio_hash,
        market_inputs_hash,
        snapshot_id: market.snapshot_id.clone(),
        as_of: market.as_of,
        currency,
        year_basis: market.year_basis,
        source_contracts,
        rows,
    })
}

fn validate_canonical_portfolio(portfolio: &CanonicalPortfolio) -> Result<(), ComputeError> {
    if portfolio.schema_version != CANONICAL_PORTFOLIO_SCHEMA_VERSION {
        return Err(invalid(format!(
            "unsupported portfolio schema {}",
            portfolio.schema_version
        )));
    }
    if portfolio.positions.is_empty() {
        return Err(invalid("option valuation requires a nonempty portfolio"));
    }
    let positions = portfolio
        .positions
        .iter()
        .map(|position| Position::new(position.contract.clone(), position.quantity))
        .collect::<Result<Vec<_>, _>>()
        .map_err(volt_error)?;
    let canonical = CanonicalPortfolio::from_positions(positions).map_err(volt_error)?;
    if canonical != *portfolio {
        return Err(invalid(
            "portfolio representation is not canonical; construct it through Volt",
        ));
    }
    Ok(())
}

fn validate_observation_time(
    observed_at: TimestampNs,
    market: &PortfolioMarketInputs,
) -> Result<(), ComputeError> {
    if observed_at.0 < 0 || observed_at > market.as_of {
        return Err(invalid(
            "market observation must be nonnegative and no later than as_of",
        ));
    }
    if market.as_of.0 - observed_at.0 > market.max_observation_age_ns {
        return Err(invalid("market observation exceeds maximum permitted age"));
    }
    Ok(())
}

fn check_consistent<K: Ord>(
    values: &mut BTreeMap<K, f64>,
    key: K,
    value: f64,
    description: &str,
) -> Result<(), ComputeError> {
    if let Some(previous) = values.insert(key, value)
        && previous != value
    {
        return Err(invalid(format!("conflicting {description}")));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> ComputeError {
    ComputeError::InvalidInput(message.into())
}

fn volt_error(error: volt::VoltError) -> ComputeError {
    invalid(format!("Volt validation failed: {error}"))
}
