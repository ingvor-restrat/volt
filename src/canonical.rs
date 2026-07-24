//! Exact portfolio normalization and terminal-payoff projection.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::contract::{
    ContractId, Currency, OptionContract, OptionRight, Position, PriceMicros, TimestampNs,
    UnderlyingId,
};
use crate::hash::canonical_hash;
use crate::{CogError, CogResult};

/// Schema identifier serialized with [`CanonicalPortfolio`].
pub const CANONICAL_PORTFOLIO_SCHEMA_VERSION: &str = "volt.portfolio.v1";
/// Schema identifier serialized with [`TerminalPayoffProjection`].
pub const TERMINAL_PAYOFF_SCHEMA_VERSION: &str = "volt.terminal_payoff.v1";

/// One exact contract after quantities with the same identity are aggregated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalPosition {
    /// Exact listed contract definition.
    pub contract: OptionContract,
    /// Nonzero net signed quantity.
    pub quantity: i64,
}

/// Order-independent exact representation of an options portfolio.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalPortfolio {
    /// Must equal [`CANONICAL_PORTFOLIO_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Net nonzero positions ordered by contract identity.
    pub positions: Vec<CanonicalPosition>,
}

impl CanonicalPortfolio {
    /// Aggregates positions with the same contract identifier, rejects
    /// conflicting definitions for one identifier, removes zero net positions,
    /// and orders the result by identifier.
    pub fn from_positions(positions: impl IntoIterator<Item = Position>) -> CogResult<Self> {
        let mut by_id: BTreeMap<ContractId, (OptionContract, i64)> = BTreeMap::new();
        for position in positions {
            position.contract.validate()?;
            let contract_id = position.contract.contract_id.clone();
            let entry = by_id
                .entry(contract_id.clone())
                .or_insert_with(|| (position.contract.clone(), 0));
            if entry.0 != position.contract {
                return Err(CogError::InvalidDefinition(format!(
                    "conflicting definitions for contract {}",
                    contract_id.0
                )));
            }
            entry.1 = entry
                .1
                .checked_add(position.quantity)
                .ok_or_else(|| CogError::NumericOverflow("position aggregation".to_owned()))?;
        }
        let positions = by_id
            .into_values()
            .filter_map(|(contract, quantity)| {
                (quantity != 0).then_some(CanonicalPosition { contract, quantity })
            })
            .collect();
        Ok(Self {
            schema_version: CANONICAL_PORTFOLIO_SCHEMA_VERSION.to_owned(),
            positions,
        })
    }

    /// Hash the exact canonical representation with its schema domain.
    pub fn semantic_hash(&self) -> CogResult<String> {
        canonical_hash(CANONICAL_PORTFOLIO_SCHEMA_VERSION, self)
    }

    /// Project a compatible nonempty portfolio into terminal hinge form.
    pub fn terminal_payoff_projection(&self) -> CogResult<TerminalPayoffProjection> {
        TerminalPayoffProjection::from_portfolio(self)
    }

    /// Evaluate terminal intrinsic payoff directly from exact positions.
    pub fn direct_terminal_payoff(&self, spot: PriceMicros) -> CogResult<i128> {
        if spot.0 < 0 {
            return Err(CogError::InvalidDefinition(
                "terminal spot must be non-negative".to_owned(),
            ));
        }
        self.positions.iter().try_fold(0_i128, |total, position| {
            let units = share_equivalent_units(position)?;
            let payoff = i128::from(position.contract.intrinsic_per_unit(spot))
                .checked_mul(i128::from(units))
                .ok_or_else(|| CogError::NumericOverflow("direct terminal payoff".to_owned()))?;
            total
                .checked_add(payoff)
                .ok_or_else(|| CogError::NumericOverflow("direct payoff sum".to_owned()))
        })
    }
}

/// One signed call-hinge coefficient in terminal payoff space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayoffKnot {
    /// Hinge strike in currency micro-units.
    pub strike: PriceMicros,
    /// Signed share-equivalent weight on `(spot - strike)+`.
    pub weight_units: i64,
}

/// Canonical terminal-payoff projection:
/// `cash_micros + underlying_units * spot + sum(weight * (spot - strike)+)`.
///
/// This intentionally forgets early-exercise, assignment, settlement, and
/// execution-realization differences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalPayoffProjection {
    /// Must equal [`TERMINAL_PAYOFF_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Common underlying required of every projected position.
    pub underlying: UnderlyingId,
    /// Common currency required of every projected position.
    pub currency: Currency,
    /// Common expiry required of every projected position.
    pub expiry: TimestampNs,
    /// Constant payoff term in currency micro-units.
    pub cash_micros: i128,
    /// Coefficient on terminal spot.
    pub underlying_units: i64,
    /// Ordered nonzero coefficients on call-style payoff hinges.
    pub call_knots: Vec<PayoffKnot>,
}

impl TerminalPayoffProjection {
    /// Build hinge form from a nonempty, compatible exact portfolio.
    pub fn from_portfolio(portfolio: &CanonicalPortfolio) -> CogResult<Self> {
        let first = portfolio.positions.first().ok_or_else(|| {
            CogError::IncompatiblePortfolio(
                "terminal payoff projection requires a non-empty portfolio".to_owned(),
            )
        })?;
        let underlying = first.contract.underlying.clone();
        let currency = first.contract.currency.clone();
        let expiry = first.contract.expiry();

        let mut cash_micros = 0_i128;
        let mut underlying_units = 0_i64;
        let mut knots: BTreeMap<PriceMicros, i64> = BTreeMap::new();

        for position in &portfolio.positions {
            let contract = &position.contract;
            if contract.underlying != underlying
                || contract.currency != currency
                || contract.expiry() != expiry
            {
                return Err(CogError::IncompatiblePortfolio(
                    "terminal payoff projection requires one underlying, currency, and expiry"
                        .to_owned(),
                ));
            }
            let units = share_equivalent_units(position)?;
            let knot = knots.entry(contract.strike).or_default();
            *knot = knot
                .checked_add(units)
                .ok_or_else(|| CogError::NumericOverflow("payoff knot aggregation".to_owned()))?;

            if contract.right == OptionRight::Put {
                let put_cash = i128::from(units)
                    .checked_mul(i128::from(contract.strike.0))
                    .ok_or_else(|| CogError::NumericOverflow("put cash projection".to_owned()))?;
                cash_micros = cash_micros
                    .checked_add(put_cash)
                    .ok_or_else(|| CogError::NumericOverflow("payoff cash sum".to_owned()))?;
                underlying_units = underlying_units
                    .checked_sub(units)
                    .ok_or_else(|| CogError::NumericOverflow("underlying projection".to_owned()))?;
            }
        }

        let call_knots = knots
            .into_iter()
            .filter_map(|(strike, weight_units)| {
                (weight_units != 0).then_some(PayoffKnot {
                    strike,
                    weight_units,
                })
            })
            .collect();

        Ok(Self {
            schema_version: TERMINAL_PAYOFF_SCHEMA_VERSION.to_owned(),
            underlying,
            currency,
            expiry,
            cash_micros,
            underlying_units,
            call_knots,
        })
    }

    /// Evaluate hinge form at a nonnegative terminal spot.
    pub fn evaluate(&self, spot: PriceMicros) -> CogResult<i128> {
        if spot.0 < 0 {
            return Err(CogError::InvalidDefinition(
                "terminal spot must be non-negative".to_owned(),
            ));
        }
        let linear = i128::from(self.underlying_units)
            .checked_mul(i128::from(spot.0))
            .ok_or_else(|| CogError::NumericOverflow("linear payoff".to_owned()))?;
        let mut total = self
            .cash_micros
            .checked_add(linear)
            .ok_or_else(|| CogError::NumericOverflow("payoff base".to_owned()))?;
        for knot in &self.call_knots {
            let intrinsic = spot.0.saturating_sub(knot.strike.0).max(0);
            let contribution = i128::from(knot.weight_units)
                .checked_mul(i128::from(intrinsic))
                .ok_or_else(|| CogError::NumericOverflow("knot payoff".to_owned()))?;
            total = total
                .checked_add(contribution)
                .ok_or_else(|| CogError::NumericOverflow("payoff sum".to_owned()))?;
        }
        Ok(total)
    }

    /// Hash the canonical payoff representation with its schema domain.
    pub fn semantic_hash(&self) -> CogResult<String> {
        canonical_hash(TERMINAL_PAYOFF_SCHEMA_VERSION, self)
    }
}

fn share_equivalent_units(position: &CanonicalPosition) -> CogResult<i64> {
    position
        .quantity
        .checked_mul(i64::from(position.contract.multiplier))
        .ok_or_else(|| CogError::NumericOverflow("share-equivalent units".to_owned()))
}
