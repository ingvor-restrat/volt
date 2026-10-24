//! Explicit quantization bridge from a numerical hedge path to the exact
//! self-financing research ledger. This does not construct listed options or
//! authorize trading. The batch adapter already enforces r=q=0; a raw batch has
//! no rate provenance, so callers explicitly accept ZERO-rate/no-dividend
//! accounting here. No financing, margin, corporate actions or extra flows are
//! inferred from marks. Timestamps are model years*365days, from epoch0.
//!
//! Price/quantity/cash/rate/time conversion rounds nearest, ties away from0.
//! Fees use nearest integer ppm and disclose the residual, including e.g.
//! 1/512 ->1953ppm, not an exact representation of1953.125ppm. The ledger's
//! integer picounits reconcile EXACTLY; an independent rational identity exposes
//! per-trade fee rounding, while the unquantized P&L gap is separately bounded.

use crate::{
    ComputeError,
    chunked::CompensatedSum,
    hedge_application::{HedgeCostConfig, HedgePathBatch},
    hedge_ledger::{
        HedgeInstrument, HedgeInstrumentKind, HedgeJournalEntry, HedgeLedger, HedgeLedgerConfig,
        HedgeLedgerState, HedgeMarketSnapshot, IntervalAccrual, LiabilityCashflow,
        ProportionalCost, QuantityMicros,
    },
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use volt::{Currency, PriceMicros, TimestampNs, UnderlyingId};

const MICROS: f64 = 1_000_000.0;
const PICOS: f64 = 1_000_000_000_000.0;
const FEE_DENOMINATOR: i128 = 1_000_000;
const PICOS_PER_MICRO: i128 = 1_000_000;
const NS_PER_YEAR: f64 = 365.0 * 86_400.0 * 1_000_000_000.0;
pub const MAX_AUDIT_DECISIONS: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditModelFamily {
    BlackScholes,
    Heston,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeeQuantization {
    pub requested_rate: f64,
    pub parts_per_million: u32,
    pub ledger_rate: f64,
    /// Requested minus ledger rate, in cost fraction units.
    pub rate_residual: f64,
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct RoundingResiduals {
    pub max_price_currency: f64,
    pub max_quantity_units: f64,
    pub premium_currency: f64,
    pub terminal_liability_currency: f64,
    pub max_time_nanoseconds: f64,
    pub max_fee_picounits: f64,
}
#[derive(Debug, Clone, Serialize)]
pub struct ReconciliationAudit {
    pub schema: String,
    pub source_hash: String,
    pub selected_input_hash: String,
    pub path_index: usize,
    pub model_family: AuditModelFamily,
    pub dt_years: f64,
    pub accounting_contract: String,
    pub trading_fee: FeeQuantization,
    pub terminal_fee: FeeQuantization,
    pub max_rounding_residual: RoundingResiduals,
    pub unquantized_float_pnl: f64,
    /// Exact rational quantized-input identity with unrounded rational fees,
    /// converted once to f64 currency units. This is not the original P&L.
    pub float_quantized_identity_pnl: f64,
    pub exact_ledger_pnl_picos: String,
    pub exact_independent_identity_picos: String,
    pub exact_reconciliation: bool,
    pub float_identity_difference_currency: f64,
    pub fee_rounding_bound_picos: String,
    pub float_identity_conversion_bound_currency: f64,
    pub float_identity_within_bound: bool,
    /// Exact-ledger currency equivalent minus original floating path P&L.
    pub quantization_gap: f64,
    pub quantization_error_bound_currency: f64,
    pub quantization_within_bound: bool,
    pub initial_premium_micros: String,
    pub terminal_liability_micros: String,
    pub initial_market: HedgeMarketSnapshot,
    pub ledger_config: HedgeLedgerConfig,
    pub rounded_holdings: Vec<QuantityMicros>,
    pub final_state: HedgeLedgerState,
    pub journal: Vec<HedgeJournalEntry>,
}

/// `holdings` is ONLY this selected path, `[decision, instrument]`, with exactly
/// steps*hedge_count entries. Heston mark order is stock,H per system; BS uses
/// stocks. All positions are in quoted units (multiplier1), not option lots.
pub fn reconcile_path(
    batch: &HedgePathBatch,
    path_index: usize,
    holdings: &[f64],
    family: AuditModelFamily,
    dt: f64,
    premium: f64,
    cost: HedgeCostConfig,
) -> Result<ReconciliationAudit, ComputeError> {
    batch.validate()?;
    let decisions = batch
        .steps
        .checked_mul(batch.hedge_count)
        .ok_or_else(|| invalid("audit decision shape overflow"))?;
    if path_index >= batch.paths
        || holdings.len() != decisions
        || decisions > MAX_AUDIT_DECISIONS
        || batch.steps > 4096
        || batch.hedge_count > 128
        || holdings.iter().any(|x| !x.is_finite())
        || !dt.is_finite()
        || dt <= 0.0
        || !premium.is_finite()
        || (family == AuditModelFamily::Heston && !batch.hedge_count.is_multiple_of(2))
    {
        return Err(invalid(
            "audit path/holdings/family/time shape or finite-value limit invalid",
        ));
    }
    let trading_fee = quantize_fee(cost.proportional)?;
    let terminal_fee = quantize_fee(cost.terminal_proportional)?;
    let mut residuals = RoundingResiduals::default();
    let mut times = Vec::with_capacity(batch.steps + 1);
    for time in 0..=batch.steps {
        let source = time as f64 * dt * NS_PER_YEAR;
        let rounded = round_i64(source, "timestamp nanoseconds")?;
        if rounded < 0 || times.last().is_some_and(|t: &TimestampNs| rounded <= t.0) {
            return Err(invalid(
                "rounded audit times must be strictly chronological",
            ));
        }
        residuals.max_time_nanoseconds = residuals
            .max_time_nanoseconds
            .max((source - rounded as f64).abs());
        times.push(TimestampNs(rounded));
    }
    let mut quantities = Vec::with_capacity(decisions);
    for value in holdings {
        let q = round_i64(*value * MICROS, "quantity micros")?;
        residuals.max_quantity_units = residuals
            .max_quantity_units
            .max((*value - q as f64 / MICROS).abs());
        quantities.push(QuantityMicros(q));
    }
    // Validate changes before constructing a mutable ledger (both endpoints may
    // fit i64 while their difference does not).
    for t in 0..batch.steps {
        for h in 0..batch.hedge_count {
            let i = t * batch.hedge_count + h;
            let previous = if t == 0 {
                0
            } else {
                quantities[i - batch.hedge_count].0
            };
            quantities[i]
                .0
                .checked_sub(previous)
                .ok_or_else(|| invalid("audit quantity change overflows i64"))?;
        }
    }
    for q in &quantities[(batch.steps - 1) * batch.hedge_count..] {
        q.0.checked_neg()
            .ok_or_else(|| invalid("terminal inventory change overflows i64"))?;
    }
    let stride = (batch.steps + 1) * batch.hedge_count;
    let marks = &batch.marks[path_index * stride..(path_index + 1) * stride];
    let mut prices = Vec::with_capacity(stride);
    for (i, value) in marks.iter().enumerate() {
        let p = round_i64(*value * MICROS, "price micros")?;
        let stock =
            family == AuditModelFamily::BlackScholes || (i % batch.hedge_count).is_multiple_of(2);
        if p < 0 || (stock && p == 0) {
            return Err(invalid(
                "quantized stock marks must stay positive and funded claim marks nonnegative",
            ));
        }
        residuals.max_price_currency = residuals
            .max_price_currency
            .max((*value - p as f64 / MICROS).abs());
        prices.push(p);
    }
    let initial_micros = round_i128(premium * MICROS, "premium micros")?;
    let liability = batch.payoffs[path_index];
    let liability_micros = round_i128(liability * MICROS, "liability micros")?;
    let initial_picos = mul(initial_micros, PICOS_PER_MICRO, "premium picounits")?;
    let liability_picos = mul(liability_micros, PICOS_PER_MICRO, "liability picounits")?;
    residuals.premium_currency = (premium - initial_micros as f64 / MICROS).abs();
    residuals.terminal_liability_currency = (liability - liability_micros as f64 / MICROS).abs();
    let mut original = CompensatedSum::default();
    original.add(premium);
    original.add(-liability);
    let mut error_bound = CompensatedSum::default();
    error_bound.add(residuals.premium_currency);
    error_bound.add(residuals.terminal_liability_currency);
    let mut absolute_work = CompensatedSum::default();
    absolute_work.add(premium.abs());
    absolute_work.add(liability.abs());
    let mut gains = 0_i128;
    let mut fees = FeeTotals::default();
    for t in 0..batch.steps {
        for h in 0..batch.hedge_count {
            let i = t * batch.hedge_count + h;
            let j = (t + 1) * batch.hedge_count + h;
            let q = quantities[i].0;
            let previous_q = if t == 0 {
                0
            } else {
                quantities[i - batch.hedge_count].0
            };
            let before = if t == 0 {
                0.0
            } else {
                holdings[i - batch.hedge_count]
            };
            let before_error = (before - previous_q as f64 / MICROS).abs();
            let q_float = q as f64 / MICROS;
            let q_error = (holdings[i] - q_float).abs();
            let p_error = (marks[i] - prices[i] as f64 / MICROS).abs();
            let next_error = (marks[j] - prices[j] as f64 / MICROS).abs();
            let gain = holdings[i] * (marks[j] - marks[i]);
            original.add(gain);
            original.add(-cost.proportional * (holdings[i] - before).abs() * marks[i]);
            absolute_work.add(holdings[i].abs() * (marks[i].abs() + marks[j].abs()));
            absolute_work.add(cost.proportional * (holdings[i] - before).abs() * marks[i]);
            error_bound.add(
                q_error * (marks[j] - marks[i]).abs() + q_float.abs() * (p_error + next_error),
            );
            let change = q
                .checked_sub(previous_q)
                .ok_or_else(|| invalid("audit quantity change overflow"))?;
            error_bound.add(
                trading_fee.rate_residual.abs() * (holdings[i] - before).abs() * marks[i]
                    + trading_fee.ledger_rate
                        * ((q_error + before_error) * marks[i]
                            + (change as f64 / MICROS).abs() * p_error),
            );
            gains = add(
                gains,
                mul(
                    i128::from(q),
                    i128::from(prices[j]) - i128::from(prices[i]),
                    "independent mark gain",
                )?,
                "independent gains",
            )?;
            if change != 0 {
                accumulate_fee(
                    change,
                    prices[i],
                    trading_fee.parts_per_million,
                    &mut fees,
                    &mut residuals,
                )?;
            }
            if t + 1 == batch.steps {
                original.add(-cost.terminal_proportional * holdings[i].abs() * marks[j]);
                absolute_work.add(cost.terminal_proportional * holdings[i].abs() * marks[j]);
                error_bound.add(
                    terminal_fee.rate_residual.abs() * holdings[i].abs() * marks[j]
                        + terminal_fee.ledger_rate
                            * (q_error * marks[j] + q_float.abs() * next_error),
                );
                if q != 0 {
                    accumulate_fee(
                        q,
                        prices[j],
                        terminal_fee.parts_per_million,
                        &mut fees,
                        &mut residuals,
                    )?;
                }
            }
        }
    }
    let base = sub(
        add(initial_picos, gains, "identity initial plus gains")?,
        liability_picos,
        "identity liability",
    )?;
    let independent_identity = sub(base, fees.rounded, "identity rounded fees")?;
    let rational_numerator = sub(
        mul(base, FEE_DENOMINATOR, "rational identity numerator")?,
        fees.numerators,
        "rational fees",
    )?;
    let float_identity = rational_numerator as f64 / (PICOS * FEE_DENOMINATOR as f64);
    let terminal = times[batch.steps];
    let currency = Currency("USD".into());
    let mut instruments = Vec::with_capacity(batch.hedge_count);
    let mut ids = Vec::with_capacity(batch.hedge_count);
    for h in 0..batch.hedge_count {
        let system = if family == AuditModelFamily::Heston {
            h / 2
        } else {
            h
        };
        let underlying = UnderlyingId(format!("numerical-system-{system:04}"));
        let kind = if family == AuditModelFamily::BlackScholes || h.is_multiple_of(2) {
            HedgeInstrumentKind::Stock { underlying }
        } else {
            HedgeInstrumentKind::FundedVarianceClaim {
                underlying,
                maturity: terminal,
            }
        };
        let name = if matches!(kind, HedgeInstrumentKind::Stock { .. }) {
            "stock"
        } else {
            "funded_variance"
        };
        let id = format!("system-{system:04}.{name}");
        ids.push(id.clone());
        instruments.push(HedgeInstrument {
            instrument_id: id,
            currency: currency.clone(),
            kind,
            quantity_step: QuantityMicros(1),
            multiplier: 1,
        });
    }
    let ledger_config = HedgeLedgerConfig {
        currency: currency.clone(),
        terminal_time: terminal,
        instruments,
    };
    let snapshot = |t: usize, label: &str| HedgeMarketSnapshot {
        snapshot_id: format!("audit-path-{path_index}-{label}"),
        as_of: times[t],
        currency: currency.clone(),
        prices: ids
            .iter()
            .enumerate()
            .map(|(h, id)| (id.clone(), PriceMicros(prices[t * batch.hedge_count + h])))
            .collect(),
    };
    let initial_market = snapshot(0, "inception");
    let mut ledger = HedgeLedger::new(
        ledger_config.clone(),
        initial_market.clone(),
        initial_micros,
    )
    .map_err(ledger_error)?;
    for t in 0..batch.steps {
        let targets: BTreeMap<_, _> = ids
            .iter()
            .enumerate()
            .map(|(h, id)| (id.clone(), quantities[t * batch.hedge_count + h]))
            .collect();
        let from = if t == 0 { times[0] } else { times[t - 1] };
        ledger
            .rebalance(
                snapshot(t, &format!("rebalance-{t}")),
                targets,
                IntervalAccrual::zero_rate(from, times[t]),
                ProportionalCost {
                    parts_per_million: trading_fee.parts_per_million,
                },
            )
            .map_err(ledger_error)?;
    }
    ledger
        .settle(
            snapshot(batch.steps, "terminal"),
            LiabilityCashflow {
                settlement_id: format!("audit-path-{path_index}-liability"),
                currency,
                amount_micros: liability_micros,
            },
            IntervalAccrual::zero_rate(times[batch.steps - 1], terminal),
            ProportionalCost {
                parts_per_million: terminal_fee.parts_per_million,
            },
        )
        .map_err(ledger_error)?;
    let state = ledger.state().clone();
    let ledger_currency = state.net_wealth_picos as f64 / PICOS;
    let exact_reconciliation = state.settled
        && state.holdings.values().all(|q| q.0 == 0)
        && state.net_wealth_picos == independent_identity
        && state.cumulative_gain_picos == gains
        && state.cumulative_cost_picos == fees.rounded
        && state.cumulative_interest_picos == 0
        && sub(
            rational_numerator,
            mul(
                state.net_wealth_picos,
                FEE_DENOMINATOR,
                "rational ledger comparison",
            )?,
            "rational ledger difference",
        )? == -fees.signed
        && ledger
            .journal()
            .iter()
            .all(|entry| entry.reconciled_wealth_picos == entry.after.net_wealth_picos);
    // Two exact integer-to-f64 conversions and division by exactly representable
    // integer powers of10. This bounds conversion error separately from the
    // exact rational per-trade fee rounding bound, not an arbitrary epsilon.
    let conversion_bound =
        4.0 * f64::EPSILON * (float_identity.abs() + ledger_currency.abs()) + f64::MIN_POSITIVE;
    let fee_bound_currency = fees.absolute as f64 / (PICOS * FEE_DENOMINATOR as f64);
    let identity_difference = float_identity - ledger_currency;
    let source_operations = (decisions * 32 + 16) as f64;
    let gamma = source_operations * f64::EPSILON / (1.0 - source_operations * f64::EPSILON);
    error_bound.add(gamma * absolute_work.value());
    error_bound.add(fee_bound_currency);
    error_bound.add(4.0 * f64::EPSILON * ledger_currency.abs());
    let original_pnl = original.value();
    let quantization_gap = ledger_currency - original_pnl;
    let bound = error_bound.value();
    if [
        float_identity,
        ledger_currency,
        conversion_bound,
        identity_difference,
        original_pnl,
        quantization_gap,
        bound,
    ]
    .iter()
    .any(|v| !v.is_finite())
        || bound < 0.0
    {
        return Err(execution(
            "nonfinite audit numerical identity or error bound",
        ));
    }
    let mut hash = Sha256::new();
    hash.update(b"volt.hedge_reconciliation_inputs.v1\0");
    hash.update((batch.source_hash.len() as u64).to_le_bytes());
    hash.update(batch.source_hash.as_bytes());
    hash.update((path_index as u64).to_le_bytes());
    hash.update((batch.steps as u64).to_le_bytes());
    hash.update((batch.hedge_count as u64).to_le_bytes());
    hash.update(serde_json::to_vec(&family).map_err(|e| execution(&e.to_string()))?);
    for value in marks.iter().chain(holdings).chain([
        &dt,
        &premium,
        &liability,
        &cost.proportional,
        &cost.terminal_proportional,
    ]) {
        hash.update(value.to_le_bytes());
    }
    Ok(ReconciliationAudit {schema:"volt.compute.hedge_reconciliation.v1".into(),source_hash:batch.source_hash.clone(),
        selected_input_hash:format!("{:x}",hash.finalize()),path_index,model_family:family,dt_years:dt,
        accounting_contract:"explicit r=q=0,USD,unit multipliers,funded stock/variance assets; raw batch carries no proof of rate or listed-contract provenance".into(),
        trading_fee,terminal_fee,max_rounding_residual:residuals,unquantized_float_pnl:original_pnl,
        float_quantized_identity_pnl:float_identity,exact_ledger_pnl_picos:state.net_wealth_picos.to_string(),
        exact_independent_identity_picos:independent_identity.to_string(),exact_reconciliation,
        float_identity_difference_currency:identity_difference,fee_rounding_bound_picos:((fees.absolute+FEE_DENOMINATOR-1)/FEE_DENOMINATOR).to_string(),
        float_identity_conversion_bound_currency:conversion_bound,float_identity_within_bound:identity_difference.abs()<=fee_bound_currency+conversion_bound,
        quantization_gap,quantization_error_bound_currency:bound,quantization_within_bound:quantization_gap.abs()<=bound,
        initial_premium_micros:initial_micros.to_string(),terminal_liability_micros:liability_micros.to_string(),
        initial_market,ledger_config,rounded_holdings:quantities,final_state:state,journal:ledger.journal().to_vec()})
}

#[derive(Default)]
struct FeeTotals {
    rounded: i128,
    numerators: i128,
    signed: i128,
    absolute: i128,
}
fn accumulate_fee(
    quantity: i64,
    price: i64,
    ppm: u32,
    totals: &mut FeeTotals,
    residuals: &mut RoundingResiduals,
) -> Result<(), ComputeError> {
    let notional = mul(i128::from(quantity), i128::from(price), "fee notional")?
        .checked_abs()
        .ok_or_else(|| execution("fee absolute notional overflow"))?;
    let numerator = mul(notional, i128::from(ppm), "fee numerator")?;
    let q = numerator / FEE_DENOMINATOR;
    let r = numerator % FEE_DENOMINATOR;
    let fee = add(q, i128::from(r >= FEE_DENOMINATOR / 2), "fee rounding")?;
    let remainder = sub(
        numerator,
        mul(fee, FEE_DENOMINATOR, "fee residual")?,
        "fee remainder",
    )?;
    totals.rounded = add(totals.rounded, fee, "rounded fee sum")?;
    totals.numerators = add(totals.numerators, numerator, "fee numerator sum")?;
    totals.signed = add(totals.signed, remainder, "signed fee residual sum")?;
    totals.absolute = add(
        totals.absolute,
        remainder.abs(),
        "absolute fee residual sum",
    )?;
    residuals.max_fee_picounits = residuals
        .max_fee_picounits
        .max(remainder.abs() as f64 / FEE_DENOMINATOR as f64);
    Ok(())
}
fn quantize_fee(rate: f64) -> Result<FeeQuantization, ComputeError> {
    let ppm = (rate * MICROS).round();
    if !rate.is_finite() || rate < 0.0 || !ppm.is_finite() || ppm < 0.0 || ppm > u32::MAX as f64 {
        return Err(invalid(
            "fee fraction cannot be represented by nonnegative u32 ppm",
        ));
    }
    let parts_per_million = ppm as u32;
    let ledger_rate = parts_per_million as f64 / MICROS;
    Ok(FeeQuantization {
        requested_rate: rate,
        parts_per_million,
        ledger_rate,
        rate_residual: rate - ledger_rate,
    })
}
fn round_i64(scaled: f64, name: &str) -> Result<i64, ComputeError> {
    let rounded = scaled.round();
    if !rounded.is_finite() || rounded < i64::MIN as f64 || rounded >= -(i64::MIN as f64) {
        return Err(invalid(&format!("{name} outside exact cast bounds")));
    }
    Ok(rounded as i64)
}
fn round_i128(scaled: f64, name: &str) -> Result<i128, ComputeError> {
    let rounded = scaled.round();
    if !rounded.is_finite() || rounded < i128::MIN as f64 || rounded >= -(i128::MIN as f64) {
        return Err(invalid(&format!("{name} outside exact cast bounds")));
    }
    Ok(rounded as i128)
}
fn add(a: i128, b: i128, name: &str) -> Result<i128, ComputeError> {
    a.checked_add(b)
        .ok_or_else(|| execution(&format!("{name} overflow")))
}
fn sub(a: i128, b: i128, name: &str) -> Result<i128, ComputeError> {
    a.checked_sub(b)
        .ok_or_else(|| execution(&format!("{name} overflow")))
}
fn mul(a: i128, b: i128, name: &str) -> Result<i128, ComputeError> {
    a.checked_mul(b)
        .ok_or_else(|| execution(&format!("{name} overflow")))
}
fn invalid(message: &str) -> ComputeError {
    ComputeError::InvalidInput(message.into())
}
fn execution(message: &str) -> ComputeError {
    ComputeError::Execution(message.into())
}
fn ledger_error(error: crate::hedge_ledger::HedgeLedgerError) -> ComputeError {
    execution(&format!("exact hedge ledger: {error}"))
}
