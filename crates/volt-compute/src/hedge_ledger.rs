//! Self-financing research hedge accounting, independent of Volt's lifecycle.
//!
//! Prices use Volt currency micros; quantities use millionths of a quoted unit.
//! Their product is currency picounits and is kept exactly in checked i128. Costs
//! and interval financing round to the nearest picounit, ties away from zero;
//! their rational residuals are journaled. No floating-point conversion occurs.
//! A caller converting cash to Volt micros receives the explicit residual.
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use volt::{Currency, PriceMicros, TimestampNs, UnderlyingId};

pub const QUANTITY_SCALE: i64 = 1_000_000;
pub const PICOS_PER_MICRO: i128 = 1_000_000;
pub const HEDGE_LEDGER_SCHEMA: &str = "volt.compute.hedge_ledger.v1";

#[derive(Debug, thiserror::Error)]
pub enum HedgeLedgerError {
    #[error("invalid hedge ledger input: {0}")]
    InvalidInput(String),
    #[error("hedge ledger integer overflow: {0}")]
    Overflow(&'static str),
    #[error("hedge ledger self-financing reconciliation failed")]
    Reconciliation,
}
type Result<T> = std::result::Result<T, HedgeLedgerError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QuantityMicros(#[serde(with = "i64_decimal")] pub i64);

/// Both instruments are funded assets quoted in currency micros per unit.
/// A variance claim pays integrated (not annualized) variance in currency units
/// at terminal time. A conventional zero-value swap is not supported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HedgeInstrumentKind {
    Stock {
        underlying: UnderlyingId,
    },
    FundedVarianceClaim {
        underlying: UnderlyingId,
        maturity: TimestampNs,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HedgeInstrument {
    pub instrument_id: String,
    pub currency: Currency,
    pub kind: HedgeInstrumentKind,
    /// Positive quantity-micro increment; 1 allows millionths, 1_000_000 whole units.
    pub quantity_step: QuantityMicros,
    /// Positive economic units per quoted instrument; applied exactly once.
    pub multiplier: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HedgeLedgerConfig {
    pub currency: Currency,
    pub terminal_time: TimestampNs,
    pub instruments: Vec<HedgeInstrument>,
}

/// Atomic point-in-time marks of exactly the declared complete instrument set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HedgeMarketSnapshot {
    pub snapshot_id: String,
    pub as_of: TimestampNs,
    pub currency: Currency,
    pub prices: BTreeMap<String, PriceMicros>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccrualFactor {
    /// Gross cash growth over the declared interval, not an annualized rate.
    pub numerator: u64,
    pub denominator: u64,
}
impl AccrualFactor {
    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };
}

/// Caller supplies exact interval factors from a separately declared rate model.
/// Positive cash earns lending; negative cash grows under borrowing. Factors
/// below one are supported (negative rates). Zero factors are rejected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntervalAccrual {
    pub source_id: String,
    pub from: TimestampNs,
    pub to: TimestampNs,
    pub lending: AccrualFactor,
    pub borrowing: AccrualFactor,
}
impl IntervalAccrual {
    pub fn zero_rate(from: TimestampNs, to: TimestampNs) -> Self {
        Self {
            source_id: "zero_rate.v1".into(),
            from,
            to,
            lending: AccrualFactor::ONE,
            borrowing: AccrualFactor::ONE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProportionalCost {
    /// Cost = abs(quoted fill notional) * ppm / 1_000_000.
    pub parts_per_million: u32,
}
impl ProportionalCost {
    pub const ZERO: Self = Self {
        parts_per_million: 0,
    };
}

/// Exact unrounded numerator = rounded result * denominator + remainder.
/// The remainder may be negative after rounding up. It is not a cash flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RationalResidual {
    #[serde(with = "i128_decimal")]
    pub numerator: i128,
    pub denominator: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HedgeTrade {
    pub instrument_id: String,
    pub before: QuantityMicros,
    pub change: QuantityMicros,
    pub after: QuantityMicros,
    pub price: PriceMicros,
    pub multiplier: u32,
    /// Signed cash paid to buy the change; negative for sale proceeds.
    #[serde(with = "i128_decimal")]
    pub notional_picos: i128,
    #[serde(with = "i128_decimal")]
    pub cost_picos: i128,
    pub cost_rounding: RationalResidual,
}

/// Signed net cash paid for the liability: positive pays, negative receives.
/// This explicitly declared cash flow does not exercise/mutate a Volt contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiabilityCashflow {
    pub settlement_id: String,
    pub currency: Currency,
    #[serde(with = "i128_decimal")]
    pub amount_micros: i128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HedgeLedgerState {
    pub as_of: TimestampNs,
    #[serde(with = "i128_decimal")]
    pub cash_picos: i128,
    pub holdings: BTreeMap<String, QuantityMicros>,
    #[serde(with = "i128_decimal")]
    pub market_value_picos: i128,
    #[serde(with = "i128_decimal")]
    pub net_wealth_picos: i128,
    #[serde(with = "i128_decimal")]
    pub cumulative_gain_picos: i128,
    #[serde(with = "i128_decimal")]
    pub cumulative_interest_picos: i128,
    #[serde(with = "i128_decimal")]
    pub cumulative_cost_picos: i128,
    #[serde(with = "i128_decimal")]
    pub cumulative_liability_picos: i128,
    pub settled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HedgeJournalEntry {
    pub schema_version: String,
    pub sequence: u64,
    pub market: HedgeMarketSnapshot,
    pub accrual: IntervalAccrual,
    pub cost_model: ProportionalCost,
    #[serde(with = "i128_decimal")]
    pub interest_picos: i128,
    pub financing_rounding: RationalResidual,
    #[serde(with = "i128_decimal")]
    pub mark_gain_picos: i128,
    pub trades: Vec<HedgeTrade>,
    pub liability: Option<LiabilityCashflow>,
    pub before: HedgeLedgerState,
    pub after: HedgeLedgerState,
    /// Initial cash + gains + interest - costs - liability; equals after wealth.
    #[serde(with = "i128_decimal")]
    pub reconciled_wealth_picos: i128,
}

/// Checked application ledger; mutation is transactional and journals are ordered
/// by instrument ID. Replays must reuse declared marks, factors and costs.
#[derive(Debug, Clone)]
pub struct HedgeLedger {
    config: HedgeLedgerConfig,
    instruments: BTreeMap<String, HedgeInstrument>,
    initial_cash_picos: i128,
    market: HedgeMarketSnapshot,
    state: HedgeLedgerState,
    journal: Vec<HedgeJournalEntry>,
    snapshot_ids: BTreeSet<String>,
}

impl HedgeLedger {
    pub fn new(
        config: HedgeLedgerConfig,
        market: HedgeMarketSnapshot,
        initial_cash_micros: i128,
    ) -> Result<Self> {
        nonempty(&config.currency.0, "currency")?;
        if config.instruments.is_empty()
            || market.as_of.0 < 0
            || config.terminal_time <= market.as_of
        {
            return invalid(
                "instruments must be nonempty and terminal time after nonnegative inception",
            );
        }
        let mut instruments = BTreeMap::new();
        for instrument in &config.instruments {
            nonempty(&instrument.instrument_id, "instrument ID")?;
            if instrument.currency != config.currency
                || instrument.quantity_step.0 <= 0
                || instrument.multiplier == 0
            {
                return invalid("instrument currency, quantity step or multiplier is invalid");
            }
            let underlying = match &instrument.kind {
                HedgeInstrumentKind::Stock { underlying } => underlying,
                HedgeInstrumentKind::FundedVarianceClaim {
                    underlying,
                    maturity,
                } => {
                    if *maturity != config.terminal_time {
                        return invalid("funded claim maturity must equal ledger terminal time");
                    }
                    underlying
                }
            };
            nonempty(&underlying.0, "underlying")?;
            if instruments
                .insert(instrument.instrument_id.clone(), instrument.clone())
                .is_some()
            {
                return invalid("duplicate instrument ID");
            }
        }
        validate_market(&config.currency, &instruments, &market)?;
        let initial_cash_picos = mul(initial_cash_micros, PICOS_PER_MICRO, "initial cash")?;
        let holdings = instruments
            .keys()
            .map(|k| (k.clone(), QuantityMicros(0)))
            .collect();
        let state = HedgeLedgerState {
            as_of: market.as_of,
            cash_picos: initial_cash_picos,
            holdings,
            market_value_picos: 0,
            net_wealth_picos: initial_cash_picos,
            cumulative_gain_picos: 0,
            cumulative_interest_picos: 0,
            cumulative_cost_picos: 0,
            cumulative_liability_picos: 0,
            settled: false,
        };
        Ok(Self {
            config,
            instruments,
            initial_cash_picos,
            market,
            state,
            journal: vec![],
            snapshot_ids: BTreeSet::new(),
        })
    }

    pub fn state(&self) -> &HedgeLedgerState {
        &self.state
    }
    pub fn journal(&self) -> &[HedgeJournalEntry] {
        &self.journal
    }
    pub fn config(&self) -> &HedgeLedgerConfig {
        &self.config
    }

    /// Rounded Volt-compatible cash micros and the exact leftover picounits.
    pub fn cash_micros(&self) -> Result<(i128, RationalResidual)> {
        round_ratio(self.state.cash_picos, PICOS_PER_MICRO as u64)
    }

    /// Targets are absolute post-trade holdings, with exactly one entry per
    /// declared instrument. The first rebalance may share inception time; every
    /// subsequent event must have a strictly later timestamp and unique ID.
    pub fn rebalance(
        &mut self,
        market: HedgeMarketSnapshot,
        targets: BTreeMap<String, QuantityMicros>,
        accrual: IntervalAccrual,
        cost: ProportionalCost,
    ) -> Result<&HedgeJournalEntry> {
        self.apply(market, targets, accrual, cost, None)
    }

    /// At exactly terminal time, sell/buy all inventory at declared marks and
    /// pay the signed liability. Pass ZERO cost to match paper terminal handling.
    pub fn settle(
        &mut self,
        market: HedgeMarketSnapshot,
        liability: LiabilityCashflow,
        accrual: IntervalAccrual,
        terminal_cost: ProportionalCost,
    ) -> Result<&HedgeJournalEntry> {
        let targets = self
            .instruments
            .keys()
            .map(|k| (k.clone(), QuantityMicros(0)))
            .collect();
        self.apply(market, targets, accrual, terminal_cost, Some(liability))
    }

    fn apply(
        &mut self,
        market: HedgeMarketSnapshot,
        targets: BTreeMap<String, QuantityMicros>,
        accrual: IntervalAccrual,
        cost_model: ProportionalCost,
        liability: Option<LiabilityCashflow>,
    ) -> Result<&HedgeJournalEntry> {
        if self.state.settled {
            return invalid("ledger already settled");
        }
        validate_market(&self.config.currency, &self.instruments, &market)?;
        if market.as_of < self.state.as_of
            || (!self.journal.is_empty() && market.as_of <= self.state.as_of)
            || market.as_of > self.config.terminal_time
            || self.snapshot_ids.contains(&market.snapshot_id)
        {
            return invalid("event time or snapshot identity is invalid");
        }
        let terminal = liability.is_some();
        if (terminal && market.as_of != self.config.terminal_time)
            || (!terminal && market.as_of >= self.config.terminal_time)
        {
            return invalid("use settlement only at terminal time and rebalance before it");
        }
        if targets.len() != self.instruments.len() || targets.keys().ne(self.instruments.keys()) {
            return invalid("targets must cover exactly all instruments");
        }
        nonempty(&accrual.source_id, "accrual source")?;
        if accrual.from != self.state.as_of || accrual.to != market.as_of {
            return invalid("accrual interval does not match event interval");
        }
        for factor in [accrual.lending, accrual.borrowing] {
            if factor.numerator == 0
                || factor.denominator == 0
                || (accrual.from == accrual.to && factor.numerator != factor.denominator)
            {
                return invalid(
                    "gross factors must be positive and zero intervals have unit growth",
                );
            }
        }
        if let Some(flow) = &liability {
            nonempty(&flow.settlement_id, "settlement ID")?;
            if flow.currency != self.config.currency {
                return invalid("liability currency mismatch");
            }
        }
        let factor = if self.state.cash_picos < 0 {
            accrual.borrowing
        } else {
            accrual.lending
        };
        let (mut cash, financing_rounding) = round_ratio(
            mul(
                self.state.cash_picos,
                i128::from(factor.numerator),
                "financing",
            )?,
            factor.denominator,
        )?;
        let interest = sub(cash, self.state.cash_picos, "interest")?;
        let mut gain = 0;
        let mut costs = 0;
        let mut market_value = 0;
        let mut trades = Vec::new();
        for (id, instrument) in &self.instruments {
            let before = self.state.holdings[id];
            let after = targets[id];
            if after.0 % instrument.quantity_step.0 != 0 {
                return invalid("target is not an exact multiple of instrument quantity step");
            }
            let change = QuantityMicros(
                after
                    .0
                    .checked_sub(before.0)
                    .ok_or(HedgeLedgerError::Overflow("quantity change"))?,
            );
            let price = market.prices[id];
            let price_change = i128::from(price.0) - i128::from(self.market.prices[id].0);
            gain = add(
                gain,
                notional(before, price_change, instrument.multiplier)?,
                "mark gains",
            )?;
            market_value = add(
                market_value,
                notional(after, i128::from(price.0), instrument.multiplier)?,
                "market value",
            )?;
            if change.0 == 0 {
                continue;
            }
            let paid = notional(change, i128::from(price.0), instrument.multiplier)?;
            let absolute = paid
                .checked_abs()
                .ok_or(HedgeLedgerError::Overflow("absolute notional"))?;
            let (fee, cost_rounding) = round_ratio(
                mul(
                    absolute,
                    i128::from(cost_model.parts_per_million),
                    "transaction cost",
                )?,
                1_000_000,
            )?;
            cash = sub(sub(cash, paid, "trade cash")?, fee, "cost cash")?;
            costs = add(costs, fee, "total costs")?;
            trades.push(HedgeTrade {
                instrument_id: id.clone(),
                before,
                change,
                after,
                price,
                multiplier: instrument.multiplier,
                notional_picos: paid,
                cost_picos: fee,
                cost_rounding,
            });
        }
        let liability_picos = liability
            .as_ref()
            .map(|x| mul(x.amount_micros, PICOS_PER_MICRO, "liability"))
            .transpose()?
            .unwrap_or(0);
        cash = sub(cash, liability_picos, "liability cash")?;
        let after = HedgeLedgerState {
            as_of: market.as_of,
            cash_picos: cash,
            holdings: targets,
            market_value_picos: market_value,
            net_wealth_picos: add(cash, market_value, "net wealth")?,
            cumulative_gain_picos: add(self.state.cumulative_gain_picos, gain, "cumulative gain")?,
            cumulative_interest_picos: add(
                self.state.cumulative_interest_picos,
                interest,
                "cumulative interest",
            )?,
            cumulative_cost_picos: add(self.state.cumulative_cost_picos, costs, "cumulative cost")?,
            cumulative_liability_picos: add(
                self.state.cumulative_liability_picos,
                liability_picos,
                "cumulative liability",
            )?,
            settled: terminal,
        };
        let reconciled = sub(
            sub(
                add(
                    add(
                        self.initial_cash_picos,
                        after.cumulative_gain_picos,
                        "reconciliation gains",
                    )?,
                    after.cumulative_interest_picos,
                    "reconciliation interest",
                )?,
                after.cumulative_cost_picos,
                "reconciliation costs",
            )?,
            after.cumulative_liability_picos,
            "reconciliation liability",
        )?;
        if reconciled != after.net_wealth_picos {
            return Err(HedgeLedgerError::Reconciliation);
        }
        let sequence = u64::try_from(self.journal.len())
            .ok()
            .and_then(|x| x.checked_add(1))
            .ok_or(HedgeLedgerError::Overflow("sequence"))?;
        let entry = HedgeJournalEntry {
            schema_version: HEDGE_LEDGER_SCHEMA.into(),
            sequence,
            market: market.clone(),
            accrual,
            cost_model,
            interest_picos: interest,
            financing_rounding,
            mark_gain_picos: gain,
            trades,
            liability,
            before: self.state.clone(),
            after: after.clone(),
            reconciled_wealth_picos: reconciled,
        };
        self.state = after;
        self.snapshot_ids.insert(market.snapshot_id.clone());
        self.market = market;
        self.journal.push(entry);
        Ok(self.journal.last().expect("journal just appended"))
    }
}

fn validate_market(
    currency: &Currency,
    instruments: &BTreeMap<String, HedgeInstrument>,
    market: &HedgeMarketSnapshot,
) -> Result<()> {
    nonempty(&market.snapshot_id, "snapshot ID")?;
    if &market.currency != currency
        || market.as_of.0 < 0
        || market.prices.len() != instruments.len()
        || market.prices.keys().ne(instruments.keys())
        || market.prices.values().any(|x| x.0 < 0)
    {
        return invalid("market requires same currency and complete nonnegative funded marks");
    }
    for (id, instrument) in instruments {
        if matches!(instrument.kind, HedgeInstrumentKind::Stock { .. }) && market.prices[id].0 == 0
        {
            return invalid(
                "stock marks must be strictly positive; funded variance claims may be zero",
            );
        }
    }
    Ok(())
}
fn nonempty(value: &str, name: &str) -> Result<()> {
    if value.trim().is_empty() {
        invalid(&format!("{name} must not be empty"))
    } else {
        Ok(())
    }
}
fn invalid<T>(message: &str) -> Result<T> {
    Err(HedgeLedgerError::InvalidInput(message.into()))
}
fn add(a: i128, b: i128, what: &'static str) -> Result<i128> {
    a.checked_add(b).ok_or(HedgeLedgerError::Overflow(what))
}
fn sub(a: i128, b: i128, what: &'static str) -> Result<i128> {
    a.checked_sub(b).ok_or(HedgeLedgerError::Overflow(what))
}
fn mul(a: i128, b: i128, what: &'static str) -> Result<i128> {
    a.checked_mul(b).ok_or(HedgeLedgerError::Overflow(what))
}
fn notional(quantity: QuantityMicros, price: i128, multiplier: u32) -> Result<i128> {
    mul(
        mul(i128::from(quantity.0), price, "price times quantity")?,
        i128::from(multiplier),
        "instrument multiplier",
    )
}
fn round_ratio(numerator: i128, denominator: u64) -> Result<(i128, RationalResidual)> {
    if denominator == 0 {
        return invalid("zero rational denominator");
    }
    let den = i128::from(denominator);
    let q = numerator / den;
    let r = numerator % den;
    let adjustment = if r.abs() >= (den + 1) / 2 {
        r.signum()
    } else {
        0
    };
    let rounded = add(q, adjustment, "rational rounding")?;
    Ok((
        rounded,
        RationalResidual {
            numerator: r - adjustment * den,
            denominator,
        },
    ))
}

// Decimal strings prevent JSON consumers from losing exact integer cash above 2^53.
mod i128_decimal {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(
        value: &i128,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<i128, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

mod i64_decimal {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(
        value: &i64,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<i64, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}
