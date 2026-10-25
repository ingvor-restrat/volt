//! European straddle mechanism study, native scalar f64. Physical zero-drift
//! GBM sigma_P and Black--Scholes quote sigma_Q are separate model inputs.
//! Coarse hedge grids SUBSAMPLE one finest Brownian path; changing inventory,
//! quote volatility, costs, batching or hedge frequency never redraws markets.
//! Cash is self-financing at r=q=0. No option spread, funding/margin constraint,
//! empirical forecast or trading-edge claim is inferred from simulated profit.
use crate::{
    ComputeError, OptionRow, OptionSide,
    aad::analytic_greeks,
    chunked::CompensatedSum,
    hedge_application::{HedgeCostConfig, HedgePathBatch},
    hedge_band::banded_holdings,
    hedge_reconciliation::{AuditModelFamily, ReconciliationAudit, reconcile_path},
    monte_carlo::normal_at,
    risk_objective::{RiskObjective, weighted_risk},
    second_order::analytic_gamma,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

const PURPOSE: u64 = 0x5653_4742_4d30_3031;
const RNG: &str = "volt.straddle.gbm.v1:normal_at(seed XOR 0x565347424d303031,path_id*finest_steps+step);cumulativeBrownian;zeroDrift";
const NS_YEAR: f64 = 365.0 * 86400.0 * 1e9;
fn default_audits() -> usize {
    8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StraddleStudyConfig {
    pub schema: String,
    pub initial_spot: f64,
    pub strike: f64,
    pub maturity_years: f64,
    /// Annualized physical diffusion volatility; zero is deterministic spot.
    pub physical_volatility: f64,
    /// Positive annualized BS quote volatility, held fixed during repricing.
    pub pricing_volatility: f64,
    pub quantity: f64,
    pub multiplier: u32,
    pub finest_steps: usize,
    /// Distinct positive divisors of finest_steps, in caller-declared order.
    pub hedge_steps: Vec<usize>,
    pub paths: usize,
    pub first_path_id: u64,
    pub seed: u64,
    pub batch_paths: usize,
    pub cost: HedgeCostConfig,
    /// Optional PREDECLARED width in total stock inventory units. No tuning on
    /// these paths is performed; a learned validation choice is a caller input.
    pub band_width: Option<f64>,
    pub max_working_bytes: u64,
    #[serde(default = "default_audits")]
    pub audit_paths: usize,
}
impl Default for StraddleStudyConfig {
    fn default() -> Self {
        Self {
            schema: "volt.compute.straddle_study.v1".into(),
            initial_spot: 100.0,
            strike: 100.0,
            maturity_years: 0.25,
            physical_volatility: 0.25,
            pricing_volatility: 0.20,
            quantity: 1.0,
            multiplier: 1,
            finest_steps: 24,
            hedge_steps: vec![6, 12, 24],
            paths: 1024,
            first_path_id: 0,
            seed: 77119,
            batch_paths: 128,
            cost: HedgeCostConfig::default(),
            band_width: None,
            max_working_bytes: 128 * 1024 * 1024,
            audit_paths: 8,
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct StraddleStudyPlan {
    pub signed_underlying_units: f64,
    pub cases: usize,
    pub retained_pnl_bytes: u64,
    pub statistic_workspace_bytes: u64,
    pub path_and_audit_workspace_bytes: u64,
    /// Conservative admission model, not an allocator or OS reservation.
    pub modeled_working_bytes: u64,
}
impl StraddleStudyConfig {
    pub fn validate(&self) -> Result<StraddleStudyPlan, ComputeError> {
        let units = self.quantity * f64::from(self.multiplier);
        let max_step = self.hedge_steps.iter().copied().max().unwrap_or(0);
        if self.schema != "volt.compute.straddle_study.v1"
            || self.paths < 2
            || self.batch_paths == 0
            || self.batch_paths > self.paths
            || self.finest_steps == 0
            || self.finest_steps > 4096
            || self.hedge_steps.is_empty()
            || self.hedge_steps.len() > 32
            || self
                .hedge_steps
                .iter()
                .any(|n| *n == 0 || !self.finest_steps.is_multiple_of(*n))
            || self
                .hedge_steps
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.hedge_steps.len()
            || !self.initial_spot.is_finite()
            || self.initial_spot <= 0.0
            || !self.strike.is_finite()
            || self.strike <= 0.0
            || !self.maturity_years.is_finite()
            || self.maturity_years <= 0.0
            || self.maturity_years * NS_YEAR >= i64::MAX as f64
            || self.maturity_years / max_step.max(1) as f64 * NS_YEAR < 1.0
            || !self.physical_volatility.is_finite()
            || self.physical_volatility < 0.0
            || !self.pricing_volatility.is_finite()
            || self.pricing_volatility <= 0.0
            || !units.is_finite()
            || units == 0.0
            || units.abs() > 1e9
            || self.multiplier == 0
            || self.audit_paths > 8
            || self.audit_paths > self.paths
            || [self.cost.proportional, self.cost.terminal_proportional]
                .iter()
                .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
            || self
                .band_width
                .is_some_and(|w| !w.is_finite() || w < 0.0 || w > 1e9)
        {
            return Err(invalid(
                "invalid straddle model, grid, position, fee, path or audit configuration",
            ));
        }
        let end = self
            .first_path_id
            .checked_add(self.paths as u64)
            .ok_or_else(|| invalid("path ID overflow"))?;
        if end
            .checked_mul(self.finest_steps as u64)
            .is_none_or(|n| n > 1u64 << 63)
        {
            return Err(invalid(
                "normal_at counter range exceeds unique 63-bit stream",
            ));
        }
        let cases = self
            .hedge_steps
            .len()
            .checked_mul(if self.band_width.is_some() { 3 } else { 2 })
            .ok_or_else(|| invalid("case count overflow"))?;
        let size = |x: usize, n: u64| {
            (x as u64)
                .checked_mul(n)
                .ok_or_else(|| invalid("memory size overflow"))
        };
        let retained_pnl_bytes = size(self.paths, 8)?
            .checked_mul(cases as u64)
            .ok_or_else(|| invalid("retained PNL overflow"))?;
        // Risk sorting/probabilities, quantile copy, and native serialization.
        let statistic_workspace_bytes = size(self.paths, 192)?;
        let path_and_audit_workspace_bytes =
            size(self.batch_paths, (self.finest_steps as u64 + 1) * 8)?
                .checked_add(size(self.finest_steps, 512)?)
                .and_then(|n| n.checked_add((max_step as u64) * 16384))
                .and_then(|n| n.checked_add((cases as u64) * 16384 + 131072))
                .ok_or_else(|| invalid("working size overflow"))?;
        let modeled_working_bytes = retained_pnl_bytes
            .checked_add(statistic_workspace_bytes)
            .and_then(|n| n.checked_add(path_and_audit_workspace_bytes))
            .ok_or_else(|| invalid("total size overflow"))?;
        if modeled_working_bytes > self.max_working_bytes {
            return Err(ComputeError::BudgetExceeded(format!(
                "straddle study needs modeled {modeled_working_bytes} bytes, allowance {}",
                self.max_working_bytes
            )));
        }
        quote_straddle(
            self.initial_spot,
            self.strike,
            self.maturity_years,
            self.pricing_volatility,
        )?;
        Ok(StraddleStudyPlan {
            signed_underlying_units: units,
            cases,
            retained_pnl_bytes,
            statistic_workspace_bytes,
            path_and_audit_workspace_bytes,
            modeled_working_bytes,
        })
    }
}
#[derive(Debug, Clone, Copy, Serialize)]
pub struct StraddleQuote {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
}
/// Per underlying unit; theta is calendar currency/year and vega per unit of
/// annual volatility fraction. Terminal payoff is separate, no terminal Greek.
pub fn quote_straddle(
    spot: f64,
    strike: f64,
    time: f64,
    volatility: f64,
) -> Result<StraddleQuote, ComputeError> {
    let row = OptionRow {
        row_id: "straddle-quote".into(),
        contract_id: "same-expiry-call-put".into(),
        side: OptionSide::Call,
        spot,
        strike,
        maturity_years: time,
        volatility,
        rate: 0.0,
        dividend_yield: 0.0,
        quantity: 1,
        multiplier: 1,
    };
    let call = analytic_greeks(&row)?;
    let put = analytic_greeks(&OptionRow {
        side: OptionSide::Put,
        ..row.clone()
    })?;
    Ok(StraddleQuote {
        price: finite(call.price + put.price)?,
        delta: finite(call.delta + put.delta)?,
        gamma: finite(2.0 * analytic_gamma(&row)?)?,
        theta: finite(call.theta + put.theta)?,
        vega: finite(call.vega + put.vega)?,
    })
}
/// Actual fine-grid spot values. Coarse observations must select indices from
/// this path, rather than regenerating normals at another hedge step count.
pub fn fine_gbm_path(config: &StraddleStudyConfig, path_id: u64) -> Result<Vec<f64>, ComputeError> {
    config.validate()?;
    let end = config.first_path_id + config.paths as u64;
    if !(config.first_path_id..end).contains(&path_id) {
        return Err(invalid("path ID outside admitted cohort"));
    }
    fine_path(config, path_id)
}
fn fine_path(c: &StraddleStudyConfig, id: u64) -> Result<Vec<f64>, ComputeError> {
    let dt = c.maturity_years / c.finest_steps as f64;
    let mut w = 0.0;
    let mut result = Vec::with_capacity(c.finest_steps + 1);
    result.push(c.initial_spot);
    for step in 0..c.finest_steps {
        w += dt.sqrt() * normal_at(c.seed ^ PURPOSE, id * c.finest_steps as u64 + step as u64);
        let t = (step + 1) as f64 * dt;
        let spot = finite(
            c.initial_spot
                * (-0.5 * c.physical_volatility * c.physical_volatility * t
                    + c.physical_volatility * w)
                    .exp(),
        )?;
        if spot <= 0.0 {
            return Err(invalid("physical GBM stock underflow"));
        }
        result.push(spot);
    }
    Ok(result)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StraddleHedge {
    Delta,
    NoHedge,
    FixedBand,
}
#[derive(Debug, Clone, Serialize)]
pub struct StraddlePathResult {
    pub path_id: u64,
    pub hedge_steps: usize,
    pub strategy: StraddleHedge,
    pub terminal_spot: f64,
    pub pnl: f64,
    pub option_mark_gain: f64,
    pub stock_mark_gain: f64,
    pub transaction_cost: f64,
    pub turnover: f64,
    pub terminal_turnover: f64,
    pub theta_approximation: f64,
    pub gamma_approximation: f64,
    pub delta_approximation: f64,
    pub higher_order_residual: f64,
    pub inventory_tracking_gain: f64,
    pub continuous_variance_drift_approximation: f64,
    /// Annualized fine-grid realized log-return variance, common across hedge grids.
    pub realized_variance: f64,
    pub hedge_grid_realized_variance: f64,
    pub initial_cash: f64,
    pub terminal_cash: f64,
    pub cash_identity_error: f64,
    pub attribution_identity_error: f64,
}
/// Caller replay helper for one already observed fine path; it does not claim
/// that explicit caller values were sampled from the configured physical law.
pub fn evaluate_straddle_path(
    c: &StraddleStudyConfig,
    id: u64,
    fine: &[f64],
    steps: usize,
    strategy: StraddleHedge,
) -> Result<StraddlePathResult, ComputeError> {
    c.validate()?;
    Ok(calculate_path(c, id, fine, steps, strategy)?.0)
}
fn calculate_path(
    c: &StraddleStudyConfig,
    id: u64,
    fine: &[f64],
    steps: usize,
    strategy: StraddleHedge,
) -> Result<(StraddlePathResult, Vec<f64>), ComputeError> {
    if fine.len() != c.finest_steps + 1
        || !c.hedge_steps.contains(&steps)
        || fine.iter().any(|s| !s.is_finite() || *s <= 0.0)
        || fine[0].to_bits() != c.initial_spot.to_bits()
        || (strategy == StraddleHedge::FixedBand && c.band_width.is_none())
    {
        return Err(invalid("invalid explicit fine path, hedge grid or band"));
    }
    let units = c.quantity * f64::from(c.multiplier);
    let dt = c.maturity_years / steps as f64;
    let stride = c.finest_steps / steps;
    let initial = quote_straddle(
        c.initial_spot,
        c.strike,
        c.maturity_years,
        c.pricing_volatility,
    )?
    .price;
    let mut quotes = Vec::with_capacity(steps);
    let mut targets = Vec::with_capacity(steps);
    for k in 0..steps {
        let q = quote_straddle(
            fine[k * stride],
            c.strike,
            (steps - k) as f64 * dt,
            c.pricing_volatility,
        )?;
        targets.push(finite(-units * q.delta)?);
        quotes.push(q);
    }
    let holdings = match strategy {
        StraddleHedge::Delta => targets,
        StraddleHedge::NoHedge => vec![0.0; steps],
        StraddleHedge::FixedBand => banded_holdings(
            &targets,
            1,
            steps,
            &[c.band_width.unwrap()],
            c.max_working_bytes,
        )?,
    };
    let terminal = fine[c.finest_steps];
    let payoff = (terminal - c.strike).abs();
    let mut stock = CompensatedSum::default();
    let mut fees = CompensatedSum::default();
    let mut turnover = CompensatedSum::default();
    let mut delta = CompensatedSum::default();
    let mut theta = CompensatedSum::default();
    let mut gamma = CompensatedSum::default();
    let mut residual = CompensatedSum::default();
    let mut tracking = CompensatedSum::default();
    let mut continuous = CompensatedSum::default();
    let mut rv = CompensatedSum::default();
    let mut cash = CompensatedSum::default();
    cash.add(-units * initial);
    let mut previous = 0.0;
    for k in 0..steps {
        let s = fine[k * stride];
        let next = fine[(k + 1) * stride];
        let ds = next - s;
        let h = holdings[k];
        let q = quotes[k];
        let next_price = if k + 1 == steps {
            payoff
        } else {
            quotes[k + 1].price
        };
        let notional = (h - previous) * s;
        let fee = c.cost.proportional * notional.abs();
        cash.add(-notional);
        cash.add(-fee);
        fees.add(fee);
        turnover.add(notional.abs());
        stock.add(h * ds);
        delta.add(units * q.delta * ds);
        theta.add(units * q.theta * dt);
        gamma.add(0.5 * units * q.gamma * ds * ds);
        residual.add(
            units
                * ((next_price - q.price) - q.delta * ds - q.theta * dt - 0.5 * q.gamma * ds * ds),
        );
        tracking.add((h + units * q.delta) * ds);
        continuous.add(
            0.5 * units
                * q.gamma
                * s
                * s
                * (c.physical_volatility * c.physical_volatility
                    - c.pricing_volatility * c.pricing_volatility)
                * dt,
        );
        rv.add((next / s).ln().powi(2));
        previous = h;
    }
    let terminal_turnover = previous.abs() * terminal;
    let terminal_fee = c.cost.terminal_proportional * terminal_turnover;
    turnover.add(terminal_turnover);
    fees.add(terminal_fee);
    cash.add(previous * terminal);
    cash.add(-terminal_fee);
    cash.add(units * payoff);
    let stock = stock.value();
    let fees = fees.value();
    let turnover = turnover.value();
    let delta = delta.value();
    let theta = theta.value();
    let gamma = gamma.value();
    let residual = residual.value();
    let tracking = tracking.value();
    let continuous = continuous.value();
    let rv = rv.value();
    let cash = cash.value();
    let option_gain = units * (payoff - initial);
    let pnl = finite(option_gain + stock - fees)?;
    let attribution = theta + gamma + residual + tracking - fees;
    let cash_error = cash - pnl;
    let attribution_error = attribution - pnl;
    // Accumulated floating summation bound, independent of ledger quantization.
    let work = units.abs() * (initial + payoff)
        + turnover
        + fees
        + stock.abs()
        + delta.abs()
        + theta.abs()
        + gamma.abs()
        + residual.abs()
        + tracking.abs();
    let bound = 512.0 * (steps + 1) as f64 * f64::EPSILON * work.max(1.0);
    if !cash_error.is_finite()
        || !attribution_error.is_finite()
        || cash_error.abs() > bound
        || attribution_error.abs() > bound
    {
        return Err(ComputeError::Execution(
            "straddle self-financing/attribution identity failed".into(),
        ));
    }
    Ok((
        StraddlePathResult {
            path_id: id,
            hedge_steps: steps,
            strategy,
            terminal_spot: terminal,
            pnl,
            option_mark_gain: finite(option_gain)?,
            stock_mark_gain: finite(stock)?,
            transaction_cost: finite(fees)?,
            turnover: finite(turnover)?,
            terminal_turnover: finite(terminal_turnover)?,
            theta_approximation: finite(theta)?,
            gamma_approximation: finite(gamma)?,
            delta_approximation: finite(delta)?,
            higher_order_residual: finite(residual)?,
            inventory_tracking_gain: finite(tracking)?,
            continuous_variance_drift_approximation: finite(continuous)?,
            realized_variance: finite(
                fine.windows(2)
                    .map(|p| (p[1] / p[0]).ln().powi(2))
                    .sum::<f64>()
                    / c.maturity_years,
            )?,
            hedge_grid_realized_variance: finite(rv / c.maturity_years)?,
            initial_cash: finite(-units * initial)?,
            terminal_cash: finite(cash)?,
            cash_identity_error: cash_error,
            attribution_identity_error: attribution_error,
        },
        holdings,
    ))
}
#[derive(Debug, Clone, Serialize)]
pub struct StraddleAudit {
    pub path_id: u64,
    pub exact_reconciliation: bool,
    pub float_identity_within_bound: bool,
    pub quantization_within_bound: bool,
    pub exact_ledger_pnl_picos: String,
    pub quantization_gap: f64,
    pub quantization_error_bound_currency: f64,
    pub requested_fee: f64,
    pub ledger_fee: f64,
    pub requested_terminal_fee: f64,
    pub ledger_terminal_fee: f64,
}
impl StraddleAudit {
    fn from_audit(id: u64, a: ReconciliationAudit) -> Result<Self, ComputeError> {
        if !a.exact_reconciliation || !a.float_identity_within_bound || !a.quantization_within_bound
        {
            return Err(ComputeError::Execution(
                "straddle exact-ledger audit failed".into(),
            ));
        }
        Ok(Self {
            path_id: id,
            exact_reconciliation: a.exact_reconciliation,
            float_identity_within_bound: a.float_identity_within_bound,
            quantization_within_bound: a.quantization_within_bound,
            exact_ledger_pnl_picos: a.exact_ledger_pnl_picos,
            quantization_gap: a.quantization_gap,
            quantization_error_bound_currency: a.quantization_error_bound_currency,
            requested_fee: a.trading_fee.requested_rate,
            ledger_fee: a.trading_fee.ledger_rate,
            requested_terminal_fee: a.terminal_fee.requested_rate,
            ledger_terminal_fee: a.terminal_fee.ledger_rate,
        })
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct StraddleCaseReport {
    pub hedge_steps: usize,
    pub strategy: StraddleHedge,
    pub mean_profit: f64,
    pub standard_error: f64,
    pub quantiles: [f64; 5],
    pub quantile_probabilities: [f64; 5],
    pub loss_cvar_95: f64,
    pub analytic_expected_profit_before_costs: f64,
    pub mean_cost: f64,
    pub mean_turnover: f64,
    pub mean_terminal_turnover: f64,
    pub mean_option_mark_gain: f64,
    pub mean_stock_mark_gain: f64,
    pub mean_theta_approximation: f64,
    pub mean_gamma_approximation: f64,
    pub mean_higher_order_residual: f64,
    pub mean_inventory_tracking_gain: f64,
    pub mean_continuous_variance_drift_approximation: f64,
    pub mean_realized_variance: f64,
    pub mean_hedge_grid_realized_variance: f64,
    pub mean_delta_approximation: f64,
    pub max_cash_identity_error: f64,
    pub max_attribution_identity_error: f64,
    pub audits: Vec<StraddleAudit>,
}
#[derive(Debug, Clone, Serialize)]
pub struct StraddleSamplesArtifact {
    pub path: PathBuf,
    pub rows: u64,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct StraddleStudyReport {
    pub schema: String,
    pub config: StraddleStudyConfig,
    pub plan: StraddleStudyPlan,
    pub execution_policy: String,
    pub rng_contract: String,
    pub source_sha256: String,
    pub lockfile_sha256: String,
    pub executable_sha256: String,
    pub recipe_sha256: String,
    pub physical_market_recipe_sha256: String,
    pub physical_market_stream_sha256: String,
    pub initial_quote_per_unit: StraddleQuote,
    pub elapsed_seconds: f64,
    pub cases: Vec<StraddleCaseReport>,
    pub samples: Option<StraddleSamplesArtifact>,
    pub interpretation: String,
}
#[derive(Default)]
struct Accumulator {
    pnl: Vec<f64>,
    sums: [CompensatedSum; 13],
    max_cash: f64,
    max_attribution: f64,
    audits: Vec<StraddleAudit>,
}
impl Accumulator {
    fn add(&mut self, p: &StraddlePathResult) {
        self.pnl.push(p.pnl);
        let values = [
            p.transaction_cost,
            p.turnover,
            p.terminal_turnover,
            p.option_mark_gain,
            p.stock_mark_gain,
            p.theta_approximation,
            p.gamma_approximation,
            p.higher_order_residual,
            p.inventory_tracking_gain,
            p.continuous_variance_drift_approximation,
            p.realized_variance,
            p.delta_approximation,
            p.hedge_grid_realized_variance,
        ];
        for (s, v) in self.sums.iter_mut().zip(values) {
            s.add(v);
        }
        self.max_cash = self.max_cash.max(p.cash_identity_error.abs());
        self.max_attribution = self.max_attribution.max(p.attribution_identity_error.abs());
    }
}
/// Pure admission before files, paths or pricing buffers are allocated. Exact
/// empirical quantiles/CVaR retain only admitted per-case PNL arrays; fine paths
/// are chunked and discarded. Sorting workspace is explicitly included.
pub fn run_straddle_study(
    c: &StraddleStudyConfig,
    output_samples: Option<&Path>,
) -> Result<StraddleStudyReport, ComputeError> {
    let start = Instant::now();
    let plan = c.validate()?;
    let quote = quote_straddle(
        c.initial_spot,
        c.strike,
        c.maturity_years,
        c.pricing_volatility,
    )?;
    let physical_price = if c.physical_volatility == 0.0 {
        (c.initial_spot - c.strike).abs()
    } else {
        quote_straddle(
            c.initial_spot,
            c.strike,
            c.maturity_years,
            c.physical_volatility,
        )?
        .price
    };
    let expected = finite(plan.signed_underlying_units * (physical_price - quote.price))?;
    let recipe = hash_json(c)?;
    let physical_recipe = hash_json(&(
        "volt.straddle.physical_market.v1",
        RNG,
        c.seed,
        c.first_path_id,
        c.paths,
        c.initial_spot,
        c.physical_volatility,
        c.maturity_years,
        c.finest_steps,
    ))?;
    let mut market_hash = Sha256::new();
    market_hash.update(b"volt.straddle.market_rows.v1\0");
    market_hash.update(physical_recipe.as_bytes());
    let mut writer = output_samples.map(AtomicSamples::new).transpose()?;
    let strategies = if c.band_width.is_some() {
        vec![
            StraddleHedge::Delta,
            StraddleHedge::NoHedge,
            StraddleHedge::FixedBand,
        ]
    } else {
        vec![StraddleHedge::Delta, StraddleHedge::NoHedge]
    };
    let mut accumulators: Vec<_> = (0..plan.cases)
        .map(|_| Accumulator {
            pnl: Vec::with_capacity(c.paths),
            ..Default::default()
        })
        .collect();
    for batch_start in (0..c.paths).step_by(c.batch_paths) {
        let count = (c.paths - batch_start).min(c.batch_paths);
        let chunk = (0..count)
            .map(|i| fine_path(c, c.first_path_id + (batch_start + i) as u64))
            .collect::<Result<Vec<_>, _>>()?;
        for (i, fine) in chunk.iter().enumerate() {
            let offset = batch_start + i;
            let id = c.first_path_id + offset as u64;
            market_hash.update(id.to_le_bytes());
            for spot in fine {
                market_hash.update(spot.to_le_bytes());
            }
            for (grid, &steps) in c.hedge_steps.iter().enumerate() {
                for (j, &strategy) in strategies.iter().enumerate() {
                    let (result, holdings) = calculate_path(c, id, fine, steps, strategy)?;
                    let accumulator = &mut accumulators[grid * strategies.len() + j];
                    accumulator.add(&result);
                    if offset < c.audit_paths {
                        let stride = c.finest_steps / steps;
                        let batch = HedgePathBatch {
                            paths: 1,
                            steps,
                            hedge_count: 1,
                            feature_count: 1,
                            marks: (0..=steps).map(|k| fine[k * stride]).collect(),
                            features: vec![0.0; steps],
                            // Ledger views an option buyer as a negative liability.
                            payoffs: vec![
                                -plan.signed_underlying_units
                                    * (fine[c.finest_steps] - c.strike).abs(),
                            ],
                            source_hash: format!("{physical_recipe}:path:{id}"),
                        };
                        let audit = reconcile_path(
                            &batch,
                            0,
                            &holdings,
                            AuditModelFamily::BlackScholes,
                            c.maturity_years / steps as f64,
                            -plan.signed_underlying_units * quote.price,
                            c.cost,
                        )?;
                        if (audit.unquantized_float_pnl - result.pnl).abs()
                            > 256.0 * f64::EPSILON * (1.0 + result.pnl.abs() + result.turnover)
                        {
                            return Err(ComputeError::Execution(
                                "straddle audit/accounting PNL mismatch".into(),
                            ));
                        }
                        accumulator
                            .audits
                            .push(StraddleAudit::from_audit(id, audit)?);
                    }
                    if let Some(writer) = &mut writer {
                        writer.write(&result)?;
                    }
                }
            }
        }
    }
    let mut cases = Vec::with_capacity(plan.cases);
    for (index, mut a) in accumulators.into_iter().enumerate() {
        let risk = weighted_risk(&a.pnl, None, RiskObjective::Cvar { alpha: 0.95 })
            .map_err(|e| ComputeError::Execution(e.to_string()))?;
        a.pnl.sort_by(f64::total_cmp);
        let probabilities = [0.01, 0.05, 0.5, 0.95, 0.99];
        // Empirical inverse CDF, left-continuous at declared probabilities.
        let quantiles = probabilities.map(|p| {
            a.pnl[((p * c.paths as f64).ceil() as usize)
                .saturating_sub(1)
                .min(c.paths - 1)]
        });
        let means = a
            .sums
            .map(|s| finite(s.value() / c.paths as f64))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        cases.push(StraddleCaseReport {
            hedge_steps: c.hedge_steps[index / strategies.len()],
            strategy: strategies[index % strategies.len()],
            mean_profit: risk.mean_pnl,
            standard_error: risk.pnl_standard_deviation / ((c.paths - 1) as f64).sqrt(),
            quantiles,
            quantile_probabilities: probabilities,
            loss_cvar_95: risk.value,
            analytic_expected_profit_before_costs: expected,
            mean_cost: means[0],
            mean_turnover: means[1],
            mean_terminal_turnover: means[2],
            mean_option_mark_gain: means[3],
            mean_stock_mark_gain: means[4],
            mean_theta_approximation: means[5],
            mean_gamma_approximation: means[6],
            mean_higher_order_residual: means[7],
            mean_inventory_tracking_gain: means[8],
            mean_continuous_variance_drift_approximation: means[9],
            mean_realized_variance: means[10],
            mean_hedge_grid_realized_variance: means[12],
            mean_delta_approximation: means[11],
            max_cash_identity_error: a.max_cash,
            max_attribution_identity_error: a.max_attribution,
            audits: a.audits,
        });
    }
    let mut source = Sha256::new();
    for (name, bytes) in [
        (
            "volatility_strategy.rs",
            include_bytes!("volatility_strategy.rs").as_slice(),
        ),
        ("aad.rs", include_bytes!("aad.rs").as_slice()),
        (
            "second_order.rs",
            include_bytes!("second_order.rs").as_slice(),
        ),
        (
            "monte_carlo.rs",
            include_bytes!("monte_carlo.rs").as_slice(),
        ),
        ("hedge_band.rs", include_bytes!("hedge_band.rs").as_slice()),
        (
            "hedge_reconciliation.rs",
            include_bytes!("hedge_reconciliation.rs").as_slice(),
        ),
        (
            "hedge_ledger.rs",
            include_bytes!("hedge_ledger.rs").as_slice(),
        ),
        (
            "risk_objective.rs",
            include_bytes!("risk_objective.rs").as_slice(),
        ),
    ] {
        source.update(name.as_bytes());
        source.update((bytes.len() as u64).to_le_bytes());
        source.update(bytes);
    }
    let source_sha256 = format!("{:x}", source.finalize());
    let lockfile_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!("../../../Cargo.lock"))
    );
    let executable = std::env::current_exe().map_err(io_error)?;
    let executable_sha256 = hash_file(&executable)?;
    // Publish only after every path, risk summary, audit and metadata operation
    // has succeeded. Existing output files are never overwritten.
    let samples = writer.map(AtomicSamples::finish).transpose()?;
    Ok(StraddleStudyReport {schema:"volt.compute.straddle_report.v1".into(),config:c.clone(),plan,execution_policy:"native_scalar_cpu_f64;analytic_bs_quotes;exact_increment_zero_drift_physical_gbm".into(),rng_contract:RNG.into(),source_sha256,lockfile_sha256,executable_sha256,recipe_sha256:recipe,physical_market_recipe_sha256:physical_recipe,physical_market_stream_sha256:format!("{:x}",market_hash.finalize()),initial_quote_per_unit:quote,elapsed_seconds:start.elapsed().as_secs_f64(),cases,samples,interpretation:"Simulated zero-carry European mechanism control. Physical and quote volatility are distinct; expected pre-cost profit is the analytic physical payoff expectation minus the quote purchase price. Finite-frequency PNL uses full repricing and actual inventory fees; theta/gamma attribution retains its residual. No empirical edge, funding/margin feasibility, option spread or learned band selection is claimed.".into()})
}
fn finite(x: f64) -> Result<f64, ComputeError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(ComputeError::Execution(
            "nonfinite straddle arithmetic".into(),
        ))
    }
}
fn invalid(s: &str) -> ComputeError {
    ComputeError::InvalidInput(s.into())
}
fn io_error(e: impl std::fmt::Display) -> ComputeError {
    ComputeError::Execution(format!("straddle study I/O: {e}"))
}
fn hash_json(v: &impl Serialize) -> Result<String, ComputeError> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(v).map_err(io_error)?)
    ))
}
fn hash_file(path: &Path) -> Result<String, ComputeError> {
    use std::io::Read;
    let mut f = File::open(path).map_err(io_error)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let n = f.read(&mut buffer).map_err(io_error)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
struct AtomicSamples {
    path: PathBuf,
    temporary: PathBuf,
    writer: Option<BufWriter<File>>,
    hash: Sha256,
    rows: u64,
    bytes: u64,
}
impl AtomicSamples {
    fn new(path: &Path) -> Result<Self, ComputeError> {
        static NONCE: AtomicU64 = AtomicU64::new(0);
        if path.exists() {
            return Err(invalid("sample output already exists"));
        }
        let temporary = path.with_extension(format!(
            "partial-{}-{}",
            std::process::id(),
            NONCE.fetch_add(1, Ordering::Relaxed)
        ));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(io_error)?;
        Ok(Self {
            path: path.into(),
            temporary,
            writer: Some(BufWriter::new(file)),
            hash: Sha256::new(),
            rows: 0,
            bytes: 0,
        })
    }
    fn write(&mut self, value: &StraddlePathResult) -> Result<(), ComputeError> {
        let mut row = serde_json::to_vec(value).map_err(io_error)?;
        row.push(b'\n');
        self.writer
            .as_mut()
            .unwrap()
            .write_all(&row)
            .map_err(io_error)?;
        self.hash.update(&row);
        self.rows += 1;
        self.bytes += row.len() as u64;
        Ok(())
    }
    fn finish(mut self) -> Result<StraddleSamplesArtifact, ComputeError> {
        let mut writer = self.writer.take().unwrap();
        writer.flush().map_err(io_error)?;
        writer.get_ref().sync_all().map_err(io_error)?;
        drop(writer);
        std::fs::hard_link(&self.temporary, &self.path).map_err(io_error)?;
        Ok(StraddleSamplesArtifact {
            path: self.path.clone(),
            rows: self.rows,
            bytes: self.bytes,
            sha256: format!("{:x}", self.hash.clone().finalize()),
        })
    }
}
impl Drop for AtomicSamples {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.temporary);
    }
}
