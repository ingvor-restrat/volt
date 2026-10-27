//! Dated one-stock GBM episodes with a numerical Bermudan holder and delta control.
//!
//! Risk-neutral PDE volatility/rate price the option. Physical stock drift is an
//! explicit separate parameter. At a cash dividend the holder first decides on
//! pre-dividend spot; exercised paths liquidate and settle there. Continuing
//! paths receive min(D,S) and use post-dividend stock marks. A stopped episode
//! then contains cash-only financing. Terminal `payoffs` are zero: settlement
//! amounts live exclusively in `EpisodeBatch.exercise`.
use crate::{
    ComputeError,
    exercise_pricing::{ExerciseGrid, ExerciseOption, HolderGrid, HolderGridCache},
    joint_option_market::ScenarioRange,
    monte_carlo::normal_at,
    native_learner::{BatchProvenance, EpisodeBatch, ExerciseEpisode},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use volt::TimestampNs;
const YEAR_NS: f64 = 31_536_000_000_000_000.;
fn invalid(s: impl Into<String>) -> ComputeError {
    ComputeError::InvalidInput(s.into())
}
fn hash<T: Serialize>(x: &T) -> Result<String, ComputeError> {
    struct Writer(Sha256);
    impl std::io::Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = Writer(Sha256::new());
    serde_json::to_writer(&mut writer, x).map_err(|e| invalid(e.to_string()))?;
    Ok(format!("{:x}", writer.0.finalize()))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BermudanMarketConfig {
    pub option: ExerciseOption,
    pub grid: ExerciseGrid,
    /// All hedge dates, including valuation/expiry and every exercise/dividend date.
    pub dates: Vec<TimestampNs>,
    pub seed: u64,
    /// Annual arithmetic GBM drift between fixed cash dividends; Q drift is rate.
    pub physical_drift: f64,
    /// Signed amount owed by writer, independently of holder's stopping decision.
    pub quantity: i64,
    pub multiplier: u32,
    pub trading_cost: f64,
    pub liquidation_cost: f64,
    pub max_working_bytes: u64,
    pub max_holder_scalar_values: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BermudanEconomics {
    /// Signed initial cash for the whole writer position, after quantity/multiplier.
    pub premium: f64,
    pub premium_basis: String,
    /// Signed optimal-holder PDE value, even when a supplied holder is used.
    pub numerical_reference_premium: f64,
    pub numerical_control_basis: String,
    pub cash_growth: Vec<f64>,
    pub income_growth: Vec<f64>,
    pub trading_costs: Vec<f64>,
    pub liquidation_costs: Vec<f64>,
    pub feature_names: Vec<String>,
    pub event_convention: String,
    pub volatility_measure: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BermudanScenarios {
    pub batch: EpisodeBatch,
    /// Numerical writer delta target [path,decision]; zero from settlement onward.
    pub numerical_holdings: Vec<f64>,
    pub numerical_pnl: Vec<f64>,
    /// Physical pre-dividend stock states [path,date], independent of holder stop.
    pub pre_dividend_spots: Vec<f64>,
    pub economics: BermudanEconomics,
    pub holder_cache_key: String,
    pub holder_policy: HolderPolicyProvenance,
    pub config_sha256: String,
    pub physical_sha256: String,
    pub content_sha256: String,
    pub estimated_working_bytes: u64,
}
/// Caller-owned holder policy/checkpoint identity, bound into dataset/content hashes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HolderPolicyProvenance {
    pub policy_id: String,
    pub policy_sha256: String,
}
/// A chosen holder's admissible stopping dates. The producer derives cash amounts
/// from contract intrinsic and spot; supplying a stopping date never supplies or
/// overrides a payoff. Causality of the policy selecting these dates is caller-owned.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuppliedStopping {
    pub settlement_steps: Vec<usize>,
    pub provenance: HolderPolicyProvenance,
    /// Optional signed initial cash for the complete quantity/multiplier position.
    /// This records the caller-declared initial capital convention.
    pub premium_override: Option<f64>,
}
impl SuppliedStopping {
    fn validate(&self, c: &BermudanMarketConfig, paths: usize) -> Result<(), ComputeError> {
        let p = &self.provenance;
        if self.settlement_steps.len() != paths
            || p.policy_id.is_empty()
            || p.policy_id.len() > 256
            || p.policy_id.chars().any(char::is_control)
            || p.policy_sha256.len() != 64
            || !p.policy_sha256.bytes().all(|v| v.is_ascii_hexdigit())
            || self.premium_override.is_some_and(|v| !v.is_finite())
            || self
                .settlement_steps
                .iter()
                .any(|t| *t > c.steps() || !c.option.exercise.permits_exercise(c.dates[*t]))
        {
            return Err(invalid(
                "invalid supplied holder provenance, premium, stop count or exercise opportunity",
            ));
        }
        Ok(())
    }
}
impl BermudanMarketConfig {
    pub fn steps(&self) -> usize {
        self.dates.len().saturating_sub(1)
    }
    pub fn validate(&self) -> Result<(), ComputeError> {
        self.option.validate()?;
        self.grid.validate()?;
        let steps = self.steps();
        let units = i128::from(self.quantity) * i128::from(self.multiplier);
        if steps == 0
            || steps > 512
            || self.dates[0] != self.option.valuation_time
            || self.dates.last() != Some(&self.option.exercise.expiry())
            || self.dates.windows(2).any(|w| w[0] >= w[1])
            || self.option.dividend_yield != 0.
            || !self.physical_drift.is_finite()
            || self.physical_drift.abs() > 2.
            || self.multiplier == 0
            || self.multiplier > 1_000_000
            || units.abs() > 1_000_000_000
            || [self.trading_cost, self.liquidation_cost]
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || *v > 1.)
            || self.max_working_bytes == 0
            || self.max_working_bytes > 1_073_741_824
            || self.max_holder_scalar_values == 0
            || self.max_holder_scalar_values > 8_000_000
        {
            return Err(invalid(
                "invalid Bermudan scenario dates/carry/units/cost/admission",
            ));
        }
        if matches!(
            self.option.exercise,
            volt::ExerciseSchedule::American { .. }
        ) {
            return Err(ComputeError::Unsupported("scenario holder requires explicit Bermudan dates, rather than an implicit American discretization".into()));
        }
        if self
            .option
            .cash_dividends
            .iter()
            .any(|d| self.dates.binary_search(&d.ex_date).is_err())
        {
            return Err(invalid("hedge dates must include every cash-dividend date"));
        }
        Ok(())
    }
    fn reserve(&self, paths: usize) -> Result<u64, ComputeError> {
        let rows = paths
            .checked_mul(self.steps() + 1)
            .and_then(|v| v.checked_mul(16))
            .and_then(|v| v.checked_add(self.max_holder_scalar_values))
            .and_then(|v| v.checked_add((self.grid.space_steps + 1) * 32))
            .and_then(|v| v.checked_mul(8))
            .ok_or_else(|| invalid("scenario working size overflow"))? as u64;
        if paths == 0 || paths > 1_000_000 || rows > self.max_working_bytes {
            return Err(invalid("Bermudan scenario working-byte admission exceeded"));
        }
        Ok(rows)
    }
}
/// Deterministic global-path-indexed generation. Changing chunk boundaries leaves
/// every physical path unchanged. The Box–Muller stream contract is normal_at.v1.
pub fn generate(
    config: &BermudanMarketConfig,
    range: ScenarioRange,
    cache: &mut HolderGridCache,
) -> Result<BermudanScenarios, ComputeError> {
    config.validate()?;
    config.reserve(range.paths)?;
    let steps = config.steps();
    // Admit/build the numerical domain before allocating a physical cohort.
    cache.get_or_build(&config.option, &config.grid, &config.dates)?;
    range
        .first_path
        .checked_add(range.paths as u64)
        .and_then(|end| end.checked_mul(steps as u64))
        .filter(|end| *end <= 1_u64 << 63)
        .ok_or_else(|| invalid("Bermudan RNG path counter overflow"))?;
    let mut spots = Vec::with_capacity(range.paths * (steps + 1));
    for p in 0..range.paths {
        let mut spot = config.option.spot;
        spots.push(spot);
        for t in 0..steps {
            let dividend = config
                .option
                .cash_dividends
                .iter()
                .find(|d| d.ex_date == config.dates[t])
                .map_or(0., |d| d.amount.min(spot));
            let post = (spot - dividend).max(0.);
            let dt = (config.dates[t + 1].0 - config.dates[t].0) as f64 / YEAR_NS;
            let index = (range.first_path + p as u64) * steps as u64 + t as u64;
            spot = post
                * ((config.physical_drift - 0.5 * config.option.volatility.powi(2)) * dt
                    + config.option.volatility * dt.sqrt() * normal_at(config.seed, index))
                .exp();
            if !spot.is_finite() || spot < 0. {
                return Err(invalid("GBM stock overflow"));
            }
            spots.push(spot);
        }
    }
    generate_from_paths(config, range, &spots, cache)
}
/// Wrap an explicitly supplied cohort using the same holder and financed control.
/// This supports paired comparisons with learned holder rules on identical paths.
/// Supplied states are pre-dividend; their realized distribution remains caller-owned.
pub fn generate_from_paths(
    config: &BermudanMarketConfig,
    range: ScenarioRange,
    spots: &[f64],
    cache: &mut HolderGridCache,
) -> Result<BermudanScenarios, ComputeError> {
    config.validate()?;
    let reserved = config.reserve(range.paths)?;
    let steps = config.steps();
    let stride = steps + 1;
    if spots.len()
        != range
            .paths
            .checked_mul(stride)
            .ok_or_else(|| invalid("scenario shape overflow"))?
        || spots.iter().any(|v| !v.is_finite() || *v < 0.)
        || (0..range.paths).any(|p| spots[p * stride] != config.option.spot)
    {
        return Err(invalid("invalid explicit pre-dividend stock paths"));
    }
    range
        .first_path
        .checked_add(range.paths as u64)
        .ok_or_else(|| invalid("scenario path id overflow"))?;
    let holder = cache.get_or_build(&config.option, &config.grid, &config.dates)?;
    assemble(config, range, spots, &holder, reserved, None)
}
/// Assemble a fixed chosen holder's episodes on the same complete market cohort.
/// Exercise steps must be contractual opportunities, including expiry. Settlement
/// amounts and cash-dividend ordering are derived here. Only states through the
/// chosen stop are consulted for the numerical control; policy causality remains
/// with the caller supplying stops and immutable checkpoint provenance.
pub fn generate_from_paths_with_stopping(
    config: &BermudanMarketConfig,
    range: ScenarioRange,
    spots: &[f64],
    stopping: &SuppliedStopping,
    cache: &mut HolderGridCache,
) -> Result<BermudanScenarios, ComputeError> {
    config.validate()?;
    let reserved = config.reserve(range.paths)?;
    stopping.validate(config, range.paths)?;
    let stride = config.steps() + 1;
    if spots.len()
        != range
            .paths
            .checked_mul(stride)
            .ok_or_else(|| invalid("scenario shape overflow"))?
        || spots.iter().any(|v| !v.is_finite() || *v < 0.)
        || (0..range.paths).any(|p| spots[p * stride] != config.option.spot)
    {
        return Err(invalid("invalid explicit pre-dividend stock paths"));
    }
    range
        .first_path
        .checked_add(range.paths as u64)
        .ok_or_else(|| invalid("scenario path id overflow"))?;
    let holder = cache.get_or_build(&config.option, &config.grid, &config.dates)?;
    assemble(config, range, spots, &holder, reserved, Some(stopping))
}

fn assemble(
    c: &BermudanMarketConfig,
    r: ScenarioRange,
    spots: &[f64],
    holder: &HolderGrid,
    reserved: u64,
    supplied: Option<&SuppliedStopping>,
) -> Result<BermudanScenarios, ComputeError> {
    let steps = c.steps();
    let stride = steps + 1;
    let units = c.quantity as f64 * c.multiplier as f64;
    let numerical_reference_premium = units * holder.price0;
    let premium = supplied
        .and_then(|v| v.premium_override)
        .unwrap_or(numerical_reference_premium);
    let holder_policy = supplied.map_or_else(
        || HolderPolicyProvenance {
            policy_id: "numerical_bermudan_fd_holder.v1".into(),
            policy_sha256: holder.cache_key.clone(),
        },
        |v| v.provenance.clone(),
    );
    let cash_growth = c
        .dates
        .windows(2)
        .map(|w| (c.option.rate * (w[1].0 - w[0].0) as f64 / YEAR_NS).exp())
        .collect::<Vec<_>>();
    if cash_growth.iter().any(|v| !v.is_finite() || *v <= 0.) {
        return Err(invalid("cash growth overflow"));
    }
    let mut marks = Vec::with_capacity(spots.len());
    let mut features = Vec::with_capacity(r.paths * steps * 2);
    let mut holdings = Vec::with_capacity(r.paths * steps);
    let mut income = Vec::with_capacity(r.paths * steps);
    let mut settlement_step = Vec::with_capacity(r.paths);
    let mut amounts = Vec::with_capacity(r.paths);
    let mut pnl = Vec::with_capacity(r.paths);
    for p in 0..r.paths {
        let path = &spots[p * stride..(p + 1) * stride];
        let mut decisions = Vec::with_capacity(stride);
        let mut stopped = None;
        for (t, spot) in path.iter().enumerate() {
            let decision = holder.evaluate(c.dates[t], *spot)?;
            let exercise = supplied.map_or(decision.exercise_now, |chosen| {
                t == chosen.settlement_steps[p]
            });
            decisions.push(decision);
            if exercise {
                stopped = Some(t);
                break;
            }
        }
        let tau = stopped.ok_or_else(|| invalid("holder omitted terminal settlement"))?;
        settlement_step.push(tau);
        amounts.push(units * decisions[tau].immediate_payoff);
        for t in 0..=steps {
            let div = c
                .option
                .cash_dividends
                .iter()
                .find(|d| d.ex_date == c.dates[t])
                .map_or(0., |d| d.amount.min(path[t]));
            let mark = if t == tau {
                path[t]
            } else {
                (path[t] - div).max(0.)
            };
            marks.push(mark);
            if t > 0 {
                income.push(if t < tau { div } else { 0. });
            }
            if t < steps {
                features.extend([
                    (mark / c.option.strike).max(1e-12).ln(),
                    (c.option.exercise.expiry().0 - c.dates[t].0) as f64 / YEAR_NS,
                ]);
                holdings.push(if t < tau {
                    units * decisions[t].delta
                } else {
                    0.
                });
            }
        }
        let base = p * stride;
        let hbase = p * steps;
        let mut value = premium;
        let mut previous = 0.;
        if tau == 0 {
            value -= amounts[p];
        }
        for t in 0..steps {
            if t >= tau {
                value *= cash_growth[t];
                continue;
            }
            let target = holdings[hbase + t];
            let mark = marks[base + t];
            let charge = (target - previous).abs() * mark * c.trading_cost;
            let cash = value - target * mark - charge;
            value = cash * cash_growth[t] + target * (marks[base + t + 1] + income[hbase + t]);
            previous = target;
            if t + 1 == tau {
                value -= target.abs() * marks[base + t + 1] * c.liquidation_cost + amounts[p];
                previous = 0.;
            }
        }
        if !value.is_finite() {
            return Err(invalid("numerical hedge account overflow"));
        }
        pnl.push(value);
    }
    let config_sha256 = hash(c)?;
    let dataset = format!(
        "bermudan-gbm-v1:{}",
        hash(&(
            c,
            &holder_policy,
            supplied.map(|v| v.premium_override),
            include_bytes!("bermudan_market.rs").as_slice()
        ))?
    );
    let batch = EpisodeBatch {
        paths: r.paths,
        provenance: BatchProvenance {
            dataset_id: dataset,
            first_path: r.first_path,
        },
        marks,
        features,
        payoffs: vec![0.; r.paths],
        force_trade_signal: None,
        initial_wealth: None,
        exercise: Some(ExerciseEpisode {
            settlement_step,
            settlement_amounts: amounts,
        }),
        cash_income: Some(income),
    };
    let economics=BermudanEconomics{premium,
        premium_basis:if supplied.is_some_and(|s|s.premium_override.is_some()) {"caller_supplied_signed_cash".into()} else {"numerical_pde_position_value".into()},
        numerical_reference_premium,
        numerical_control_basis:"optimal-holder Bermudan PDE delta evaluated until chosen holder settles; model-based control for the supplied stopping rule".into(),cash_growth,income_growth:vec![1.;steps],
        trading_costs:vec![c.trading_cost],liquidation_costs:vec![c.liquidation_cost],
        feature_names:vec!["log_post_dividend_spot_over_strike".into(),"remaining_years".into()],
        event_convention:"holder pre-dividend decision; exercised path pre-dividend mark and zero dividend credit; continuation post-dividend mark plus capped dividend income; immediate cash settlement then bank financing".into(),
        volatility_measure:"constant annual volatility shared by P and Q; physical arithmetic drift separately specified; q=0 with explicit cash dividends".into()};
    let physical_sha256 = hash(&(
        c.seed,
        c.physical_drift,
        c.option.volatility,
        c.option.spot,
        &c.dates,
        &c.option.cash_dividends,
        r.first_path,
        spots,
    ))?;
    let content_sha256 = hash(&(&batch, &economics, &holdings, &pnl, &holder_policy))?;
    Ok(BermudanScenarios {
        batch,
        numerical_holdings: holdings,
        numerical_pnl: pnl,
        pre_dividend_spots: spots.to_vec(),
        economics,
        holder_cache_key: holder.cache_key.clone(),
        holder_policy,
        config_sha256,
        physical_sha256,
        content_sha256,
        estimated_working_bytes: reserved,
    })
}
