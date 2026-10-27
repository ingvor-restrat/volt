//! Explicit factor-risk / predictive-return scenarios for statistical hedging.
//!
//! Loadings map decimal factor returns to decimal stock returns. Covariances
//! are annualized; idiosyncratic volatility is annual decimal standard deviation.
//! A persistent observed signal changes physical drift. Per-path coefficient
//! uncertainty is latent and stays out of observations. European option marks
//! use a distinct constant-volatility risk-neutral Black–Scholes reference.
use crate::{
    ComputeError, OptionRow, OptionSide,
    aad::analytic_greeks,
    joint_option_market::{ScenarioRange, validate_ranges},
    native_learner::{BatchProvenance, DatasetPartition, EpisodeBatch},
    second_order::analytic_gamma,
    volatility_surface::{EuropeanInputs, european_price_and_vega},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
fn invalid(x: impl Into<String>) -> ComputeError {
    ComputeError::InvalidInput(x.into())
}
fn finite(x: f64, label: &str) -> Result<f64, ComputeError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(invalid(format!("nonfinite factor {label}")))
    }
}
pub const RNG_CONTRACT: &str = "factor-scenarios.v1; nested SplitMix64 seed/path/step/domain; open52uniform; BoxMuller; independent factor/idio/signal/coefficient domains";
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorProvenance {
    pub model_id: String,
    pub source_sha256: String,
    pub synthetic: bool,
    pub decision_origin_ns: Option<i64>,
    pub known_at_ns: Option<i64>,
    pub assumptions: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorModel {
    pub asset_ids: Vec<String>,
    pub factor_ids: Vec<String>,
    pub initial_prices: Vec<f64>,
    pub loadings: Vec<Vec<f64>>,
    pub annual_factor_covariance: Vec<Vec<f64>>,
    pub annual_idiosyncratic_volatility: Vec<f64>,
    pub dividend_yields: Vec<f64>,
    pub annual_base_excess_return: Vec<f64>,
    pub annual_signal_coefficients: Vec<f64>,
    pub coefficient_uncertainty_std: Vec<f64>,
    /// Discrete per-interval AR(1); signal shock std is per interval, not annual.
    pub signal_initial: f64,
    pub signal_persistence: f64,
    pub signal_innovation_std: f64,
    pub provenance: FactorProvenance,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorRisk {
    pub factor_impact: Vec<Vec<f64>>,
    pub annual_stock_covariance: Vec<Vec<f64>>,
    pub annual_stock_volatility: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorOption {
    pub asset: usize,
    pub side: OptionSide,
    pub strike: f64,
    pub maturity_steps: usize,
    pub quantity: i64,
    pub multiplier: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactorObservation {
    MarketOnly,
    MarketAndSignal,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorConfig {
    pub seed: u64,
    pub steps: usize,
    pub dt: f64,
    pub rate: f64,
    pub liabilities: Vec<FactorOption>,
    pub observation: FactorObservation,
    pub trading_costs: Vec<f64>,
    pub liquidation_costs: Vec<f64>,
    pub max_working_bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorEconomics {
    pub premium: f64,
    pub currency: String,
    pub cash_growth: Vec<f64>,
    pub income_growth: Vec<f64>,
    pub trading_costs: Vec<f64>,
    pub liquidation_costs: Vec<f64>,
    pub units: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorScenarios {
    pub batch: EpisodeBatch,
    pub config: FactorConfig,
    pub model: FactorModel,
    pub range: ScenarioRange,
    pub risk: FactorRisk,
    pub economics: FactorEconomics,
    pub feature_names: Vec<String>,
    pub signal_states: Vec<f64>,
    pub latent_coefficients: Vec<f64>,
    pub physical_sha256: String,
    pub content_sha256: String,
    pub estimated_working_bytes: u64,
}
impl FactorModel {
    pub fn validate(&self) -> Result<(), ComputeError> {
        let n = self.asset_ids.len();
        let k = self.factor_ids.len();
        if !(1..=16).contains(&n)
            || !(1..=8).contains(&k)
            || self.loadings.len() != n
            || self.loadings.iter().any(|r| r.len() != k)
            || self.annual_factor_covariance.len() != k
            || self.annual_factor_covariance.iter().any(|r| r.len() != k)
        {
            return Err(invalid(
                "factor dimensions require1..16 assets,1..8 factors and aligned matrices",
            ));
        }
        for ids in [&self.asset_ids, &self.factor_ids] {
            let mut seen = std::collections::HashSet::new();
            for s in ids {
                if s.trim().is_empty() || s.len() > 128 || !seen.insert(s) {
                    return Err(invalid(
                        "empty/duplicate/excessive factor or asset identity",
                    ));
                }
            }
        }
        for v in [
            &self.initial_prices,
            &self.annual_idiosyncratic_volatility,
            &self.dividend_yields,
            &self.annual_base_excess_return,
            &self.annual_signal_coefficients,
            &self.coefficient_uncertainty_std,
        ] {
            if v.len() != n || v.iter().any(|x| !x.is_finite()) {
                return Err(invalid("invalid factor asset vector"));
            }
        }
        if self.initial_prices.iter().any(|x| *x <= 0.)
            || self
                .annual_idiosyncratic_volatility
                .iter()
                .chain(&self.coefficient_uncertainty_std)
                .any(|x| *x < 0.)
            || !self.signal_initial.is_finite()
            || !self.signal_persistence.is_finite()
            || self.signal_persistence.abs() >= 1.
            || !self.signal_innovation_std.is_finite()
            || self.signal_innovation_std < 0.
            || self.loadings.iter().flatten().any(|x| !x.is_finite())
        {
            return Err(invalid("invalid factor risk/signal values"));
        }
        let p = &self.provenance;
        if p.model_id.trim().is_empty()
            || p.model_id.len() > 512
            || p.assumptions.trim().is_empty()
            || p.assumptions.len() > 8192
            || p.source_sha256.len() != 64
            || !p.source_sha256.bytes().all(|c| c.is_ascii_hexdigit())
        {
            return Err(invalid("invalid factor provenance"));
        }
        match (p.decision_origin_ns, p.known_at_ns) {
            (Some(origin), Some(known)) if known <= origin => {}
            (None, None) if p.synthetic => {}
            _ => {
                return Err(invalid(
                    "factor calibration knowledge must precede decision origin",
                ));
            }
        }
        cholesky_psd(&self.annual_factor_covariance)?;
        Ok(())
    }
    pub fn risk(&self) -> Result<FactorRisk, ComputeError> {
        self.validate()?;
        let n = self.asset_ids.len();
        let k = self.factor_ids.len();
        let l = cholesky_psd(&self.annual_factor_covariance)?;
        let mut impact = vec![vec![0.; k]; n];
        for (i, row) in impact.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = finite(
                    (0..k).map(|a| self.loadings[i][a] * l[a][j]).sum(),
                    "factor impact",
                )?;
            }
        }
        let mut cov = vec![vec![0.; n]; n];
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = finite(
                    (0..k).map(|a| impact[i][a] * impact[j][a]).sum::<f64>()
                        + if i == j {
                            self.annual_idiosyncratic_volatility[i].powi(2)
                        } else {
                            0.
                        },
                    "stock covariance",
                )?;
            }
        }
        let vols = (0..n)
            .map(|i| {
                let x = cov[i][i].sqrt();
                if !x.is_finite() || x <= 0. {
                    Err(invalid("positive stock reference volatility required"))
                } else {
                    Ok(x)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(FactorRisk {
            factor_impact: impact,
            annual_stock_covariance: cov,
            annual_stock_volatility: vols,
        })
    }
}
fn cholesky_psd(c: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ComputeError> {
    let n = c.len();
    let scale = c
        .iter()
        .enumerate()
        .map(|(i, r)| r[i].abs())
        .fold(0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let tol = 1e-12 * scale;
    let mut l = vec![vec![0.; n]; n];
    for i in 0..n {
        for j in 0..=i {
            if !c[i][j].is_finite()
                || !c[j][i].is_finite()
                || (c[i][j] - c[j][i]).abs() > tol
                || c[i][i] < 0.
            {
                return Err(invalid("factor covariance must be finite symmetric PSD"));
            }
            let value = c[i][j] - (0..j).map(|a| l[i][a] * l[j][a]).sum::<f64>();
            if i == j {
                if value < -tol {
                    return Err(invalid("factor covariance is not PSD"));
                }
                l[i][j] = value.max(0.).sqrt();
            } else if l[j][j] > 0. {
                l[i][j] = value / l[j][j];
            } else if value.abs() > tol {
                return Err(invalid("inconsistent singular factor covariance"));
            }
        }
    }
    Ok(l)
}
impl FactorConfig {
    pub fn validate(&self, m: &FactorModel) -> Result<(), ComputeError> {
        m.validate()?;
        let n = m.asset_ids.len();
        if self.steps == 0
            || self.steps > 252
            || !self.dt.is_finite()
            || self.dt <= 0.
            || self.dt > 1.
            || !self.rate.is_finite()
            || self.liabilities.len() > 64
            || self.trading_costs.len() != n
            || self.liquidation_costs.len() != n
            || self
                .trading_costs
                .iter()
                .chain(&self.liquidation_costs)
                .any(|x| !x.is_finite() || *x < 0.)
            || self.max_working_bytes == 0
            || self.max_working_bytes > 1024 * 1024 * 1024
        {
            return Err(invalid("invalid factor scenario horizon/carry/cost/budget"));
        }
        for r in std::iter::once(&self.rate).chain(&m.dividend_yields) {
            let g = (*r * self.dt).exp();
            if !g.is_finite() || g <= 0. {
                return Err(invalid("factor funding/income overflow or underflow"));
            }
        }
        for leg in &self.liabilities {
            if leg.asset >= n
                || !leg.strike.is_finite()
                || leg.strike <= 0.
                || leg.maturity_steps < self.steps
                || leg.maturity_steps > 2520
                || leg.quantity.unsigned_abs() > 1_000_000
                || leg.multiplier == 0
                || leg.multiplier > 1_000_000
            {
                return Err(invalid("invalid factor European liability"));
            }
        }
        Ok(())
    }
}
pub fn feature_names(m: &FactorModel) -> Vec<String> {
    let mut v = m
        .asset_ids
        .iter()
        .map(|s| format!("log_price_relative:{s}"))
        .collect::<Vec<_>>();
    v.extend(["horizon_years_left".into(), "liability_mark".into()]);
    for s in &m.asset_ids {
        v.push(format!("liability_fixed_vol_delta:{s}"));
    }
    for s in &m.asset_ids {
        v.push(format!("liability_fixed_vol_gamma:{s}"));
    }
    v.push("observed_signal".into());
    for s in &m.asset_ids {
        v.push(format!("annual_excess_return_forecast:{s}"));
    }
    for s in &m.asset_ids {
        v.push(format!("annual_drift_uncertainty_prior_sd:{s}"));
    }
    v
}
/// Conservative bounded host model includes output/hash/serialization workspace.
pub fn plan(m: &FactorModel, c: &FactorConfig, r: &ScenarioRange) -> Result<u64, ComputeError> {
    c.validate(m)?;
    validate_ranges(std::slice::from_ref(r))?;
    let points = r
        .paths
        .checked_mul(c.steps + 1)
        .ok_or_else(|| invalid("factor path shape overflow"))?;
    let bytes = points
        .checked_mul(1024 + 1024 * m.asset_ids.len())
        .and_then(|x| x.checked_add(8 * 1024 * 1024))
        .ok_or_else(|| invalid("factor path memory overflow"))?;
    let bytes = u64::try_from(bytes).map_err(|_| invalid("factor memory overflow"))?;
    if bytes > c.max_working_bytes {
        return Err(invalid(format!(
            "factor scenarios need{bytes} modeled bytes; budget{}",
            c.max_working_bytes
        )));
    }
    Ok(bytes)
}
fn digest(v: &impl Serialize) -> Result<String, ComputeError> {
    let b = serde_json::to_vec(v).map_err(|e| invalid(e.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(b)))
}
/// Contemporaneous state and explicit independent innovations for one interval.
#[derive(Clone, Copy)]
pub struct FactorStep<'a> {
    pub prices: &'a [f64],
    pub signal: f64,
    pub coefficient: &'a [f64],
    pub factor_normal: &'a [f64],
    pub idio_normal: &'a [f64],
}
/// Fixed standardized shocks for an auditable one-interval financial transition.
/// `coefficient` is the latent path-specific annual signal response. The signal
/// available at the beginning of the interval determines its physical drift.
pub fn advance_given_shocks(
    m: &FactorModel,
    risk: &FactorRisk,
    c: &FactorConfig,
    step: FactorStep<'_>,
) -> Result<Vec<f64>, ComputeError> {
    let FactorStep {
        prices,
        signal,
        coefficient,
        factor_normal,
        idio_normal,
    } = step;
    c.validate(m)?;
    let expected = m.risk()?;
    if digest(&expected)? != digest(risk)? {
        return Err(invalid("risk decomposition inconsistent with factor model"));
    }
    let n = m.asset_ids.len();
    if prices.len() != n
        || coefficient.len() != n
        || idio_normal.len() != n
        || factor_normal.len() != m.factor_ids.len()
        || !signal.is_finite()
        || prices.iter().any(|x| !x.is_finite() || *x <= 0.)
        || coefficient
            .iter()
            .chain(idio_normal)
            .chain(factor_normal)
            .any(|x| !x.is_finite())
    {
        return Err(invalid("invalid supplied factor transition"));
    }
    advance(m, risk, c, step)
}
fn advance(
    m: &FactorModel,
    risk: &FactorRisk,
    c: &FactorConfig,
    step: FactorStep<'_>,
) -> Result<Vec<f64>, ComputeError> {
    let FactorStep {
        prices,
        signal,
        coefficient,
        factor_normal,
        idio_normal,
    } = step;
    let mut next = Vec::with_capacity(prices.len());
    for i in 0..prices.len() {
        let drift = c.rate - m.dividend_yields[i]
            + m.annual_base_excess_return[i]
            + coefficient[i] * signal;
        let shock = risk.factor_impact[i]
            .iter()
            .zip(factor_normal)
            .map(|(a, z)| a * z)
            .sum::<f64>()
            + m.annual_idiosyncratic_volatility[i] * idio_normal[i];
        let y =
            (drift - 0.5 * risk.annual_stock_volatility[i].powi(2)) * c.dt + c.dt.sqrt() * shock;
        let x = finite(prices[i] * y.exp(), "stock mark")?;
        if x <= 0. {
            return Err(invalid("factor stock underflow"));
        }
        next.push(x);
    }
    Ok(next)
}
/// Stateless seeded paths. Range changes affect output content hashes, while
/// generator/dataset identity remains stable for chunked dataset registration.
pub fn generate(
    m: &FactorModel,
    c: &FactorConfig,
    r: ScenarioRange,
) -> Result<FactorScenarios, ComputeError> {
    let reserved = plan(m, c, &r)?;
    let risk = m.risk()?;
    let n = m.asset_ids.len();
    let k = m.factor_ids.len();
    let f = feature_names(m);
    let mut marks = Vec::with_capacity(r.paths * (c.steps + 1) * n);
    let mut signals = Vec::with_capacity(r.paths * (c.steps + 1));
    let mut coefficients = Vec::with_capacity(r.paths * n);
    let mut features = Vec::with_capacity(r.paths * c.steps * f.len());
    let mut payoffs = Vec::with_capacity(r.paths);
    let mut premium = 0.;
    for local in 0..r.paths {
        let path = r.first_path + local as u64;
        let mut prices = m.initial_prices.clone();
        let mut signal = m.signal_initial;
        let coefficient = (0..n)
            .map(|i| {
                m.annual_signal_coefficients[i]
                    + m.coefficient_uncertainty_std[i] * normal(c.seed, path, 0, 1, i as u64)
            })
            .collect::<Vec<_>>();
        if coefficient.iter().any(|x| !x.is_finite()) {
            return Err(invalid("factor coefficient overflow"));
        }
        coefficients.extend_from_slice(&coefficient);
        for t in 0..=c.steps {
            let (value, delta, gamma) = liability(m, &risk, c, &prices, t)?;
            marks.extend_from_slice(&prices);
            signals.push(signal);
            if t == 0 {
                premium = value;
            }
            if t == c.steps {
                payoffs.push(value);
                break;
            }
            features.extend(
                prices
                    .iter()
                    .zip(&m.initial_prices)
                    .map(|(s, s0)| (s / s0).ln()),
            );
            features.extend([(c.steps - t) as f64 * c.dt, value]);
            features.extend(delta);
            features.extend(gamma);
            let observed = if c.observation == FactorObservation::MarketAndSignal {
                signal
            } else {
                0.
            };
            features.push(observed);
            features.extend((0..n).map(|i| {
                if c.observation == FactorObservation::MarketAndSignal {
                    m.annual_base_excess_return[i] + m.annual_signal_coefficients[i] * signal
                } else {
                    0.
                }
            }));
            features.extend((0..n).map(|i| m.coefficient_uncertainty_std[i] * observed.abs()));
            let factor = (0..k)
                .map(|i| normal(c.seed, path, t as u64, 2, i as u64))
                .collect::<Vec<_>>();
            let idio = (0..n)
                .map(|i| normal(c.seed, path, t as u64, 3, i as u64))
                .collect::<Vec<_>>();
            prices = advance(
                m,
                &risk,
                c,
                FactorStep {
                    prices: &prices,
                    signal,
                    coefficient: &coefficient,
                    factor_normal: &factor,
                    idio_normal: &idio,
                },
            )
            .map_err(|e| invalid(format!("path{path} step{t}: {e}")))?;
            signal = finite(
                m.signal_persistence * signal
                    + m.signal_innovation_std * normal(c.seed, path, t as u64, 4, 0),
                "observed signal",
            )?;
        }
    }
    if features.iter().any(|x| !x.is_finite()) {
        return Err(invalid("factor feature overflow"));
    }
    let physical = digest(&(&marks, &signals, &coefficients))?;
    let dataset = digest(&("volt.factor_scenarios.v1", RNG_CONTRACT, m, c))?;
    let batch = EpisodeBatch {
        force_trade_signal: None,
        exercise: None,
        cash_income: None,
        initial_wealth: None,
        paths: r.paths,
        provenance: BatchProvenance {
            dataset_id: dataset,
            first_path: r.first_path,
        },
        marks,
        features,
        payoffs,
    };
    let cash_growth = vec![(c.rate * c.dt).exp(); c.steps];
    let mut income_growth = Vec::with_capacity(c.steps * n);
    for _ in 0..c.steps {
        income_growth.extend(m.dividend_yields.iter().map(|q| (q * c.dt).exp()));
    }
    let economics=FactorEconomics{premium,currency:"USD".into(),cash_growth,income_growth,trading_costs:c.trading_costs.clone(),liquidation_costs:c.liquidation_costs.clone(),units:"Per-share stock marks; factor covariance annual decimal-return²; option quantity*multiplier signed units; signal dimensionless; drift/uncertainty annual decimal excess returns per signal unit. Latent path coefficients are audit output, excluded from all decision features.".into()};
    let content = digest(&(&batch, &economics, &physical))?;
    Ok(FactorScenarios {
        batch,
        config: c.clone(),
        model: m.clone(),
        range: r,
        risk,
        economics,
        feature_names: f,
        signal_states: signals,
        latent_coefficients: coefficients,
        physical_sha256: physical,
        content_sha256: content,
        estimated_working_bytes: reserved,
    })
}
fn add(sum: &mut f64, correction: &mut f64, x: f64) {
    let t = *sum + x;
    *correction += if sum.abs() >= x.abs() {
        (*sum - t) + x
    } else {
        (x - t) + *sum
    };
    *sum = t;
}
fn liability(
    m: &FactorModel,
    risk: &FactorRisk,
    c: &FactorConfig,
    prices: &[f64],
    t: usize,
) -> Result<(f64, Vec<f64>, Vec<f64>), ComputeError> {
    let n = prices.len();
    let (mut value, mut correction) = (0., 0.);
    let mut delta = vec![0.; n];
    let mut gamma = vec![0.; n];
    let mut dc = vec![0.; n];
    let mut gc = vec![0.; n];
    for (index, leg) in c.liabilities.iter().enumerate() {
        let a = leg.asset;
        let tau = (leg.maturity_steps - t) as f64 * c.dt;
        let units = leg.quantity as f64 * leg.multiplier as f64;
        if tau == 0. {
            let payoff = match leg.side {
                OptionSide::Call => (prices[a] - leg.strike).max(0.),
                OptionSide::Put => (leg.strike - prices[a]).max(0.),
            };
            add(
                &mut value,
                &mut correction,
                finite(units * payoff, "terminal liability")?,
            );
            continue;
        }
        let input = EuropeanInputs {
            spot: prices[a],
            strike: leg.strike,
            maturity_years: tau,
            rate: c.rate,
            dividend_yield: m.dividend_yields[a],
            side: leg.side,
        };
        let sigma = risk.annual_stock_volatility[a];
        let price = european_price_and_vega(&input, sigma)?.0;
        let row = OptionRow {
            row_id: format!("factor-leg{index}"),
            contract_id: "factor-European".into(),
            side: leg.side,
            spot: prices[a],
            strike: leg.strike,
            maturity_years: tau,
            volatility: sigma,
            rate: c.rate,
            dividend_yield: m.dividend_yields[a],
            quantity: 1,
            multiplier: 1,
        };
        let g = analytic_greeks(&row)?;
        add(
            &mut value,
            &mut correction,
            finite(units * price, "liability mark")?,
        );
        add(
            &mut delta[a],
            &mut dc[a],
            finite(units * g.delta, "liability delta")?,
        );
        add(
            &mut gamma[a],
            &mut gc[a],
            finite(units * analytic_gamma(&row)?, "liability gamma")?,
        );
    }
    for i in 0..n {
        delta[i] = finite(delta[i] + dc[i], "aggregate delta")?;
        gamma[i] = finite(gamma[i] + gc[i], "aggregate gamma")?;
    }
    Ok((
        finite(value + correction, "aggregate liability")?,
        delta,
        gamma,
    ))
}
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn uniform(seed: u64, path: u64, step: u64, domain: u64, stream: u64) -> f64 {
    let x = mix(mix(mix(mix(seed ^ 0xc4b2863e5a019e71) ^ path) ^ step) ^ mix(domain));
    let y = mix(x ^ stream);
    ((y >> 12) as f64 + 0.5) / 4503599627370496.
}
fn normal(seed: u64, path: u64, step: u64, domain: u64, index: u64) -> f64 {
    let u = uniform(seed, path, step, domain, index * 2);
    let v = uniform(seed, path, step, domain, index * 2 + 1);
    (-2. * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}
pub fn partitions(ranges: &[ScenarioRange]) -> Result<Vec<DatasetPartition>, ComputeError> {
    crate::joint_option_market::dataset_partitions(ranges)
}
