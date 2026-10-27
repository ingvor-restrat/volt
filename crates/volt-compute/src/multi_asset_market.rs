//! Correlated exact-transition GBM and cash-settled Bermudan max-call episodes.
//!
//! Under Q, dS_i=(r-q_i)S_i dt+sigma_i S_i dW_i. Correlation may be
//! positive semidefinite (including perfect correlation); no diagonal jitter is
//! introduced. Episodes use discounted reinvested-security marks
//! P_i(t)=exp((q_i-r)t)S_i(t), constant cash, and discounted cash settlement.
//! Policy features retain raw current stocks and the discounted unit payoff.
use crate::{
    ComputeError,
    native_learner::{BatchProvenance, EpisodeBatch, ExerciseEpisode},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
fn invalid(s: impl Into<String>) -> ComputeError {
    ComputeError::InvalidInput(s.into())
}
fn digest<T: Serialize>(value: &T) -> Result<String, ComputeError> {
    struct Writer(Sha256);
    impl std::io::Write for Writer {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.update(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut w = Writer(Sha256::new());
    serde_json::to_writer(&mut w, value).map_err(|e| invalid(e.to_string()))?;
    Ok(format!("{:x}", w.0.finalize()))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxCallMarket {
    pub spots: Vec<f64>,
    pub volatilities: Vec<f64>,
    pub dividend_yields: Vec<f64>,
    pub correlation: Vec<Vec<f64>>,
    pub rate: f64,
    pub strike: f64,
    pub times: Vec<f64>,
    pub exercise: Vec<bool>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationFactor {
    pub lower: Vec<Vec<f64>>,
    pub rank: usize,
    pub roundoff_clamped_pivots: usize,
    pub max_reconstruction_error: f64,
}
impl MaxCallMarket {
    pub fn paper_five() -> Self {
        let d = 5;
        Self {
            spots: vec![100.; d],
            volatilities: vec![0.2; d],
            dividend_yields: vec![0.1; d],
            correlation: (0..d)
                .map(|i| (0..d).map(|j| if i == j { 1. } else { 0. }).collect())
                .collect(),
            rate: 0.05,
            strike: 100.,
            times: (0..=9).map(|i| i as f64 / 3.).collect(),
            exercise: vec![true; 10],
        }
    }
    pub fn assets(&self) -> usize {
        self.spots.len()
    }
    pub fn validate(&self) -> Result<CorrelationFactor, ComputeError> {
        let d = self.assets();
        let n = self.times.len();
        if d == 0
            || d > 100
            || self.volatilities.len() != d
            || self.dividend_yields.len() != d
            || self.correlation.len() != d
            || !(2..=1025).contains(&n)
            || self.exercise.len() != n
            || !self.exercise[n - 1]
            || self.times[0] != 0.
            || self
                .spots
                .iter()
                .any(|x| !x.is_finite() || *x <= 0. || *x > 1e12)
            || self
                .volatilities
                .iter()
                .any(|x| !x.is_finite() || *x < 0. || *x > 3.)
            || self
                .dividend_yields
                .iter()
                .any(|x| !x.is_finite() || x.abs() > 1.)
            || !self.rate.is_finite()
            || self.rate.abs() > 1.
            || !self.strike.is_finite()
            || self.strike <= 0.
            || self.strike > 1e12
            || self
                .times
                .iter()
                .any(|x| !x.is_finite() || *x < 0. || *x > 30.)
            || self.times.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(invalid(
                "invalid max-call dimensions, calendar or GBM parameters",
            ));
        }
        factor_correlation(&self.correlation)
    }
    pub fn intrinsic(&self, state: &[f64]) -> Result<f64, ComputeError> {
        if state.len() != self.assets() || state.iter().any(|x| !x.is_finite() || *x < 0.) {
            return Err(invalid("invalid max-call state"));
        }
        Ok((state.iter().copied().fold(0., f64::max) - self.strike).max(0.))
    }
    /// Globally time-zero discounted payoff, per max-call unit.
    pub fn payoff(&self, date: usize, state: &[f64]) -> Result<f64, ComputeError> {
        let t = *self
            .times
            .get(date)
            .ok_or_else(|| invalid("payoff date outside model"))?;
        let value = self.intrinsic(state)? * (-self.rate * t).exp();
        if !value.is_finite() {
            return Err(invalid("discounted payoff overflow"));
        }
        Ok(value)
    }
}
/// Unpivoted semidefinite Cholesky with an explicit roundoff tolerance. Negative
/// residual pivots in[-1e-12,0) are clamped to zero and counted; positive pivots
/// retain their magnitude. A zero pivot must have zero residual column. The
/// reconstructed matrix is checked; the algorithm never adds diagonal noise.
pub fn factor_correlation(c: &[Vec<f64>]) -> Result<CorrelationFactor, ComputeError> {
    let d = c.len();
    let tol = 1e-12;
    if d == 0
        || d > 100
        || c.iter()
            .any(|r| r.len() != d || r.iter().any(|v| !v.is_finite() || v.abs() > 1.))
    {
        return Err(invalid("invalid correlation dimensions or entries"));
    }
    for (i, row) in c.iter().enumerate() {
        if row[i] != 1. || (0..i).any(|j| (c[i][j] - c[j][i]).abs() > tol) {
            return Err(invalid("correlation requires unit diagonal and symmetry"));
        }
    }
    let mut l = vec![vec![0.; d]; d];
    let mut rank = 0;
    let mut clamped = 0;
    for i in 0..d {
        for j in 0..=i {
            let residual = c[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if !residual.is_finite() {
                return Err(invalid("nonfinite correlation factor residual"));
            }
            if i == j {
                if residual < -tol {
                    return Err(invalid("correlation is not positive semidefinite"));
                }
                if residual < 0. {
                    clamped += 1;
                }
                if residual > 0. {
                    l[i][j] = residual.sqrt();
                    rank += 1;
                }
            } else if l[j][j] > 0. {
                l[i][j] = residual / l[j][j];
            } else if residual.abs() > tol {
                return Err(invalid(
                    "nonzero residual column at semidefinite zero pivot",
                ));
            }
        }
    }
    let mut error = 0_f64;
    for i in 0..d {
        for j in 0..d {
            error = error.max((c[i][j] - (0..d).map(|k| l[i][k] * l[j][k]).sum::<f64>()).abs());
        }
    }
    if error > 1e-10 {
        return Err(invalid("correlation factor reconstruction failed"));
    }
    Ok(CorrelationFactor {
        lower: l,
        rank,
        roundoff_clamped_pivots: clamped,
        max_reconstruction_error: error,
    })
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomKey {
    pub seed: u64,
    pub namespace: u64,
    pub path_id: u64,
    pub branch: u64,
}
fn stream_seed(key: RandomKey, purpose: u64) -> [u64; 2] {
    let mut h = Sha256::new();
    h.update(b"volt.correlated-gbm.v1\0");
    for x in [key.seed, key.namespace, key.path_id, key.branch, purpose] {
        h.update(x.to_le_bytes());
    }
    let bytes = h.finalize();
    [
        u64::from_le_bytes(bytes[..8].try_into().expect("eight-byte SHA prefix")),
        u64::from_le_bytes(bytes[8..16].try_into().expect("eight-byte SHA suffix")),
    ]
}
// Two independent64bit key words address Box-Muller uniforms; SHA128stream
// identity avoids reducing whole conditional streams to a64bit seed namespace.
fn keyed_normal(seed: [u64; 2], index: u64) -> f64 {
    fn mix(mut x: u64) -> u64 {
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    let counter = index.wrapping_mul(0x9e3779b97f4a7c15);
    let u1 = ((mix(seed[0].wrapping_add(counter)) >> 12) as f64 + 0.5) / 4_503_599_627_370_496.;
    let u2 = ((mix(seed[1].wrapping_add(counter)) >> 12) as f64 + 0.5) / 4_503_599_627_370_496.;
    (-2. * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
fn advance(
    m: &MaxCallMarket,
    f: &CorrelationFactor,
    date: usize,
    state: &mut [f64],
    seed: [u64; 2],
) -> Result<(), ComputeError> {
    if state.len() != m.assets()
        || state.iter().any(|v| !v.is_finite() || *v < 0.)
        || date + 1 >= m.times.len()
    {
        return Err(invalid("invalid transition state/date"));
    }
    let d = m.assets();
    let dt = m.times[date + 1] - m.times[date];
    let mut z = [0.; 100];
    for (i, value) in z.iter_mut().enumerate().take(d) {
        *value = keyed_normal(seed, (date * d + i) as u64);
    }
    for (i, current) in state.iter_mut().enumerate() {
        let shock = (0..=i).map(|j| f.lower[i][j] * z[j]).sum::<f64>();
        let s = *current
            * ((m.rate - m.dividend_yields[i] - 0.5 * m.volatilities[i].powi(2)) * dt
                + m.volatilities[i] * dt.sqrt() * shock)
                .exp();
        if !s.is_finite() || s < 0. || (*current > 0. && s == 0.) {
            return Err(invalid("GBM transition overflow or underflow"));
        }
        *current = s;
    }
    Ok(())
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxCallPaths {
    pub values: Vec<f64>,
    pub paths: usize,
    pub dates: usize,
    pub assets: usize,
    pub first_path: u64,
    pub seed: u64,
    pub namespace: u64,
    pub market_sha256: String,
    pub content_sha256: String,
    pub correlation_factor: CorrelationFactor,
}
impl MaxCallPaths {
    pub fn state(&self, path: usize, date: usize) -> Result<&[f64], ComputeError> {
        if self.assets == 0
            || self.assets > 100
            || self.dates < 2
            || self.dates > 1025
            || path >= self.paths
            || date >= self.dates
        {
            return Err(invalid("path/date outside cohort"));
        }
        let i = path
            .checked_mul(self.dates)
            .and_then(|i| i.checked_add(date))
            .and_then(|i| i.checked_mul(self.assets))
            .ok_or_else(|| invalid("malformed path index overflow"))?;
        self.values
            .get(
                i..i.checked_add(self.assets)
                    .ok_or_else(|| invalid("malformed state extent"))?,
            )
            .ok_or_else(|| invalid("malformed path storage"))
    }
}
fn cells(
    m: &MaxCallMarket,
    paths: usize,
    maxbytes: u64,
    multiplier: usize,
) -> Result<usize, ComputeError> {
    let n = paths
        .checked_mul(m.times.len())
        .and_then(|n| n.checked_mul(m.assets()))
        .ok_or_else(|| invalid("path shape overflow"))?;
    let bytes = n
        .checked_mul(multiplier)
        .and_then(|n| n.checked_add(m.assets() * m.assets() * 2 + m.assets() * 8 + 512))
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| invalid("path bytes overflow"))? as u64;
    if paths == 0
        || paths > 2_000_000
        || maxbytes == 0
        || maxbytes > 8_589_934_592
        || bytes > maxbytes
    {
        return Err(invalid("max-call working-byte admission exceeded"));
    }
    Ok(n)
}
pub fn generate_paths(
    m: &MaxCallMarket,
    paths: usize,
    seed: u64,
    first_path: u64,
    namespace: u64,
    max_working_bytes: u64,
) -> Result<MaxCallPaths, ComputeError> {
    let factor = m.validate()?;
    let n = cells(m, paths, max_working_bytes, 2)?;
    first_path
        .checked_add(paths as u64)
        .ok_or_else(|| invalid("global path id overflow"))?;
    let mut values = Vec::with_capacity(n);
    for p in 0..paths {
        let mut state = m.spots.clone();
        values.extend_from_slice(&state);
        let key = RandomKey {
            seed,
            namespace,
            path_id: first_path + p as u64,
            branch: 0,
        };
        let stream = stream_seed(key, 0);
        for t in 0..m.times.len() - 1 {
            advance(m, &factor, t, &mut state, stream)?;
            values.extend_from_slice(&state);
        }
    }
    Ok(MaxCallPaths {
        content_sha256: digest(&(&values, first_path, seed, namespace))?,
        values,
        paths,
        dates: m.times.len(),
        assets: m.assets(),
        first_path,
        seed,
        namespace,
        market_sha256: digest(m)?,
        correlation_factor: factor,
    })
}
/// Independently addressed conditional GBM branch. The conditional purpose salt
/// differs from outer generation even when caller supplies identical keys.
/// Result starts with current state, then contains[date-start,asset] states.
pub fn simulate_tail(
    m: &MaxCallMarket,
    start_date: usize,
    state: &[f64],
    key: RandomKey,
    max_working_bytes: u64,
) -> Result<Vec<f64>, ComputeError> {
    let factor = m.validate()?;
    cells(m, 1, max_working_bytes, 2)?;
    if start_date >= m.times.len()
        || state.len() != m.assets()
        || state.iter().any(|v| !v.is_finite() || *v < 0.)
    {
        return Err(invalid("invalid conditional initial state"));
    }
    let mut output = Vec::with_capacity((m.times.len() - start_date) * m.assets());
    let mut state = state.to_vec();
    output.extend_from_slice(&state);
    let seed = stream_seed(key, 1);
    for t in start_date..m.times.len() - 1 {
        advance(m, &factor, t, &mut state, seed)?;
        output.extend_from_slice(&state);
    }
    Ok(output)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxCallEpisodeConfig {
    pub quantity: i64,
    pub multiplier: u32,
    /// Total signed initial cash for the whole position; no unit scaling is applied.
    pub premium: f64,
    pub trading_costs: Vec<f64>,
    pub liquidation_costs: Vec<f64>,
    pub holder_id: String,
    pub holder_sha256: String,
    pub max_working_bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxCallEpisodes {
    pub assets: usize,
    pub dates: usize,
    pub trading_costs: Vec<f64>,
    pub liquidation_costs: Vec<f64>,
    pub account_basis: String,
    pub batch: EpisodeBatch,
    pub cash_growth: Vec<f64>,
    pub income_growth: Vec<f64>,
    pub premium: f64,
    pub feature_names: Vec<String>,
    pub content_sha256: String,
    pub market_sha256: String,
    pub physical_cohort_sha256: String,
    pub holder_id: String,
    pub holder_sha256: String,
}
/// Supplied stops are dates, not cash amounts. Only contractual dates are valid;
/// settlement derives from observed state with signed quantity/multiplier once.
/// Features are raw current stocks followed by globally discounted unit payoff.
pub fn assemble_episodes(
    m: &MaxCallMarket,
    p: &MaxCallPaths,
    stops: &[usize],
    c: &MaxCallEpisodeConfig,
) -> Result<MaxCallEpisodes, ComputeError> {
    m.validate()?;
    let expected = cells(m, p.paths, c.max_working_bytes, 6)?;
    let d = m.assets();
    let n = m.times.len();
    let units = i128::from(c.quantity) * i128::from(c.multiplier);
    if p.dates != n
        || p.assets != d
        || p.values.len() != expected
        || p.market_sha256 != digest(m)?
        || p.content_sha256 != digest(&(&p.values, p.first_path, p.seed, p.namespace))?
        || p.values.iter().any(|v| !v.is_finite() || *v < 0.)
        || stops.len() != p.paths
        || c.multiplier == 0
        || c.multiplier > 1_000_000
        || units.abs() > 1_000_000_000
        || !c.premium.is_finite()
        || c.trading_costs.len() != d
        || c.liquidation_costs.len() != d
        || c.trading_costs
            .iter()
            .chain(&c.liquidation_costs)
            .any(|x| !x.is_finite() || *x < 0. || *x > 1.)
        || c.holder_id.is_empty()
        || c.holder_id.len() > 256
        || c.holder_sha256.len() != 64
        || !c.holder_sha256.bytes().all(|x| x.is_ascii_hexdigit())
        || stops.iter().any(|t| *t >= n || !m.exercise[*t])
    {
        return Err(invalid(
            "invalid max-call episode/cohort/holder/units/costs/stops",
        ));
    }
    let mut features = Vec::with_capacity(p.paths * (n - 1) * (d + 1));
    let mut amounts = Vec::with_capacity(p.paths);
    for (path, tau) in stops.iter().enumerate() {
        if p.state(path, 0)? != m.spots.as_slice() {
            return Err(invalid("initial path state differs from model"));
        }
        amounts.push(units as f64 * m.payoff(*tau, p.state(path, *tau)?)?);
        for t in 0..n - 1 {
            let state = p.state(path, t)?;
            features.extend_from_slice(state);
            features.push(m.payoff(t, state)?);
        }
    }
    let cash_growth = vec![1.; n - 1];
    let income_growth = vec![1.; (n - 1) * d];
    let mut marks = Vec::with_capacity(p.values.len());
    for path in 0..p.paths {
        for t in 0..n {
            for i in 0..d {
                let value =
                    p.state(path, t)?[i] * ((m.dividend_yields[i] - m.rate) * m.times[t]).exp();
                if !value.is_finite() {
                    return Err(invalid("discounted reinvested mark overflow"));
                }
                marks.push(value);
            }
        }
    }
    let dataset = format!(
        "maxcall-gbm-v1:{}",
        digest(&(
            m,
            p.seed,
            p.namespace,
            c,
            include_bytes!("multi_asset_market.rs").as_slice()
        ))?
    );
    let batch = EpisodeBatch {
        paths: p.paths,
        provenance: BatchProvenance {
            dataset_id: dataset,
            first_path: p.first_path,
        },
        marks,
        features,
        payoffs: vec![0.; p.paths],
        force_trade_signal: None,
        exercise: Some(ExerciseEpisode {
            settlement_step: stops.to_vec(),
            settlement_amounts: amounts,
        }),
        cash_income: None,
        initial_wealth: None,
    };
    let mut feature_names = (0..d).map(|i| format!("stock_{i}")).collect::<Vec<_>>();
    feature_names.push("discounted_unit_maxcall_payoff".into());
    Ok(MaxCallEpisodes {
        assets:d,dates:n,trading_costs:c.trading_costs.clone(),liquidation_costs:c.liquidation_costs.clone(),
        account_basis:"discounted reinvested-security marks exp((q_i-r)t)S_i(t); discounted settlement and fees; holdings in reinvested-security units; physical share count h_i exp(q_i t)".into(),
        content_sha256: digest(&(&batch, c, m))?,
        batch,
        cash_growth,
        income_growth,
        premium: c.premium,
        feature_names,
        market_sha256: p.market_sha256.clone(),
        physical_cohort_sha256:p.content_sha256.clone(),
        holder_id: c.holder_id.clone(),
        holder_sha256: c.holder_sha256.clone(),
    })
}

/// Fixed selected policy. Batch overrides permit resident GPU inference; inputs
/// are flat[current state,asset] and outputs one decision per state. Implementors
/// keep checkpoint weights and inference normalization unchanged during evaluation.
pub trait MaxCallStoppingPolicy {
    fn should_stop(&mut self, date: usize, state: &[f64]) -> Result<bool, ComputeError>;
    fn should_stop_batch(
        &mut self,
        date: usize,
        states: &[f64],
        assets: usize,
    ) -> Result<Vec<bool>, ComputeError> {
        if assets == 0 || !states.len().is_multiple_of(assets) {
            return Err(invalid("malformed policy state batch"));
        }
        states
            .chunks_exact(assets)
            .map(|s| self.should_stop(date, s))
            .collect()
    }
}
impl<F: FnMut(usize, &[f64]) -> Result<bool, ComputeError>> MaxCallStoppingPolicy for F {
    fn should_stop(&mut self, date: usize, state: &[f64]) -> Result<bool, ComputeError> {
        self(date, state)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyTailConfig {
    pub seed: u64,
    pub namespace: u64,
    pub lower_first_path: u64,
    pub upper_first_path: u64,
    pub lower_paths: usize,
    pub upper_paths: usize,
    pub inner_samples: usize,
    pub outer_batch_rows: usize,
    #[serde(default = "one_parent")]
    pub conditional_outer_batch_rows: usize,
    pub max_working_bytes: u64,
    pub max_transition_asset_steps: u64,
    pub policy_id: String,
    pub policy_sha256: String,
}
fn one_parent() -> usize {
    1
}
impl Default for PolicyTailConfig {
    fn default() -> Self {
        Self {
            seed: 196271,
            namespace: 301,
            lower_first_path: 0,
            upper_first_path: 0,
            lower_paths: 4096,
            upper_paths: 256,
            inner_samples: 256,
            outer_batch_rows: 128,
            conditional_outer_batch_rows: 1,
            max_working_bytes: 64 * 1024 * 1024,
            max_transition_asset_steps: 1_000_000_000,
            policy_id: "selected-policy".into(),
            policy_sha256: "0".repeat(64),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleEstimate {
    pub mean: f64,
    pub standard_error: f64,
    pub paths: usize,
}
fn estimate(samples: &[f64]) -> Result<SampleEstimate, ComputeError> {
    if samples.len() < 2 {
        return Err(invalid("at least two independent outer samples required"));
    }
    let (mut mean, mut m2) = (0., 0.);
    for (i, x) in samples.iter().enumerate() {
        if !x.is_finite() {
            return Err(invalid("nonfinite Monte Carlo sample"));
        }
        let delta = x - mean;
        mean += delta / (i + 1) as f64;
        m2 += delta * (x - mean);
    }
    let se = (m2 / (samples.len() - 1) as f64 / samples.len() as f64).sqrt();
    if !mean.is_finite() || !se.is_finite() {
        return Err(invalid("Monte Carlo moment overflow"));
    }
    Ok(SampleEstimate {
        mean,
        standard_error: se,
        paths: samples.len(),
    })
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyTailBounds {
    pub lower: SampleEstimate,
    pub upper: SampleEstimate,
    pub midpoint: f64,
    pub asymptotic_joint_95_enclosure: [f64; 2],
    pub lower_samples: Vec<f64>,
    pub upper_samples: Vec<f64>,
    pub mean_noisy_martingale_increment: Vec<f64>,
    pub market_sha256: String,
    pub evaluation: PolicyTailConfig,
    pub estimator_id: String,
    pub estimator: String,
    pub confidence_scope: String,
    pub lower_cohort_sha256: String,
    pub upper_cohort_sha256: String,
}
fn checked_decisions<P: MaxCallStoppingPolicy + ?Sized>(
    p: &mut P,
    date: usize,
    states: &[f64],
    assets: usize,
) -> Result<Vec<bool>, ComputeError> {
    let out = p.should_stop_batch(date, states, assets)?;
    if out.len() != states.len() / assets {
        return Err(invalid("policy returned wrong batch decision count"));
    }
    Ok(out)
}
/// Mean discounted payoff when forced to CONTINUE at start, then applying the
/// fixed policy from start+1. Fresh conditionally independent full-policy tails
/// supply unbiased Cbar_start, rather than a learned value-proxy substitution.
fn policy_tail_means<P: MaxCallStoppingPolicy + ?Sized>(
    m: &MaxCallMarket,
    factor: &CorrelationFactor,
    p: &mut P,
    start: usize,
    parent_states: &[f64],
    keys: &[RandomKey],
    draws: usize,
) -> Result<Vec<f64>, ComputeError> {
    let d = m.assets();
    let n = m.times.len();
    let parents = keys.len();
    if parent_states.len() != parents * d || parents == 0 {
        return Err(invalid("malformed conditional parent group"));
    }
    let total = parents * draws;
    let mut states = Vec::with_capacity(total * d);
    for parent in parent_states.chunks_exact(d) {
        for _ in 0..draws {
            states.extend_from_slice(parent);
        }
    }
    let mut active = vec![true; total];
    let mut rewards = vec![0.; total];
    let seeds = (0..total)
        .map(|index| {
            let j = index % draws;
            let key = keys[index / draws];
            stream_seed(
                RandomKey {
                    branch: ((start as u64) << 32) | j as u64,
                    ..key
                },
                1,
            )
        })
        .collect::<Vec<_>>();
    for t in start..n - 1 {
        let mut indices = Vec::new();
        let mut current = Vec::new();
        for j in 0..total {
            if active[j] {
                let next = &mut states[j * d..(j + 1) * d];
                advance(m, factor, t, next, seeds[j])?;
                if m.exercise[t + 1] {
                    indices.push(j);
                    current.extend_from_slice(next);
                }
            }
        }
        if !indices.is_empty() {
            let flags = if t + 1 == n - 1 {
                vec![true; indices.len()]
            } else {
                checked_decisions(p, t + 1, &current, d)?
            };
            for (j, stop) in indices.into_iter().zip(flags) {
                if stop {
                    rewards[j] = m.payoff(t + 1, &states[j * d..(j + 1) * d])?;
                    active[j] = false;
                }
            }
        }
        if !active.iter().any(|v| *v) {
            break;
        }
    }
    rewards
        .chunks_exact(draws)
        .map(|values| {
            let mean = values.iter().sum::<f64>() / draws as f64;
            if !mean.is_finite() {
                return Err(invalid("conditional policy-tail payoff overflow"));
            }
            Ok(mean)
        })
        .collect()
}
/// Selected-policy-tail nested dual: Becker/Cheridito/Jentzen2019 Proposition7
/// (https://jmlr.org/papers/v20/18-232.html), and2020 section3.2
/// (https://arxiv.org/abs/1912.11060). With Cbar_n an independent conditional
/// policy-tail mean and f_n the fixed policy on the OUTER state,
/// deltaM_n=f_n*g_n+(1-f_n)*Cbar_n-Cbar_(n-1), Cbar_N=0.
/// The noisy process equals the policy-value Doob martingale plus errors whose
/// conditional means vanish; conditional Jensen gives an upper EXPECTATION.
/// It need not be a martingale under a filtration revealing auxiliary estimates.
/// Max-call payoffs are unbounded: confidence output uses asymptotic CLT only.
pub fn evaluate_policy_tail<P: MaxCallStoppingPolicy + ?Sized>(
    m: &MaxCallMarket,
    policy: &mut P,
    c: &PolicyTailConfig,
) -> Result<PolicyTailBounds, ComputeError> {
    let factor = m.validate()?;
    let n = m.times.len();
    let d = m.assets();
    if c.lower_paths < 2
        || c.upper_paths < 2
        || c.lower_paths > 8_000_000
        || c.upper_paths > 2_000_000
        || c.inner_samples == 0
        || c.inner_samples > 65536
        || c.outer_batch_rows == 0
        || c.outer_batch_rows > 65536
        || !(1..=32).contains(&c.conditional_outer_batch_rows)
        || c.policy_id.is_empty()
        || c.policy_id.len() > 256
        || c.policy_sha256.len() != 64
        || !c.policy_sha256.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(invalid("invalid policy-tail evaluation counts/provenance"));
    }
    c.lower_first_path
        .checked_add(c.lower_paths as u64)
        .ok_or_else(|| invalid("lower path range overflow"))?;
    c.upper_first_path
        .checked_add(c.upper_paths as u64)
        .ok_or_else(|| invalid("upper path range overflow"))?;
    let worst =
        (c.upper_paths as u128) * (c.inner_samples as u128) * (n as u128) * ((n - 1) as u128) / 2
            * (d as u128)
            + ((c.lower_paths + c.upper_paths) as u128) * ((n - 1) as u128) * (d as u128);
    let working = ((c.outer_batch_rows as u128) * (n as u128) * (d as u128) * 3
        + (c.inner_samples as u128)
            * (c.conditional_outer_batch_rows as u128)
            * ((d as u128) * 8 + 8)
        + (d as u128) * (d as u128) * 4
        + 512
        + (c.lower_paths + c.upper_paths) as u128 * 2)
        * 8;
    if c.max_transition_asset_steps == 0
        || c.max_transition_asset_steps > 100_000_000_000
        || worst > c.max_transition_asset_steps as u128
        || c.max_working_bytes == 0
        || c.max_working_bytes > 8_589_934_592
        || working > c.max_working_bytes as u128
    {
        return Err(invalid(
            "policy-tail transition or memory admission exceeded",
        ));
    }
    let lower_namespace = c.namespace ^ 0x4c4f574552;
    let upper_namespace = c.namespace ^ 0x5550504552;
    let mut lower = Vec::with_capacity(c.lower_paths);
    let mut upper = Vec::with_capacity(c.upper_paths);
    let mut increments = vec![0.; n - 1];
    let mut lower_hash = Sha256::new();
    let mut upper_hash = Sha256::new();
    for (is_upper, total, first, namespace) in [
        (false, c.lower_paths, c.lower_first_path, lower_namespace),
        (true, c.upper_paths, c.upper_first_path, upper_namespace),
    ] {
        for offset in (0..total).step_by(c.outer_batch_rows) {
            let count = c.outer_batch_rows.min(total - offset);
            let paths = generate_paths(
                m,
                count,
                c.seed,
                first + offset as u64,
                namespace,
                c.max_working_bytes,
            )?;
            let cohort_hash = if is_upper {
                &mut upper_hash
            } else {
                &mut lower_hash
            };
            if offset == 0 {
                cohort_hash.update(digest(&(m, c.seed, namespace, first, total))?.as_bytes());
            }
            for value in &paths.values {
                cohort_hash.update(value.to_le_bytes());
            }
            // All upper-date decisions are evaluated, including dates after an
            // outer holder would exercise. Lower evaluation uses its first stop.
            let mut flags = vec![vec![false; count]; n];
            for (t, date_flags) in flags.iter_mut().enumerate() {
                if m.exercise[t] {
                    if t == n - 1 {
                        date_flags.fill(true);
                    } else {
                        let mut states = Vec::with_capacity(count * d);
                        for path in 0..count {
                            states.extend_from_slice(paths.state(path, t)?);
                        }
                        *date_flags = checked_decisions(policy, t, &states, d)?;
                    }
                }
            }
            let mut all_conditionals = vec![vec![0.; count]; if is_upper { n } else { 0 }];
            if is_upper {
                for (date, date_values) in all_conditionals.iter_mut().enumerate().take(n - 1) {
                    for start in (0..count).step_by(c.conditional_outer_batch_rows) {
                        let stop = count.min(start + c.conditional_outer_batch_rows);
                        let mut states = Vec::with_capacity((stop - start) * d);
                        let mut keys = Vec::with_capacity(stop - start);
                        for path in start..stop {
                            states.extend_from_slice(paths.state(path, date)?);
                            keys.push(RandomKey {
                                seed: c.seed,
                                namespace,
                                path_id: first + offset as u64 + path as u64,
                                branch: 0,
                            });
                        }
                        let values = policy_tail_means(
                            m,
                            &factor,
                            policy,
                            date,
                            &states,
                            &keys,
                            c.inner_samples,
                        )?;
                        for (path, value) in (start..stop).zip(values) {
                            date_values[path] = value;
                        }
                    }
                }
            }
            for (path, _) in paths.values.chunks_exact(n * d).enumerate() {
                if !is_upper {
                    let tau = (0..n)
                        .find(|t| flags[*t][path])
                        .ok_or_else(|| invalid("policy failed terminal stopping"))?;
                    lower.push(m.payoff(tau, paths.state(path, tau)?)?);
                    continue;
                }

                let mut maximum = if m.exercise[0] {
                    m.payoff(0, paths.state(path, 0)?)?
                } else {
                    f64::NEG_INFINITY
                };
                let mut stopped_sum = 0.;
                let mut previous_m = 0.;
                for t in 1..n {
                    let g = m.payoff(t, paths.state(path, t)?)?;
                    if flags[t][path] {
                        stopped_sum += g - all_conditionals[t][path];
                    }
                    let martingale =
                        stopped_sum + all_conditionals[t][path] - all_conditionals[0][path];
                    increments[t - 1] += martingale - previous_m;
                    previous_m = martingale;
                    if m.exercise[t] {
                        maximum = maximum.max(g - martingale);
                    }
                }
                if !maximum.is_finite() {
                    return Err(invalid("nonfinite policy-tail upper sample"));
                }
                upper.push(maximum);
            }
        }
    }
    let l = estimate(&lower)?;
    let u = estimate(&upper)?;
    Ok(PolicyTailBounds{midpoint:0.5*(l.mean+u.mean),asymptotic_joint_95_enclosure:[l.mean-1.959963984540054*l.standard_error,u.mean+1.959963984540054*u.standard_error],lower:l,upper:u,
        lower_samples:lower,upper_samples:upper,mean_noisy_martingale_increment:increments.into_iter().map(|x|x/c.upper_paths as f64).collect(),market_sha256:digest(m)?,evaluation:c.clone(),
        lower_cohort_sha256:format!("{:x}",lower_hash.finalize()),upper_cohort_sha256:format!("{:x}",upper_hash.finalize()),
        estimator_id:"becker_policy_tail_nested_v1".into(),estimator:"Selected-policy forced-continuation nested tails; globally discounted rewards; noisy policy-value Doob increments; conditional-error Jensen upper expectation".into(),
        confidence_scope:"Independent selected-model evaluation, independent per-outer auxiliary streams; asymptotic CLT one-sided intervals combined at95%; unbounded max-call, no bounded-payoff finite-sample guarantee".into()})
}

/// Independent scalar account for a supplied reinvested-security control.
/// Holdings[path,decision,asset] are ignored from settlement onward; opening,
/// rebalancing and a single liquidation are charged on discounted notionals.
pub fn evaluate_episode_holdings(
    e: &MaxCallEpisodes,
    holdings: &[f64],
) -> Result<Vec<f64>, ComputeError> {
    let (p, n, d) = (e.batch.paths, e.dates, e.assets);
    let count = p
        .checked_mul(n)
        .and_then(|v| v.checked_mul(d))
        .ok_or_else(|| invalid("episode account shape overflow"))?;
    let exercise = e
        .batch
        .exercise
        .as_ref()
        .ok_or_else(|| invalid("stopped episode required"))?;
    if p == 0
        || p > 2_000_000
        || !(2..=1025).contains(&n)
        || d == 0
        || d > 100
        || count > 100_000_000
        || e.batch.marks.len() != count
        || holdings.len() != p * (n - 1) * d
        || holdings.iter().any(|v| !v.is_finite())
        || e.batch.marks.iter().any(|v| !v.is_finite() || *v < 0.)
        || exercise.settlement_step.len() != p
        || exercise.settlement_amounts.len() != p
        || exercise.settlement_step.iter().any(|v| *v >= n)
        || exercise.settlement_amounts.iter().any(|v| !v.is_finite())
        || !e.premium.is_finite()
        || e.trading_costs.len() != d
        || e.liquidation_costs.len() != d
        || e.trading_costs
            .iter()
            .chain(&e.liquidation_costs)
            .any(|v| !v.is_finite() || *v < 0.)
        || e.cash_growth != vec![1.; n - 1]
        || e.income_growth != vec![1.; (n - 1) * d]
    {
        return Err(invalid("invalid discounted stopped account"));
    }
    let mut output = Vec::with_capacity(p);
    for path in 0..p {
        let tau = exercise.settlement_step[path];
        let mut value = e.premium;
        let mut previous = vec![0.; d];
        for t in 0..tau {
            let mut cash = value;
            let mut next = 0.;
            for i in 0..d {
                let target = holdings[(path * (n - 1) + t) * d + i];
                let mark = e.batch.marks[(path * n + t) * d + i];
                cash -= target * mark + (target - previous[i]).abs() * mark * e.trading_costs[i];
                next += target * e.batch.marks[(path * n + t + 1) * d + i];
                previous[i] = target;
            }
            value = cash + next;
        }
        for (i, h) in previous.iter().enumerate() {
            value -= h.abs() * e.batch.marks[(path * n + tau) * d + i] * e.liquidation_costs[i];
        }
        value -= exercise.settlement_amounts[path];
        if !value.is_finite() {
            return Err(invalid("discounted account overflow"));
        }
        output.push(value);
    }
    Ok(output)
}
