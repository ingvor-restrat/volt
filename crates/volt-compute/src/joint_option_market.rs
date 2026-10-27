//! Bounded joint index-return / five-coefficient IV-surface scenarios.
//!
//! Transition and surface equations follow the published author repository for
//! François et al., arXiv:2504.06208v3, commit64aef8c. Parameters and innovation
//! quantiles are explicit inputs. Historical calibration is never inferred from
//! filenames. The source's random-history selection and extreme-path deletion
//! are separate sampling policies; this module rejects invalid paths.
use crate::{
    ComputeError, OptionRow, OptionSide,
    aad::analytic_greeks,
    native_learner::{BatchProvenance, DatasetPartition, EpisodeBatch, Split},
    volatility_surface::{EuropeanInputs, european_price_and_vega},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub const AUTHOR_REPOSITORY: &str = "https://github.com/cpmendoza/deep-hedging_with_options";
pub const AUTHOR_COMMIT: &str = "64aef8c35b3cc4d31de8c499e2cc9e9f23fb5b3d";
pub const RNG_CONTRACT: &str = "splitmix64-counter.v1; seed/path/step/domain keyed; BoxMuller six normals; original copula order return,beta1..5; output permutation1,2,3,4,5,0; CDFclip[.0001,0.9995]; linear quantiles";
pub const FEATURE_NAMES: [&str; 17] = [
    "log_stock",
    "horizon_years_left",
    "beta1",
    "beta2",
    "beta3",
    "beta4",
    "beta5",
    "h_beta1",
    "h_beta2",
    "h_beta3",
    "h_beta4",
    "h_beta5",
    "h_return",
    "straddle_value",
    "straddle_fixed_vol_delta",
    "straddle_gamma_feature",
    "hedge_call_value",
];
fn invalid(s: impl Into<String>) -> ComputeError {
    ComputeError::InvalidInput(s.into())
}
fn finite(x: f64, label: &str) -> Result<f64, ComputeError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(invalid(format!("nonfinite {label}")))
    }
}
/// Exact row/column convention of author Parameters_2.csv (excluding labels).
/// Columns beta1..5,return. Rows intercept/lambda, five VAR rows, beta2 second
/// lag, annual beta sigma, level scales, persistence, ARCH, leverage, zeta,phi.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointParameters {
    pub rows: [[f64; 6]; 14],
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointState {
    pub stock: f64,
    pub betas: [f64; 5],
    pub lag_betas: [f64; 5],
    pub conditional_variances: [f64; 6],
    pub previous_innovations: [f64; 6],
}
/// Quantile rows have columns beta1..5,return. The author uses1000 evenly
/// spaced probabilities. Explicit grids permit a source-produced table or
/// labelled synthetic fixtures; their distribution provenance remains caller-owned.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InnovationModel {
    pub original_correlation: [[f64; 6]; 6],
    pub probabilities: Vec<f64>,
    pub quantiles: Vec<[f64; 6]>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointProvenance {
    pub calibration_id: String,
    pub calibration_sha256: String,
    pub initial_state_id: String,
    pub initial_state_sha256: String,
    pub innovation_table_id: String,
    pub innovation_table_sha256: String,
    pub synthetic: bool,
    pub availability_note: String,
    #[serde(default)]
    pub decision_origin_ns: Option<i64>,
    #[serde(default)]
    pub calibration_known_ns: Option<i64>,
    #[serde(default)]
    pub initial_state_known_ns: Option<i64>,
}
/// The author's BS gamma feature discounts by sqrt(tau). Preserve it explicitly
/// for source parity; canonical price derivatives use exp(-q*tau).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GammaFeatureConvention {
    #[default]
    AuthorSqrtTimeDividend,
    FixedVolatilityDerivative,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointOptionConfig {
    pub seed: u64,
    pub steps: usize,
    pub dt: f64,
    pub rate: f64,
    pub dividend_yield: f64,
    pub volatility_floor: Option<f64>,
    pub straddle_days: usize,
    pub hedge_call_days: usize,
    pub stock_cost: f64,
    pub option_cost: f64,
    pub terminal_costs: [f64; 2],
    #[serde(default)]
    pub gamma_feature: GammaFeatureConvention,
    pub max_working_bytes: u64,
    pub provenance: JointProvenance,
}
impl JointOptionConfig {
    /// Financial settings of the63-day paper case; all statistical inputs supplied.
    pub fn paper_case(seed: u64, provenance: JointProvenance) -> Self {
        Self {
            seed,
            steps: 63,
            dt: 1. / 252.,
            rate: 0.026623194,
            dividend_yield: 0.01772245,
            volatility_floor: Some(0.01),
            straddle_days: 63,
            hedge_call_days: 84,
            stock_cost: 0.0005,
            option_cost: 0.02,
            terminal_costs: [0., 0.02],
            gamma_feature: GammaFeatureConvention::AuthorSqrtTimeDividend,
            max_working_bytes: 64 * 1024 * 1024,
            provenance,
        }
    }
    pub fn validate(&self) -> Result<(), ComputeError> {
        if self.steps == 0
            || self.steps > 252
            || self.straddle_days < self.steps
            || self.straddle_days > 2520
            || self.hedge_call_days <= self.steps
            || self.hedge_call_days > 2520
            || !self.dt.is_finite()
            || self.dt <= 0.
            || self.dt > 1.
            || ![
                self.rate,
                self.dividend_yield,
                self.stock_cost,
                self.option_cost,
                self.terminal_costs[0],
                self.terminal_costs[1],
            ]
            .iter()
            .all(|x| x.is_finite())
            || self.stock_cost < 0.
            || self.option_cost < 0.
            || self.terminal_costs.iter().any(|x| *x < 0.)
            || self.max_working_bytes == 0
            || self.max_working_bytes > 1024 * 1024 * 1024
        {
            return Err(invalid("invalid joint market horizon/carry/cost/budget"));
        }
        if self
            .volatility_floor
            .is_some_and(|x| !x.is_finite() || x <= 0.)
        {
            return Err(invalid("volatility floor must be positive finite"));
        }
        if [self.rate, self.dividend_yield].iter().any(|rate| {
            let growth = (*rate * self.dt).exp();
            !growth.is_finite() || growth <= 0.
        }) {
            return Err(invalid("cash/dividend growth overflow or underflow"));
        }
        for s in [
            &self.provenance.calibration_id,
            &self.provenance.calibration_sha256,
            &self.provenance.initial_state_id,
            &self.provenance.initial_state_sha256,
            &self.provenance.innovation_table_id,
            &self.provenance.innovation_table_sha256,
            &self.provenance.availability_note,
        ] {
            if s.trim().is_empty() || s.len() > 4096 {
                return Err(invalid("missing or excessive joint provenance"));
            }
        }
        for h in [
            &self.provenance.calibration_sha256,
            &self.provenance.initial_state_sha256,
            &self.provenance.innovation_table_sha256,
        ] {
            if h.len() != 64 || !h.bytes().all(|c| c.is_ascii_hexdigit()) {
                return Err(invalid(
                    "provenance digests require64 hexadecimal characters",
                ));
            }
        }
        match (
            self.provenance.decision_origin_ns,
            self.provenance.calibration_known_ns,
            self.provenance.initial_state_known_ns,
        ) {
            (Some(origin), Some(calibration), Some(initial))
                if calibration <= origin && initial <= origin => {}
            (None, None, None) if self.provenance.synthetic => {}
            _ => {
                return Err(invalid(
                    "point-in-time source knowledge must precede decision origin; real inputs require timestamps",
                ));
            }
        }
        Ok(())
    }
}
impl JointParameters {
    pub fn validate(&self) -> Result<(), ComputeError> {
        if self.rows.iter().flatten().any(|x| !x.is_finite())
            || self.rows[13].iter().any(|x| *x <= 0.)
        {
            return Err(invalid("invalid joint parameter matrix"));
        }
        Ok(())
    }
}
impl JointState {
    pub fn validate(&self) -> Result<(), ComputeError> {
        if !self.stock.is_finite()
            || self.stock <= 0.
            || self
                .betas
                .iter()
                .chain(&self.lag_betas)
                .chain(&self.previous_innovations)
                .any(|x| !x.is_finite())
            || self
                .conditional_variances
                .iter()
                .any(|x| !x.is_finite() || *x < 0.)
        {
            return Err(invalid(
                "invalid joint state; conditional variances must be nonnegative",
            ));
        }
        Ok(())
    }
}
impl InnovationModel {
    pub fn validate(&self) -> Result<(), ComputeError> {
        let n = self.probabilities.len();
        if !(2..=4096).contains(&n)
            || n != self.quantiles.len()
            || self.probabilities[0] != 0.0001
            || self.probabilities[n - 1] != 0.9995
            || self
                .probabilities
                .windows(2)
                .any(|x| !x[0].is_finite() || x[0] >= x[1])
            || self.quantiles.iter().flatten().any(|x| !x.is_finite())
        {
            return Err(invalid("invalid innovation probability/quantile grid"));
        }
        for col in 0..6 {
            if self.quantiles.windows(2).any(|x| x[0][col] > x[1][col]) {
                return Err(invalid("innovation quantiles must be monotone"));
            }
        }
        self.cholesky()?;
        Ok(())
    }
    fn cholesky(&self) -> Result<[[f64; 6]; 6], ComputeError> {
        let mut l = [[0.; 6]; 6];
        for i in 0..6 {
            for j in 0..=i {
                let c = self.original_correlation[i][j];
                if !c.is_finite()
                    || c.abs() > 1.
                    || !self.original_correlation[j][i].is_finite()
                    || (c - self.original_correlation[j][i]).abs() > 1e-12
                    || (i == j && (c - 1.).abs() > 1e-12)
                {
                    return Err(invalid("invalid copula correlation"));
                }
                let a = c - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
                l[i][j] = if i == j {
                    if a <= 1e-14 {
                        return Err(invalid("copula correlation must be positive definite"));
                    }
                    a.sqrt()
                } else {
                    a / l[j][j]
                };
            }
        }
        Ok(l)
    }
    /// Explicit independent standard normals; preserves author copula ordering.
    pub fn from_standard_normals(&self, z: [f64; 6]) -> Result<[f64; 6], ComputeError> {
        self.validate()?;
        self.transform(z, &self.cholesky()?)
    }
    fn transform(&self, z: [f64; 6], l: &[[f64; 6]; 6]) -> Result<[f64; 6], ComputeError> {
        if z.iter().any(|x| !x.is_finite()) {
            return Err(invalid("nonfinite copula normal"));
        }
        let mut e = [0.; 6];
        for (out, original) in [1, 2, 3, 4, 5, 0].into_iter().enumerate() {
            let x = (0..=original).map(|j| l[original][j] * z[j]).sum::<f64>();
            let u =
                (0.5 * (1. + libm::erf(x * std::f64::consts::FRAC_1_SQRT_2))).clamp(0.0001, 0.9995);
            let hi = self
                .probabilities
                .partition_point(|p| *p <= u)
                .min(self.probabilities.len() - 1);
            let lo = hi.saturating_sub(1);
            let w =
                (u - self.probabilities[lo]) / (self.probabilities[hi] - self.probabilities[lo]);
            e[out] = finite(
                self.quantiles[lo][out] + w * (self.quantiles[hi][out] - self.quantiles[lo][out]),
                "interpolated innovation",
            )?;
        }
        Ok(e)
    }
}
/// Annual IV at author's M=ln(F/K)/sqrt(tau), including asymmetric smile.
pub fn surface_volatility(
    b: [f64; 5],
    m: f64,
    tau: f64,
    floor: Option<f64>,
) -> Result<f64, ComputeError> {
    if !m.is_finite()
        || !tau.is_finite()
        || tau <= 0.
        || b.iter().any(|x| !x.is_finite())
        || floor.is_some_and(|x| !x.is_finite() || x <= 0.)
    {
        return Err(invalid("invalid surface coordinate"));
    }
    let negative = m.min(0.);
    let slope = if m < 0. { (negative).tanh() } else { m };
    let value = b[0]
        + b[1] * (-(tau / 0.25).sqrt()).exp()
        + b[2] * slope
        + b[3] * (-(-m * m).exp_m1()) * (tau / 5.).ln()
        + if m < 0. {
            b[4] * (-((3. * negative).powi(3)).exp_m1()) * (tau / 5.).ln()
        } else {
            0.
        };
    let value = finite(value, "surface volatility")?;
    let value = floor.map_or(value, |f| value.max(f));
    if value <= 0. {
        return Err(invalid(
            "nonpositive surface volatility without positive floor",
        ));
    }
    Ok(value)
}
/// Source cumulant of the standardized return innovation. A finite moment
/// domain is required; invalid paths are reported rather than discarded/retried.
pub fn return_cumulant(p: &JointParameters, z: f64) -> Result<f64, ComputeError> {
    let phi = p.rows[13][5];
    let zeta = p.rows[12][5];
    let a = phi * phi + zeta * zeta;
    let rad = a - (zeta + z) * (zeta + z);
    if !z.is_finite() || !rad.is_finite() || rad < 0. || phi <= 0. {
        return Err(invalid("NIG return cumulant outside moment domain"));
    }
    finite(
        phi * phi / a * (-zeta * z + phi * phi - phi * rad.sqrt()),
        "NIG cumulant",
    )
}
/// Reconstruct source initial innovations from three beta rows and last h/return.
/// The author historical return column is Ret_m_Rf_dvdy; no extra carry is
/// subtracted here. Historical availability must be validated by the producer.
pub fn state_from_history(
    p: &JointParameters,
    c: &JointOptionConfig,
    stock: f64,
    beta_last3: [[f64; 5]; 3],
    h: [f64; 6],
    last_excess_return: f64,
) -> Result<JointState, ComputeError> {
    p.validate()?;
    c.validate()?;
    if !last_excess_return.is_finite()
        || beta_last3.iter().flatten().any(|x| !x.is_finite())
        || h.iter().any(|x| !x.is_finite() || *x <= 0.)
    {
        return Err(invalid("invalid initial history"));
    }
    let mut e = [0.; 6];
    for i in 0..5 {
        let ar = (0..5)
            .map(|j| p.rows[j + 1][i] * beta_last3[1][j])
            .sum::<f64>();
        e[i] = (beta_last3[2][i]
            - p.rows[0][i]
            - ar
            - if i == 1 {
                p.rows[6][1] * beta_last3[0][1]
            } else {
                0.
            })
            / (h[i] * c.dt).sqrt();
    }
    let z = (h[5] * c.dt).sqrt();
    e[5] = (last_excess_return - return_cumulant(p, -p.rows[0][5] * z)?
        + return_cumulant(p, (1. - p.rows[0][5]) * z)?)
        / z;
    let s = JointState {
        stock,
        betas: beta_last3[2],
        lag_betas: beta_last3[1],
        conditional_variances: h,
        previous_innovations: e,
    };
    s.validate()?;
    Ok(s)
}
pub fn advance(
    p: &JointParameters,
    c: &JointOptionConfig,
    s: &JointState,
    e: [f64; 6],
) -> Result<(JointState, f64), ComputeError> {
    p.validate()?;
    c.validate()?;
    s.validate()?;
    if e.iter().any(|x| !x.is_finite()) {
        return Err(invalid("nonfinite innovation"));
    }
    advance_checked(p, c, s, e)
}
fn advance_checked(
    p: &JointParameters,
    c: &JointOptionConfig,
    s: &JointState,
    e: [f64; 6],
) -> Result<(JointState, f64), ComputeError> {
    let atm = surface_volatility(s.betas, 0., 1. / 12., c.volatility_floor)?;
    let mut h = [0.; 6];
    for (i, value) in h.iter_mut().enumerate() {
        let level = if i == 0 || i == 5 {
            (p.rows[8][i] * atm).powi(2)
        } else {
            p.rows[7][i].powi(2) / 252.
        };
        *value = finite(
            level
                + p.rows[9][i] * (s.conditional_variances[i] - level)
                + p.rows[10][i]
                    * s.conditional_variances[i]
                    * (s.previous_innovations[i].powi(2)
                        - 1.
                        - 2. * s.previous_innovations[i] * p.rows[11][i]),
            "next conditional variance",
        )?;
        if *value < 0. {
            return Err(invalid(format!(
                "negative conditional variance at channel{i}"
            )));
        }
    }
    let z = (h[5] * c.dt).sqrt();
    let ret = finite(
        (c.rate - c.dividend_yield) * c.dt + return_cumulant(p, -p.rows[0][5] * z)?
            - return_cumulant(p, (1. - p.rows[0][5]) * z)?
            + z * e[5],
        "log return",
    )?;
    let mut b = [0.; 5];
    for i in 0..5 {
        b[i] = finite(
            p.rows[0][i]
                + (0..5).map(|j| p.rows[j + 1][i] * s.betas[j]).sum::<f64>()
                + if i == 1 {
                    p.rows[6][1] * s.lag_betas[1]
                } else {
                    0.
                }
                + (h[i] * c.dt).sqrt() * e[i],
            "next surface coefficient",
        )?;
    }
    let next = JointState {
        stock: finite(s.stock * ret.exp(), "stock")?,
        betas: b,
        lag_betas: s.betas,
        conditional_variances: h,
        previous_innovations: e,
    };
    next.validate()?;
    Ok((next, ret))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioRange {
    pub split: Split,
    pub first_path: u64,
    pub paths: usize,
}
/// Disjoint IDs under one seeded generator; split labels alone do not prevent reuse.
pub fn validate_ranges(ranges: &[ScenarioRange]) -> Result<(), ComputeError> {
    if ranges.is_empty() || ranges.len() > 64 {
        return Err(invalid("invalid split count"));
    }
    let mut spans = Vec::new();
    for r in ranges {
        if r.paths == 0 {
            return Err(invalid("empty scenario range"));
        }
        let end = r
            .first_path
            .checked_add(u64::try_from(r.paths).map_err(|_| invalid("range count overflow"))?)
            .ok_or_else(|| invalid("scenario range overflow"))?;
        spans.push((r.first_path, end));
    }
    spans.sort_unstable();
    if spans.windows(2).any(|r| r[0].1 > r[1].0) {
        return Err(invalid("overlapping scenario ranges across splits"));
    }
    Ok(())
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EconomicTerms {
    pub premium: f64,
    pub currency: String,
    pub units: String,
    pub trading_costs: [f64; 2],
    pub liquidation_costs: [f64; 2],
    pub cash_growth: Vec<f64>,
    pub income_growth: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointOptionScenarios {
    pub config: JointOptionConfig,
    pub range: ScenarioRange,
    pub dataset_id: String,
    pub content_sha256: String,
    pub batch: EpisodeBatch,
    pub economics: EconomicTerms,
    pub feature_names: Vec<String>,
    pub states: Vec<JointState>,
    pub log_returns: Vec<f64>,
    pub innovations: Vec<[f64; 6]>,
    pub estimated_working_bytes: u64,
    pub sampling_note: String,
}
fn digest(v: &impl Serialize) -> Result<String, ComputeError> {
    let b = serde_json::to_vec(v).map_err(|e| invalid(e.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(b)))
}
/// Checked conservative host reservation, including transient JSON/hash copies.
pub fn plan(c: &JointOptionConfig, range: &ScenarioRange) -> Result<u64, ComputeError> {
    c.validate()?;
    validate_ranges(std::slice::from_ref(range))?;
    let points = range
        .paths
        .checked_mul(c.steps + 1)
        .ok_or_else(|| invalid("scenario shape overflow"))?;
    let bytes = points
        .checked_mul(4096)
        .and_then(|n| n.checked_add(8 * 1024 * 1024))
        .ok_or_else(|| invalid("scenario memory overflow"))?;
    let bytes = u64::try_from(bytes).map_err(|_| invalid("scenario memory overflow"))?;
    if bytes > c.max_working_bytes {
        return Err(invalid(format!(
            "joint scenarios need{bytes} modeled host bytes; budget{}",
            c.max_working_bytes
        )));
    }
    Ok(bytes)
}
/// Seeded paths are stable across chunk boundaries. Caller must register all
/// splits together with validate_ranges; a single generation cannot detect
/// another process reusing the same dataset/path IDs.
pub fn generate(
    p: &JointParameters,
    c: &JointOptionConfig,
    initial: &JointState,
    model: &InnovationModel,
    range: ScenarioRange,
) -> Result<JointOptionScenarios, ComputeError> {
    plan(c, &range)?;
    p.validate()?;
    initial.validate()?;
    model.validate()?;
    let l = model.cholesky()?;
    let count = range
        .paths
        .checked_mul(c.steps)
        .ok_or_else(|| invalid("innovation shape overflow"))?;
    let mut e = Vec::with_capacity(count);
    for path in range.first_path..range.first_path + range.paths as u64 {
        for step in 0..c.steps {
            let mut normals = [0.; 6];
            for pair in 0..3 {
                let u = uniform(c.seed, path, step as u64, (pair * 2) as u64);
                let v = uniform(c.seed, path, step as u64, (pair * 2 + 1) as u64);
                let radius = (-2. * u.ln()).sqrt();
                let angle = std::f64::consts::TAU * v;
                normals[pair * 2] = radius * angle.cos();
                normals[pair * 2 + 1] = radius * angle.sin();
            }
            e.push(model.transform(normals, &l)?);
        }
    }
    let id = dataset_identity(p, c, initial, Some(model))?;
    assemble(
        p,
        c,
        initial,
        range,
        e,
        id,
        "Seeded counter innovations; no path deletion/resampling; supplied quantile-grid provenance.",
    )
}
/// Exact supplied standardized innovations in [path,step,channel] order.
/// This route makes no claim of random sampling; identity binds their bytes.
pub fn generate_with_innovations(
    p: &JointParameters,
    c: &JointOptionConfig,
    initial: &JointState,
    range: ScenarioRange,
    e: Vec<[f64; 6]>,
) -> Result<JointOptionScenarios, ComputeError> {
    plan(c, &range)?;
    p.validate()?;
    initial.validate()?;
    if e.len() != range.paths * c.steps || e.iter().flatten().any(|x| !x.is_finite()) {
        return Err(invalid("supplied innovation shape/finiteness mismatch"));
    }
    let id = dataset_identity(p, c, initial, None)?;
    let id = digest(&(id, digest(&e)?, range.first_path, range.paths))?;
    assemble(
        p,
        c,
        initial,
        range,
        e,
        id,
        "Explicit fixed innovations; dataset identity includes input bytes/range; no random-distribution claim.",
    )
}
fn dataset_identity(
    p: &JointParameters,
    c: &JointOptionConfig,
    s: &JointState,
    m: Option<&InnovationModel>,
) -> Result<String, ComputeError> {
    digest(&(
        "volt.joint_option_market.v1",
        AUTHOR_COMMIT,
        RNG_CONTRACT,
        p,
        c,
        s,
        m,
    ))
}
fn assemble(
    p: &JointParameters,
    c: &JointOptionConfig,
    initial: &JointState,
    range: ScenarioRange,
    e: Vec<[f64; 6]>,
    id: String,
    note: &str,
) -> Result<JointOptionScenarios, ComputeError> {
    let reserved = plan(c, &range)?;
    let mut states = Vec::with_capacity(range.paths * (c.steps + 1));
    let mut returns = Vec::with_capacity(e.len());
    let mut marks = Vec::with_capacity(range.paths * (c.steps + 1) * 2);
    let mut features = Vec::with_capacity(range.paths * c.steps * 17);
    let mut payoffs = Vec::with_capacity(range.paths);
    let mut premium = None;
    for path in 0..range.paths {
        let mut state = initial.clone();
        for t in 0..=c.steps {
            let remaining = (c.straddle_days - t) as f64 * c.dt;
            let call_tau = (c.hedge_call_days - t) as f64 * c.dt;
            let liability = option_values(c, &state, initial.stock, remaining)?;
            let call = option_value(c, &state, initial.stock, call_tau, OptionSide::Call)?;
            marks.extend([state.stock, call.0]);
            if t == 0 {
                premium = Some(liability.0);
            }
            if t < c.steps {
                features.extend([state.stock.ln(), (c.steps - t) as f64 * c.dt]);
                features.extend(state.betas);
                features.extend(state.conditional_variances);
                let gamma_feature = match c.gamma_feature {
                    GammaFeatureConvention::AuthorSqrtTimeDividend => {
                        liability.2 * (c.dividend_yield * (remaining - remaining.sqrt())).exp()
                    }
                    GammaFeatureConvention::FixedVolatilityDerivative => liability.2,
                };
                features.extend([
                    liability.0,
                    liability.1,
                    finite(gamma_feature, "gamma feature")?,
                    call.0,
                ]);
            } else {
                payoffs.push(liability.0);
            }
            states.push(state.clone());
            if t < c.steps {
                let (next, ret) =
                    advance_checked(p, c, &state, e[path * c.steps + t]).map_err(|err| {
                        invalid(format!(
                            "path{} step{t}: {err}",
                            range.first_path + path as u64
                        ))
                    })?;
                state = next;
                returns.push(ret);
            }
        }
    }
    let cash = finite((c.rate * c.dt).exp(), "cash growth")?;
    let dividend = finite((c.dividend_yield * c.dt).exp(), "stock income growth")?;
    let mut income = Vec::with_capacity(c.steps * 2);
    for _ in 0..c.steps {
        income.extend([dividend, 1.]);
    }
    let batch = EpisodeBatch {
        force_trade_signal: None,
        exercise: None,
        cash_income: None,
        initial_wealth: None,
        paths: range.paths,
        provenance: BatchProvenance {
            dataset_id: id.clone(),
            first_path: range.first_path,
        },
        marks,
        features,
        payoffs,
    };
    let economics=EconomicTerms{premium:premium.ok_or_else(||invalid("empty scenarios"))?,currency:"USD".into(),units:"Per underlying unit; stock shares, one-unit European call; long straddle liability owed by short dealer. Costs fractions of traded mark, annual continuously compounded r/q, h uses source channel scales: each innovation enters as sqrt(h*dt); beta2..5 baseline sigma is row7/sqrt(252).".into(),trading_costs:[c.stock_cost,c.option_cost],liquidation_costs:c.terminal_costs,cash_growth:vec![cash;c.steps],income_growth:income};
    let content = digest(&(&batch, &economics, &states, &returns, &e))?;
    Ok(JointOptionScenarios {
        config: c.clone(),
        range,
        dataset_id: id,
        content_sha256: content,
        batch,
        economics,
        feature_names: FEATURE_NAMES.iter().map(|x| x.to_string()).collect(),
        states,
        log_returns: returns,
        innovations: e,
        estimated_working_bytes: reserved,
        sampling_note: note.into(),
    })
}
fn option_values(
    c: &JointOptionConfig,
    s: &JointState,
    k: f64,
    tau: f64,
) -> Result<(f64, f64, f64), ComputeError> {
    if tau == 0. {
        return Ok(((s.stock - k).abs(), 0., 0.));
    }
    let a = option_value(c, s, k, tau, OptionSide::Call)?;
    let b = option_value(c, s, k, tau, OptionSide::Put)?;
    Ok((
        finite(a.0 + b.0, "straddle mark")?,
        finite(a.1 + b.1, "straddle delta")?,
        finite(a.2 + b.2, "straddle gamma")?,
    ))
}
/// Source Black–Scholes marking at the current fitted smile; reported feature
/// Greeks hold this implied volatility fixed, as the source's BS convention.
pub fn option_value(
    c: &JointOptionConfig,
    s: &JointState,
    strike: f64,
    tau: f64,
    side: OptionSide,
) -> Result<(f64, f64, f64), ComputeError> {
    let m = ((s.stock / strike).ln() + (c.rate - c.dividend_yield) * tau) / tau.sqrt();
    let vol = surface_volatility(s.betas, m, tau, c.volatility_floor)?;
    let input = EuropeanInputs {
        spot: s.stock,
        strike,
        maturity_years: tau,
        rate: c.rate,
        dividend_yield: c.dividend_yield,
        side,
    };
    let price = european_price_and_vega(&input, vol)?.0;
    let row = OptionRow {
        row_id: "joint-mark".into(),
        contract_id: "joint-european".into(),
        side,
        spot: s.stock,
        strike,
        maturity_years: tau,
        volatility: vol,
        rate: c.rate,
        dividend_yield: c.dividend_yield,
        quantity: 1,
        multiplier: 1,
    };
    let g = analytic_greeks(&row)?;
    Ok((price, g.delta, crate::second_order::analytic_gamma(&row)?))
}
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn uniform(seed: u64, path: u64, step: u64, stream: u64) -> f64 {
    let root = mix(seed ^ 0xa3c59ac3d8451f77);
    let path_key = mix(root ^ path);
    let step_key = mix(path_key ^ step.wrapping_add(0x1436729));
    let x = mix(step_key ^ stream.wrapping_add(0x8a135));
    ((x >> 12) as f64 + 0.5) * (1. / 4503599627370496.)
}
/// Helper for registering all nonoverlapping path partitions with native_learner.
pub fn dataset_partitions(ranges: &[ScenarioRange]) -> Result<Vec<DatasetPartition>, ComputeError> {
    validate_ranges(ranges)?;
    Ok(ranges
        .iter()
        .map(|r| DatasetPartition {
            split: r.split,
            first_path: r.first_path,
            paths: r.paths as u64,
        })
        .collect())
}
