//! Reproducible BS and simplified Broadie–Kaya Heston market paths.
//!
//! CIR ENDPOINTS are sampled from their exact noncentral-chi-square law via a
//! Poisson/gamma mixture, including the attainable zero boundary. Integrated
//! variance is left or trapezoidal quadrature, so the joint Heston path is an
//! approximation and is generally NOT a martingale. No Euler clipping or silent
//! drift correction is applied. See Andersen/Jäckel/Kahl, *Simulation of
//! Square-Root Processes*, §4.2.2–4.2.3, Eqs.29/32:
//! <http://www.jaeckel.org/eqf013_009.pdf>.
//!
//! Random123 Philox4x32-10 counters isolate each (path,step,stream), including
//! variable-length rejection draws. Poisson uses product inversion below30 and
//! Hörmann PTRS otherwise; gamma uses Marsaglia–Tsang plus the shape<1 identity.
//! See <https://www.thesalmons.org/john/random123/papers/random123sc11.pdf> and
//! <https://doi.org/10.1145/358407.358414>. Every rejection loop has a hard failure
//! limit; there is no normal approximation fallback for CIR endpoints.
//!
//! These are numerical market states, not canonical Volt contract/ledger data.
//! No return-predictor, trend, PCA, or signal is hidden in the path state.

use crate::{ComputeError, Precision, Session, heston::HestonParams};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const STOCK: usize = 0;
pub const VARIANCE: usize = 1;
pub const INTEGRATED_VARIANCE: usize = 2;
pub const FUNDED_VARIANCE_CLAIM: usize = 3;
pub const MARKET_FIELDS: usize = 4;
pub const MAX_PATH_VALUES: usize = 128_000_000;
pub const RNG_CONTRACT: &str = "philox4x32-10.v1:key=seed64;counter=(path_lo32,path_hi32,step32,stream4|block28);stream0=cir,stream1=stock;open52-uniform;box-muller";
const MAX_REJECTION_ATTEMPTS: usize = 1_000_000;
const MAX_POISSON_MEAN: f64 = 1e9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegratedVarianceRule {
    Left,
    Trapezoid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MarketModel {
    BlackScholes { volatility: f64 },
    Heston { params: HestonParams },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarketPathConfig {
    pub seed: u64,
    pub first_path_id: u64,
    pub paths: usize,
    pub steps: usize,
    pub dt: f64,
    pub initial_spot: f64,
    pub rate: f64,
    pub dividend_yield: f64,
    pub model: MarketModel,
    /// Irrelevant for BS and deterministic-variance xi=0, which use exact I.
    pub integrated_variance_rule: IntegratedVarianceRule,
}
impl MarketPathConfig {
    /// Original single-system Deep Hedging market parameters. The paper's
    /// piecewise-constant footnote does not identify the endpoint convention;
    /// Left is a declared matching assumption, not recovered author code.
    pub fn paper_heston(paths: usize, seed: u64) -> Self {
        Self {
            seed,
            first_path_id: 0,
            paths,
            steps: 30,
            dt: 1.0 / 365.0,
            initial_spot: 100.0,
            rate: 0.0,
            dividend_yield: 0.0,
            model: MarketModel::Heston {
                params: HestonParams {
                    initial_variance: 0.04,
                    kappa: 1.0,
                    long_run_variance: 0.04,
                    vol_of_variance: 2.0,
                    correlation: -0.7,
                },
            },
            integrated_variance_rule: IntegratedVarianceRule::Left,
        }
    }
    pub fn horizon(&self) -> f64 {
        self.steps as f64 * self.dt
    }
    pub fn validate(&self) -> Result<MarketPathShape, ComputeError> {
        validate_config(self)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MarketPathShape {
    pub path_points: usize,
    pub transitions: usize,
    pub output_values: usize,
    pub input_bytes: u64,
    pub output_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathDiagnostics {
    /// Positive gamma/CIR outcomes rounded to0 by f64 representation, not
    /// negative variance clipped by a discretization algorithm.
    pub gamma_underflow_endpoints: u64,
    /// Includes exact absorbing outcomes when theta=0, and numerical underflow.
    pub zero_variance_endpoints: u64,
    /// Positive host endpoints rounded to0 when replayed on an f32 device.
    pub device_variance_underflow_endpoints: u64,
}
impl PathDiagnostics {
    fn merge(&mut self, rhs: &Self) {
        self.gamma_underflow_endpoints += rhs.gamma_underflow_endpoints;
        self.zero_variance_endpoints += rhs.zero_variance_endpoints;
        self.device_variance_underflow_endpoints += rhs.device_variance_underflow_endpoints;
    }
}

/// Explicit replay inputs. Endpoint layout `[path, time]`, independent-normal
/// layout `[path, step]`. Generated endpoints have exact CIR transitions; caller
/// replay buffers alone do not establish that distributional provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketPathInputs {
    pub config: MarketPathConfig,
    pub variance_endpoints: Vec<f64>,
    pub independent_normals: Vec<f64>,
    pub diagnostics: PathDiagnostics,
    pub source: String,
    pub inputs_hash: String,
}
impl MarketPathInputs {
    pub fn from_explicit(
        config: MarketPathConfig,
        variance_endpoints: Vec<f64>,
        independent_normals: Vec<f64>,
    ) -> Result<Self, ComputeError> {
        let mut inputs = Self {
            config,
            variance_endpoints,
            independent_normals,
            diagnostics: PathDiagnostics::default(),
            source: "explicit_caller_replay.v1".into(),
            inputs_hash: String::new(),
        };
        validate_input_values(&inputs)?;
        inputs.diagnostics.zero_variance_endpoints = inputs
            .variance_endpoints
            .chunks_exact(inputs.config.steps + 1)
            .map(|path| path[1..].iter().filter(|v| **v == 0.0).count() as u64)
            .sum();
        inputs.inputs_hash = hash_inputs(&inputs)?;
        Ok(inputs)
    }
    pub fn validate(&self) -> Result<MarketPathShape, ComputeError> {
        let shape = validate_input_values(self)?;
        if self.inputs_hash != hash_inputs(self)? {
            return Err(invalid("market replay input hash mismatch"));
        }
        Ok(shape)
    }
}

/// Row-major `[path, time, field]`, shape `[paths, steps + 1, 4]`. Tradable field indices are
/// stock0 and funded-variance-claim3; instantaneous variance and A are state.
/// The continuous-CIR model quote is H_t=e^(-r*(T-t)) E[A_T|F_t]; at r=0,
/// A_t+(V_t-theta)*(1-exp(-kappa*(T-t)))/kappa+theta*(T-t).
/// Evaluating that formula on quadrature A and sampled V does not make H an
/// exact martingale for the discrete simulation scheme.
/// In BS H is deterministic (a redundant cash-control, not an extra hedge).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketPaths {
    pub schema: String,
    pub config: MarketPathConfig,
    pub times: Vec<f64>,
    pub values: Vec<f64>,
    pub diagnostics: PathDiagnostics,
    pub rng_contract: String,
    pub input_source: String,
    pub inputs_hash: String,
    pub config_hash: String,
    pub content_hash: String,
}
impl MarketPaths {
    /// Consistency validation for untrusted serialized numerical batches. Hashes
    /// detect mutation; they do not establish that an external author sampled
    /// the declared distribution or provide canonical Volt contract authority.
    pub fn validate(&self) -> Result<MarketPathShape, ComputeError> {
        let shape = self.config.validate()?;
        if self.schema != "volt.compute.market_paths.v1"
            || self.rng_contract != RNG_CONTRACT
            || self.input_source.is_empty()
            || self.times.len() != self.config.steps + 1
            || self
                .times
                .iter()
                .enumerate()
                .any(|(t, v)| *v != t as f64 * self.config.dt)
            || self.inputs_hash.len() != 64
            || !self.inputs_hash.bytes().all(|b| b.is_ascii_hexdigit())
            || self.config_hash != hash_json(&self.config)?
        {
            return Err(invalid("market path metadata/grid/config hash mismatch"));
        }
        validate_output_values(&self.config, &self.values)?;
        if self.content_hash
            != hash_output(
                &self.config_hash,
                &self.inputs_hash,
                &self.input_source,
                &self.diagnostics,
                &self.values,
            )?
        {
            return Err(invalid("market path content hash mismatch"));
        }
        Ok(shape)
    }
    pub fn shape(&self) -> [usize; 3] {
        [self.config.paths, self.config.steps + 1, MARKET_FIELDS]
    }
    pub fn path_id(&self, path: usize) -> Option<u64> {
        (path < self.config.paths).then(|| self.config.first_path_id + path as u64)
    }
    pub fn value(&self, path: usize, time: usize, field: usize) -> Option<f64> {
        if path >= self.config.paths || time > self.config.steps || field >= MARKET_FIELDS {
            return None;
        }
        self.values
            .get((path * (self.config.steps + 1) + time) * MARKET_FIELDS + field)
            .copied()
    }
}

/// Shape/numeric checks only; no path allocations or device initialization.
/// Actual working-memory admission belongs to the application/job wrapper.
pub fn validate_config(config: &MarketPathConfig) -> Result<MarketPathShape, ComputeError> {
    if config.paths == 0 || config.steps == 0 || config.steps > 4096 {
        return Err(invalid(
            "market paths need positive count and1..=4096 steps",
        ));
    }
    config
        .first_path_id
        .checked_add(config.paths as u64)
        .ok_or_else(|| invalid("path identity range overflow"))?;
    if !config.dt.is_finite()
        || config.dt <= 0.0
        || !config.horizon().is_finite()
        || !config.initial_spot.is_finite()
        || config.initial_spot <= 0.0
        || !config.rate.is_finite()
        || !config.dividend_yield.is_finite()
    {
        return Err(invalid(
            "market spot/time must be positive finite and rates finite",
        ));
    }
    match config.model {
        MarketModel::BlackScholes { volatility } => {
            if !volatility.is_finite() || volatility < 0.0 || !(volatility * volatility).is_finite()
            {
                return Err(invalid(
                    "BS path volatility must be nonnegative finite with finite variance",
                ));
            }
        }
        MarketModel::Heston { params } => {
            CirLaw::new(params, config.dt)?;
        }
    }
    let path_points = config
        .paths
        .checked_mul(config.steps + 1)
        .ok_or_else(|| invalid("path point shape overflow"))?;
    let transitions = config
        .paths
        .checked_mul(config.steps)
        .ok_or_else(|| invalid("path transition shape overflow"))?;
    let output_values = path_points
        .checked_mul(MARKET_FIELDS)
        .ok_or_else(|| invalid("path field shape overflow"))?;
    if output_values > MAX_PATH_VALUES {
        return Err(invalid(
            "market path shape exceeds128million-value hard cap; stream chunks",
        ));
    }
    let input_bytes = (path_points as u64)
        .checked_add(transitions as u64)
        .and_then(|v| v.checked_mul(8))
        .ok_or_else(|| invalid("path byte count overflow"))?;
    let output_bytes = (output_values as u64)
        .checked_mul(8)
        .ok_or_else(|| invalid("path byte count overflow"))?;
    Ok(MarketPathShape {
        path_points,
        transitions,
        output_values,
        input_bytes,
        output_bytes,
    })
}

pub fn generate_inputs(config: &MarketPathConfig) -> Result<MarketPathInputs, ComputeError> {
    let shape = validate_config(config)?;
    let mut variance_endpoints = vec![0.0; shape.path_points];
    let mut independent_normals = vec![0.0; shape.transitions];
    let workers = if config.paths >= 4096 {
        std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(8)
    } else {
        1
    };
    let paths_per_worker = config.paths.div_ceil(workers);
    let diagnostics = std::thread::scope(|scope| -> Result<PathDiagnostics, ComputeError> {
        let mut jobs = Vec::new();
        for (worker, (endpoints, normals)) in variance_endpoints
            .chunks_mut(paths_per_worker * (config.steps + 1))
            .zip(independent_normals.chunks_mut(paths_per_worker * config.steps))
            .enumerate()
        {
            jobs.push(scope.spawn(move || {
                generate_chunk(config, worker * paths_per_worker, endpoints, normals)
            }));
        }
        let mut result = PathDiagnostics::default();
        for job in jobs {
            result.merge(
                &job.join()
                    .map_err(|_| execution("path generator worker panicked"))??,
            );
        }
        Ok(result)
    })?;
    let mut result = MarketPathInputs {
        config: config.clone(),
        variance_endpoints,
        independent_normals,
        diagnostics,
        source: "philox_exact_cir_endpoints_independent_stock_normals.v1".into(),
        inputs_hash: String::new(),
    };
    result.inputs_hash = hash_inputs(&result)?;
    Ok(result)
}

fn generate_chunk(
    config: &MarketPathConfig,
    start: usize,
    endpoints: &mut [f64],
    normals: &mut [f64],
) -> Result<PathDiagnostics, ComputeError> {
    let mut diagnostics = PathDiagnostics::default();
    let law = match config.model {
        MarketModel::Heston { params } => Some(CirLaw::new(params, config.dt)?),
        _ => None,
    };
    for (local, (v, z)) in endpoints
        .chunks_mut(config.steps + 1)
        .zip(normals.chunks_mut(config.steps))
        .enumerate()
    {
        let path = config.first_path_id + (start + local) as u64;
        v[0] = initial_variance(config.model);
        for step in 0..config.steps {
            let mut stock_rng = CounterRng::new(config.seed, path, step as u32, 1);
            z[step] = stock_rng.normal()?;
            if let Some(law) = &law {
                let mut variance_rng = CounterRng::new(config.seed, path, step as u32, 0);
                let (next, underflow) = law.sample(v[step], &mut variance_rng)?;
                v[step + 1] = next;
                diagnostics.gamma_underflow_endpoints += u64::from(underflow);
                diagnostics.zero_variance_endpoints += u64::from(next == 0.0);
            } else {
                v[step + 1] = v[0];
            }
        }
    }
    Ok(diagnostics)
}

pub fn generate_paths(config: &MarketPathConfig) -> Result<MarketPaths, ComputeError> {
    evolve_host(&generate_inputs(config)?)
}

pub fn evolve_host(inputs: &MarketPathInputs) -> Result<MarketPaths, ComputeError> {
    let shape = inputs.validate()?;
    let config = &inputs.config;
    let mut values = vec![0.0; shape.output_values];
    for path in 0..config.paths {
        let mut log_spot = config.initial_spot.ln();
        let mut integrated = 0.0;
        for time in 0..=config.steps {
            let v = inputs.variance_endpoints[path * (config.steps + 1) + time];
            let offset = (path * (config.steps + 1) + time) * MARKET_FIELDS;
            if time > 0 {
                let previous = inputs.variance_endpoints[path * (config.steps + 1) + time - 1];
                let increment = integral_increment(config, previous, v);
                integrated += increment;
                let normal = inputs.independent_normals[path * config.steps + time - 1];
                log_spot += stock_increment(config, previous, v, increment, normal);
            }
            let spot = if time == 0 {
                config.initial_spot
            } else {
                log_spot.exp()
            };
            let claim = funded_claim(config, integrated, v, time);
            values[offset..offset + MARKET_FIELDS].copy_from_slice(&[spot, v, integrated, claim]);
        }
    }
    finish(inputs, values, inputs.diagnostics.clone())
}

/// Replay the same host-generated CIR endpoints and independent normals while
/// evolving all stock/A/claim arithmetic on the selected device. RNG and exact
/// CIR rejection sampling remain scalar f64. Session cap counts OUTPUT VALUES,
/// and the caller must separately admit inputs and intermediate device states.
pub fn evolve_on_device(
    session: &Session,
    inputs: &MarketPathInputs,
) -> Result<MarketPaths, ComputeError> {
    let shape = inputs.validate()?;
    if shape.output_values > session.max_batch_rows() {
        return Err(invalid(
            "market device replay exceeds session output-value cap",
        ));
    }
    let config = &inputs.config;
    if session.precision() == Precision::F32 {
        for value in inputs
            .variance_endpoints
            .iter()
            .chain(&inputs.independent_normals)
            .chain([
                &config.initial_spot,
                &config.dt,
                &config.rate,
                &config.dividend_yield,
            ])
        {
            if !(*value as f32).is_finite() {
                return Err(invalid("market inputs overflow device f32"));
            }
        }
        if (config.initial_spot as f32) <= 0.0 || (config.dt as f32) <= 0.0 {
            return Err(invalid("market spot/dt underflow device f32"));
        }
    }
    let variances = session
        .vector(inputs.variance_endpoints.clone())
        .reshape([config.paths, config.steps + 1]);
    let normals = session
        .vector(inputs.independent_normals.clone())
        .reshape([config.paths, config.steps]);
    let mut spot = session.vector(vec![config.initial_spot; config.paths]);
    let mut log_spot = spot.clone().log();
    let mut integrated = spot.zeros_like();
    let mut states = Vec::with_capacity(config.steps + 1);
    for time in 0..=config.steps {
        let v = variances
            .clone()
            .slice([0..config.paths, time..time + 1])
            .reshape([config.paths]);
        if time > 0 {
            let previous = variances
                .clone()
                .slice([0..config.paths, time - 1..time])
                .reshape([config.paths]);
            let normal = normals
                .clone()
                .slice([0..config.paths, time - 1..time])
                .reshape([config.paths]);
            let increment = match config.model {
                MarketModel::BlackScholes { .. } => previous.clone() * config.dt,
                MarketModel::Heston { params } if params.vol_of_variance == 0.0 => {
                    let (_, a, b) = time_factors(params.kappa, config.dt);
                    previous.clone() * a + params.long_run_variance * b
                }
                MarketModel::Heston { .. } => match config.integrated_variance_rule {
                    IntegratedVarianceRule::Left => previous.clone() * config.dt,
                    IntegratedVarianceRule::Trapezoid => {
                        (previous.clone() + v.clone()) * (0.5 * config.dt)
                    }
                },
            };
            integrated = integrated + increment.clone();
            let drift = (config.rate - config.dividend_yield) * config.dt;
            let change = match config.model {
                MarketModel::Heston { params } if params.vol_of_variance > 0.0 => {
                    (v.clone() - previous - params.kappa * params.long_run_variance * config.dt)
                        * (params.correlation / params.vol_of_variance)
                        + increment.clone()
                            * (params.kappa * params.correlation / params.vol_of_variance - 0.5)
                        + (increment * (1.0 - params.correlation * params.correlation)).sqrt()
                            * normal
                        + drift
                }
                _ => increment.clone() * -0.5 + increment.sqrt() * normal + drift,
            };
            log_spot = log_spot + change;
            spot = log_spot.clone().exp();
        }
        let remaining = (config.steps - time) as f64 * config.dt;
        let expected_future = match config.model {
            MarketModel::BlackScholes { .. } => v.clone() * remaining,
            MarketModel::Heston { params } => {
                let (_, a, b) = time_factors(params.kappa, remaining);
                v.clone() * a + params.long_run_variance * b
            }
        };
        let claim = (integrated.clone() + expected_future) * (-config.rate * remaining).exp();
        states.push(Tensor::<1>::stack::<2>(
            vec![spot.clone(), v, integrated.clone(), claim],
            1,
        ));
    }
    let output = Tensor::<2>::stack::<3>(states, 1).reshape([shape.output_values]);
    session.synchronize()?;
    let values = session.read_vector(&output)?;
    let mut diagnostics = inputs.diagnostics.clone();
    if session.precision() == Precision::F32 {
        diagnostics.device_variance_underflow_endpoints = inputs
            .variance_endpoints
            .iter()
            .filter(|v| **v > 0.0 && (**v as f32) == 0.0)
            .count() as u64;
    }
    finish(inputs, values, diagnostics)
}

fn initial_variance(model: MarketModel) -> f64 {
    match model {
        MarketModel::BlackScholes { volatility } => volatility * volatility,
        MarketModel::Heston { params } => params.initial_variance,
    }
}
fn integral_increment(config: &MarketPathConfig, previous: f64, next: f64) -> f64 {
    match config.model {
        MarketModel::BlackScholes { .. } => previous * config.dt,
        MarketModel::Heston { params } if params.vol_of_variance == 0.0 => {
            let (_, a, b) = time_factors(params.kappa, config.dt);
            previous * a + params.long_run_variance * b
        }
        MarketModel::Heston { .. } => match config.integrated_variance_rule {
            IntegratedVarianceRule::Left => previous * config.dt,
            IntegratedVarianceRule::Trapezoid => (previous + next) * (0.5 * config.dt),
        },
    }
}
fn stock_increment(
    config: &MarketPathConfig,
    previous: f64,
    next: f64,
    integral: f64,
    normal: f64,
) -> f64 {
    let drift = (config.rate - config.dividend_yield) * config.dt;
    match config.model {
        MarketModel::Heston { params } if params.vol_of_variance > 0.0 => {
            drift
                + params.correlation / params.vol_of_variance
                    * (next - previous - params.kappa * params.long_run_variance * config.dt)
                + (params.kappa * params.correlation / params.vol_of_variance - 0.5) * integral
                + ((1.0 - params.correlation * params.correlation) * integral).sqrt() * normal
        }
        _ => drift - 0.5 * integral + integral.sqrt() * normal,
    }
}
fn funded_claim(config: &MarketPathConfig, integrated: f64, variance: f64, time: usize) -> f64 {
    let remaining = (config.steps - time) as f64 * config.dt;
    let future = match config.model {
        MarketModel::BlackScholes { .. } => variance * remaining,
        MarketModel::Heston { params } => {
            let (_, a, b) = time_factors(params.kappa, remaining);
            variance * a + params.long_run_variance * b
        }
    };
    (integrated + future) * (-config.rate * remaining).exp()
}
/// exp(-kh), (1-exp(-kh))/k, h-(1-exp(-kh))/k, stable at small kh.
fn time_factors(kappa: f64, h: f64) -> (f64, f64, f64) {
    let x = kappa * h;
    let decay = (-x).exp();
    let b = if x.abs() < 1e-4 {
        h * (x * 0.5 - x * x / 6.0 + x * x * x / 24.0 - x * x * x * x / 120.0)
    } else {
        h + (-x).exp_m1() / kappa
    };
    (decay, h - b, b)
}

struct CirLaw {
    params: HestonParams,
    decay: f64,
    scale: f64,
    shape: f64,
    lambda_per_variance: f64,
    dt: f64,
}
impl CirLaw {
    fn new(params: HestonParams, dt: f64) -> Result<Self, ComputeError> {
        if [
            params.initial_variance,
            params.kappa,
            params.long_run_variance,
            params.vol_of_variance,
            params.correlation,
        ]
        .iter()
        .any(|v| !v.is_finite())
            || params.initial_variance < 0.0
            || params.kappa <= 0.0
            || params.long_run_variance < 0.0
            || params.vol_of_variance < 0.0
            || params.correlation.abs() > 1.0
        {
            return Err(invalid(
                "CIR needs nonnegative variance/theta/xi, positive kappa and|rho|<=1",
            ));
        }
        let decay = (-params.kappa * dt).exp();
        let xi2 = params.vol_of_variance * params.vol_of_variance;
        if params.vol_of_variance == 0.0 {
            return Ok(Self {
                params,
                decay,
                scale: 0.0,
                shape: 0.0,
                lambda_per_variance: 0.0,
                dt,
            });
        }
        let scale = xi2 * (-(-params.kappa * dt).exp_m1()) / (4.0 * params.kappa);
        let shape = 2.0 * params.kappa * params.long_run_variance / xi2;
        let lambda_per_variance = decay / (2.0 * scale);
        if !scale.is_finite()
            || scale <= 0.0
            || !shape.is_finite()
            || shape > MAX_POISSON_MEAN
            || !lambda_per_variance.is_finite()
            || params.initial_variance * lambda_per_variance > MAX_POISSON_MEAN
        {
            return Err(invalid(
                "CIR mixture exceeds supported finite f64 scale/shape/Poisson range",
            ));
        }
        Ok(Self {
            params,
            decay,
            scale,
            shape,
            lambda_per_variance,
            dt,
        })
    }
    fn sample(&self, current: f64, rng: &mut CounterRng) -> Result<(f64, bool), ComputeError> {
        if self.params.vol_of_variance == 0.0 {
            let one_minus = -(-self.params.kappa * self.dt).exp_m1();
            return Ok((
                current * self.decay + self.params.long_run_variance * one_minus,
                false,
            ));
        }
        let n = poisson(current * self.lambda_per_variance, rng)?;
        let shape = self.shape + n as f64;
        if shape > 2.0 * MAX_POISSON_MEAN {
            return Err(execution("CIR gamma shape exceeds supported range"));
        }
        if shape == 0.0 {
            return Ok((0.0, false));
        }
        let (gamma, underflow) = gamma(shape, rng)?;
        let value = (2.0 * self.scale) * gamma;
        if !value.is_finite() || value < 0.0 {
            return Err(execution("nonfinite CIR endpoint"));
        }
        Ok((value, underflow || (gamma > 0.0 && value == 0.0)))
    }
}

fn poisson(mean: f64, rng: &mut CounterRng) -> Result<u64, ComputeError> {
    if !mean.is_finite() || !(0.0..=MAX_POISSON_MEAN).contains(&mean) {
        return Err(execution("CIR Poisson mean exceeds supported range"));
    }
    if mean == 0.0 {
        return Ok(0);
    }
    if mean < 30.0 {
        let limit = (-mean).exp();
        let mut product = 1.0;
        for k in 0..MAX_REJECTION_ATTEMPTS {
            product *= rng.uniform()?;
            if product <= limit {
                return Ok(k as u64);
            }
        }
    } else {
        let root = mean.sqrt();
        let log_mean = mean.ln();
        let b = 0.931 + 2.53 * root;
        let a = -0.059 + 0.02483 * b;
        let inverse_alpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);
        for _ in 0..MAX_REJECTION_ATTEMPTS {
            let u = rng.uniform()? - 0.5;
            let v = rng.uniform()?;
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + mean + 0.43).floor();
            if k < 0.0 || (us < 0.013 && v > us) {
                continue;
            }
            if us >= 0.07 && v <= vr {
                return Ok(k as u64);
            }
            if (v * inverse_alpha / (a / (us * us) + b)).ln()
                <= -mean + k * log_mean - libm::lgamma(k + 1.0)
            {
                return Ok(k as u64);
            }
        }
    }
    Err(execution("CIR Poisson rejection iteration limit exceeded"))
}
fn gamma(shape: f64, rng: &mut CounterRng) -> Result<(f64, bool), ComputeError> {
    let boosted = if shape < 1.0 { shape + 1.0 } else { shape };
    let d = boosted - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    for _ in 0..MAX_REJECTION_ATTEMPTS {
        let normal = rng.normal()?;
        let base = 1.0 + c * normal;
        if base <= 0.0 {
            continue;
        }
        let v = base * base * base;
        let u = rng.uniform()?;
        if u < 1.0 - 0.0331 * normal.powi(4)
            || u.ln() < 0.5 * normal * normal + d * (1.0 - v + v.ln())
        {
            let value = d * v;
            if shape < 1.0 {
                let transformed = (value.ln() + rng.uniform()?.ln() / shape).exp();
                return Ok((transformed, transformed == 0.0));
            }
            return Ok((value, false));
        }
    }
    Err(execution("CIR gamma rejection iteration limit exceeded"))
}

struct CounterRng {
    seed: u64,
    path: u64,
    step: u32,
    stream: u32,
    block: u32,
    words: [u32; 4],
    cursor: usize,
}
impl CounterRng {
    fn new(seed: u64, path: u64, step: u32, stream: u32) -> Self {
        Self {
            seed,
            path,
            step,
            stream,
            block: 0,
            words: [0; 4],
            cursor: 4,
        }
    }
    fn word(&mut self) -> Result<u32, ComputeError> {
        if self.cursor == 4 {
            if self.block >= 1 << 28 {
                return Err(execution("Philox local rejection stream exhausted"));
            }
            self.words = philox(
                [
                    self.path as u32,
                    (self.path >> 32) as u32,
                    self.step,
                    (self.stream << 28) | self.block,
                ],
                [self.seed as u32, (self.seed >> 32) as u32],
            );
            self.block += 1;
            self.cursor = 0;
        }
        let value = self.words[self.cursor];
        self.cursor += 1;
        Ok(value)
    }
    fn uniform(&mut self) -> Result<f64, ComputeError> {
        let bits = ((self.word()? as u64) << 20) | ((self.word()? as u64) >> 12);
        Ok((bits as f64 + 0.5) / 4_503_599_627_370_496.0)
    }
    fn normal(&mut self) -> Result<f64, ComputeError> {
        Ok((-2.0 * self.uniform()?.ln()).sqrt() * (std::f64::consts::TAU * self.uniform()?).cos())
    }
}
fn philox(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for _ in 0..10 {
        let p0 = 0xd2511f53_u64 * counter[0] as u64;
        let p1 = 0xcd9e8d57_u64 * counter[2] as u64;
        counter = [
            (p1 >> 32) as u32 ^ counter[1] ^ key[0],
            p1 as u32,
            (p0 >> 32) as u32 ^ counter[3] ^ key[1],
            p0 as u32,
        ];
        key[0] = key[0].wrapping_add(0x9e3779b9);
        key[1] = key[1].wrapping_add(0xbb67ae85);
    }
    counter
}

fn validate_input_values(inputs: &MarketPathInputs) -> Result<MarketPathShape, ComputeError> {
    let shape = validate_config(&inputs.config)?;
    if inputs.variance_endpoints.len() != shape.path_points
        || inputs.independent_normals.len() != shape.transitions
        || inputs
            .variance_endpoints
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        || inputs.independent_normals.iter().any(|v| !v.is_finite())
    {
        return Err(invalid(
            "market replay inputs have malformed shape or invalid values",
        ));
    }
    let initial = initial_variance(inputs.config.model);
    for path in inputs.variance_endpoints.chunks(inputs.config.steps + 1) {
        if path[0] != initial {
            return Err(invalid(
                "market replay initial variance differs from config",
            ));
        }
        if matches!(inputs.config.model, MarketModel::BlackScholes { .. })
            && path.iter().any(|v| *v != initial)
        {
            return Err(invalid("BS replay variance endpoints must be constant"));
        }
    }
    Ok(shape)
}
fn finish(
    inputs: &MarketPathInputs,
    values: Vec<f64>,
    diagnostics: PathDiagnostics,
) -> Result<MarketPaths, ComputeError> {
    let config = &inputs.config;
    validate_output_values(config, &values)?;
    let times = (0..=config.steps).map(|t| t as f64 * config.dt).collect();
    let config_hash = hash_json(config)?;
    let content_hash = hash_output(
        &config_hash,
        &inputs.inputs_hash,
        &inputs.source,
        &diagnostics,
        &values,
    )?;
    Ok(MarketPaths {
        schema: "volt.compute.market_paths.v1".into(),
        config: config.clone(),
        times,
        values,
        diagnostics,
        rng_contract: RNG_CONTRACT.into(),
        input_source: inputs.source.clone(),
        inputs_hash: inputs.inputs_hash.clone(),
        config_hash,
        content_hash,
    })
}
fn validate_output_values(config: &MarketPathConfig, values: &[f64]) -> Result<(), ComputeError> {
    if values.len() != config.paths * (config.steps + 1) * MARKET_FIELDS
        || values.as_chunks::<MARKET_FIELDS>().0.iter().any(|row| {
            row.iter().any(|v| !v.is_finite())
                || row[STOCK] <= 0.0
                || row[VARIANCE] < 0.0
                || row[INTEGRATED_VARIANCE] < 0.0
                || row[FUNDED_VARIANCE_CLAIM] < 0.0
        })
    {
        return Err(execution(
            "market evolution produced nonfinite/nonpositive stock or negative variance/claim",
        ));
    }
    Ok(())
}
pub(crate) fn hash_output(
    config_hash: &str,
    inputs_hash: &str,
    input_source: &str,
    diagnostics: &PathDiagnostics,
    values: &[f64],
) -> Result<String, ComputeError> {
    let mut digest = Sha256::new();
    digest.update(b"volt.market_paths.v1\0");
    digest.update(config_hash.as_bytes());
    digest.update(inputs_hash.as_bytes());
    digest.update(input_source.as_bytes());
    digest.update(hash_json(diagnostics)?.as_bytes());
    hash_values(&mut digest, values);
    Ok(format!("{:x}", digest.finalize()))
}
fn hash_inputs(inputs: &MarketPathInputs) -> Result<String, ComputeError> {
    let mut digest = Sha256::new();
    digest.update(b"volt.market_path_inputs.v1\0");
    digest.update(hash_json(&inputs.config)?.as_bytes());
    digest.update(RNG_CONTRACT.as_bytes());
    digest.update(inputs.source.as_bytes());
    digest.update(hash_json(&inputs.diagnostics)?.as_bytes());
    hash_values(&mut digest, &inputs.variance_endpoints);
    hash_values(&mut digest, &inputs.independent_normals);
    Ok(format!("{:x}", digest.finalize()))
}
pub(crate) fn hash_json<T: Serialize>(value: &T) -> Result<String, ComputeError> {
    let json = serde_json::to_vec(value)
        .map_err(|e| execution(&format!("path hash serialization: {e}")))?;
    Ok(format!("{:x}", Sha256::digest(json)))
}
fn hash_values(hash: &mut Sha256, values: &[f64]) {
    let mut buffer = [0_u8; 8192];
    for chunk in values.chunks(1024) {
        for (value, slot) in chunk.iter().zip(buffer.as_chunks_mut::<8>().0.iter_mut()) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        hash.update(&buffer[..chunk.len() * 8]);
    }
}
fn invalid(message: &str) -> ComputeError {
    ComputeError::InvalidInput(message.into())
}
fn execution(message: &str) -> ComputeError {
    ComputeError::Execution(message.into())
}

#[cfg(test)]
mod tests {
    #[test]
    fn philox_matches_random123_known_answer() {
        assert_eq!(
            super::philox([0; 4], [0; 2]),
            [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
        );
    }
}
