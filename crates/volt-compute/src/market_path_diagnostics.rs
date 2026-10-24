//! Streaming, independent-path sampling controls for the declared path scheme.
//! Continuous-model comparisons disclose discretization bias rather than
//! disguising a biased scheme as an exact Monte Carlo confidence failure.

use crate::{
    ComputeError,
    market_paths::{
        FUNDED_VARIANCE_CLAIM, INTEGRATED_VARIANCE, IntegratedVarianceRule, MarketModel,
        MarketPathConfig, PathDiagnostics, RNG_CONTRACT, STOCK, VARIANCE, generate_paths,
    },
    monte_carlo::{Estimate, OnlineMoments},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Instant;

pub const PAPER_HESTON_CALL_REFERENCE: f64 = 1.691_833_925_110_386;
pub const PAPER_REFERENCE_SOURCE: &str = "QuantLib1.40 Gatheral adaptive-Lobatto rel1e-12 abs1e-13 integrationepsilon1e-14; independently agrees exp-sinh1.691833925110385;30/365 maturity";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathDiagnosticConfig {
    pub market: MarketPathConfig,
    pub call_strike: f64,
    pub requested_chunk_paths: usize,
    pub host_working_bytes: u64,
    pub call_reference: Option<f64>,
    pub reference_source: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct PathDiagnosticPlan {
    pub chunk_paths: usize,
    pub chunks: usize,
    pub estimated_working_bytes: u64,
    pub host_admission_limit_bytes: u64,
    pub working_bytes_per_path: u64,
}
#[derive(Debug, Clone, Serialize)]
pub struct SamplingControl {
    pub estimate: Estimate,
    pub reference: f64,
    pub signed_error: f64,
    /// Descriptive only: a continuous-model comparison may have scheme bias.
    pub standard_errors_from_reference: Option<f64>,
}
#[derive(Debug, Clone, Serialize)]
pub struct TimeControls {
    pub time: f64,
    pub growth_adjusted_stock: SamplingControl,
    pub variance: SamplingControl,
    pub integrated_variance: SamplingControl,
    pub continuous_integrated_variance_reference: f64,
    pub discounted_variance_claim_discrete_expectation: SamplingControl,
    pub discounted_variance_claim_continuous_martingale_target: SamplingControl,
}
#[derive(Debug, Clone, Serialize)]
pub struct ConditionalDriftControl {
    pub current_variance: f64,
    /// E[S_next/S * exp(-(r-q)h)-1 | V], analytically integrated over exact CIR
    /// endpoints and the independent normal for the DECLARED quadrature rule.
    /// None means the exponential moment does not exist for this scheme.
    pub adjusted_stock_fractional_drift: Option<f64>,
    /// E[e^-rh H_next-H_now | V,A], at time0. Exposes conditional quadrature
    /// bias even when V0=theta hides it in an unconditional mean.
    pub discounted_claim_absolute_drift: f64,
}
#[derive(Debug, Serialize)]
pub struct PathDiagnosticReport {
    pub schema: String,
    pub config: PathDiagnosticConfig,
    pub plan: PathDiagnosticPlan,
    pub rng_contract: String,
    pub independent_observation_unit: String,
    pub diagnostics: PathDiagnostics,
    pub times: Vec<TimeControls>,
    pub conditional_one_step_drift: Vec<ConditionalDriftControl>,
    pub terminal_discounted_call: Estimate,
    pub call_reference_comparison: Option<SamplingControl>,
    pub chunk_content_hashes: Vec<String>,
    pub chunk_manifest_hash: String,
    pub generation_seconds: f64,
    pub analysis_seconds: f64,
    pub total_seconds: f64,
    pub scheme_disclosure: String,
}

pub fn plan_path_diagnostics(
    config: &PathDiagnosticConfig,
) -> Result<PathDiagnosticPlan, ComputeError> {
    config.market.validate()?;
    if !config.call_strike.is_finite()
        || config.call_strike <= 0.0
        || config.requested_chunk_paths == 0
        || config.host_working_bytes == 0
        || config
            .call_reference
            .is_some_and(|x| !x.is_finite() || x < 0.0)
        || (config.call_reference.is_some()
            && config
                .reference_source
                .as_ref()
                .is_none_or(|s| s.trim().is_empty()))
    {
        return Err(ComputeError::InvalidInput(
            "path diagnostic strike/count/budget/reference invalid".into(),
        ));
    }
    let mut one = config.market.clone();
    one.paths = 1;
    let shape = one.validate()?;
    // Actual generator holds endpoint/normal buffers while creating four-field
    // outputs. Hashing uses bounded stack buffers. Caller inputs/report JSON
    // serialization and other processes are outside this admission estimate.
    let per_path = shape
        .input_bytes
        .checked_add(shape.output_bytes)
        .ok_or_else(overflow)?;
    let limit =
        (config.host_working_bytes / 100) * 80 + (config.host_working_bytes % 100) * 80 / 100;
    let base_fixed = 8192_u64
        + (config.market.steps as u64 + 1) * 4 * std::mem::size_of::<OnlineMoments>() as u64;
    // Each retained String hash has24bytes of Vec metadata and64 content bytes.
    // Solve the small monotone chunk/manifest admission dependency before any
    // allocation; a tiny budget must not silently retain millions of hashes.
    let mut fixed = base_fixed;
    let mut accepted = None;
    for _ in 0..64 {
        let chunk = ((limit.checked_sub(fixed).ok_or_else(|| {
            ComputeError::BudgetExceeded(
                "path diagnostic global controls/manifest exceed host limit".into(),
            )
        })? / per_path) as usize)
            .min(config.requested_chunk_paths)
            .min(config.market.paths);
        if chunk == 0 {
            return Err(ComputeError::BudgetExceeded(
                "path diagnostic cannot admit one complete path".into(),
            ));
        }
        let required = base_fixed + config.market.paths.div_ceil(chunk) as u64 * 88;
        if required <= fixed {
            accepted = Some(chunk);
            break;
        }
        fixed = required;
    }
    let chunk_paths = accepted.ok_or_else(|| {
        ComputeError::BudgetExceeded("path diagnostic chunk/manifest plan did not converge".into())
    })?;
    Ok(PathDiagnosticPlan {
        chunk_paths,
        chunks: config.market.paths.div_ceil(chunk_paths),
        estimated_working_bytes: fixed + per_path * chunk_paths as u64,
        host_admission_limit_bytes: limit,
        working_bytes_per_path: per_path,
    })
}

pub fn diagnose_paths(config: &PathDiagnosticConfig) -> Result<PathDiagnosticReport, ComputeError> {
    let started = Instant::now();
    let plan = plan_path_diagnostics(config)?;
    let mut time_moments: Vec<[OnlineMoments; 4]> = (0..=config.market.steps)
        .map(|_| std::array::from_fn(|_| OnlineMoments::default()))
        .collect();
    let mut call = OnlineMoments::default();
    let mut diagnostics = PathDiagnostics::default();
    let mut chunk_hashes = Vec::with_capacity(plan.chunks);
    let mut generation_seconds = 0.0;
    let mut analysis_seconds = 0.0;
    for first in (0..config.market.paths).step_by(plan.chunk_paths) {
        let mut chunk = config.market.clone();
        chunk.first_path_id += first as u64;
        chunk.paths = plan.chunk_paths.min(config.market.paths - first);
        let tick = Instant::now();
        let paths = generate_paths(&chunk)?;
        generation_seconds += tick.elapsed().as_secs_f64();
        let tick = Instant::now();
        diagnostics.gamma_underflow_endpoints += paths.diagnostics.gamma_underflow_endpoints;
        diagnostics.zero_variance_endpoints += paths.diagnostics.zero_variance_endpoints;
        diagnostics.device_variance_underflow_endpoints +=
            paths.diagnostics.device_variance_underflow_endpoints;
        for path in 0..chunk.paths {
            for (time, moments) in time_moments.iter_mut().enumerate() {
                let t = time as f64 * chunk.dt;
                let values = [
                    paths.value(path, time, STOCK).unwrap()
                        * (-(chunk.rate - chunk.dividend_yield) * t).exp(),
                    paths.value(path, time, VARIANCE).unwrap(),
                    paths.value(path, time, INTEGRATED_VARIANCE).unwrap(),
                    paths.value(path, time, FUNDED_VARIANCE_CLAIM).unwrap()
                        * (-chunk.rate * t).exp(),
                ];
                for (moment, value) in moments.iter_mut().zip(values) {
                    moment.add(value);
                }
            }
            let payoff = (paths.value(path, chunk.steps, STOCK).unwrap() - config.call_strike)
                .max(0.0)
                * (-chunk.rate * chunk.horizon()).exp();
            call.add(payoff);
        }
        chunk_hashes.push(paths.content_hash);
        analysis_seconds += tick.elapsed().as_secs_f64();
    }
    let mut times = Vec::with_capacity(time_moments.len());
    for (step, moments) in time_moments.iter().enumerate() {
        let time = step as f64 * config.market.dt;
        let remaining = config.market.horizon() - time;
        let variance = expected_variance(&config.market, time);
        let expected_integral = expected_discrete_integral(&config.market, step);
        let continuous_integral =
            expected_future_integral(&config.market, initial_variance(config.market.model), time);
        let discount = (-config.market.rate * config.market.horizon()).exp();
        let expected_claim = discount
            * (expected_integral + expected_future_integral(&config.market, variance, remaining));
        let martingale_claim = discount
            * expected_future_integral(
                &config.market,
                initial_variance(config.market.model),
                config.market.horizon(),
            );
        times.push(TimeControls {
            time,
            growth_adjusted_stock: control(&moments[0], config.market.initial_spot)?,
            variance: control(&moments[1], variance)?,
            integrated_variance: control(&moments[2], expected_integral)?,
            continuous_integrated_variance_reference: continuous_integral,
            discounted_variance_claim_discrete_expectation: control(&moments[3], expected_claim)?,
            discounted_variance_claim_continuous_martingale_target: control(
                &moments[3],
                martingale_claim,
            )?,
        });
    }
    let conditional_one_step_drift = [0.0, 1e-8, 1e-6, 1e-4, 0.001, 0.01, 0.04, 0.1, 0.5, 1.0]
        .into_iter()
        .map(|v| conditional_drift(&config.market, v))
        .collect::<Result<Vec<_>, _>>()?;
    let comparison = config
        .call_reference
        .map(|reference| control(&call, reference))
        .transpose()?;
    let mut digest = Sha256::new();
    digest.update(b"volt.path_diagnostic_manifest.v1\0");
    for hash in &chunk_hashes {
        digest.update(hash.as_bytes());
    }
    Ok(PathDiagnosticReport {schema:"volt.compute.market_path_diagnostics.v1".into(),config:config.clone(),plan,
        rng_contract:RNG_CONTRACT.into(),independent_observation_unit:"one complete independent path; no within-path independence assumption".into(),
        diagnostics,times,conditional_one_step_drift,terminal_discounted_call:call.estimate(),call_reference_comparison:comparison,
        chunk_content_hashes:chunk_hashes,chunk_manifest_hash:format!("{:x}",digest.finalize()),generation_seconds,analysis_seconds,
        total_seconds:started.elapsed().as_secs_f64(),scheme_disclosure:"BS log paths are exact at grid times. Heston has exact CIR endpoints, quadrature integrated variance, and simplified BK stock. Continuous Fourier payoff and martingale references can exhibit genuine scheme bias; sigma distances are descriptive, not exact-scheme pass gates. Conditional drift uses CIR MGF, not correlated within-path samples. No hidden drift correction.".into()})
}

/// Exact expectation for one step OF THE SIMPLIFIED SCHEME, not a promise that
/// the scheme is a martingale. Integrating the independent normal leaves an
/// exponential affine function of the CIR endpoint, whose MGF is analytic.
pub fn conditional_drift(
    config: &MarketPathConfig,
    current_variance: f64,
) -> Result<ConditionalDriftControl, ComputeError> {
    config.validate()?;
    if !current_variance.is_finite() || current_variance < 0.0 {
        return Err(ComputeError::InvalidInput(
            "conditional variance must be finite nonnegative".into(),
        ));
    }
    let (stock, claim) = match config.model {
        MarketModel::BlackScholes { .. } => (Some(0.0), 0.0),
        MarketModel::Heston { params } if params.vol_of_variance == 0.0 => (Some(0.0), 0.0),
        MarketModel::Heston { params } => {
            let h = config.dt;
            let decay = (-params.kappa * h).exp();
            let one_minus = -(-params.kappa * h).exp_m1();
            let scale = params.vol_of_variance.powi(2) * one_minus / (4.0 * params.kappa);
            let shape =
                2.0 * params.kappa * params.long_run_variance / params.vol_of_variance.powi(2);
            let (a, b) = match config.integrated_variance_rule {
                IntegratedVarianceRule::Left => (1.0, 0.0),
                IntegratedVarianceRule::Trapezoid => (0.5, 0.5),
            };
            let c = params.correlation / params.vol_of_variance;
            let d = params.kappa * c - 0.5 * params.correlation.powi(2);
            let alpha = c + d * h * b;
            let denominator = 1.0 - 2.0 * scale * alpha;
            let stock = if denominator > 0.0 {
                let constant = -c
                    * (current_variance + params.kappa * params.long_run_variance * h)
                    + d * h * a * current_variance;
                let log_moment = constant - shape * (-2.0 * scale * alpha).ln_1p()
                    + current_variance * decay * alpha / denominator;
                let drift = log_moment.exp_m1();
                drift.is_finite().then_some(drift)
            } else {
                None
            };
            let mean_next =
                params.long_run_variance + (current_variance - params.long_run_variance) * decay;
            let expected_quadrature = h * (a * current_variance + b * mean_next);
            let true_integral = expected_future_integral(config, current_variance, h);
            (
                stock,
                (-config.rate * config.horizon()).exp() * (expected_quadrature - true_integral),
            )
        }
    };
    if !claim.is_finite() {
        return Err(ComputeError::Execution(
            "conditional claim drift overflow".into(),
        ));
    }
    Ok(ConditionalDriftControl {
        current_variance,
        adjusted_stock_fractional_drift: stock,
        discounted_claim_absolute_drift: claim,
    })
}

fn initial_variance(model: MarketModel) -> f64 {
    match model {
        MarketModel::BlackScholes { volatility } => volatility * volatility,
        MarketModel::Heston { params } => params.initial_variance,
    }
}
fn expected_variance(config: &MarketPathConfig, time: f64) -> f64 {
    match config.model {
        MarketModel::BlackScholes { .. } => initial_variance(config.model),
        MarketModel::Heston { params } => {
            params.long_run_variance
                + (params.initial_variance - params.long_run_variance)
                    * (-params.kappa * time).exp()
        }
    }
}
fn expected_future_integral(config: &MarketPathConfig, variance: f64, time: f64) -> f64 {
    match config.model {
        MarketModel::BlackScholes { .. } => variance * time,
        MarketModel::Heston { params } => {
            let x = params.kappa * time;
            let b = if x.abs() < 1e-4 {
                time * (x * 0.5 - x * x / 6.0 + x * x * x / 24.0 - x * x * x * x / 120.0)
            } else {
                time + (-x).exp_m1() / params.kappa
            };
            variance * (time - b) + params.long_run_variance * b
        }
    }
}
fn expected_discrete_integral(config: &MarketPathConfig, steps: usize) -> f64 {
    let t = steps as f64 * config.dt;
    match config.model {
        MarketModel::BlackScholes { .. } => initial_variance(config.model) * t,
        MarketModel::Heston { params } if params.vol_of_variance == 0.0 => {
            expected_future_integral(config, params.initial_variance, t)
        }
        MarketModel::Heston { params } => {
            let left = params.long_run_variance * t
                + (params.initial_variance - params.long_run_variance)
                    * config.dt
                    * (-(-params.kappa * t).exp_m1())
                    / (-(-params.kappa * config.dt).exp_m1());
            match config.integrated_variance_rule {
                IntegratedVarianceRule::Left => left,
                IntegratedVarianceRule::Trapezoid => {
                    left + 0.5
                        * config.dt
                        * (expected_variance(config, t) - params.initial_variance)
                }
            }
        }
    }
}
fn control(moments: &OnlineMoments, reference: f64) -> Result<SamplingControl, ComputeError> {
    if !moments.mean.is_finite()
        || !moments.m2.is_finite()
        || moments.m2 < 0.0
        || !reference.is_finite()
    {
        return Err(ComputeError::Execution(
            "nonfinite path control moments/reference".into(),
        ));
    }
    let estimate = moments.estimate();
    let error = estimate.mean - reference;
    let sigma = estimate
        .standard_error
        .filter(|se| *se > 0.0)
        .map(|se| error / se)
        .filter(|v| v.is_finite());
    Ok(SamplingControl {
        estimate,
        reference,
        signed_error: error,
        standard_errors_from_reference: sigma,
    })
}
fn overflow() -> ComputeError {
    ComputeError::InvalidInput("path diagnostic memory estimate overflow".into())
}
