use crate::{
    ComputeError, ExecutionTarget, OptionRow, Precision,
    budget::{MemoryBudget, MemoryPlan},
    chunked::ChunkedStats,
    job::{ValidationSummary, input_hash, validate_scaled_result},
    persistent::PreparedValuation,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentWorkloadConfig {
    pub target: ExecutionTarget,
    pub precision: Precision,
    pub repeats: usize,
    pub spot_scales: Vec<f64>,
    pub budget: MemoryBudget,
}

#[derive(Debug, Serialize)]
pub struct FrameReport {
    pub spot_scale: f64,
    /// Evaluation, readback and deterministic aggregation. Reference checks and
    /// report construction are outside the timed job interval.
    pub job_seconds: f64,
    pub stats: ChunkedStats,
    pub option_position_value: f64,
    pub validation: ValidationSummary,
    pub sample_prices: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct PersistentWorkloadReport {
    pub schema: &'static str,
    pub config: PersistentWorkloadConfig,
    pub rows: usize,
    pub plan: MemoryPlan,
    pub os: &'static str,
    pub architecture: &'static str,
    pub build_profile: &'static str,
    pub fusion_feature_enabled: bool,
    pub numeric_input_hash: String,
    pub lockfile_sha256: String,
    /// Validation, device setup, metadata preparation and initial input upload.
    pub preparation_seconds: f64,
    pub packing_and_upload_seconds: f64,
    pub input_batches_uploaded: usize,
    pub first_frame: FrameReport,
    pub warm_frames: Vec<FrameReport>,
    pub warm_job_median_seconds: f64,
    pub passed: bool,
}

pub fn run(
    rows: &[OptionRow],
    config: PersistentWorkloadConfig,
) -> Result<PersistentWorkloadReport, ComputeError> {
    if config.repeats == 0
        || config.repeats > 10_000
        || config.spot_scales.is_empty()
        || config.spot_scales.len() > 1000
        || config
            .spot_scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(ComputeError::InvalidInput(
            "repeats must be 1..=10000 and spot scales must be 1..=1000 positive finite values"
                .into(),
        ));
    }
    let numeric_input_hash = input_hash(rows)?;
    let started = Instant::now();
    let job = PreparedValuation::new(rows, config.target, config.precision, &config.budget)?;
    let preparation_seconds = started.elapsed().as_secs_f64();
    let first_frame = frame(&job, config.spot_scales[0], config.precision)?;
    let mut warm_frames = Vec::with_capacity(config.repeats);
    for i in 0..config.repeats {
        warm_frames.push(frame(
            &job,
            config.spot_scales[i % config.spot_scales.len()],
            config.precision,
        )?);
    }
    let mut times: Vec<f64> = warm_frames.iter().map(|frame| frame.job_seconds).collect();
    times.sort_by(f64::total_cmp);
    let warm_job_median_seconds = if times.len().is_multiple_of(2) {
        (times[times.len() / 2 - 1] + times[times.len() / 2]) * 0.5
    } else {
        times[times.len() / 2]
    };
    let passed =
        first_frame.validation.passed && warm_frames.iter().all(|frame| frame.validation.passed);
    Ok(PersistentWorkloadReport {
        schema: "volt.compute.persistent_workload.v1",
        rows: rows.len(),
        plan: job.plan().clone(),
        os: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        fusion_feature_enabled: cfg!(feature = "fusion"),
        config,
        numeric_input_hash,
        lockfile_sha256: format!(
            "{:x}",
            Sha256::digest(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../Cargo.lock"
            )))
        ),
        preparation_seconds,
        packing_and_upload_seconds: job.packing_and_upload_seconds,
        input_batches_uploaded: 1,
        first_frame,
        warm_frames,
        warm_job_median_seconds,
        passed,
    })
}

fn frame(
    job: &PreparedValuation<'_>,
    scale: f64,
    precision: Precision,
) -> Result<FrameReport, ComputeError> {
    let started = Instant::now();
    let result = job.evaluate(scale)?;
    let job_seconds = started.elapsed().as_secs_f64();
    let validation = validate_scaled_result(job.source_rows(), &result, precision, scale)?;
    Ok(FrameReport {
        spot_scale: scale,
        job_seconds,
        stats: result.stats,
        option_position_value: result.portfolio_value,
        validation,
        sample_prices: result.prices.into_iter().take(6).collect(),
    })
}
