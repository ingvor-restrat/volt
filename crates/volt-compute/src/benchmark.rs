use crate::{
    ComputeError, ExecutionTarget, OptionRow, OptionSide, Precision, ResidentOptions, Session,
    reference_price,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub target: ExecutionTarget,
    pub precision: Precision,
    pub rows: usize,
    pub repeats: usize,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkReport {
    pub schema: &'static str,
    pub timestamp_unix_seconds: u64,
    pub config: BenchmarkConfig,
    pub os: &'static str,
    pub architecture: &'static str,
    pub build_profile: &'static str,
    pub device: String,
    pub burn_version: &'static str,
    pub cuda_compiler: Option<&'static str>,
    pub allocator_live_bytes_after_evaluation: Option<u64>,
    pub allocator_reserved_bytes_after_evaluation: Option<u64>,
    pub lockfile_sha256: String,
    pub input_sha256: String,
    pub executable_sha256: Option<String>,
    pub fixture_seconds: f64,
    pub reference_seconds: f64,
    /// Packing, host-to-device conversion/upload, and synchronization.
    pub packing_and_upload_seconds: f64,
    /// First evaluation, including any JIT compilation and runtime setup.
    pub first_evaluation_seconds: f64,
    pub warm_evaluation_seconds: Vec<f64>,
    pub readback_seconds: f64,
    pub total_seconds: f64,
    pub logical_resident_input_bytes: usize,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub max_absolute_price_error: f64,
    pub max_tolerance_ratio: f64,
    pub mismatched_prices: usize,
    pub reference_portfolio_value: f64,
    pub device_portfolio_value: f64,
    pub portfolio_absolute_error: f64,
    pub portfolio_tolerance: f64,
    pub sample_prices: Vec<f64>,
    pub passed: bool,
}

/// Deterministic heterogeneous options, including dividend yields, negative
/// rates, long/short positions, and varied strike, expiry, and volatility.
pub fn fixture(count: usize) -> Vec<OptionRow> {
    (0..count)
        .map(|i| OptionRow {
            row_id: format!("scenario-row-{i}"),
            contract_id: format!("option-{}", i % 997),
            side: if i % 2 == 0 {
                OptionSide::Call
            } else {
                OptionSide::Put
            },
            spot: 70.0 + (i % 71) as f64,
            strike: 75.0 + (i % 51) as f64,
            maturity_years: (7 + i % 724) as f64 / 365.0,
            volatility: 0.10 + (i % 41) as f64 / 100.0,
            rate: -0.01 + (i % 8) as f64 / 100.0,
            dividend_yield: (i % 5) as f64 / 100.0,
            quantity: if i % 3 == 0 { -2 } else { 1 },
            multiplier: 100,
        })
        .collect()
}

pub fn run(config: BenchmarkConfig) -> Result<BenchmarkReport, ComputeError> {
    run_with_rows(config, None)
}

/// Validate and measure application-supplied numeric option rows, or generate the
/// deterministic fixture when no rows are supplied. Input file parsing is outside
/// these timings; hashing and packing are included.
pub fn run_with_rows(
    config: BenchmarkConfig,
    supplied_rows: Option<Vec<OptionRow>>,
) -> Result<BenchmarkReport, ComputeError> {
    if config.rows == 0 || config.repeats == 0 || config.rows > 4_000_000 || config.repeats > 10_000
    {
        return Err(ComputeError::InvalidInput(
            "rows must be 1..=4000000 and repeats 1..=10000".into(),
        ));
    }
    let total = Instant::now();
    let started = Instant::now();
    let rows = supplied_rows.unwrap_or_else(|| fixture(config.rows));
    if rows.len() != config.rows {
        return Err(ComputeError::InvalidInput(
            "configured row count does not match input".into(),
        ));
    }
    let input_sha256 =
        hash(&serde_json::to_vec(&rows).map_err(|e| ComputeError::Execution(e.to_string()))?);
    let fixture_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let reference = rows
        .iter()
        .map(reference_price)
        .collect::<Result<Vec<_>, _>>()?;
    let reference_portfolio_value = rows
        .iter()
        .zip(&reference)
        .map(|(r, p)| p * r.quantity as f64 * r.multiplier as f64)
        .sum::<f64>();
    let reference_seconds = started.elapsed().as_secs_f64();
    let session = Session::new(config.target, config.precision, config.rows)?;
    let started = Instant::now();
    let resident = ResidentOptions::upload(&session, &rows)?;
    session.synchronize()?;
    let packing_and_upload_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut prices = resident.prices();
    let mut value = resident.portfolio_value(prices.clone());
    session.synchronize()?;
    let first_evaluation_seconds = started.elapsed().as_secs_f64();
    let mut warm_evaluation_seconds = Vec::with_capacity(config.repeats);
    for _ in 0..config.repeats {
        let started = Instant::now();
        prices = resident.prices();
        value = resident.portfolio_value(prices.clone());
        session.synchronize()?;
        warm_evaluation_seconds.push(started.elapsed().as_secs_f64());
    }
    let started = Instant::now();
    let actual = session.read_vector(&prices)?;
    let device_portfolio_value = session.read_vector(&value)?[0];
    let readback_seconds = started.elapsed().as_secs_f64();
    // f64 acceptance requires accurate math functions, not only f64 storage.
    let (absolute_tolerance, relative_tolerance) = match config.precision {
        Precision::F32 => (0.0002, 0.0001),
        Precision::F64 => (1e-10, 1e-10),
    };
    let mut max_absolute_price_error = 0.0_f64;
    let mut max_tolerance_ratio = 0.0_f64;
    let mut mismatched_prices = 0;
    let mut portfolio_tolerance = 0.0;
    for ((expected, observed), row) in reference.iter().zip(&actual).zip(&rows) {
        if !observed.is_finite() {
            return Err(ComputeError::Execution("nonfinite output price".into()));
        }
        let tolerance = absolute_tolerance + relative_tolerance * expected.abs();
        let error = (observed - expected).abs();
        max_absolute_price_error = max_absolute_price_error.max(error);
        max_tolerance_ratio = max_tolerance_ratio.max(error / tolerance);
        mismatched_prices += usize::from(error > tolerance);
        portfolio_tolerance += tolerance * (row.quantity as f64 * row.multiplier as f64).abs();
    }
    // Include reduction rounding against gross exposure, so long/short
    // cancellation cannot make a relative tolerance artificially strict.
    let gross = rows
        .iter()
        .zip(&reference)
        .map(|(r, p)| (p * r.quantity as f64 * r.multiplier as f64).abs())
        .sum::<f64>();
    portfolio_tolerance += match config.precision {
        Precision::F32 => 0.00001,
        Precision::F64 => 1e-12,
    } * gross;
    let portfolio_absolute_error = (reference_portfolio_value - device_portfolio_value).abs();
    let allocator = session.device().memory_pool_usage();
    let passed = actual.len() == reference.len()
        && mismatched_prices == 0
        && device_portfolio_value.is_finite()
        && portfolio_absolute_error <= portfolio_tolerance;
    Ok(BenchmarkReport {
        schema: "volt.compute.benchmark.v1",
        timestamp_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        config,
        os: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        device: format!("{:?}", session.device()),
        burn_version: "0.22.1",
        cuda_compiler: if matches!(session.target(), ExecutionTarget::Cuda { .. }) {
            Some("cuda_cpp_nvrtc")
        } else {
            None
        },
        allocator_live_bytes_after_evaluation: allocator.map(|usage| usage.bytes_in_use),
        allocator_reserved_bytes_after_evaluation: allocator.map(|usage| usage.bytes_reserved),
        lockfile_sha256: hash(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../Cargo.lock"
        ))),
        input_sha256,
        executable_sha256: std::env::current_exe()
            .ok()
            .and_then(|p| std::fs::read(p).ok())
            .map(|bytes| hash(&bytes)),
        fixture_seconds,
        reference_seconds,
        packing_and_upload_seconds,
        first_evaluation_seconds,
        warm_evaluation_seconds,
        readback_seconds,
        total_seconds: total.elapsed().as_secs_f64(),
        logical_resident_input_bytes: rows.len()
            * 8
            * match session.precision() {
                Precision::F32 => 4,
                Precision::F64 => 8,
            },
        absolute_tolerance,
        relative_tolerance,
        max_absolute_price_error,
        max_tolerance_ratio,
        mismatched_prices,
        reference_portfolio_value,
        device_portfolio_value,
        portfolio_absolute_error,
        portfolio_tolerance,
        sample_prices: actual.into_iter().take(6).collect(),
        passed,
    })
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
