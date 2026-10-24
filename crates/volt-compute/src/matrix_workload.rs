//! Matrix correctness and timing workloads with explicit logical memory leases.

use crate::{
    ComputeError, ExecutionTarget, Precision,
    matrix::{MatrixBudget, MatrixMemorySnapshot, MatrixWorkspace, checked_elements},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixWorkloadConfig {
    pub target: ExecutionTarget,
    pub precision: Precision,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub repeats: usize,
    #[serde(default)]
    pub budget: MatrixBudget,
}

#[derive(Debug, Serialize)]
pub struct MatrixWorkloadReport {
    pub schema: &'static str,
    pub timestamp_unix_seconds: u64,
    pub config: MatrixWorkloadConfig,
    pub fixture: &'static str,
    pub reference_method: &'static str,
    pub os: &'static str,
    pub architecture: &'static str,
    pub build_profile: &'static str,
    pub device: String,
    pub burn_version: &'static str,
    pub cubecl_version: &'static str,
    pub fusion_feature_enabled: bool,
    pub cuda_compiler: Option<&'static str>,
    pub matmul_compute_policy: &'static str,
    pub source_sha256: String,
    pub input_sha256: String,
    pub lockfile_sha256: String,
    pub executable_sha256: Option<String>,
    pub output_sha256: String,
    pub packing_and_input_hash_seconds: f64,
    pub upload_seconds: f64,
    pub first_matmul_seconds: f64,
    pub warm_matmul_seconds: Vec<f64>,
    pub readback_seconds: f64,
    pub independent_reference_seconds: f64,
    pub total_seconds: f64,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub max_absolute_error: f64,
    pub root_mean_squared_error: f64,
    pub max_tolerance_ratio: f64,
    pub mismatched_values: usize,
    pub sample_output: Vec<f64>,
    pub modeled_memory_at_readback: MatrixMemorySnapshot,
    pub max_sampled_allocator_live_bytes: Option<u64>,
    pub max_sampled_allocator_reserved_bytes: Option<u64>,
    pub sampled_allocator_reserved_exceeds_compute_budget: bool,
    pub passed: bool,
}

pub fn run(config: MatrixWorkloadConfig) -> Result<MatrixWorkloadReport, ComputeError> {
    let total = Instant::now();
    if config.repeats == 0 || config.repeats > 10_000 {
        return Err(ComputeError::InvalidInput(
            "matrix repeats must be 1..=10000".into(),
        ));
    }
    let left_count = checked_elements(config.m, config.k)?;
    let right_count = checked_elements(config.k, config.n)?;
    checked_elements(config.m, config.n)?;
    let operations = (config.m as u128) * (config.k as u128) * (config.n as u128);
    let dense_reference = operations <= 16_777_216;
    let workspace = MatrixWorkspace::new(config.target, config.precision, config.budget.clone())?;
    let host_source_bytes = left_count
        .checked_add(right_count)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(config.repeats * std::mem::size_of::<f64>() + 4096))
        .ok_or_else(|| ComputeError::InvalidInput("matrix host fixture size overflow".into()))?;
    // Admission precedes allocation. Source buffers stay charged throughout both
    // execution and independent validation; readback is separately leased.
    let _source_lease = workspace.reserve_host_bytes(host_source_bytes as u64)?;
    let started = Instant::now();
    let (left, right) = if dense_reference {
        (
            (0..left_count)
                .map(|index| dense_value(index, 0x9e37_79b9))
                .collect::<Vec<_>>(),
            (0..right_count)
                .map(|index| dense_value(index, 0x243f_6a88))
                .collect::<Vec<_>>(),
        )
    } else {
        (
            (0..left_count)
                .map(|index| row_term(index / config.k) + inner_left_term(index % config.k))
                .collect(),
            (0..right_count)
                .map(|index| inner_right_term(index / config.n) + column_term(index % config.n))
                .collect(),
        )
    };
    let mut input_digest = Sha256::new();
    for dimension in [config.m, config.k, config.n] {
        input_digest.update((dimension as u64).to_le_bytes());
    }
    for value in left.iter().chain(&right) {
        input_digest.update(value.to_le_bytes());
    }
    let input_sha256 = format!("{:x}", input_digest.finalize());
    let packing_and_input_hash_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let a = workspace.upload(config.m, config.k, &left)?;
    let b = workspace.upload(config.k, config.n, &right)?;
    let upload_seconds = started.elapsed().as_secs_f64();
    let mut telemetry = AllocatorTelemetry::default();
    telemetry.sample(&workspace);
    let started = Instant::now();
    let mut product = Some(workspace.matmul(&a, &b)?);
    let first_matmul_seconds = started.elapsed().as_secs_f64();
    telemetry.sample(&workspace);
    let mut warm_matmul_seconds = Vec::with_capacity(config.repeats);
    for _ in 0..config.repeats {
        drop(product.take());
        let started = Instant::now();
        product = Some(workspace.matmul(&a, &b)?);
        warm_matmul_seconds.push(started.elapsed().as_secs_f64());
        telemetry.sample(&workspace);
    }
    let started = Instant::now();
    let output = workspace.download(product.as_ref().expect("product exists after repetitions"))?;
    let readback_seconds = started.elapsed().as_secs_f64();
    telemetry.sample(&workspace);
    let modeled_memory_at_readback = workspace.snapshot();
    let (absolute_tolerance, relative_tolerance) = match config.precision {
        Precision::F32 => (0.0005, 0.0002),
        Precision::F64 => (1e-10, 1e-10),
    };
    let started = Instant::now();
    // The large structured fixture is rank two by construction. Its full
    // reference comes from expanding (a_i+b_k)(c_k+d_j), not another matmul.
    let mut sum_b = 0.0;
    let mut sum_c = 0.0;
    let mut sum_bc = 0.0;
    if !dense_reference {
        for inner in 0..config.k {
            let b = inner_left_term(inner);
            let c = inner_right_term(inner);
            sum_b += b;
            sum_c += c;
            sum_bc += b * c;
        }
    }
    let mut max_absolute_error = 0.0_f64;
    let mut sum_squared_error = 0.0;
    let mut max_tolerance_ratio = 0.0_f64;
    let mut mismatched_values = 0;
    for (index, &actual) in output.values().iter().enumerate() {
        let row = index / config.n;
        let column = index % config.n;
        let expected = if dense_reference {
            (0..config.k)
                .map(|inner| left[row * config.k + inner] * right[inner * config.n + column])
                .sum::<f64>()
        } else {
            let a = row_term(row);
            let d = column_term(column);
            a * sum_c + config.k as f64 * a * d + sum_bc + d * sum_b
        };
        let error = (actual - expected).abs();
        let tolerance = absolute_tolerance + relative_tolerance * expected.abs();
        max_absolute_error = max_absolute_error.max(error);
        sum_squared_error += error * error;
        max_tolerance_ratio = max_tolerance_ratio.max(error / tolerance);
        mismatched_values +=
            usize::from(!actual.is_finite() || !expected.is_finite() || error > tolerance);
    }
    let independent_reference_seconds = started.elapsed().as_secs_f64();
    let mut output_digest = Sha256::new();
    for value in output.values() {
        output_digest.update(value.to_le_bytes());
    }
    let mut source_digest = Sha256::new();
    source_digest.update(include_bytes!("matrix.rs"));
    source_digest.update(include_bytes!("matrix_workload.rs"));
    source_digest.update(include_bytes!("runtime.rs"));
    let allocator_exceeded = telemetry
        .reserved
        .is_some_and(|bytes| bytes > config.budget.compute_bytes);
    Ok(MatrixWorkloadReport {
        schema: "volt.compute.matrix_workload.v1",
        timestamp_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        fixture: if dense_reference {
            "dense_deterministic_nondyadic"
        } else {
            "structured_rank_two_dyadic"
        },
        reference_method: if dense_reference {
            "independent_scalar_f64_dot_products"
        } else {
            "independent_structured_closed_form_f64"
        },
        os: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        device: workspace.device_description(),
        burn_version: "0.22.1",
        cubecl_version: "0.11.3",
        fusion_feature_enabled: cfg!(feature = "fusion"),
        cuda_compiler: if matches!(config.target, ExecutionTarget::Cuda { .. }) {
            Some("cuda_cpp_nvrtc")
        } else {
            None
        },
        matmul_compute_policy: match config.target {
            ExecutionTarget::Cpu => "burn_flex_native",
            ExecutionTarget::Metal => "burn_cubecl_native_metal",
            ExecutionTarget::Cuda { .. } => "cubek_simple_unit_full_precision",
        },
        source_sha256: format!("{:x}", source_digest.finalize()),
        input_sha256,
        lockfile_sha256: hash(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../Cargo.lock"
        ))),
        executable_sha256: std::env::current_exe()
            .ok()
            .and_then(|path| hash_file(&path).ok()),
        output_sha256: format!("{:x}", output_digest.finalize()),
        packing_and_input_hash_seconds,
        upload_seconds,
        first_matmul_seconds,
        warm_matmul_seconds,
        readback_seconds,
        independent_reference_seconds,
        total_seconds: total.elapsed().as_secs_f64(),
        absolute_tolerance,
        relative_tolerance,
        max_absolute_error,
        root_mean_squared_error: (sum_squared_error / output.values().len() as f64).sqrt(),
        max_tolerance_ratio,
        mismatched_values,
        sample_output: output.values().iter().take(6).copied().collect(),
        modeled_memory_at_readback,
        max_sampled_allocator_live_bytes: telemetry.live,
        max_sampled_allocator_reserved_bytes: telemetry.reserved,
        sampled_allocator_reserved_exceeds_compute_budget: allocator_exceeded,
        passed: mismatched_values == 0 && !allocator_exceeded,
        config,
    })
}

fn dense_value(index: usize, seed: u64) -> f64 {
    let mut value = (index as u64).wrapping_add(seed);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    // A non-power-of-two denominator exercises input rounding as well as layout.
    ((value % 257) as i64 - 128) as f64 / 127.0
}
fn row_term(index: usize) -> f64 {
    ((index % 17) as i64 - 8) as f64 / 16.0
}
fn inner_left_term(index: usize) -> f64 {
    ((index % 13) as i64 - 6) as f64 / 16.0
}
fn inner_right_term(index: usize) -> f64 {
    ((index % 11) as i64 - 5) as f64 / 32.0
}
fn column_term(index: usize) -> f64 {
    ((index % 19) as i64 - 9) as f64 / 32.0
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash_file(path: &std::path::Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Default)]
struct AllocatorTelemetry {
    live: Option<u64>,
    reserved: Option<u64>,
}
impl AllocatorTelemetry {
    fn sample(&mut self, workspace: &MatrixWorkspace) {
        if let Some((live, reserved)) = workspace.allocator_usage() {
            self.live = Some(self.live.unwrap_or(0).max(live));
            self.reserved = Some(self.reserved.unwrap_or(0).max(reserved));
        }
    }
}
