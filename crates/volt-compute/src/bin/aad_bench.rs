use clap::Parser;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use volt_compute::{
    ExecutionTarget, OptionRow, Precision,
    aad::{AadResult, analytic_greeks},
    aad_job::{differentiate_job_with_gamma, plan_aad_rows_with_gamma},
    budget::MemoryBudget,
    volt_adapter::VoltValuationRequest,
};

#[derive(Parser)]
#[command(about = "Budgeted reverse-mode option Greeks with independent analytic validation")]
struct Args {
    #[arg(long, default_value = "cpu")]
    device: ExecutionTarget,
    #[arg(long, default_value = "f32", value_parser = ["f32", "f64"])]
    precision: String,
    #[arg(long, default_value_t = 65536)]
    rows: usize,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, default_value_t = 65536)]
    max_batch_rows: usize,
    #[arg(long, conflicts_with = "volt_input")]
    input: Option<PathBuf>,
    #[arg(long)]
    volt_input: Option<PathBuf>,
    #[arg(long, default_value_t = 256)]
    compute_budget_mib: u64,
    #[arg(long, default_value_t = 256)]
    host_working_mib: u64,
    #[arg(long, default_value_t = 32)]
    compute_reserve_mib: u64,
    #[arg(long)]
    plan_only: bool,
    /// Include spot gamma through a second-order forward-mode pass.
    #[arg(long)]
    gamma: bool,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    greeks_output: Option<PathBuf>,
}

#[derive(Serialize)]
struct MetricCheck {
    name: &'static str,
    max_absolute_error: f64,
    max_tolerance_ratio: f64,
    mismatches: usize,
}

fn validate(
    rows: &[OptionRow],
    values: &AadResult,
    precision: Precision,
) -> Result<Vec<MetricCheck>, Box<dyn std::error::Error>> {
    let names = [
        "price",
        "delta",
        "vega_per_unit_volatility",
        "rho_per_unit_rate",
        "theta_per_year",
        "dividend_per_unit_yield",
    ];
    let mut checks = names.map(|name| MetricCheck {
        name,
        max_absolute_error: 0.0,
        max_tolerance_ratio: 0.0,
        mismatches: 0,
    });
    let observed = [
        &values.prices,
        &values.delta,
        &values.vega,
        &values.rho,
        &values.theta,
        &values.dividend_sensitivity,
    ];
    for (i, row) in rows.iter().enumerate() {
        let analytic = analytic_greeks(row)?;
        let reference = [
            volt_compute::reference_price(row)?,
            analytic.delta,
            analytic.vega,
            analytic.rho,
            analytic.theta,
            analytic.dividend_sensitivity,
        ];
        for j in 0..6 {
            let error = (observed[j][i] - reference[j]).abs();
            let tolerance = match precision {
                Precision::F32 => 0.0002 + 0.0002 * reference[j].abs(),
                Precision::F64 => 1e-9 + 1e-10 * reference[j].abs(),
            };
            let ratio = error / tolerance;
            checks[j].max_absolute_error = checks[j].max_absolute_error.max(error);
            checks[j].max_tolerance_ratio = checks[j].max_tolerance_ratio.max(ratio);
            checks[j].mismatches += usize::from(!ratio.is_finite() || ratio > 1.0);
        }
    }
    Ok(checks.into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.repeats == 0 || args.repeats > 1000 {
        return Err("repeats must be 1..=1000".into());
    }
    if args.input.is_none()
        && args.volt_input.is_none()
        && (args.rows == 0 || args.rows > 4_000_000)
    {
        return Err("fixture rows must be 1..=4000000".into());
    }
    let outputs: Vec<_> = [&args.output, &args.greeks_output]
        .into_iter()
        .flatten()
        .collect();
    for output in &outputs {
        for input in [&args.input, &args.volt_input].into_iter().flatten() {
            if resolved(output)? == resolved(input)? {
                return Err("output cannot overwrite input".into());
            }
        }
    }
    if outputs.len() == 2 && resolved(outputs[0])? == resolved(outputs[1])? {
        return Err("report and Greek outputs must differ".into());
    }
    let mut provenance = None;
    let rows: Vec<OptionRow> = if let Some(path) = args.volt_input {
        let request: VoltValuationRequest = serde_json::from_reader(File::open(path)?)?;
        let batch = request.project()?;
        provenance = Some(serde_json::json!({"portfolio_hash": batch.portfolio_hash,
            "market_inputs_hash": batch.market_inputs_hash, "snapshot_id": batch.snapshot_id,
            "as_of": batch.as_of, "currency": batch.currency, "year_basis": batch.year_basis}));
        batch.rows
    } else if let Some(path) = args.input {
        serde_json::from_reader(File::open(path)?)?
    } else {
        volt_compute::benchmark::fixture(args.rows)
    };
    let mib = |n: u64| {
        n.checked_mul(1024 * 1024)
            .ok_or("budget byte count overflow")
    };
    let precision = if args.precision == "f64" {
        Precision::F64
    } else {
        Precision::F32
    };
    let budget = MemoryBudget {
        compute_bytes: mib(args.compute_budget_mib)?,
        host_working_bytes: mib(args.host_working_mib)?,
        compute_reserve_bytes: mib(args.compute_reserve_mib)?,
        max_batch_rows: args.max_batch_rows,
        ..MemoryBudget::default()
    };
    let numeric_input_hash = volt_compute::job::input_hash(&rows)?;
    let plan = plan_aad_rows_with_gamma(&rows, args.device, precision, &budget, args.gamma)?;
    if args.plan_only {
        let report = serde_json::json!({"schema":"volt.compute.aad_plan.v1", "plan":plan,
            "include_gamma":args.gamma, "numeric_input_hash":numeric_input_hash,
            "workload_model":if args.gamma {"black_scholes_reverse_and_spot_jet_384_slots_seven_outputs"} else {"black_scholes_reverse_256_slots_six_outputs"}, "volt_provenance":provenance});
        if let Some(path) = args.output {
            write_json(&path, &report)?;
        }
        serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
        return Ok(());
    }
    let mut dispatches = Vec::with_capacity(args.repeats);
    let mut passed = true;
    for i in 0..args.repeats {
        let result =
            differentiate_job_with_gamma(&rows, args.device, precision, &budget, args.gamma)?;
        let mut checks = validate(&rows, &result.values, precision)?;
        if let Some(gammas) = &result.gamma {
            let mut check = MetricCheck {
                name: "gamma_per_squared_spot_unit",
                max_absolute_error: 0.0,
                max_tolerance_ratio: 0.0,
                mismatches: 0,
            };
            for (row, value) in rows.iter().zip(gammas) {
                let expected = volt_compute::second_order::analytic_gamma(row)?;
                let tolerance = match precision {
                    Precision::F32 => 1e-6 + 0.0002 * expected.abs(),
                    Precision::F64 => 1e-10 + 1e-10 * expected.abs(),
                };
                let error = (value - expected).abs();
                let ratio = error / tolerance;
                check.max_absolute_error = check.max_absolute_error.max(error);
                check.max_tolerance_ratio = check.max_tolerance_ratio.max(ratio);
                check.mismatches += usize::from(!ratio.is_finite() || ratio > 1.0);
            }
            checks.push(check);
        }
        passed &= checks.iter().all(|check| check.mismatches == 0);
        if i + 1 == args.repeats
            && let Some(path) = &args.greeks_output
        {
            create_parent(path)?;
            let mut writer = BufWriter::new(File::create(path)?);
            for (j, row) in rows.iter().enumerate() {
                serde_json::to_writer(
                    &mut writer,
                    &serde_json::json!({"schema":"volt.compute.option_greeks.v1",
                        "numeric_input_hash":numeric_input_hash,
                        "row_id":row.row_id, "contract_id":row.contract_id, "price":result.values.prices[j],
                        "gamma":result.gamma.as_ref().map(|g| g[j]),
                        "delta":result.values.delta[j], "vega":result.values.vega[j], "rho":result.values.rho[j],
                        "theta":result.values.theta[j], "dividend_sensitivity":result.values.dividend_sensitivity[j]}),
                )?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
        }
        dispatches.push(serde_json::json!({"dispatch":i,"job_seconds":result.job_seconds,
            "completed_chunks":result.completed_chunks, "validation":checks,
            "parallel_sensitivities":result.parallel_sensitivities,"parallel_gamma":result.parallel_gamma,
            "max_sampled_allocator_live_bytes":result.values.max_sampled_allocator_live_bytes,
            "max_sampled_allocator_reserved_bytes":result.values.max_sampled_allocator_reserved_bytes}));
    }
    let executable_sha256 = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)));
    let report = serde_json::json!({"schema":"volt.compute.aad_run.v1", "device":args.device,
        "precision":precision,"budget":budget,"plan":plan,"rows":rows.len(), "volt_provenance":provenance,
        "numeric_input_hash":numeric_input_hash,"include_gamma":args.gamma,"os":std::env::consts::OS,
        "architecture":std::env::consts::ARCH,"build_profile":if cfg!(debug_assertions) {"debug"} else {"release"},
        "fusion_feature_enabled":cfg!(feature="fusion"), "executable_sha256":executable_sha256,
        "lockfile_sha256":format!("{:x}", Sha256::digest(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock")))),
        "method":"reverse-mode first derivatives; optional second-order forward spot gamma; independent analytic validation",
        "units":{"vega":"per 1.0 annual volatility fraction", "rho":"per 1.0 annual rate fraction",
            "dividend_sensitivity":"per 1.0 annual dividend yield fraction", "theta":"calendar decay per year",
            "delta":"per spot currency unit", "gamma":"per squared spot currency unit"},
        "timing":"each dispatch includes device setup, upload, forward/backward, readback and host aggregation; analytic validation excluded",
        "dispatches":dispatches,"passed":passed});
    if let Some(path) = args.output {
        write_json(&path, &report)?;
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    if !passed {
        return Err("AAD validation failed".into());
    }
    Ok(())
}
fn resolved(path: &Path) -> std::io::Result<PathBuf> {
    path.canonicalize().or_else(|_| std::path::absolute(path))
}
fn create_parent(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
fn write_json(path: &Path, value: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
    create_parent(path)?;
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
