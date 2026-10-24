use clap::Parser;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use volt_compute::{
    ExecutionTarget, OptionRow, Precision,
    budget::MemoryBudget,
    monte_carlo_job::{MonteCarloConfig, plan_monte_carlo, run_monte_carlo},
    monte_carlo_validation::{VALIDATION_STANDARD_ERRORS, validate_monte_carlo},
    volt_adapter::VoltValuationRequest,
};

#[derive(Parser)]
#[command(
    about = "Bounded antithetic GBM Monte Carlo with reverse Greeks, LR gamma and pair standard errors"
)]
struct Args {
    #[arg(long, default_value = "cpu")]
    device: ExecutionTarget,
    #[arg(long,default_value="f32",value_parser=["f32","f64"])]
    precision: String,
    #[arg(long, default_value_t = 4)]
    rows: usize,
    /// Independent pairs: two terminal outcomes per pair and option row.
    #[arg(long, default_value_t = 131072)]
    pairs: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 1)]
    repeats: usize,
    /// Ceiling on expanded row*pair*2 outcomes, not independent option rows.
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
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    estimates_output: Option<PathBuf>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.repeats == 0 || args.repeats > 1000 {
        return Err("repeats must be 1..=1000".into());
    }
    if args.input.is_none() && args.volt_input.is_none() && (args.rows == 0 || args.rows > 4096) {
        return Err("fixture row count must be 1..=4096".into());
    }
    let outputs: Vec<_> = [&args.output, &args.estimates_output]
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
        return Err("report and estimates outputs must differ".into());
    }
    let mut provenance = None;
    let rows: Vec<OptionRow> = if let Some(path) = args.volt_input {
        let request: VoltValuationRequest = serde_json::from_reader(File::open(path)?)?;
        let batch = request.project()?;
        provenance = Some(
            serde_json::json!({"portfolio_hash":batch.portfolio_hash,"market_inputs_hash":batch.market_inputs_hash,
            "snapshot_id":batch.snapshot_id,"as_of":batch.as_of,"currency":batch.currency,"year_basis":batch.year_basis}),
        );
        batch.rows
    } else if let Some(path) = args.input {
        serde_json::from_reader(File::open(path)?)?
    } else {
        let mut rows = volt_compute::benchmark::fixture(args.rows);
        for (i, row) in rows.iter_mut().enumerate() {
            row.spot = 100.0 + (i % 5) as f64 * 5.0;
            row.strike = 100.0;
            row.maturity_years = 0.25 + (i % 7) as f64 * 0.25;
            row.volatility = 0.2 + (i % 3) as f64 * 0.05;
            row.rate = 0.03;
            row.dividend_yield = 0.01;
        }
        rows
    };
    let mib = |n: u64| {
        n.checked_mul(1024 * 1024)
            .ok_or("budget byte count overflow")
    };
    let config = MonteCarloConfig {
        target: args.device,
        precision: if args.precision == "f64" {
            Precision::F64
        } else {
            Precision::F32
        },
        seed: args.seed,
        antithetic_pairs: args.pairs,
        budget: MemoryBudget {
            compute_bytes: mib(args.compute_budget_mib)?,
            host_working_bytes: mib(args.host_working_mib)?,
            compute_reserve_bytes: mib(args.compute_reserve_mib)?,
            max_batch_rows: args.max_batch_rows,
            ..MemoryBudget::default()
        },
    };
    let plan = plan_monte_carlo(&rows, &config)?;
    let input_hash = volt_compute::job::input_hash(&rows)?;
    if args.plan_only {
        let report = serde_json::json!({"schema":"volt.compute.monte_carlo_admission.v1","config":config,"plan":plan,
            "numeric_input_hash":input_hash,"volt_provenance":provenance});
        if let Some(path) = args.output {
            write_json(&path, &report)?;
        }
        serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
        return Ok(());
    }
    let mut dispatches = Vec::with_capacity(args.repeats);
    let mut passed = true;
    for i in 0..args.repeats {
        let run = run_monte_carlo(&rows, &config)?;
        let validation = validate_monte_carlo(&rows, &run.result, config.precision)?;
        passed &= validation.iter().all(|check| check.passed);
        if i + 1 == args.repeats
            && let Some(path) = &args.estimates_output
        {
            create_parent(path)?;
            let mut writer = BufWriter::new(File::create(path)?);
            for (row, observed) in rows.iter().zip(&run.result.rows) {
                serde_json::to_writer(
                    &mut writer,
                    &serde_json::json!({"schema":"volt.compute.monte_carlo_estimate.v1",
                    "numeric_input_hash":input_hash,"row_id":row.row_id,"contract_id":row.contract_id,
                    "price":observed.moments.price.estimate(),"delta":observed.moments.delta.estimate(),
                    "vega":observed.moments.vega.estimate(),"rho":observed.moments.rho.estimate(),
                    "theta":observed.moments.theta.estimate(),"dividend_sensitivity":observed.moments.dividend_sensitivity.estimate(),
                    "gamma":observed.moments.gamma.estimate()}),
                )?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
        }
        dispatches.push(serde_json::json!({"dispatch":i,"job_seconds":run.job_seconds,"completed_batches":run.completed_batches,
            "max_sampled_allocator_live_bytes":run.result.max_sampled_allocator_live_bytes,
            "max_sampled_allocator_reserved_bytes":run.result.max_sampled_allocator_reserved_bytes,
            "validation":validation}));
    }
    let mut normal_hasher = Sha256::new();
    for pair in 0..config.antithetic_pairs {
        normal_hasher
            .update(volt_compute::monte_carlo::normal_at(config.seed, pair as u64).to_le_bytes());
    }
    let normal_f64_sha256 = format!("{:x}", normal_hasher.finalize());
    let executable_sha256 = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)));
    let report = serde_json::json!({"schema":"volt.compute.monte_carlo_run.v1","config":config,"plan":plan,
        "numeric_input_hash":input_hash,"volt_provenance":provenance,"executable_sha256":executable_sha256,
        "lockfile_sha256":format!("{:x}",Sha256::digest(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock")))),
        "os":std::env::consts::OS,"architecture":std::env::consts::ARCH,
        "build_profile":if cfg!(debug_assertions){"debug"}else{"release"},"fusion_feature_enabled":cfg!(feature="fusion"),
        "method":"exact terminal GBM; antithetic pairs; reverse pathwise first derivatives; likelihood-ratio gamma",
        "normal_f64_sha256":normal_f64_sha256,
        "sampling":"splitmix64 indexed box-muller; one independent observation per antithetic pair; row-shared common random numbers",
        "portfolio_se":"uncertainty of numerical estimator with pair covariance; not a joint market P&L risk model",
        "units":{"vega_rho_dividend":"per 1.0 annual fraction","theta":"calendar decay per year","gamma":"per squared spot currency unit"},
        "validation_standard_errors":VALIDATION_STANDARD_ERRORS,"dispatches":dispatches,"passed":passed,
        "timing":"complete dispatch including RNG/upload/reverse/readback/moments; independent analytic checks excluded"});
    if let Some(path) = args.output {
        write_json(&path, &report)?;
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    if !passed {
        return Err("Monte Carlo analytic validation failed".into());
    }
    Ok(())
}
fn resolved(path: &Path) -> std::io::Result<PathBuf> {
    path.canonicalize().or_else(|_| std::path::absolute(path))
}
fn create_parent(path: &Path) -> std::io::Result<()> {
    if let Some(p) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p)?;
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
