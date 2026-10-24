use clap::Parser;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use volt_compute::{
    ExecutionTarget, OptionRow, Precision,
    budget::MemoryBudget,
    persistent_workload::{self, PersistentWorkloadConfig},
    volt_adapter::VoltValuationRequest,
};

#[derive(Parser)]
#[command(about = "Validate complete warm valuation jobs with persistent device inputs")]
struct Args {
    #[arg(long, default_value = "cpu")]
    device: ExecutionTarget,
    #[arg(long, default_value = "f32", value_parser = ["f32", "f64"])]
    precision: String,
    #[arg(long, default_value_t = 65536)]
    rows: usize,
    #[arg(long, default_value_t = 10)]
    repeats: usize,
    #[arg(long, value_delimiter = ',', default_value = "0.99,1,1.01")]
    spot_scales: Vec<f64>,
    #[arg(long, conflicts_with = "volt_input")]
    input: Option<PathBuf>,
    #[arg(long)]
    volt_input: Option<PathBuf>,
    #[arg(long, default_value_t = 1024)]
    compute_budget_mib: u64,
    #[arg(long, default_value_t = 1024)]
    host_working_mib: u64,
    #[arg(long, default_value_t = 32)]
    compute_reserve_mib: u64,
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.input.is_none()
        && args.volt_input.is_none()
        && (args.rows == 0 || args.rows > 4_000_000)
    {
        return Err("fixture rows must be 1..=4000000".into());
    }
    if let Some(output) = &args.output {
        for input in [&args.input, &args.volt_input].into_iter().flatten() {
            if resolved(output)? == resolved(input)? {
                return Err("output cannot overwrite input".into());
            }
        }
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
    let report = persistent_workload::run(
        &rows,
        PersistentWorkloadConfig {
            target: args.device,
            precision: if args.precision == "f64" {
                Precision::F64
            } else {
                Precision::F32
            },
            repeats: args.repeats,
            spot_scales: args.spot_scales,
            budget: MemoryBudget {
                compute_bytes: mib(args.compute_budget_mib)?,
                host_working_bytes: mib(args.host_working_mib)?,
                compute_reserve_bytes: mib(args.compute_reserve_mib)?,
                max_batch_rows: rows.len(),
                ..MemoryBudget::default()
            },
        },
    )?;
    let passed = report.passed;
    let value = serde_json::json!({"schema": "volt.compute.persistent_run.v1", "workload": report, "volt_provenance": provenance});
    if let Some(path) = args.output {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let mut writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(&mut writer, &value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &value)?;
    if !passed {
        return Err("persistent valuation accuracy failed".into());
    }
    Ok(())
}
fn resolved(path: &Path) -> std::io::Result<PathBuf> {
    path.canonicalize().or_else(|_| std::path::absolute(path))
}
