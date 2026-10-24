use clap::Parser;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use volt_compute::{
    ExecutionTarget, OptionRow, Precision,
    benchmark::{self, BenchmarkConfig},
    budget::MemoryBudget,
    chunked::{plan_rows, value_rows},
    job::{input_hash, validate_result},
    volt_adapter::VoltValuationRequest,
};

#[derive(Parser)]
#[command(about = "Validate resident or memory-budgeted option valuation on CPU, Metal, or CUDA")]
struct Args {
    #[arg(long, default_value = "cpu")]
    device: ExecutionTarget,
    #[arg(long, default_value = "f32", value_parser = ["f32", "f64"])]
    precision: String,
    #[arg(long, default_value_t = 65536)]
    rows: usize,
    #[arg(long, default_value_t = 10)]
    repeats: usize,
    #[arg(long)]
    output: Option<PathBuf>,
    /// JSON array of raw numeric OptionRow records. Input length replaces --rows.
    #[arg(long, conflicts_with = "volt_input")]
    input: Option<PathBuf>,
    /// Versioned exact Volt portfolio plus point-in-time market snapshot.
    #[arg(long, requires = "chunked")]
    volt_input: Option<PathBuf>,
    /// Execute the job in memory-budgeted chunks rather than resident timing mode.
    #[arg(long)]
    chunked: bool,
    /// Validate and print the admission plan without initializing a device.
    #[arg(long, requires = "chunked", conflicts_with = "marks_output")]
    plan_only: bool,
    #[arg(long, requires = "chunked", default_value_t = 256)]
    compute_budget_mib: u64,
    #[arg(long, requires = "chunked", default_value_t = 256)]
    host_working_mib: u64,
    #[arg(long, requires = "chunked", default_value_t = 32)]
    compute_reserve_mib: u64,
    #[arg(long, requires = "chunked", default_value_t = 20)]
    headroom_percent: u8,
    #[arg(long, requires = "chunked", default_value_t = 65536)]
    max_batch_rows: usize,
    /// Write ordered per-row marks as JSON Lines without another full JSON copy.
    #[arg(long, requires = "chunked")]
    marks_output: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.input.is_none()
        && args.volt_input.is_none()
        && (args.rows == 0 || args.rows > 4_000_000)
    {
        return Err("generated fixture rows must be 1..=4000000".into());
    }
    for output in [&args.output, &args.marks_output].into_iter().flatten() {
        for input in [&args.input, &args.volt_input].into_iter().flatten() {
            if resolved(output)? == resolved(input)? {
                return Err("output must not overwrite an input file".into());
            }
        }
    }
    if let (Some(summary), Some(marks)) = (&args.output, &args.marks_output)
        && resolved(summary)? == resolved(marks)?
    {
        return Err("summary and marks outputs must differ".into());
    }
    let precision = if args.precision == "f64" {
        Precision::F64
    } else {
        Precision::F32
    };
    let mut provenance = None;
    let supplied_rows: Option<Vec<OptionRow>> = if let Some(path) = &args.volt_input {
        let request: VoltValuationRequest = serde_json::from_reader(File::open(path)?)?;
        let batch = request.project()?;
        provenance = Some(serde_json::json!({
            "portfolio_hash": batch.portfolio_hash, "market_inputs_hash": batch.market_inputs_hash,
            "snapshot_id": batch.snapshot_id, "as_of": batch.as_of,
            "currency": batch.currency, "year_basis": batch.year_basis,
        }));
        Some(batch.rows)
    } else {
        args.input
            .as_ref()
            .map(|path| {
                Ok::<_, Box<dyn std::error::Error>>(serde_json::from_reader(File::open(path)?)?)
            })
            .transpose()?
    };
    if !args.chunked {
        let report = benchmark::run_with_rows(
            BenchmarkConfig {
                target: args.device,
                precision,
                rows: supplied_rows.as_ref().map_or(args.rows, Vec::len),
                repeats: args.repeats,
            },
            supplied_rows,
        )?;
        emit(&report, args.output.as_deref())?;
        if !report.passed {
            return Err("numerical validation failed; inspect report".into());
        }
        return Ok(());
    }
    let rows = supplied_rows.unwrap_or_else(|| benchmark::fixture(args.rows));
    let mib = |n: u64| {
        n.checked_mul(1024 * 1024)
            .ok_or("memory budget overflows byte count")
    };
    let budget = MemoryBudget {
        compute_bytes: mib(args.compute_budget_mib)?,
        host_working_bytes: mib(args.host_working_mib)?,
        compute_reserve_bytes: mib(args.compute_reserve_mib)?,
        headroom_percent: args.headroom_percent,
        max_batch_rows: args.max_batch_rows,
    };
    let numeric_input_hash = input_hash(&rows)?;
    let lockfile_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../Cargo.lock"
        )))
    );
    if args.plan_only {
        let plan = plan_rows(&rows, args.device, precision, &budget)?;
        emit(
            &serde_json::json!({
                "schema": "volt.compute.admission_report.v1", "target": args.device,
                "precision": precision, "budget": budget, "plan": plan,
                "numeric_input_hash": numeric_input_hash, "volt_provenance": provenance,
                "lockfile_sha256": lockfile_sha256,
            }),
            args.output.as_deref(),
        )?;
        return Ok(());
    }
    let started = Instant::now();
    let result = value_rows(&rows, args.device, precision, &budget)?;
    let execution_seconds = started.elapsed().as_secs_f64();
    let validation = validate_result(&rows, &result, precision)?;
    if validation.passed
        && let Some(path) = &args.marks_output
    {
        prepare_parent(path)?;
        let mut writer = BufWriter::new(File::create(path)?);
        for (row, price) in rows.iter().zip(&result.prices) {
            serde_json::to_writer(
                &mut writer,
                &serde_json::json!({
                    "schema": "volt.compute.option_mark.v1", "numeric_input_hash": numeric_input_hash,
                    "row_id": row.row_id, "contract_id": row.contract_id, "price": price,
                    "quantity": row.quantity, "multiplier": row.multiplier,
                }),
            )?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
    }
    emit(
        &serde_json::json!({
            "schema": "volt.compute.chunked_valuation.v1", "target": args.device,
            "precision": precision, "budget": budget, "plan": result.plan, "stats": result.stats,
            "numeric_input_hash": numeric_input_hash, "volt_provenance": provenance,
            "lockfile_sha256": lockfile_sha256, "execution_seconds": execution_seconds,
            "option_position_value": result.portfolio_value,
            "device_reduced_value_diagnostic": result.device_reduced_value,
            "sample_prices": result.prices.iter().take(6).collect::<Vec<_>>(),
            "validation": validation, "marks_output": args.marks_output,
        }),
        args.output.as_deref(),
    )?;
    if !validation.passed {
        return Err("chunked numerical validation failed; inspect report".into());
    }
    Ok(())
}

fn emit(value: &impl Serialize, output: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = output {
        prepare_parent(path)?;
        let mut writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(&mut writer, value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, value)?;
    writeln!(stdout)?;
    Ok(())
}

fn prepare_parent(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn resolved(path: &Path) -> std::io::Result<PathBuf> {
    path.canonicalize().or_else(|_| std::path::absolute(path))
}
