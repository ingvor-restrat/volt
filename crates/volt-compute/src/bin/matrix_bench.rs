use clap::Parser;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};
use volt_compute::{
    ExecutionTarget, Precision,
    matrix::MatrixBudget,
    matrix_workload::{self, MatrixWorkloadConfig},
};
#[derive(Parser)]
#[command(about = "Validate and time budgeted matrix exchange and multiplication")]
struct Args {
    #[arg(long, default_value = "cpu")]
    device: ExecutionTarget,
    #[arg(long, default_value = "f32", value_parser = ["f32", "f64"])]
    precision: String,
    #[arg(long, default_value_t = 257)]
    m: usize,
    #[arg(long, default_value_t = 193)]
    k: usize,
    #[arg(long, default_value_t = 129)]
    n: usize,
    #[arg(long, default_value_t = 10)]
    repeats: usize,
    #[arg(long, default_value_t = 256)]
    compute_budget_mib: u64,
    #[arg(long, default_value_t = 256)]
    host_working_mib: u64,
    #[arg(long)]
    output: Option<PathBuf>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mib = |n: u64| {
        n.checked_mul(1024 * 1024)
            .ok_or("budget byte count overflow")
    };
    let report = matrix_workload::run(MatrixWorkloadConfig {
        target: args.device,
        precision: if args.precision == "f64" {
            Precision::F64
        } else {
            Precision::F32
        },
        m: args.m,
        k: args.k,
        n: args.n,
        repeats: args.repeats,
        budget: MatrixBudget {
            compute_bytes: mib(args.compute_budget_mib)?,
            host_working_bytes: mib(args.host_working_mib)?,
            ..MatrixBudget::default()
        },
    })?;
    if let Some(path) = args.output {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let mut writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(&mut writer, &report)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    if !report.passed {
        return Err("matrix validation failed; inspect report".into());
    }
    Ok(())
}
