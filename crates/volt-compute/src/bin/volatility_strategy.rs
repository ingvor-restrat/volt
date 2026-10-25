//! JSON-request European straddle control. No private experiment inputs embedded.
use clap::Parser;
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use volt_compute::volatility_strategy::{StraddleStudyConfig, run_straddle_study};
#[derive(Parser)]
#[command(
    about = "Study European straddle delta hedging with distinct physical and quote volatility"
)]
struct Args {
    #[arg(
        long,
        required_unless_present = "example_config",
        conflicts_with = "example_config"
    )]
    config: Option<PathBuf>,
    /// Print a generic complete JSON request; does not execute an experiment.
    #[arg(long)]
    example_config: bool,
    #[arg(long)]
    output: Option<PathBuf>,
    /// Optional streamed per-path/per-strategy JSONL, atomically published.
    #[arg(long, conflicts_with = "example_config")]
    samples_output: Option<PathBuf>,
    #[arg(long, conflicts_with = "example_config")]
    plan_only: bool,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.example_config {
        let value = serde_json::to_value(StraddleStudyConfig::default())?;
        if let Some(path) = args.output {
            write_json(&path, &value)?;
        }
        serde_json::to_writer_pretty(std::io::stdout().lock(), &value)?;
        return Ok(());
    }
    let input = args.config.unwrap();
    let config: StraddleStudyConfig = serde_json::from_reader(File::open(&input)?)?;
    let plan = config.validate()?;
    let absolute = |p: &Path| -> Result<PathBuf, std::io::Error> {
        if p.exists() {
            p.canonicalize()
        } else {
            let parent = p
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            Ok(parent.canonicalize()?.join(
                p.file_name()
                    .ok_or_else(|| std::io::Error::other("output requires filename"))?,
            ))
        }
    };
    let mut outputs = Vec::new();
    for path in [&args.output, &args.samples_output].into_iter().flatten() {
        let resolved = absolute(path)?;
        if resolved == input.canonicalize()? || outputs.contains(&resolved) || path.exists() {
            return Err("outputs must be distinct new files and cannot replace the request".into());
        }
        outputs.push(resolved);
    }
    let value = if args.plan_only {
        serde_json::json!({"schema":"volt.compute.straddle_admission.v1","config":config,"plan":plan})
    } else {
        serde_json::to_value(run_straddle_study(&config, args.samples_output.as_deref())?)?
    };
    if let Some(path) = args.output {
        write_json(&path, &value)?;
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &value)?;
    Ok(())
}
fn write_json(
    path: &Path,
    value: &impl serde::Serialize,
) -> Result<(), Box<dyn std::error::Error>> {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    let temp = path.with_extension(format!(
        "partial-{}-{}",
        std::process::id(),
        NONCE.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        std::fs::hard_link(&temp, path)?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&temp);
    result
}
