//! Multi-asset continuation learner, resident inference, and reproducible timing.
use clap::{Parser, Subcommand};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    io::{self, Read, Write},
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};
use volt_compute::{
    ExecutionTarget, Precision,
    maxcall_learning::*,
    multi_asset_market::{PolicyTailConfig, evaluate_policy_tail, generate_paths},
};
#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    Example {
        #[arg(long)]
        output: PathBuf,
    },
    Train {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        report: PathBuf,
        #[arg(long)]
        diagnostic_checkpoint: Option<PathBuf>,
        #[arg(long)]
        progress_dir: Option<PathBuf>,
    },
    Evaluate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value = "cpu")]
        target: String,
        #[arg(long, default_value = "f32")]
        precision: String,
        #[arg(long, default_value_t = 8192)]
        batch_rows: usize,
        #[arg(long)]
        output: PathBuf,
    },
    Predict {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        states: PathBuf,
        #[arg(long)]
        date: usize,
        #[arg(long, default_value = "cpu")]
        target: String,
        #[arg(long, default_value = "f32")]
        precision: String,
        #[arg(long, default_value_t = 8192)]
        batch_rows: usize,
        #[arg(long)]
        output: PathBuf,
    },
    Bench {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        #[arg(long, default_value_t = 10)]
        updates: usize,
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f64,
        #[arg(long)]
        synchronized_stages: bool,
        #[arg(long)]
        output: PathBuf,
    },
}
fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err("input exceeds16MiB".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}
fn check_new(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if path.exists() {
        return Err(format!("refusing to overwrite {}", path.display()).into());
    }
    Ok(())
}
fn save(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    check_new(path)?;
    let temp = path.with_extension(format!("partial-{}", std::process::id()));
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp)?;
    let mut writer = io::BufWriter::new(file);
    let result = serde_json::to_writer_pretty(&mut writer, value)
        .map_err(io::Error::other)
        .and_then(|_| writer.write_all(b"\n"))
        .and_then(|_| writer.flush())
        .and_then(|_| writer.get_ref().sync_all());
    drop(writer);
    if let Err(e) = result {
        let _ = std::fs::remove_file(&temp);
        return Err(e.into());
    }
    let result = std::fs::hard_link(&temp, path);
    let _ = std::fs::remove_file(&temp);
    result?;
    Ok(())
}
fn precision(s: &str) -> Result<Precision, Box<dyn std::error::Error>> {
    match s {
        "f32" => Ok(Precision::F32),
        "f64" => Ok(Precision::F64),
        _ => Err("precision must be f32 or f64".into()),
    }
}
fn sha(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Action::Example { output } => save(
            &output,
            &serde_json::json!({"training":MaxCallTrainingConfig::default(),"evaluation":PolicyTailConfig{lower_paths:4_096_000,upper_paths:2048,inner_samples:2048,outer_batch_rows:8192,conditional_outer_batch_rows:4,max_working_bytes:1024*1024*1024,max_transition_asset_steps:2_000_000_000,..Default::default()}}),
        )?,
        Action::Train {
            config,
            resume,
            checkpoint,
            report,
            diagnostic_checkpoint,
            progress_dir,
        } => {
            for p in [&checkpoint, &report] {
                check_new(p)?;
            }
            if let Some(p) = &diagnostic_checkpoint {
                check_new(p)?;
            }
            let partial: Option<MaxCallPartialCheckpoint> =
                resume.as_deref().map(read).transpose()?;
            let config: MaxCallTrainingConfig = match config {
                Some(p) => read(&p)?,
                None => partial
                    .as_ref()
                    .ok_or("provide --config or --resume")?
                    .config
                    .clone(),
            };
            if let Some(dir) = &progress_dir {
                std::fs::create_dir_all(dir)?;
            }
            let result = train_with_progress(&config, partial.as_ref(), |progress, archive| {
                eprintln!(
                    "{}",
                    serde_json::to_string(progress)
                        .map_err(|e| volt_compute::ComputeError::Execution(e.to_string()))?
                );
                if let (Some(dir), Some(archive)) = (&progress_dir, archive) {
                    save(
                        &dir.join(format!("date-{:03}.json", progress.date)),
                        archive,
                    )
                    .map_err(|e| volt_compute::ComputeError::Execution(e.to_string()))?;
                }
                Ok(())
            })?;
            save(&checkpoint, &result.final_checkpoint)?;
            if let Some(p) = diagnostic_checkpoint {
                save(&p, &result.diagnostic_validation_checkpoint)?;
            }
            save(&report, &result)?;
        }
        Action::Evaluate {
            checkpoint,
            config,
            target,
            precision: prec,
            batch_rows,
            output,
        } => {
            check_new(&output)?;
            let start = Instant::now();
            let selected: MaxCallCheckpoint = read(&checkpoint)?;
            let mut evaluation: PolicyTailConfig = read(&config)?;
            let fingerprint = sha(&checkpoint)?;
            if evaluation.policy_sha256 != "0".repeat(64) && evaluation.policy_sha256 != fingerprint
            {
                return Err("evaluation policy SHA does not match checkpoint file".into());
            }
            evaluation.policy_sha256 = fingerprint.clone();
            let target = ExecutionTarget::from_str(&target)?;
            let prec = precision(&prec)?;
            let t = Instant::now();
            let mut predictor = MaxCallPredictor::new(
                &selected,
                target,
                prec,
                batch_rows,
                selected.config.compute_budget_bytes,
            )?;
            let setup = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let bounds = evaluate_policy_tail(selected.market(), &mut predictor, &evaluation)?;
            let elapsed = t.elapsed().as_secs_f64();
            #[derive(Serialize)]
            struct Envelope<'a> {
                #[serde(flatten)]
                bounds: &'a volt_compute::multi_asset_market::PolicyTailBounds,
                checkpoint_file_sha256: String,
                inference_target: ExecutionTarget,
                inference_precision: Precision,
                inference_batch_rows: usize,
                arithmetic: ProductArithmetic,
                predictor_setup_seconds: f64,
                evaluation_seconds: f64,
                end_to_end_seconds: f64,
            }
            save(
                &output,
                &Envelope {
                    bounds: &bounds,
                    checkpoint_file_sha256: fingerprint,
                    inference_target: target,
                    inference_precision: prec,
                    inference_batch_rows: batch_rows,
                    arithmetic: selected.config.arithmetic,
                    predictor_setup_seconds: setup,
                    evaluation_seconds: elapsed,
                    end_to_end_seconds: start.elapsed().as_secs_f64(),
                },
            )?;
        }
        Action::Predict {
            checkpoint,
            states,
            date,
            target,
            precision: prec,
            batch_rows,
            output,
        } => {
            check_new(&output)?;
            let selected: MaxCallCheckpoint = read(&checkpoint)?;
            let states: Vec<f64> = read(&states)?;
            let mut predictor = MaxCallPredictor::new(
                &selected,
                ExecutionTarget::from_str(&target)?,
                precision(&prec)?,
                batch_rows,
                selected.config.compute_budget_bytes,
            )?;
            save(&output, &predictor.predict_batch(date, &states)?)?;
        }
        Action::Bench {
            config,
            warmup,
            updates,
            learning_rate,
            synchronized_stages,
            output,
        } => {
            check_new(&output)?;
            if updates == 0 || updates > 10000 || warmup > 1000 {
                return Err("invalid benchmark counts".into());
            }
            let mut config: MaxCallTrainingConfig = read(&config)?;
            config.training_paths = config.batch_rows;
            config.validation_paths = config.batch_rows;
            config.validate()?;
            let started = Instant::now();
            let t = Instant::now();
            let market = generate_paths(
                &config.market,
                config.batch_rows,
                config.training_seed,
                0,
                777,
                config.host_budget_bytes,
            )?;
            let generation = t.elapsed().as_secs_f64();
            let d = config.market.spots.len();
            let n = config.market.times.len();
            let date = n - 2;
            let states = (0..market.paths)
                .flat_map(|p| {
                    market.values[(p * n + date) * d..(p * n + date + 1) * d]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            let targets = (0..market.paths)
                .map(|p| {
                    config
                        .market
                        .payoff(n - 1, &market.values[(p * n + n - 1) * d..(p * n + n) * d])
                })
                .collect::<Result<Vec<_>, _>>()?;
            let t = Instant::now();
            let mut learner = MaxCallDateLearner::new(&config, date, None)?;
            let setup = t.elapsed().as_secs_f64();
            let upload = learner.upload_training(&states, &targets)?;
            let rows = (0..config.batch_rows).collect::<Vec<_>>();
            let mode = if synchronized_stages {
                StepTimingMode::SynchronizedStages
            } else {
                StepTimingMode::EndToEnd
            };
            let mut warm = Vec::new();
            for _ in 0..warmup {
                warm.push(learner.update_rows(&rows, learning_rate, mode)?);
            }
            let mut measured = Vec::new();
            for _ in 0..updates {
                measured.push(learner.update_rows(&rows, learning_rate, mode)?);
            }
            let t = Instant::now();
            let inference = learner.predict_training()?;
            let inference_seconds = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let final_network = learner.snapshot()?;
            let readback = t.elapsed().as_secs_f64();
            save(
                &output,
                &serde_json::json!({"scope":"One-date resident financial regression microbenchmark; not full backward training or paper-price result","config":config,"market_sha256":market.content_sha256,"generation_seconds":generation,"setup_seconds":setup,"upload_seconds":upload,"warmup":warm,"measured":measured,"inference_forward_and_readback_seconds":inference_seconds,"snapshot_readback_seconds":readback,"final_network":final_network,"inference":inference,"end_to_end_seconds":started.elapsed().as_secs_f64(),"timing_note":"Synchronized phase clocks include their synchronization; synchronization_seconds is nested and nonadditive. EndToEnd uses one scalar readback/step. First-touch timing retains actual cache state."}),
            )?;
        }
    }
    Ok(())
}
