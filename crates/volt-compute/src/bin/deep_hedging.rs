use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use volt_compute::{
    ExecutionTarget, Precision, Session,
    hedge_policy::HedgePolicy,
    hedge_training::{self, HedgeTrainingConfig, TrainingCheckpoint},
    risk_objective::RiskObjective,
};
#[derive(Parser)]
#[command(
    about = "Train, checkpoint, resume and independently evaluate a time-specific Deep Hedging policy"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Validate a checkpoint and expose native serialization fingerprints, without a device.
    Inspect {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Train {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long,default_value="heston",value_parser=["heston","bs"])]
        profile: String,
        #[arg(long)]
        device: Option<ExecutionTarget>,
        #[arg(long)]
        updates: Option<u64>,
        #[arg(long)]
        batch_paths: Option<usize>,
        #[arg(long)]
        validation_paths: Option<usize>,
        #[arg(long)]
        test_paths: Option<usize>,
        #[arg(long)]
        validation_interval: Option<u64>,
        #[arg(long)]
        eval_batch_paths: Option<usize>,
        #[arg(long)]
        alpha: Option<f64>,
        #[arg(long)]
        entropic: Option<f64>,
        #[arg(long)]
        squared: bool,
        #[arg(long)]
        cost: Option<f64>,
        #[arg(long)]
        seed: Option<u64>,
        /// Explicit conditioning deviation: scale Heston variance actions by1000.
        #[arg(long)]
        scaled_actions: bool,
        #[arg(long)]
        no_liability: bool,
        #[arg(long)]
        plan_only: bool,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Benchmark {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        device: Option<ExecutionTarget>,
        #[arg(long, default_value_t = 4096)]
        paths: usize,
        #[arg(long, default_value_t = 900001)]
        seed: u64,
        #[arg(long, default_value_t = 256)]
        batch_paths: usize,
        #[arg(long, default_value_t = 4)]
        baseline_threads: usize,
        #[arg(long, default_value_t = 1)]
        simulation_substeps: usize,
        #[arg(long, default_value_t = 8)]
        audit_paths: usize,
        #[arg(long)]
        samples_output: Option<PathBuf>,
        #[arg(long)]
        pnl_output: Option<PathBuf>,
        #[arg(long)]
        baseline_cache_dir: Option<PathBuf>,
        #[arg(long, requires = "baseline_cache_dir")]
        baseline_cache_read_only: bool,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        plan_only: bool,
    },

    Evaluate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        device: Option<ExecutionTarget>,
        #[arg(long)]
        paths: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.command {
        Command::Inspect { checkpoint, output } => {
            if let Some(path) = &output
                && resolved(path)? == resolved(&checkpoint)?
            {
                return Err("inspection output cannot overwrite checkpoint".into());
            }
            let bytes = std::fs::read(&checkpoint)?;
            let saved = hedge_training::load_checkpoint(&checkpoint)?;
            hedge_training::validate_resume(&saved.config, &saved)?;
            let report = serde_json::json!({
                "schema":"volt.compute.hedge_checkpoint_inspection.v1",
                "checkpoint_file_sha256":format!("{:x}",Sha256::digest(&bytes)),
                "checkpoint_object_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&saved)?)),
                "selected_policy_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&saved.selected_policy)?)),
                "completed_updates":saved.completed_updates,"selected_update":saved.selected_update,
                "selected_validation_risk":saved.selected_validation_risk,"config":saved.config,
                "validated_without_device":true,"build":metadata()
            });
            if let Some(path) = output {
                write_json(&path, &report)?;
            }
            serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
        }
        Command::Train {
            config,
            resume,
            profile,
            device,
            updates,
            batch_paths,
            validation_paths,
            test_paths,
            validation_interval,
            eval_batch_paths,
            alpha,
            entropic,
            squared,
            cost,
            seed,
            scaled_actions,
            no_liability,
            plan_only,
            checkpoint,
            output,
        } => {
            for out in [&checkpoint, &output].into_iter().flatten() {
                for input in [&config, &resume].into_iter().flatten() {
                    if resolved(out)? == resolved(input)? {
                        return Err(
                            "training output cannot overwrite config or resume input".into()
                        );
                    }
                }
            }
            if let (Some(a), Some(b)) = (&checkpoint, &output)
                && resolved(a)? == resolved(b)?
            {
                return Err("checkpoint and report must differ".into());
            }
            let resumed: Option<TrainingCheckpoint> = resume
                .as_ref()
                .map(|p| hedge_training::load_checkpoint(p))
                .transpose()?;
            let mut c = if let Some(path) = config {
                serde_json::from_reader(File::open(path)?)?
            } else if let Some(r) = &resumed {
                r.config.clone()
            } else if profile == "bs" {
                HedgeTrainingConfig::black_scholes(ExecutionTarget::Cpu)
            } else {
                HedgeTrainingConfig::paper_heston(ExecutionTarget::Cpu)
            };
            if let Some(target) = device {
                c.target = target;
            }
            if let Some(v) = updates {
                c.max_updates = v;
            }
            if let Some(v) = batch_paths {
                c.batch_paths = v;
            }
            if let Some(v) = validation_paths {
                c.validation_paths = v;
            }
            if let Some(v) = test_paths {
                c.test_paths = v;
            }
            if let Some(v) = validation_interval {
                c.validation_interval = v;
            }
            if let Some(v) = eval_batch_paths {
                c.evaluation_batch_paths = v;
            }
            if let Some(v) = alpha {
                c.objective = RiskObjective::Cvar { alpha: v };
            }
            if let Some(v) = entropic {
                c.objective = RiskObjective::Entropic { risk_aversion: v };
            }
            if squared {
                c.objective = RiskObjective::SquaredLoss;
            }
            if let Some(v) = cost {
                c.cost.proportional = v;
            }
            if let Some(v) = seed {
                c.policy_seed = v;
                c.market.seed = v.wrapping_add(11);
            }
            if scaled_actions {
                c.policy.action_scales = Some(
                    (0..c.policy.hedge_count)
                        .map(|i| if i % 2 == 1 { 1000.0 } else { 1.0 })
                        .collect(),
                );
            }
            if no_liability {
                c.liability = volt_compute::hedge_application::HedgeLiability::None;
                c.premium_for_comparison = 0.0;
            }
            if c.precision == Precision::F32 && c.policy.hidden_width < 2 {
                return Err("invalid hidden width".into());
            }
            let plans = c.validate()?;
            if let Some(r) = &resumed {
                hedge_training::validate_resume(&c, r)?;
            }
            if plan_only {
                let value = serde_json::json!({"schema":"volt.compute.deep_hedging_admission.v1","config":c,"training_plan":plans.0,"evaluation_plan":plans.1});
                if let Some(path) = output {
                    write_json(&path, &value)?;
                }
                serde_json::to_writer_pretty(std::io::stdout().lock(), &value)?;
                return Ok(());
            }
            eprintln!(
                "training {:?}:{}updates,{}paths/batch,{}validation,{}held-out",
                c.target, c.max_updates, c.batch_paths, c.validation_paths, c.test_paths
            );
            let report = hedge_training::train(c, resumed, checkpoint.as_deref())?;
            let value = serde_json::json!({"run":report,"build":metadata()});
            if let Some(path) = output {
                write_json(&path, &value)?;
            }
            serde_json::to_writer_pretty(std::io::stdout().lock(), &value)?;
        }
        Command::Benchmark {
            checkpoint,
            device,
            paths,
            seed,
            batch_paths,
            baseline_threads,
            simulation_substeps,
            audit_paths,
            samples_output,
            pnl_output,
            baseline_cache_dir,
            baseline_cache_read_only,
            output,
            plan_only,
        } => {
            for out in [&samples_output, &pnl_output, &output]
                .into_iter()
                .flatten()
            {
                if resolved(out)? == resolved(&checkpoint)? {
                    return Err("benchmark cannot replace checkpoint".into());
                }
            }
            let outputs: Vec<_> = [&samples_output, &pnl_output, &output]
                .into_iter()
                .flatten()
                .collect();
            for (i, a) in outputs.iter().enumerate() {
                for b in outputs.iter().skip(i + 1) {
                    if resolved(a)? == resolved(b)? {
                        return Err("sample, PnL and report paths must differ".into());
                    }
                }
            }
            let saved = hedge_training::load_checkpoint(&checkpoint)?;
            let model = volt_compute::model_hedge::ModelHedgeConfig {
                heston_method: volt_compute::model_hedge::HestonBaselineMethod::AdaptiveScalar,
                adaptive_threads: baseline_threads,
                budget: saved.config.budget.clone(),
                ..Default::default()
            };
            let config = volt_compute::hedge_benchmark::BenchmarkConfig {
                target: device.unwrap_or(saved.config.target),
                paths,
                seed,
                batch_paths,
                model,
                simulation_substeps,
                bootstrap_paths: paths.min(saved.config.bootstrap_paths),
                bootstrap_replicates: saved.config.bootstrap_replicates,
                audit_paths,
                output_samples: samples_output,
                terminal_pnl_output: pnl_output,
                baseline_cache: baseline_cache_dir.map(|directory| {
                    volt_compute::model_hedge::BaselineCacheConfig {
                        directory,
                        mode: if baseline_cache_read_only {
                            volt_compute::model_hedge::BaselineCacheMode::ReadOnly
                        } else {
                            volt_compute::model_hedge::BaselineCacheMode::ReadWrite
                        },
                        max_entry_bytes: 64 * 1024 * 1024,
                    }
                }),
            };
            if plan_only {
                let plan = volt_compute::hedge_benchmark::plan_benchmark(&saved, &config)?;
                let value = serde_json::json!({"schema":"volt.compute.hedge_benchmark_admission.v1","plan":plan,"config":config});
                if let Some(path) = output {
                    write_json(&path, &value)?;
                }
                serde_json::to_writer_pretty(std::io::stdout().lock(), &value)?;
            } else {
                let report = volt_compute::hedge_benchmark::benchmark_checkpoint(&saved, &config)?;
                let value = serde_json::json!({"benchmark":report,"build":metadata()});
                if let Some(path) = output {
                    write_json(&path, &value)?;
                }
                serde_json::to_writer_pretty(std::io::stdout().lock(), &value)?;
            }
        }
        Command::Evaluate {
            checkpoint,
            device,
            paths,
            seed,
            output,
        } => {
            if let Some(out) = &output
                && resolved(out)? == resolved(&checkpoint)?
            {
                return Err("evaluation output cannot replace checkpoint".into());
            }
            let ckp = hedge_training::load_checkpoint(&checkpoint)?;
            let mut c = ckp.config.clone();
            if let Some(v) = device {
                c.target = v;
            }
            if let Some(v) = paths {
                c.test_paths = v;
            }
            if let Some(v) = seed {
                c.test_seed = v;
            }
            c.validate()?;
            // Evaluation deliberately changes the held-out stream/count only;
            // architecture and selected snapshot remain validated and fixed.
            volt_compute::hedge_policy::validate_policy_snapshot(&ckp.selected_policy)?;
            if ckp.selected_policy.updates != ckp.selected_update {
                return Err("selected checkpoint update mismatch".into());
            }
            let session = Session::new(c.target, c.precision, c.policy.max_batch_rows)?;
            let mut policy = HedgePolicy::from_snapshot(&session, ckp.selected_policy)?;
            let evaluation = hedge_training::evaluate(&session, &c, &mut policy)?;
            let value = serde_json::json!({"schema":"volt.compute.deep_hedging_evaluation.v1","selected_update":ckp.selected_update,"config":c,"evaluation":evaluation,"build":metadata()});
            if let Some(path) = output {
                write_json(&path, &value)?;
            }
            serde_json::to_writer_pretty(std::io::stdout().lock(), &value)?;
        }
    }
    Ok(())
}
fn metadata() -> serde_json::Value {
    let exe = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .map(|v| format!("{:x}", Sha256::digest(v)));
    serde_json::json!({"executable_sha256":exe,"lockfile_sha256":format!("{:x}",Sha256::digest(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock")))),
        "os":std::env::consts::OS,"architecture":std::env::consts::ARCH,"fusion_feature_enabled":cfg!(feature="fusion"),
        "profile":if cfg!(debug_assertions){"debug"}else{"release"},"initializer":"SplitMix64 Xavier-uniform;bias0;BNgamma1beta0",
        "path_backend":"hostf64exactCIRendpoints;exogenouspathsuploadedtoexplicitpolicydevice"})
}
fn resolved(path: &Path) -> std::io::Result<PathBuf> {
    path.canonicalize().or_else(|_| std::path::absolute(path))
}
fn write_json(path: &Path, value: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(p) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
