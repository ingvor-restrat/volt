//! Validation-tuned model-target no-trade bands, independent of neural fitting.
use clap::Parser;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use volt_compute::{
    ExecutionTarget, Precision,
    hedge_application::HedgePathBatch,
    hedge_band::{NoTradeBandSelection, banded_holdings, select_no_trade_band},
    hedge_benchmark::MarketCohortHasher,
    hedge_reconciliation::{AuditModelFamily, reconcile_path},
    hedge_training::{HedgeTrainingConfig, generate_markets_with_substeps},
    market_paths::MarketModel,
    model_hedge::{
        BaselineCacheConfig, BaselineCacheMode, HestonBaselineMethod, ModelHedgeConfig,
        baseline_numeric_policy_hash, model_hedge_holdings_cached, no_hedge_pnl, scalar_hedge_pnl,
    },
    risk_objective::weighted_risk,
};

#[derive(Parser)]
#[command(about = "Tune model-target bands on validation paths, or evaluate a frozen band")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    /// Write a frozen selection with --tune; otherwise read and validate it.
    #[arg(long)]
    selection: PathBuf,
    #[arg(long)]
    tune: bool,
    #[arg(long)]
    paths: usize,
    #[arg(long)]
    seed: u64,
    #[arg(long, default_value_t = 256)]
    batch_paths: usize,
    #[arg(long, default_value_t = 8)]
    baseline_threads: usize,
    #[arg(long, default_value_t = 1)]
    simulation_substeps: usize,
    #[arg(long)]
    baseline_cache_dir: Option<PathBuf>,
    #[arg(long)]
    pnl_output: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}

fn hash<T: Serialize>(v: &T) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(v)?)))
}
fn context(c: &HedgeTrainingConfig) -> Result<String, Box<dyn std::error::Error>> {
    hash(&(
        c.market.model,
        c.market.steps,
        c.market.dt,
        c.market.initial_spot,
        c.market.rate,
        c.market.dividend_yield,
        c.market.integrated_variance_rule,
        c.systems,
        c.liability,
        c.objective,
        c.cost,
        c.premium_for_comparison,
    ))
}
fn resolved(p: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if p.exists() {
        return Ok(p.canonicalize()?);
    }
    let parent = p
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    Ok(parent
        .canonicalize()?
        .join(p.file_name().ok_or("output needs filename")?))
}
fn atomic_write(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<(), Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = resolved(path)?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let temporary = path.with_extension(format!("{}.{}.tmp", std::process::id(), stamp));
    let result = (|| {
        let mut writer = BufWriter::with_capacity(
            1024 * 1024,
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)?,
        );
        write(&mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(&temporary, &path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}
fn json_write(path: &Path, value: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
    atomic_write(path, |out| {
        serde_json::to_writer_pretty(out, value)?;
        Ok(())
    })
}
fn add_path(hash: &mut Sha256, id: usize, batch: &HedgePathBatch, row: usize) {
    hash.update((id as u64).to_le_bytes());
    let m = (batch.steps + 1) * batch.hedge_count;
    let f = batch.steps * batch.feature_count;
    for x in &batch.marks[row * m..(row + 1) * m] {
        hash.update(x.to_le_bytes());
    }
    for x in &batch.features[row * f..(row + 1) * f] {
        hash.update(x.to_le_bytes());
    }
    hash.update(batch.payoffs[row].to_le_bytes());
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.paths < 2
        || args.paths > 10_000_000
        || args.batch_paths == 0
        || (args.tune && (args.paths > 32768 || args.pnl_output.is_some()))
    {
        return Err("invalid path counts; tuning cap32768 and no tuning PnL export".into());
    }
    let inputs = if args.tune {
        vec![&args.config]
    } else {
        vec![&args.config, &args.selection]
    };
    let mut outputs = vec![&args.output];
    if args.tune {
        outputs.push(&args.selection);
    }
    if let Some(p) = &args.pnl_output {
        outputs.push(p);
    }
    for (i, a) in outputs.iter().enumerate() {
        for b in &inputs {
            if resolved(a)? == resolved(b)? {
                return Err("output cannot replace input".into());
            }
        }
        for b in outputs.iter().skip(i + 1) {
            if resolved(a)? == resolved(b)? {
                return Err("output paths must differ".into());
            }
        }
    }
    let training: HedgeTrainingConfig = serde_json::from_reader(File::open(&args.config)?)?;
    training.validate()?;
    let mut coarse = training.market.clone();
    coarse.paths = if args.tune {
        args.paths
    } else {
        args.paths.min(args.batch_paths)
    };
    let resolution = volt_compute::market_path_resolution::MarketPathResolutionConfig {
        market: coarse,
        substeps: args.simulation_substeps,
    }
    .validate()?;
    let admission = resolution.working_bytes as u128 * training.systems as u128
        + args.paths as u128 * 256
        + args.paths.min(if args.tune {
            args.paths
        } else {
            args.batch_paths
        }) as u128
            * (training.market.steps as u128 + 1)
            * training.policy.hedge_count as u128
            * 32;
    let host_limit = training.budget.host_working_bytes as u128
        * (100 - training.budget.headroom_percent) as u128
        / 100;
    if admission > host_limit {
        return Err("band resolution/output working set exceeds host allowance".into());
    }
    let stride = 0x9e3779b97f4a7c15_u64;
    let mut occupied = std::collections::BTreeSet::new();
    let mut reserved = vec![
        training.market.seed,
        training.calibration_seed,
        training.test_seed,
    ];
    if !args.tune {
        reserved.push(training.validation_seed);
    }
    for seed in reserved {
        for system in 0..training.systems {
            occupied.insert(seed.wrapping_add((system as u64).wrapping_mul(stride)));
        }
    }
    for system in 0..training.systems {
        if occupied.contains(&args.seed.wrapping_add((system as u64).wrapping_mul(stride))) {
            return Err("band cohort must be separate from training/calibration/validation/diagnostic streams".into());
        }
    }
    let is_heston = matches!(training.market.model, MarketModel::Heston { .. });
    let model = ModelHedgeConfig {
        target: ExecutionTarget::Cpu,
        precision: Precision::F64,
        heston_method: HestonBaselineMethod::AdaptiveScalar,
        adaptive_threads: args.baseline_threads,
        budget: training.budget.clone(),
        ..ModelHedgeConfig::default()
    };
    model.validate()?;
    let cache = args
        .baseline_cache_dir
        .map(|directory| BaselineCacheConfig {
            directory,
            mode: BaselineCacheMode::ReadWrite,
            max_entry_bytes: 64 * 1024 * 1024,
        });
    let fingerprint = context(&training)?;
    let numerical = baseline_numeric_policy_hash(&model, is_heston)?;
    let started = Instant::now();
    if args.tune {
        let markets = generate_markets_with_substeps(
            &training,
            args.seed,
            0,
            args.paths,
            args.simulation_substeps,
        )?;
        let batch = HedgePathBatch::from_markets(&markets, training.liability)?;
        let (targets, access) =
            model_hedge_holdings_cached(&markets, training.liability, &model, cache.as_ref())?;
        let units: Vec<_> = (0..batch.hedge_count)
            .map(|i| if is_heston && i % 2 == 1 { 1000.0 } else { 1.0 })
            .collect();
        let factors = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 1e6];
        let grid: Vec<Vec<_>> = factors
            .iter()
            .map(|f| units.iter().map(|u| f * u).collect())
            .collect();
        let selection = select_no_trade_band(
            &batch,
            &targets.holdings,
            training.cost,
            training.premium_for_comparison,
            training.objective,
            &grid,
            training.budget.host_working_bytes / 4,
        )?;
        let envelope = serde_json::json!({
            "schema":"volt.compute.model_band_selection.v1","context_sha256":fingerprint,
            "numeric_model_sha256":numerical,"validation_seed":args.seed,
            "validation_paths":args.paths,"simulation_substeps":args.simulation_substeps,
            "grid_factors":factors,"inventory_units":units,
            "selection":selection,"cache":access,"seconds":started.elapsed().as_secs_f64()
        });
        json_write(&args.selection, &envelope)?;
        json_write(&args.output, &envelope)?;
        return Ok(());
    }
    let envelope: serde_json::Value = serde_json::from_reader(File::open(&args.selection)?)?;
    if envelope["schema"] != "volt.compute.model_band_selection.v1"
        || envelope["context_sha256"] != fingerprint
        || envelope["numeric_model_sha256"] != numerical
        || envelope["validation_seed"].as_u64() == Some(args.seed)
    {
        return Err("band selection context/numerics mismatch or reused validation cohort".into());
    }
    let selection: NoTradeBandSelection = serde_json::from_value(envelope["selection"].clone())?;
    if selection.widths.len() != training.policy.hedge_count {
        return Err("band inventory dimensions mismatch".into());
    }
    let allowance = training.budget.host_working_bytes / 4;
    if (args.paths as u128) * 256 > allowance as u128 {
        return Err("band retained PnL/risk results exceed host allowance".into());
    }
    let mut pnl = Vec::with_capacity(args.paths);
    let mut baseline_pnl = Vec::with_capacity(args.paths);
    let mut unhedged_pnl = Vec::with_capacity(args.paths);
    let mut audits = Vec::new();
    let family = if is_heston {
        AuditModelFamily::Heston
    } else {
        AuditModelFamily::BlackScholes
    };
    let mut cache_hits = 0;
    let mut function_evaluations = 0_u64;
    let mut mean_cost = 0.0;
    let mut mean_turnover = 0.0;
    let mut market_cohort =
        MarketCohortHasher::new(&training, args.seed, args.paths, args.simulation_substeps)?;
    let market_cohort_spec_sha256 = market_cohort.spec_sha256().to_string();
    let mut cohort = Sha256::new();
    cohort.update(b"volt.benchmark.shared_paths.v1\0");
    cohort.update(
        hash(&(
            training.market.model,
            training.market.steps,
            training.market.dt,
            training.market.initial_spot,
            training.market.rate,
            training.market.dividend_yield,
            training.market.integrated_variance_rule,
            training.systems,
            training.liability,
            args.seed,
            args.paths,
            volt_compute::market_paths::RNG_CONTRACT,
        ))?
        .as_bytes(),
    );
    if args.simulation_substeps != 1 {
        cohort.update(b"volt.market_resolution.substeps.v1\0");
        cohort.update((args.simulation_substeps as u64).to_le_bytes());
    }
    for first in (0..args.paths).step_by(args.batch_paths) {
        let count = args.batch_paths.min(args.paths - first);
        let markets = generate_markets_with_substeps(
            &training,
            args.seed,
            first as u64,
            count,
            args.simulation_substeps,
        )?;
        let batch = HedgePathBatch::from_markets(&markets, training.liability)?;
        market_cohort.update_batch(first as u64, &batch)?;
        let (targets, access) =
            model_hedge_holdings_cached(&markets, training.liability, &model, cache.as_ref())?;
        cache_hits += usize::from(access.hit);
        function_evaluations += access.actual_function_evaluations;
        let positions = banded_holdings(
            &targets.holdings,
            count,
            batch.steps,
            &selection.widths,
            allowance,
        )?;
        let band = scalar_hedge_pnl(
            &batch,
            &positions,
            training.cost,
            training.premium_for_comparison,
        )?;
        let baseline = scalar_hedge_pnl(
            &batch,
            &targets.holdings,
            training.cost,
            training.premium_for_comparison,
        )?;
        for row in 0..count {
            add_path(&mut cohort, first + row, &batch, row);
            if first + row < 8 {
                let stride = batch.steps * batch.hedge_count;
                let audit = reconcile_path(
                    &batch,
                    row,
                    &positions[row * stride..(row + 1) * stride],
                    family,
                    training.market.dt,
                    training.premium_for_comparison,
                    training.cost,
                )?;
                if !audit.exact_reconciliation
                    || !audit.quantization_within_bound
                    || !audit.float_identity_within_bound
                    || (audit.unquantized_float_pnl - band.pnl[row]).abs()
                        > 1e-10 * (1.0 + band.pnl[row].abs())
                {
                    return Err("band path failed independent ledger audit".into());
                }
                audits.push(audit);
            }
        }
        mean_cost += band
            .costs
            .iter()
            .map(|x| x / args.paths as f64)
            .sum::<f64>();
        mean_turnover += band
            .turnover
            .iter()
            .map(|x| x / args.paths as f64)
            .sum::<f64>();
        pnl.extend(band.pnl);
        baseline_pnl.extend(baseline.pnl);
        unhedged_pnl.extend(no_hedge_pnl(&batch, training.premium_for_comparison)?.pnl);
    }
    let risk = weighted_risk(&pnl, None, training.objective)?;
    let model_risk = weighted_risk(&baseline_pnl, None, training.objective)?;
    let no_hedge_risk = weighted_risk(&unhedged_pnl, None, training.objective)?;
    let artifact = if let Some(path) = &args.pnl_output {
        let mut digest = Sha256::new();
        atomic_write(path, |out| {
            for value in &pnl {
                let b = value.to_le_bytes();
                digest.update(b);
                out.write_all(&b)?;
            }
            Ok(())
        })?;
        Some(
            serde_json::json!({"path":path,"rows":pnl.len(),"bytes":pnl.len()*8,
            "dtype":"f64_le","columns":["band_pnl"],"sha256":format!("{:x}",digest.finalize())}),
        )
    } else {
        None
    };
    let report = serde_json::json!({
        "schema":"volt.compute.model_band_evaluation.v1","context_sha256":fingerprint,
        "numeric_model_sha256":numerical,"selection_sha256":hash(&envelope)?,
        "selection":selection,"paths":args.paths,"seed":args.seed,
        "simulation_substeps":args.simulation_substeps,
        "path_stream_hash":format!("{:x}",cohort.finalize()),"candidate":risk,
        "market_cohort_spec_sha256":market_cohort_spec_sha256,
        "market_stream_hash":market_cohort.finish()?,
        "model":model_risk,"no_hedge":no_hedge_risk,"mean_cost":mean_cost,
        "mean_turnover":mean_turnover,"baseline_cache_hits":cache_hits,
        "actual_function_evaluations":function_evaluations,"terminal_pnl_artifact":artifact,
        "audits":audits,"audits_exact_reconciled":true,"audits_quantization_within_bounds":true,
        "execution":"native_scalar_cpu_f64","seconds":started.elapsed().as_secs_f64()
    });
    json_write(&args.output, &report)?;
    Ok(())
}
