use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
use volt_compute::{
    ExecutionTarget, Precision, Session,
    hedge_benchmark::{BenchmarkConfig, benchmark_checkpoint},
    hedge_policy::HedgePolicy,
    hedge_training::{HedgeTrainingConfig, TrainingCheckpoint},
    model_hedge::{BaselineCacheConfig, BaselineCacheMode, HestonBaselineMethod, ModelHedgeConfig},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
fn command(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_model_band"))
        .args(args)
        .output()
        .unwrap()
}
fn text(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn tune_freeze_evaluate_has_same_cohort_and_model_as_checkpoint_benchmark() {
    let root = std::env::temp_dir().join(format!(
        "volt-band-cli-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    let temp = Temporary(root);
    let file = |s: &str| temp.0.join(s);
    let mut c = HedgeTrainingConfig::black_scholes(ExecutionTarget::Cpu);
    c.precision = Precision::F64;
    c.market.steps = 2;
    c.policy.time_steps = 2;
    c.batch_paths = 4;
    c.market.paths = 4;
    c.validation_paths = 16;
    c.calibration_paths = 8;
    c.test_paths = 16;
    c.evaluation_batch_paths = 16;
    c.bootstrap_paths = 16;
    c.bootstrap_replicates = 20;
    c.max_updates = 2;
    c.validation_interval = 1;
    c.policy.max_batch_rows = 64;
    c.premium_for_comparison = volt_compute::reference_price(&volt_compute::OptionRow {
        row_id: "band-control".into(),
        contract_id: "synthetic".into(),
        side: volt_compute::OptionSide::Call,
        spot: 100.0,
        strike: 100.0,
        maturity_years: c.market.horizon(),
        volatility: 0.2,
        rate: 0.0,
        dividend_yield: 0.0,
        quantity: 1,
        multiplier: 1,
    })
    .unwrap();
    let config = file("config.json");
    let selection = file("selection.json");
    let cache = file("cache");
    let tune = file("tune.json");
    let report = file("evaluation.json");
    let pnl = file("band.f64");
    std::fs::write(&config, serde_json::to_vec(&c).unwrap()).unwrap();
    let out = command(&[
        "--config",
        text(&config),
        "--selection",
        text(&selection),
        "--tune",
        "--paths",
        "16",
        "--seed",
        "730101",
        "--baseline-cache-dir",
        text(&cache),
        "--output",
        text(&tune),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let frozen = std::fs::read(&selection).unwrap();
    let out = command(&[
        "--config",
        text(&config),
        "--selection",
        text(&selection),
        "--paths",
        "32",
        "--seed",
        "730103",
        "--batch-paths",
        "16",
        "--baseline-cache-dir",
        text(&cache),
        "--pnl-output",
        text(&pnl),
        "--output",
        text(&report),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(frozen, std::fs::read(&selection).unwrap());
    let band = json(&report);
    assert_eq!(band["audits_exact_reconciled"], true);
    assert_eq!(band["audits_quantization_within_bounds"], true);
    assert_eq!(band["audits"].as_array().unwrap().len(), 8);
    let payload = std::fs::read(&pnl).unwrap();
    assert_eq!(payload.len(), 32 * 8);
    assert_eq!(
        band["terminal_pnl_artifact"]["sha256"],
        format!("{:x}", Sha256::digest(&payload))
    );
    assert!(
        payload
            .as_chunks::<8>()
            .0
            .iter()
            .all(|p| f64::from_le_bytes(*p).is_finite())
    );

    let s = Session::new(ExecutionTarget::Cpu, Precision::F64, 64).unwrap();
    let snapshot = HedgePolicy::new(&s, c.policy.clone(), 23)
        .unwrap()
        .snapshot()
        .unwrap();
    drop(s);
    let checkpoint = TrainingCheckpoint {
        schema: "volt.compute.hedge_training.v1".into(),
        config: c.clone(),
        completed_updates: 0,
        current_policy: snapshot.clone(),
        current_eta: None,
        selected_policy: snapshot,
        selected_update: 0,
        selected_validation_risk: 0.0,
    };
    let benchmark = benchmark_checkpoint(
        &checkpoint,
        BenchmarkConfig {
            target: ExecutionTarget::Cpu,
            paths: 32,
            seed: 730103,
            batch_paths: 16,
            model: ModelHedgeConfig {
                heston_method: HestonBaselineMethod::AdaptiveScalar,
                adaptive_threads: 8,
                budget: c.budget.clone(),
                ..Default::default()
            },
            bootstrap_paths: 16,
            bootstrap_replicates: 20,
            audit_paths: 2,
            output_samples: None,
            baseline_cache: Some(BaselineCacheConfig {
                directory: cache.clone(),
                mode: BaselineCacheMode::ReadWrite,
                max_entry_bytes: 64 * 1024 * 1024,
            }),
            terminal_pnl_output: None,
            simulation_substeps: 1,
        },
    )
    .unwrap();
    assert_eq!(band["path_stream_hash"], benchmark.path_stream_hash);
    assert_eq!(band["model"]["value"], benchmark.model.value);

    let old = std::fs::read(&report).unwrap();
    let out = command(&[
        "--config",
        text(&config),
        "--selection",
        text(&selection),
        "--paths",
        "32",
        "--seed",
        "730101",
        "--output",
        text(&report),
    ]);
    assert!(!out.status.success());
    assert_eq!(old, std::fs::read(&report).unwrap());
    c.market.model = volt_compute::market_paths::MarketModel::BlackScholes { volatility: 0.21 };
    std::fs::write(&config, serde_json::to_vec(&c).unwrap()).unwrap();
    let out = command(&[
        "--config",
        text(&config),
        "--selection",
        text(&selection),
        "--paths",
        "32",
        "--seed",
        "730103",
        "--output",
        text(&report),
    ]);
    assert!(!out.status.success());
    assert_eq!(old, std::fs::read(&report).unwrap());
}
