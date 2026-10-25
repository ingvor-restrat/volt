//! Full configuration/CLI vertical: liability -> paths -> training -> checkpoint
//! -> resume -> genuine model benchmark and integer-ledger reconciliation.
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
use volt_compute::{
    ExecutionTarget, OptionRow, OptionSide, Precision,
    hedge_application::HedgeLiability,
    hedge_benchmark::verify_terminal_pnl_artifact,
    hedge_training::{HedgeTrainingConfig, load_checkpoint, validate_resume},
    reference_price,
    risk_objective::RiskObjective,
};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "volt-european-liabilities-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn config(liability: HedgeLiability) -> HedgeTrainingConfig {
    let mut c = HedgeTrainingConfig::black_scholes(ExecutionTarget::Cpu);
    c.precision = Precision::F64;
    c.liability = liability;
    c.market.steps = 2;
    c.market.paths = 8;
    c.policy.time_steps = 2;
    c.policy.hidden_width = 2;
    c.policy.max_batch_rows = 8;
    c.policy.batch_norm.enabled = false;
    c.batch_paths = 8;
    c.evaluation_batch_paths = 8;
    c.validation_paths = 8;
    c.test_paths = 8;
    c.bootstrap_paths = 8;
    c.bootstrap_replicates = 10;
    c.calibration_paths = 0;
    c.max_updates = 1;
    c.validation_interval = 1;
    c.objective = RiskObjective::SquaredLoss;
    let put = reference_price(&OptionRow {
        row_id: "independent-put-premium".into(),
        contract_id: "synthetic-put".into(),
        side: OptionSide::Put,
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
    c.premium_for_comparison = if matches!(liability, HedgeLiability::Straddle { .. }) {
        2.0 * put
    } else {
        put
    };
    c
}
fn cli(args: &[&str], paths: &[&Path]) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_deep_hedging"));
    // Arguments contain a literal placeholder for each OS-native file path.
    let mut path = paths.iter();
    for arg in args {
        if *arg == "@" {
            cmd.arg(path.next().unwrap());
        } else {
            cmd.arg(arg);
        }
    }
    assert!(path.next().is_none());
    let output = cmd.output().unwrap();
    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
#[test]
fn put_and_straddle_cli_train_resume_benchmark_and_readonly_cache_roundtrip() {
    for (name, liability) in [
        ("put", HedgeLiability::Put { strike: 100.0 }),
        ("straddle", HedgeLiability::Straddle { strike: 100.0 }),
    ] {
        let temp = Temp::new();
        let c = config(liability);
        c.validate().unwrap();
        let mut serialized = serde_json::to_value(&c).unwrap();
        assert_eq!(
            serialized["liability"],
            serde_json::json!({"kind":name,"strike":100.0})
        );
        serialized["liability"]["strike"] = serde_json::json!(0.0);
        assert!(
            serde_json::from_value::<HedgeTrainingConfig>(serialized)
                .unwrap()
                .validate()
                .is_err()
        );
        let input = temp.path("config.json");
        let checkpoint = temp.path("first.json");
        let resumed = temp.path("resumed.json");
        std::fs::write(&input, serde_json::to_vec(&c).unwrap()).unwrap();
        cli(
            &["train", "--config", "@", "--checkpoint", "@"],
            &[&input, &checkpoint],
        );
        cli(
            &[
                "train",
                "--resume",
                "@",
                "--updates",
                "2",
                "--checkpoint",
                "@",
            ],
            &[&checkpoint, &resumed],
        );
        let saved = load_checkpoint(&resumed).unwrap();
        assert_eq!(saved.completed_updates, 2);
        assert_eq!(
            serde_json::to_value(saved.config.liability).unwrap(),
            serde_json::to_value(liability).unwrap()
        );
        let mut different = saved.config.clone();
        different.liability = HedgeLiability::Call { strike: 100.0 };
        assert!(validate_resume(&different, &saved).is_err());
        let cache = temp.path("cache");
        let report = temp.path("cold.json");
        let warm = temp.path("warm.json");
        let pnl = temp.path("cold.pnl");
        let warm_pnl = temp.path("warm.pnl");
        cli(
            &[
                "benchmark",
                "--checkpoint",
                "@",
                "--paths",
                "8",
                "--seed",
                "730707",
                "--batch-paths",
                "4",
                "--baseline-threads",
                "1",
                "--audit-paths",
                "8",
                "--baseline-cache-dir",
                "@",
                "--pnl-output",
                "@",
                "--output",
                "@",
            ],
            &[&resumed, &cache, &pnl, &report],
        );
        cli(
            &[
                "benchmark",
                "--checkpoint",
                "@",
                "--paths",
                "8",
                "--seed",
                "730707",
                "--batch-paths",
                "4",
                "--baseline-threads",
                "1",
                "--audit-paths",
                "8",
                "--baseline-cache-dir",
                "@",
                "--baseline-cache-read-only",
                "--pnl-output",
                "@",
                "--output",
                "@",
            ],
            &[&resumed, &cache, &warm_pnl, &warm],
        );
        let cold: serde_json::Value =
            serde_json::from_slice(&std::fs::read(report).unwrap()).unwrap();
        let warm: serde_json::Value =
            serde_json::from_slice(&std::fs::read(warm).unwrap()).unwrap();
        assert_eq!(cold["benchmark"]["audits_exact_reconciled"], true);
        assert_eq!(cold["benchmark"]["audits_quantization_within_bounds"], true);
        assert_eq!(warm["benchmark"]["baseline_cache"]["hits"], 2);
        assert_eq!(warm["benchmark"]["baseline_cache"]["misses"], 0);
        assert_eq!(cold["benchmark"]["model"], warm["benchmark"]["model"]);
        assert_eq!(
            std::fs::read(&pnl).unwrap(),
            std::fs::read(&warm_pnl).unwrap()
        );
        assert_eq!(verify_terminal_pnl_artifact(&pnl).unwrap().artifact.rows, 8);
    }
}
