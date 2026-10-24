use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
use volt_compute::{
    ComputeError, ExecutionTarget, OptionRow, OptionSide, Precision, Session,
    hedge_application::HedgePathBatch,
    hedge_benchmark::{BenchmarkConfig, benchmark_checkpoint, plan_benchmark},
    hedge_policy::HedgePolicy,
    hedge_training::{
        HedgeTrainingConfig, TrainingCheckpoint, generate_markets, load_checkpoint, train,
    },
    market_paths::MarketModel,
    model_hedge::{
        BaselineExecutionPolicy, ModelHedgeConfig, model_hedge_holdings, scalar_hedge_pnl,
    },
    reference_price,
    risk_objective::weighted_risk,
};

fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "volt-benchmark-test-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn training(systems: usize) -> HedgeTrainingConfig {
    let mut c = HedgeTrainingConfig::black_scholes(target());
    c.precision = if target() == ExecutionTarget::Metal {
        Precision::F32
    } else {
        Precision::F64
    };
    c.systems = systems;
    c.market.steps = 2;
    c.market.paths = 8;
    c.policy.time_steps = 2;
    c.policy.feature_count = systems;
    c.policy.hedge_count = systems;
    c.policy.hidden_width = 2;
    c.policy.max_batch_rows = 16;
    c.batch_paths = 8;
    c.evaluation_batch_paths = 8;
    c.validation_paths = 8;
    c.test_paths = 8;
    c.bootstrap_paths = 8;
    c.bootstrap_replicates = 10;
    c.max_updates = 2;
    c.validation_interval = 1;
    c.calibration_paths = 0;
    c.premium_for_comparison = systems as f64
        * reference_price(&OptionRow {
            row_id: "premium".into(),
            contract_id: "synthetic".into(),
            side: OptionSide::Call,
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
    c
}
fn constant_checkpoint(holding: f64, systems: usize) -> TrainingCheckpoint {
    let c = training(systems);
    // Snapshot storage is portable at its exact dtype. Construct on CPU, then
    // benchmark the requested physical device without invoking training.
    let session = Session::new(ExecutionTarget::Cpu, c.precision, 16).unwrap();
    let mut snapshot = HedgePolicy::new(&session, c.policy.clone(), 23)
        .unwrap()
        .snapshot()
        .unwrap();
    for (i, p) in snapshot.parameters.iter_mut().enumerate() {
        p.values.fill(if i % 10 == 9 { holding } else { 0.0 });
    }
    TrainingCheckpoint {
        schema: "volt.compute.hedge_training.v1".into(),
        config: c,
        completed_updates: 0,
        current_policy: snapshot.clone(),
        selected_policy: snapshot,
        current_eta: None,
        selected_update: 0,
        selected_validation_risk: 0.0,
    }
}
fn config() -> BenchmarkConfig {
    let model = ModelHedgeConfig {
        max_queries_per_job: 8,
        max_cached_states: 32,
        ..Default::default()
    };
    BenchmarkConfig {
        target: target(),
        paths: 8,
        seed: 2003,
        batch_paths: 3,
        model,
        bootstrap_paths: 8,
        bootstrap_replicates: 10,
        audit_paths: 8,
        output_samples: None,
        baseline_cache: None,
        terminal_pnl_output: None,
        simulation_substeps: 1,
    }
}

#[test]
fn frozen_zero_policy_matches_nohedge_and_genuine_bs_model_on_same_paths_with_eight_audits() {
    let checkpoint = constant_checkpoint(0.0, 1);
    let before = serde_json::to_value(&checkpoint).unwrap();
    let temp = Temp::new();
    let mut config = config();
    config.output_samples = Some(temp.file("samples.jsonl"));
    let report = benchmark_checkpoint(&checkpoint, config.clone()).unwrap();
    assert_eq!(report.candidate, report.no_hedge);
    assert_eq!(
        report
            .paired_candidate_vs_no_hedge
            .risk_difference_candidate_minus_baseline,
        0.0
    );
    assert_eq!(
        report
            .paired_candidate_vs_no_hedge
            .risk_difference_interval
            .lower,
        0.0
    );
    assert_eq!(
        report
            .paired_candidate_vs_no_hedge
            .risk_difference_interval
            .upper,
        0.0
    );
    assert_eq!(report.candidate_metrics.mean_cost, 0.0);
    assert_eq!(report.candidate_metrics.mean_turnover, 0.0);
    assert_eq!(report.audits.len(), 8);
    assert!(report.audits_exact_reconciled && report.audits_quantization_within_bounds);
    assert!(report.inference_pure && report.identical_paths_for_all_strategies);
    assert_eq!(report.completed_batches, 3);
    assert_eq!(report.last_path_id, 7);
    assert!(matches!(
        report.model_execution_policy,
        BaselineExecutionPolicy::AnalyticBlackScholesCpuF64
    ));
    assert_eq!(before, serde_json::to_value(&checkpoint).unwrap());
    let markets = generate_markets(&checkpoint.config, config.seed, 0, 8).unwrap();
    let batch = HedgePathBatch::from_markets(&markets, checkpoint.config.liability).unwrap();
    let baseline =
        model_hedge_holdings(&markets, checkpoint.config.liability, &config.model).unwrap();
    let expected = scalar_hedge_pnl(
        &batch,
        &baseline.holdings,
        checkpoint.config.cost,
        checkpoint.config.premium_for_comparison,
    )
    .unwrap();
    let expected = weighted_risk(&expected.pnl, None, checkpoint.config.objective).unwrap();
    assert!((report.model.value - expected.value).abs() < 1e-12);
    let artifact = report.sample_artifact.unwrap();
    let bytes = std::fs::read(&artifact.path).unwrap();
    assert_eq!(artifact.sha256, format!("{:x}", Sha256::digest(&bytes)));
    assert_eq!(artifact.bytes, bytes.len() as u64);
    assert_eq!(artifact.rows, 8);
    for (i, line) in std::str::from_utf8(&bytes).unwrap().lines().enumerate() {
        let row: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(row["path_id"], i as u64);
        assert_eq!(row["candidate_holdings"], serde_json::json!([0.0, 0.0]));
        assert_eq!(row["path_marks_hash"].as_str().unwrap().len(), 64);
    }
}

#[test]
fn poor_candidate_is_reported_honestly_and_batch_partition_does_not_change_cohort_identity() {
    let checkpoint = constant_checkpoint(10.0, 1);
    let mut config = config();
    let chunked = benchmark_checkpoint(&checkpoint, config.clone()).unwrap();
    config.batch_paths = 8;
    let full = benchmark_checkpoint(&checkpoint, config).unwrap();
    assert!(
        full.candidate.value > full.no_hedge.value,
        "bad constant hedge must not be described as trained improvement"
    );
    assert_eq!(full.path_stream_hash, chunked.path_stream_hash);
    assert!((full.candidate.value - chunked.candidate.value).abs() < 1e-10);
    assert!((full.model.value - chunked.model.value).abs() < 1e-12);
    assert_eq!(full.selected_policy_sha256, chunked.selected_policy_sha256);
    assert!(full.inference_pure && chunked.inference_pure);
    assert!(full.candidate_device_accounting_gap.max_absolute_difference < 0.001);
}

#[test]
fn preflight_rejects_reserved_streams_counts_snapshot_errors_and_global_oom_without_device() {
    let mut checkpoint = constant_checkpoint(0.5, 1);
    checkpoint.config.calibration_paths = 8;
    let mut config = config();
    config.target = ExecutionTarget::Cuda { index: 999 };
    for seed in [
        checkpoint.config.market.seed,
        checkpoint.config.validation_seed,
        checkpoint.config.calibration_seed,
    ] {
        config.seed = seed;
        assert!(matches!(
            benchmark_checkpoint(&checkpoint, config.clone()),
            Err(ComputeError::InvalidInput(_))
        ));
    }
    config.seed = 2003;
    assert!(plan_benchmark(&checkpoint, &config).is_ok());
    config.paths = 0;
    assert!(plan_benchmark(&checkpoint, &config).is_err());
    config.paths = 8;
    config.bootstrap_paths = 9;
    assert!(plan_benchmark(&checkpoint, &config).is_err());
    config.bootstrap_paths = 8;
    config.audit_paths = 9;
    assert!(plan_benchmark(&checkpoint, &config).is_err());
    config.audit_paths = 8;
    config.model.max_queries_per_job = 0;
    assert!(plan_benchmark(&checkpoint, &config).is_err());
    config.model.max_queries_per_job = 8;
    let mut invalid = checkpoint.clone();
    invalid.selected_policy.parameters[0].values[0] = f64::NAN;
    assert!(matches!(
        benchmark_checkpoint(&invalid, config.clone()),
        Err(ComputeError::InvalidInput(_))
    ));
    let mut small = checkpoint.clone();
    small.config.budget.host_working_bytes = 96 * 1024 * 1024;
    config.paths = 1_000_000;
    assert!(matches!(
        benchmark_checkpoint(&small, config),
        Err(ComputeError::BudgetExceeded(_))
    ));
    let multi = constant_checkpoint(0.5, 2);
    let mut config = self::config();
    config.target = ExecutionTarget::Cuda { index: 999 };
    config.seed = multi.config.market.seed.wrapping_add(0x9e3779b97f4a7c15);
    assert!(matches!(
        benchmark_checkpoint(&multi, config),
        Err(ComputeError::InvalidInput(_))
    ));
}

#[test]
fn failed_genuine_model_preserves_existing_samples_and_removes_own_temporary_artifact() {
    let mut checkpoint = constant_checkpoint(0.0, 1);
    checkpoint.config.market.model = MarketModel::BlackScholes { volatility: 0.0 };
    let temp = Temp::new();
    let path = temp.file("input-to-preserve.jsonl");
    std::fs::write(&path, b"sentinel").unwrap();
    let mut config = config();
    config.output_samples = Some(path.clone());
    assert!(matches!(
        benchmark_checkpoint(&checkpoint, config),
        Err(ComputeError::Unsupported(_))
    ));
    assert_eq!(std::fs::read(&path).unwrap(), b"sentinel");
    assert_eq!(std::fs::read_dir(&temp.0).unwrap().count(), 1);
}

#[test]
fn real_two_update_checkpoint_has_pure_heldout_inference_model_comparison_and_integer_audits() {
    let temp = Temp::new();
    let path = temp.file("checkpoint.json");
    let mut training = training(1);
    training.calibration_paths = 8;
    train(training, None, Some(&path)).unwrap();
    let checkpoint = load_checkpoint(&path).unwrap();
    let frozen = serde_json::to_value(&checkpoint.selected_policy).unwrap();
    let report = benchmark_checkpoint(&checkpoint, config()).unwrap();
    assert_eq!(report.selected_update, checkpoint.selected_update);
    assert_eq!(report.candidate.observations, 8);
    assert_eq!(report.model.observations, 8);
    assert!(report.candidate.value.is_finite() && report.model.value.is_finite());
    assert!(report.audits_exact_reconciled && report.inference_pure);
    assert_eq!(
        frozen,
        serde_json::to_value(&checkpoint.selected_policy).unwrap()
    );
}

#[test]
fn matched_benchmark_cache_reuses_model_for_a_different_frozen_policy() {
    use volt_compute::model_hedge::{BaselineCacheConfig, BaselineCacheMode};
    let temp = Temp::new();
    let mut c = config();
    c.baseline_cache = Some(BaselineCacheConfig {
        directory: temp.file("cache"),
        mode: BaselineCacheMode::ReadWrite,
        max_entry_bytes: 64 * 1024 * 1024,
    });
    let first = benchmark_checkpoint(&constant_checkpoint(0.0, 1), &c).unwrap();
    assert_eq!(
        first.baseline_cache.entries_written,
        first.completed_batches
    );
    assert_eq!(first.baseline_cache.hits, 0);
    c.baseline_cache.as_mut().unwrap().mode = BaselineCacheMode::ReadOnly;
    let second = benchmark_checkpoint(&constant_checkpoint(0.5, 1), &c).unwrap();
    assert_eq!(second.baseline_cache.hits, second.completed_batches);
    assert_eq!(second.baseline_cache.entries_written, 0);
    assert_eq!(second.baseline_cache.actual_function_evaluations, 0);
    assert_eq!(first.path_stream_hash, second.path_stream_hash);
    assert_eq!(
        first.baseline_cache.manifest_sha256,
        second.baseline_cache.manifest_sha256
    );
    assert_eq!(first.model.value, second.model.value);
    assert_eq!(
        first.model_metrics.mean_turnover,
        second.model_metrics.mean_turnover
    );
    assert_ne!(first.selected_policy_sha256, second.selected_policy_sha256);
    assert!(second.audits_exact_reconciled && second.audits_quantization_within_bounds);
}

#[test]
fn compact_terminal_pnl_roundtrip_matches_optional_jsonl_and_checks_all_identities() {
    use volt_compute::hedge_benchmark::verify_terminal_pnl_artifact;
    let checkpoint = constant_checkpoint(0.5, 1);
    let temp = Temp::new();
    let mut c = config();
    c.output_samples = Some(temp.file("trace.jsonl"));
    c.terminal_pnl_output = Some(temp.file("terminal.pnl"));
    let report = benchmark_checkpoint(&checkpoint, &c).unwrap();
    let artifact = report.terminal_pnl_artifact.as_ref().unwrap();
    let verified = verify_terminal_pnl_artifact(&artifact.path).unwrap();
    assert_eq!(artifact.sha256, verified.artifact.sha256);
    assert_eq!(artifact.bytes, verified.artifact.bytes);
    assert_eq!(verified.path_stream_sha256, report.path_stream_hash);
    assert_eq!(verified.market_stream_sha256, report.market_stream_hash);
    assert_eq!(
        verified.header.selected_policy_sha256,
        report.selected_policy_sha256
    );
    assert_eq!(
        verified.header.market_cohort_spec_sha256,
        report.plan.market_cohort_spec_sha256
    );
    assert_eq!(
        verified.header.model_numeric_policy_sha256,
        report.plan.model_numeric_policy_sha256
    );
    let raw = std::fs::read(&artifact.path).unwrap();
    assert_eq!(
        raw.len() as u64,
        verified.data_offset + 64 * c.paths as u64 + 104
    );
    let trace = std::fs::read_to_string(c.output_samples.as_ref().unwrap()).unwrap();
    let keys = [
        "candidate_f64_pnl",
        "model_f64_pnl",
        "no_hedge_pnl",
        "candidate_device_pnl",
        "candidate_cost",
        "model_cost",
        "candidate_turnover",
        "model_turnover",
    ];
    for (path, line) in trace.lines().enumerate() {
        let value: serde_json::Value = serde_json::from_str(line).unwrap();
        for (column, key) in keys.into_iter().enumerate() {
            let offset = verified.data_offset as usize + path * 64 + column * 8;
            let number = f64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap());
            assert_eq!(number.to_bits(), value[key].as_f64().unwrap().to_bits());
        }
    }
    let mut broken = raw.clone();
    broken[verified.data_offset as usize + 3] ^= 1;
    std::fs::write(&artifact.path, broken).unwrap();
    assert!(verify_terminal_pnl_artifact(&artifact.path).is_err());
    std::fs::write(&artifact.path, &raw[..raw.len() - 1]).unwrap();
    assert!(verify_terminal_pnl_artifact(&artifact.path).is_err());
}

#[test]
fn binary_only_failure_preserves_existing_output_and_removes_uncommitted_files() {
    use volt_compute::model_hedge::{BaselineCacheConfig, BaselineCacheMode};
    let temp = Temp::new();
    let output = temp.file("terminal.pnl");
    std::fs::write(&output, b"existing-to-preserve").unwrap();
    let mut c = config();
    c.terminal_pnl_output = Some(output.clone());
    c.baseline_cache = Some(BaselineCacheConfig {
        directory: temp.file("missing-cache"),
        mode: BaselineCacheMode::ReadOnly,
        max_entry_bytes: 64 * 1024 * 1024,
    });
    assert!(benchmark_checkpoint(&constant_checkpoint(0.0, 1), &c).is_err());
    assert_eq!(std::fs::read(output).unwrap(), b"existing-to-preserve");
    assert!(std::fs::read_dir(&temp.0).unwrap().all(|p| {
        !p.unwrap()
            .file_name()
            .to_string_lossy()
            .contains("pnl-partial")
    }));
    c.baseline_cache = None;
    let report = benchmark_checkpoint(&constant_checkpoint(0.0, 1), &c).unwrap();
    assert!(report.sample_artifact.is_none());
    assert!(report.terminal_pnl_artifact.is_some());
    c.output_samples = c.terminal_pnl_output.clone();
    assert!(plan_benchmark(&constant_checkpoint(0.0, 1), &c).is_err());
}

#[test]
fn market_only_hash_is_partition_invariant_and_independent_of_liability_and_policy() {
    use volt_compute::hedge_benchmark::MarketCohortHasher;
    let checkpoint = constant_checkpoint(0.0, 1);
    let c = config();
    let first = benchmark_checkpoint(&checkpoint, &c).unwrap();
    let mut nil = checkpoint.clone();
    nil.config.liability = volt_compute::hedge_application::HedgeLiability::None;
    nil.config.premium_for_comparison = 0.0;
    let nil = benchmark_checkpoint(&nil, &c).unwrap();
    assert_eq!(first.market_stream_hash, nil.market_stream_hash);
    assert_ne!(first.path_stream_hash, nil.path_stream_hash);
    let mut hash = MarketCohortHasher::new(&checkpoint.config, c.seed, c.paths, 1).unwrap();
    for i in 0..c.paths {
        let markets = generate_markets(&checkpoint.config, c.seed, i as u64, 1).unwrap();
        let batch = HedgePathBatch::from_markets(
            &markets,
            volt_compute::hedge_application::HedgeLiability::None,
        )
        .unwrap();
        hash.update_batch(i as u64, &batch).unwrap();
    }
    assert_eq!(hash.finish().unwrap(), first.market_stream_hash);
}

#[test]
fn refined_simulation_is_charged_and_keeps_policy_grid_fixed() {
    let checkpoint = constant_checkpoint(0.0, 1);
    let mut c = config();
    let coarse = plan_benchmark(&checkpoint, &c).unwrap();
    c.simulation_substeps = 4;
    let fine = plan_benchmark(&checkpoint, &c).unwrap();
    assert!(fine.resolution_working_bytes > coarse.resolution_working_bytes);
    assert!(fine.retained_market_and_rollout_bytes > coarse.retained_market_and_rollout_bytes);
    let report = benchmark_checkpoint(&checkpoint, &c).unwrap();
    assert_eq!(report.paths, c.paths);
    assert_ne!(
        report.market_stream_hash,
        benchmark_checkpoint(&checkpoint, config())
            .unwrap()
            .market_stream_hash
    );
    c.simulation_substeps = usize::MAX;
    assert!(plan_benchmark(&checkpoint, &c).is_err());
}
