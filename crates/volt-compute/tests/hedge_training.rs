use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use sha2::{Digest, Sha256};
use volt_compute::{
    ComputeError, ExecutionTarget, OptionRow, OptionSide, Precision, Session,
    hedge_policy::HedgePolicy,
    hedge_training::{
        EtaSnapshot, HedgeTrainingConfig, TrainingCheckpoint, evaluate, generate_batch,
        load_checkpoint, train, validate_resume,
    },
    reference_price,
    risk_objective::RiskObjective,
};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
struct TempDirectory(PathBuf);
impl TempDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "volt-hedge-training-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}
fn tiny() -> HedgeTrainingConfig {
    let mut config = HedgeTrainingConfig::black_scholes(target());
    config.precision = if target() == ExecutionTarget::Metal {
        Precision::F32
    } else {
        Precision::F64
    };
    config.market.steps = 2;
    config.policy.time_steps = 2;
    config.policy.max_batch_rows = 16;
    config.market.paths = 8;
    config.batch_paths = 8;
    config.evaluation_batch_paths = 16;
    config.validation_paths = 16;
    config.calibration_paths = 8;
    config.test_paths = 32;
    config.bootstrap_paths = 32;
    config.bootstrap_replicates = 20;
    config.max_updates = 2;
    config.validation_interval = 1;
    config.premium_for_comparison = reference_price(&OptionRow {
        row_id: "two-day-premium".into(),
        contract_id: "synthetic-call".into(),
        side: OptionSide::Call,
        spot: 100.0,
        strike: 100.0,
        maturity_years: config.market.horizon(),
        volatility: 0.2,
        rate: 0.0,
        dividend_yield: 0.0,
        quantity: 1,
        multiplier: 1,
    })
    .unwrap();
    config
}
fn json<T: serde::Serialize>(value: &T) -> serde_json::Value {
    serde_json::to_value(value).unwrap()
}
fn checkpoint(config: HedgeTrainingConfig) -> TrainingCheckpoint {
    let temp = TempDirectory::new();
    let file = temp.file("checkpoint.json");
    train(config, None, Some(&file)).unwrap();
    load_checkpoint(&file).unwrap()
}

#[test]
fn real_training_checkpoint_inference_and_exact_resume_include_adam_and_batchnorm() {
    let temp = TempDirectory::new();
    let full_path = temp.file("full.json");
    let split_path = temp.file("split.json");
    let mut full_config = tiny();
    full_config.max_updates = 4;
    let full = train(full_config.clone(), None, Some(&full_path)).unwrap();
    let first = train(tiny(), None, Some(&split_path)).unwrap();
    let partial = load_checkpoint(&split_path).unwrap();
    assert_eq!(partial.completed_updates, 2);
    assert_eq!(partial.current_policy.updates, 2);
    assert!(
        partial
            .current_policy
            .parameters
            .iter()
            .any(|p| p.first_moment.iter().any(|x| *x != 0.0))
    );
    assert!(
        partial
            .current_policy
            .parameters
            .iter()
            .any(|p| p.second_moment.iter().any(|x| *x > 0.0))
    );
    assert!(
        partial
            .current_policy
            .batch_norm
            .iter()
            .all(|n| n.batches_tracked == 2)
    );
    assert!(
        partial
            .current_policy
            .batch_norm
            .iter()
            .any(|n| n.running_mean.iter().any(|x| *x != 0.0))
    );
    assert!(first.observations.iter().any(|o| o.gradient_norm > 0.0));
    assert!(
        first
            .observations
            .iter()
            .all(|o| o.training_loss.is_finite() && o.validation_risk.is_finite())
    );
    let resumed = train(full_config.clone(), Some(partial), Some(&split_path)).unwrap();
    let continuous = load_checkpoint(&full_path).unwrap();
    let restarted = load_checkpoint(&split_path).unwrap();
    // Compare complete state, not just reported loss: normalization statistics,
    // optimizer moments/counters, current/selected policies and selected score.
    assert_eq!(json(&continuous), json(&restarted));
    assert_eq!(full.checkpoint_sha256, resumed.checkpoint_sha256);
    assert_eq!(full.selected_update, resumed.selected_update);
    assert_eq!(
        full.evaluation.sample_candidate_pnl,
        resumed.evaluation.sample_candidate_pnl
    );
    assert_eq!(
        full.evaluation.path_stream_hash,
        resumed.evaluation.path_stream_hash
    );
    assert_eq!(
        full.observations[2].training_loss,
        resumed.observations[0].training_loss
    );
    assert_eq!(
        full.observations[3].training_loss,
        resumed.observations[1].training_loss
    );
    assert!(!full.paper_replication_complete);

    let session = Session::new(
        full_config.target,
        full_config.precision,
        full_config.policy.max_batch_rows,
    )
    .unwrap();
    let mut selected =
        HedgePolicy::from_snapshot(&session, restarted.selected_policy.clone()).unwrap();
    let before = json(&selected.snapshot().unwrap());
    let inference = evaluate(&session, &full_config, &mut selected).unwrap();
    assert_eq!(
        before,
        json(&selected.snapshot().unwrap()),
        "held-out inference must not update BatchNorm or optimizer state"
    );
    assert_eq!(
        inference.sample_candidate_pnl,
        full.evaluation.sample_candidate_pnl
    );
    assert_eq!(inference.paths, 32);
    assert_eq!(inference.bootstrap_subset_paths, 32);
    assert_eq!(
        full.checkpoint_sha256,
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&restarted.selected_policy).unwrap())
        )
    );
    let initial = HedgePolicy::new(
        &session,
        full_config.policy.clone(),
        full_config.policy_seed,
    )
    .unwrap()
    .snapshot()
    .unwrap();
    assert_ne!(
        json(&initial.parameters),
        json(&restarted.current_policy.parameters),
        "optimizer must change actual policy parameters"
    );
}

#[test]
fn cvar_eta_is_optimized_and_malformed_eta_is_rejected_before_device_selection() {
    let mut config = tiny();
    config.objective = RiskObjective::Cvar { alpha: 0.5 };
    let good = checkpoint(config.clone());
    let eta = good.current_eta.as_ref().unwrap();
    assert_eq!(eta.step, 2);
    assert!(eta.value.is_finite());
    assert!(eta.second_moment > 0.0);
    let mut complete_config = config.clone();
    complete_config.max_updates = 4;
    let continuous = checkpoint(complete_config.clone());
    let temporary = TempDirectory::new();
    let resumed_path = temporary.file("resumed-cvar.json");
    train(complete_config, Some(good.clone()), Some(&resumed_path)).unwrap();
    let resumed = load_checkpoint(&resumed_path).unwrap();
    assert_eq!(
        json(&continuous),
        json(&resumed),
        "joint eta Adam, policy Adam and BatchNorm must all resume exactly"
    );
    let mut future = config.clone();
    future.max_updates = 4;
    future.target = ExecutionTarget::Cuda { index: 999 };
    for bad_eta in [
        None,
        Some(EtaSnapshot {
            step: 1,
            ..eta.clone()
        }),
        Some(EtaSnapshot {
            second_moment: -1.0,
            ..eta.clone()
        }),
        Some(EtaSnapshot {
            value: f64::NAN,
            ..eta.clone()
        }),
    ] {
        let mut bad = good.clone();
        bad.current_eta = bad_eta;
        let error = train(future.clone(), Some(bad), None).unwrap_err();
        assert!(
            matches!(error, ComputeError::InvalidInput(_)),
            "expected checkpoint preflight, got {error}"
        );
    }
}

#[test]
fn malformed_current_selected_config_and_precision_fail_before_unavailable_cuda() {
    let config = tiny();
    let good = checkpoint(config.clone());
    let mut future = config;
    future.max_updates = 4;
    future.target = ExecutionTarget::Cuda { index: 999 };
    let mut cases = Vec::new();
    let mut bad = good.clone();
    bad.selected_policy.config.include_previous_holdings = false;
    cases.push(bad);
    let mut bad = good.clone();
    bad.current_policy.config.hidden_width += 1;
    cases.push(bad);
    let other = if future.precision == Precision::F64 {
        Precision::F32
    } else {
        Precision::F64
    };
    let mut bad = good.clone();
    bad.current_policy.precision = other;
    cases.push(bad);
    let mut bad = good.clone();
    bad.selected_policy.precision = other;
    cases.push(bad);
    let mut bad = good.clone();
    bad.selected_policy.schema = "wrong-schema".into();
    cases.push(bad);
    let mut bad = good.clone();
    bad.current_policy.parameters[0].values.pop();
    cases.push(bad);
    let mut bad = good.clone();
    bad.current_eta = Some(EtaSnapshot {
        value: 0.0,
        first_moment: 0.0,
        second_moment: 0.0,
        step: 2,
    });
    cases.push(bad);
    let mut bad = good.clone();
    bad.selected_update = bad.completed_updates + 1;
    cases.push(bad);
    let mut bad = good.clone();
    bad.selected_validation_risk = f64::NAN;
    cases.push(bad);
    for bad in cases {
        assert!(validate_resume(&future, &bad).is_err());
        let error = train(future.clone(), Some(bad), None).unwrap_err();
        assert!(
            matches!(error, ComputeError::InvalidInput(_)),
            "expected checkpoint validation before CUDA999, got {error}"
        );
    }
    assert!(
        validate_resume(&future, &good).is_ok(),
        "target-only changes are explicitly allowed"
    );
}

#[test]
fn derived_system_stream_collisions_and_path_counter_overflow_are_preflight_errors() {
    const SYSTEM_STRIDE: u64 = 0x9e3779b97f4a7c15;
    let mut config = tiny();
    config.target = ExecutionTarget::Cuda { index: 999 };
    config.systems = 2;
    config.policy.feature_count = 2;
    config.policy.hedge_count = 2;
    config.validation_seed = config.market.seed.wrapping_add(SYSTEM_STRIDE);
    assert_ne!(config.market.seed, config.validation_seed);
    let error = train(config, None, None).unwrap_err();
    assert!(
        matches!(error, ComputeError::InvalidInput(_)) && error.to_string().contains("derived"),
        "{error}"
    );
    let mut calibration = tiny();
    calibration.target = ExecutionTarget::Cuda { index: 999 };
    calibration.systems = 2;
    calibration.policy.feature_count = 2;
    calibration.policy.hedge_count = 2;
    calibration.calibration_seed = calibration.market.seed.wrapping_add(SYSTEM_STRIDE);
    let error = train(calibration, None, None).unwrap_err();
    assert!(
        matches!(error, ComputeError::InvalidInput(_)) && error.to_string().contains("derived"),
        "calibration stream collision must be preflighted: {error}"
    );
    // Wrap-around collisions must also be detected, rather than checking only
    // raw base-seed equality or a non-wrapping range.
    let mut config = tiny();
    config.target = ExecutionTarget::Cuda { index: 999 };
    config.systems = 2;
    config.policy.feature_count = 2;
    config.policy.hedge_count = 2;
    config.market.seed = u64::MAX;
    config.test_seed = u64::MAX.wrapping_add(SYSTEM_STRIDE);
    let error = train(config, None, None).unwrap_err();
    assert!(
        matches!(error, ComputeError::InvalidInput(_)) && error.to_string().contains("derived"),
        "{error}"
    );
    let mut config = tiny();
    config.target = ExecutionTarget::Cuda { index: 999 };
    config.max_updates = u64::MAX;
    let error = train(config, None, None).unwrap_err();
    assert!(
        matches!(error, ComputeError::InvalidInput(_)) && error.to_string().contains("path-ID"),
        "{error}"
    );
}

#[test]
fn compute_host_and_heldout_memory_fail_before_device_initialization() {
    for host in [false, true] {
        let mut config = tiny();
        config.target = ExecutionTarget::Cuda { index: 999 };
        if host {
            config.budget.host_working_bytes = 1;
        } else {
            config.budget.compute_bytes = 1;
        }
        let error = train(config, None, None).unwrap_err();
        assert!(
            matches!(error, ComputeError::BudgetExceeded(_)),
            "expected budget admission, got {error}"
        );
    }
    let mut config = tiny();
    config.target = ExecutionTarget::Cuda { index: 999 };
    config.test_paths = 1_000_000;
    config.validation_paths = 1_000_000;
    config.budget.host_working_bytes = 64 * 1024 * 1024;
    let error = train(config, None, None).unwrap_err();
    assert!(
        matches!(error, ComputeError::BudgetExceeded(_)) && error.to_string().contains("held-out"),
        "{error}"
    );
}

#[test]
fn fresh_path_streams_replay_exactly_and_checkpoint_configuration_is_frozen() {
    let config = tiny();
    let first = generate_batch(&config, config.market.seed, 0, 8).unwrap();
    let repeat = generate_batch(&config, config.market.seed, 0, 8).unwrap();
    let next = generate_batch(&config, config.market.seed, 8, 8).unwrap();
    assert_eq!(first.marks, repeat.marks);
    assert_eq!(first.source_hash, repeat.source_hash);
    assert_ne!(first.source_hash, next.source_hash);
    assert_ne!(first.marks, next.marks);
    let good = checkpoint(config.clone());
    for change in [0, 1, 2] {
        let mut modified = config.clone();
        modified.max_updates = 4;
        match change {
            0 => modified.learning_rate *= 0.5,
            1 => modified.validation_seed += 1,
            _ => modified.evaluation_batch_paths = 8,
        };
        assert!(validate_resume(&modified, &good).is_err());
    }
}

#[test]
fn empty_squared_liability_keeps_admissible_zero_policy_while_training_continues() {
    let temp = TempDirectory::new();
    let file = temp.file("empty-liability.json");
    let mut config = tiny();
    config.liability = volt_compute::hedge_application::HedgeLiability::None;
    config.premium_for_comparison = 0.0;
    let report = train(config, None, Some(&file)).unwrap();
    assert_eq!(report.selected_validation_risk, 0.0);
    assert_eq!(report.selected_update, 0);
    assert_eq!(report.evaluation.candidate.value, 0.0);
    let saved = load_checkpoint(&file).unwrap();
    assert_eq!(saved.current_policy.updates, 2);
    assert!(
        saved
            .selected_policy
            .parameters
            .iter()
            .all(|p| p.values.iter().all(|x| *x == 0.0))
    );
    validate_resume(&saved.config, &saved).unwrap();
}

#[test]
fn validation_patience_is_independent_of_test_data_and_survives_resume() {
    use volt_compute::hedge_training::{EarlyStoppingConfig, TrainingStopReason};
    let temp = TempDirectory::new();
    let mut config = tiny();
    config.liability = volt_compute::hedge_application::HedgeLiability::None;
    config.premium_for_comparison = 0.0;
    config.max_updates = 8;
    config.early_stopping = Some(EarlyStoppingConfig {
        minimum_updates: 6,
        patience_updates: 6,
    });
    let full_path = temp.file("early-full.json");
    let report = train(config.clone(), None, Some(&full_path)).unwrap();
    assert_eq!(report.completed_updates, 6);
    assert!(matches!(
        report.stop_reason,
        TrainingStopReason::ValidationPatience
    ));
    let full = load_checkpoint(&full_path).unwrap();
    let split_path = temp.file("early-split.json");
    let mut partial = config.clone();
    partial.max_updates = 4;
    let first = train(partial, None, Some(&split_path)).unwrap();
    assert!(matches!(
        first.stop_reason,
        TrainingStopReason::MaximumUpdates
    ));
    let saved = load_checkpoint(&split_path).unwrap();
    let resumed = train(config.clone(), Some(saved), Some(&split_path)).unwrap();
    assert_eq!(resumed.completed_updates, 6);
    assert!(matches!(
        resumed.stop_reason,
        TrainingStopReason::ValidationPatience
    ));
    assert_eq!(
        json(&full.current_policy),
        json(&load_checkpoint(&split_path).unwrap().current_policy)
    );
    config.test_seed = 777001;
    config.test_paths = 16;
    config.bootstrap_paths = 16;
    let other_path = temp.file("early-other-test.json");
    let other = train(config, None, Some(&other_path)).unwrap();
    assert_eq!(other.completed_updates, report.completed_updates);
    assert_eq!(
        json(&full.current_policy),
        json(&load_checkpoint(&other_path).unwrap().current_policy)
    );
}
