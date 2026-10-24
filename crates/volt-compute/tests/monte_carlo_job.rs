use volt_compute::{
    ComputeError, ExecutionTarget, OptionRow, Precision,
    budget::MemoryBudget,
    monte_carlo::{GreekMoments, OnlineMoments},
    monte_carlo_job::{MonteCarloConfig, plan_monte_carlo, run_monte_carlo},
    monte_carlo_validation::validate_monte_carlo,
};

fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}
fn config(cap: usize) -> MonteCarloConfig {
    MonteCarloConfig {
        target: target(),
        precision: Precision::F32,
        seed: 731,
        antithetic_pairs: 37,
        budget: MemoryBudget {
            max_batch_rows: cap,
            ..MemoryBudget::default()
        },
    }
}
fn rows(count: usize) -> Vec<OptionRow> {
    let mut rows = volt_compute::benchmark::fixture(count);
    for row in &mut rows {
        row.spot = 100.0;
        row.strike = 100.0;
        row.maturity_years = 0.7;
        row.volatility = 0.23;
        row.rate = -0.01;
        row.dividend_yield = 0.017;
    }
    rows
}
fn metrics(m: &GreekMoments) -> [&OnlineMoments; 7] {
    [
        &m.price,
        &m.delta,
        &m.vega,
        &m.rho,
        &m.theta,
        &m.dividend_sensitivity,
        &m.gamma,
    ]
}
fn compare(a: &GreekMoments, b: &GreekMoments) {
    for (a, b) in metrics(a).into_iter().zip(metrics(b)) {
        assert_eq!(a.count, b.count);
        for (a, b) in [
            (a.mean, b.mean),
            (a.m2, b.m2),
            (
                a.estimate().standard_error.unwrap(),
                b.estimate().standard_error.unwrap(),
            ),
        ] {
            assert!((a - b).abs() <= 3e-4 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }
}

#[test]
fn planning_is_offline_and_caps_expanded_outcomes_including_partial_batches() {
    let rows = rows(3);
    let mut config = config(17);
    config.target = ExecutionTarget::Cuda { index: 999 };
    let plan = plan_monte_carlo(&rows, &config).unwrap();
    assert_eq!(plan.option_rows, 3);
    assert_eq!(plan.pairs_per_batch, 2);
    assert_eq!(plan.expanded_outcomes_per_batch, 12);
    assert_eq!(plan.batches, 19);
    assert_eq!(plan.total_option_outcomes, 222);
    assert!(plan.estimated_compute_bytes <= plan.compute_admission_limit_bytes);
    assert!(plan.estimated_host_working_bytes <= plan.host_admission_limit_bytes);
    config.budget.max_batch_rows = 5;
    assert!(matches!(
        plan_monte_carlo(&rows, &config),
        Err(ComputeError::BudgetExceeded(_))
    ));
    config.budget.max_batch_rows = 6;
    assert_eq!(plan_monte_carlo(&rows, &config).unwrap().pairs_per_batch, 1);
}

#[test]
fn shared_host_admission_accounts_for_compute_reserve_and_dtype() {
    let rows = rows(2);
    let mut config = config(16);
    config.target = ExecutionTarget::Cuda { index: 999 };
    let cuda = plan_monte_carlo(&rows, &config).unwrap();
    config.target = ExecutionTarget::Cpu;
    let cpu = plan_monte_carlo(&rows, &config).unwrap();
    assert_eq!(
        cpu.estimated_host_working_bytes - cuda.estimated_host_working_bytes,
        config.budget.compute_reserve_bytes + cpu.expanded_outcomes_per_batch as u64 * 1024
    );
    config.precision = Precision::F64;
    let double = plan_monte_carlo(&rows, &config).unwrap();
    assert_eq!(double.estimated_compute_bytes_per_outcome, 2048);
    assert_eq!(
        double.estimated_compute_bytes - config.budget.compute_reserve_bytes,
        2 * (cpu.estimated_compute_bytes - config.budget.compute_reserve_bytes)
    );
}

#[test]
fn impossible_budget_and_invalid_global_inputs_fail_before_device_initialization() {
    let rows = rows(3);
    let mut config = config(16);
    config.target = ExecutionTarget::Cuda { index: 999 };
    config.budget.compute_bytes = 1;
    assert!(matches!(
        run_monte_carlo(&rows, &config),
        Err(ComputeError::BudgetExceeded(_))
    ));
    config.budget = MemoryBudget {
        host_working_bytes: 1,
        ..MemoryBudget::default()
    };
    assert!(matches!(
        run_monte_carlo(&rows, &config),
        Err(ComputeError::BudgetExceeded(_))
    ));
    config.budget = MemoryBudget::default();
    let mut duplicate = rows.clone();
    duplicate[2].row_id = duplicate[0].row_id.clone();
    assert!(matches!(
        run_monte_carlo(&duplicate, &config),
        Err(ComputeError::InvalidInput(_))
    ));
    let mut invalid = rows.clone();
    invalid[2].volatility = f64::NAN;
    assert!(matches!(
        run_monte_carlo(&invalid, &config),
        Err(ComputeError::InvalidInput(_))
    ));
    config.antithetic_pairs = usize::MAX;
    assert!(matches!(
        plan_monte_carlo(&rows, &config),
        Err(ComputeError::InvalidInput(_))
    ));
    config.antithetic_pairs = 1;
    assert!(matches!(
        plan_monte_carlo(&rows, &config),
        Err(ComputeError::InvalidInput(_))
    ));
    config.antithetic_pairs = 2;
    config.budget.headroom_percent = 100;
    assert!(matches!(
        plan_monte_carlo(&rows, &config),
        Err(ComputeError::InvalidInput(_))
    ));
}

#[test]
fn streaming_preserves_seeded_paths_counts_row_order_and_covariance() {
    let rows = rows(3);
    let full = run_monte_carlo(&rows, &config(1024)).unwrap();
    for cap in [6, 17, 78] {
        let run = run_monte_carlo(&rows, &config(cap)).unwrap();
        assert_eq!(run.completed_batches, run.plan.batches);
        assert_eq!(run.result.pair_count, 37);
        assert!(run.plan.expanded_outcomes_per_batch <= cap);
        assert!(run.job_seconds.is_finite() && run.job_seconds >= 0.0);
        for (i, (a, b)) in run.result.rows.iter().zip(&full.result.rows).enumerate() {
            assert_eq!(a.row_id, rows[i].row_id);
            compare(&a.moments, &b.moments);
            for m in metrics(&a.moments) {
                assert_eq!(m.count, 37);
            }
        }
        compare(&run.result.portfolio, &full.result.portfolio);
    }
    let mut changed = config(1024);
    changed.seed += 1;
    let changed = run_monte_carlo(&rows, &changed).unwrap();
    assert_ne!(
        changed.result.rows[0].moments.price.mean,
        full.result.rows[0].moments.price.mean
    );
}

#[test]
fn opposite_positions_cancel_pair_covariance_and_preserve_small_residual() {
    let mut rows = rows(3);
    rows[1] = rows[0].clone();
    rows[1].row_id = "residual".into();
    rows[2] = rows[0].clone();
    rows[2].row_id = "huge-short".into();
    for row in &mut rows {
        row.multiplier = 1;
    }
    rows[0].quantity = 1_i64 << 60;
    rows[1].quantity = 1;
    rows[2].quantity = -(1_i64 << 60);
    let result = run_monte_carlo(&rows, &config(18)).unwrap();
    let standalone = run_monte_carlo(&[rows[1].clone()], &config(6)).unwrap();
    compare(&result.result.portfolio, &standalone.result.portfolio);
    for (total, one) in metrics(&result.result.portfolio)
        .into_iter()
        .zip(metrics(&result.result.rows[1].moments))
    {
        assert!(
            (total.mean - one.mean).abs() < 1e-10,
            "portfolio residual {} vs {}",
            total.mean,
            one.mean
        );
        assert!(
            (total.m2 - one.m2).abs() < 1e-8,
            "portfolio covariance {} vs {}",
            total.m2,
            one.m2
        );
    }
    rows[1].quantity = 0;
    let result = run_monte_carlo(&rows, &config(18)).unwrap();
    for total in metrics(&result.result.portfolio) {
        assert_eq!(total.mean, 0.0);
        assert_eq!(total.m2, 0.0);
    }
}

#[test]
fn validation_rejects_malformed_identity_counts_and_central_moments() {
    let rows = rows(2);
    let mut result = run_monte_carlo(&rows, &config(32)).unwrap().result;
    assert!(validate_monte_carlo(&rows, &result, Precision::F32).is_ok());
    result.rows[0].row_id = "wrong".into();
    assert!(validate_monte_carlo(&rows, &result, Precision::F32).is_err());
    result.rows[0].row_id = rows[0].row_id.clone();
    result.rows[0].moments.price.count -= 1;
    assert!(validate_monte_carlo(&rows, &result, Precision::F32).is_err());
    result.rows[0].moments.price.count += 1;
    result.rows[0].moments.gamma.m2 = -1.0;
    assert!(validate_monte_carlo(&rows, &result, Precision::F32).is_err());
    result.rows[0].moments.gamma.m2 = f64::NAN;
    assert!(validate_monte_carlo(&rows, &result, Precision::F32).is_err());
}

#[test]
fn device_and_host_moment_overflows_return_errors() {
    let mut rows = rows(1);
    rows[0].rate = 1e30;
    assert!(matches!(
        run_monte_carlo(&rows, &config(16)),
        Err(ComputeError::Execution(_))
    ));
    if target() == ExecutionTarget::Metal {
        eprintln!("SKIP host-moment f64 overflow: Metal has no native f64 arithmetic");
        return;
    }
    let mut config = config(16);
    config.precision = Precision::F64;
    rows[0].rate = 0.03;
    rows[0].spot = 1e150;
    rows[0].strike = 1e150;
    rows[0].quantity = i64::MAX;
    assert!(matches!(
        run_monte_carlo(&rows, &config),
        Err(ComputeError::Execution(_))
    ));
    // Each single-pair batch has m2=0 and finite weighted observations. Only
    // combining batches overflows central moments, testing the job's merge
    // validation rather than just the low-level tensor/output checks.
    config.budget.max_batch_rows = 2;
    match run_monte_carlo(&rows, &config) {
        Err(ComputeError::Execution(message)) => {
            assert!(message.contains("merged moments"), "{message}")
        }
        other => panic!("expected merged-moment overflow, got {other:?}"),
    }
}
