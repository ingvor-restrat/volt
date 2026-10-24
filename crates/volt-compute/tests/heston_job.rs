use volt_compute::{
    ComputeError, ExecutionTarget, OptionSide, Precision,
    budget::MemoryBudget,
    heston::{HestonOption, HestonParams, QuadratureConfig},
    heston_job::{
        HestonJobConfig, HestonJobRequest, HestonPosition, metrics, plan_heston, run_heston,
    },
};

fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}
fn precisions() -> Vec<Precision> {
    if target() == ExecutionTarget::Metal {
        eprintln!("SKIP Heston job f64: Metal has no native f64 arithmetic");
        vec![Precision::F32]
    } else {
        vec![Precision::F32, Precision::F64]
    }
}
fn config(cap: usize) -> HestonJobConfig {
    HestonJobConfig {
        target: target(),
        precision: Precision::F32,
        quadrature: QuadratureConfig {
            nodes: 128,
            upper_bound: 160.0,
        },
        budget: MemoryBudget {
            max_batch_rows: cap,
            ..MemoryBudget::default()
        },
    }
}
fn request(n: usize) -> HestonJobRequest {
    HestonJobRequest {
        schema: "volt.compute.heston_request.v1".into(),
        currency: "USD".into(),
        options: (0..n)
            .map(|i| HestonOption {
                row_id: format!("row-{i}"),
                side: if i % 2 == 0 {
                    OptionSide::Call
                } else {
                    OptionSide::Put
                },
                spot: 90.0 + i as f64 * 4.0,
                strike: 100.0,
                maturity_years: 0.5 + i as f64 * 0.1,
                rate: -0.005,
                dividend_yield: 0.02,
            })
            .collect(),
        params: (0..n)
            .map(|i| HestonParams {
                initial_variance: 0.04 + i as f64 * 0.005,
                kappa: 2.0,
                long_run_variance: 0.05,
                vol_of_variance: 0.4,
                correlation: -0.6,
            })
            .collect(),
        positions: (0..n)
            .map(|i| HestonPosition {
                row_id: format!("row-{i}"),
                contract_id: format!("contract-{i}"),
                quantity: if i % 2 == 0 { 3 } else { -2 },
                multiplier: 100,
            })
            .collect(),
    }
}

#[test]
fn offline_plan_counts_expanded_quadrature_cells_and_tail_chunks() {
    let request = request(7);
    let mut config = config(128 * 2 + 17);
    config.target = ExecutionTarget::Cuda { index: 999 };
    let plan = plan_heston(&request, &config).unwrap();
    assert_eq!(plan.option_rows, 7);
    assert_eq!(plan.chunk_rows, 2);
    assert_eq!(plan.chunks, 4);
    assert_eq!(plan.quadrature_nodes, 128);
    assert_eq!(plan.expanded_cells_per_chunk, 256);
    assert_eq!(plan.total_quadrature_cells, 896);
    assert_eq!(plan.estimated_compute_bytes_per_cell, 4096);
    assert!(plan.estimated_compute_bytes <= plan.compute_admission_limit_bytes);
    assert!(plan.estimated_host_working_bytes <= plan.host_admission_limit_bytes);
    config.budget.max_batch_rows = 127;
    assert!(matches!(
        plan_heston(&request, &config),
        Err(ComputeError::BudgetExceeded(_))
    ));
    config.budget.max_batch_rows = 128;
    assert_eq!(plan_heston(&request, &config).unwrap().chunk_rows, 1);
}

#[test]
fn shared_memory_admission_and_double_precision_have_explicit_costs() {
    let request = request(3);
    let mut config = config(384);
    config.target = ExecutionTarget::Cuda { index: 999 };
    let cuda = plan_heston(&request, &config).unwrap();
    config.target = ExecutionTarget::Cpu;
    let cpu = plan_heston(&request, &config).unwrap();
    assert_eq!(
        cpu.estimated_host_working_bytes - cuda.estimated_host_working_bytes,
        config.budget.compute_reserve_bytes + 384 * 4096
    );
    config.precision = Precision::F64;
    let double = plan_heston(&request, &config).unwrap();
    assert_eq!(double.estimated_compute_bytes_per_cell, 8192);
    assert_eq!(
        double.estimated_compute_bytes - config.budget.compute_reserve_bytes,
        2 * (cpu.estimated_compute_bytes - config.budget.compute_reserve_bytes)
    );
    config.target = ExecutionTarget::Metal;
    assert!(matches!(
        plan_heston(&request, &config),
        Err(ComputeError::Unsupported(_))
    ));
}

#[test]
fn invalid_global_identities_and_late_market_inputs_fail_before_device_creation() {
    let base = request(7);
    let mut config = config(128 * 2);
    config.target = ExecutionTarget::Cuda { index: 999 };
    let mut duplicate = base.clone();
    duplicate.options[6].row_id = duplicate.options[0].row_id.clone();
    duplicate.positions[6].row_id = duplicate.options[6].row_id.clone();
    assert!(matches!(
        run_heston(&duplicate, &config),
        Err(ComputeError::InvalidInput(_))
    ));
    let mut mismatch = base.clone();
    mismatch.positions[6].row_id = "wrong-final-row".into();
    assert!(matches!(
        run_heston(&mismatch, &config),
        Err(ComputeError::InvalidInput(_))
    ));
    let mut blank = base.clone();
    blank.positions[6].contract_id = "  ".into();
    assert!(matches!(
        run_heston(&blank, &config),
        Err(ComputeError::InvalidInput(_))
    ));
    let mut invalid = base.clone();
    invalid.params[6].correlation = 1.0;
    assert!(matches!(
        run_heston(&invalid, &config),
        Err(ComputeError::InvalidInput(_))
    ));
    invalid = base.clone();
    invalid.options[6].spot = 1e-100;
    assert!(matches!(
        run_heston(&invalid, &config),
        Err(ComputeError::InvalidInput(_))
    ));
    invalid = base.clone();
    invalid.positions[6].multiplier = 0;
    assert!(matches!(
        run_heston(&invalid, &config),
        Err(ComputeError::InvalidInput(_))
    ));
    invalid = base.clone();
    invalid.currency = "usd".into();
    assert!(matches!(
        plan_heston(&invalid, &config),
        Err(ComputeError::InvalidInput(_))
    ));
    invalid = base.clone();
    invalid.schema = "canonical.volt.portfolio".into();
    assert!(matches!(
        plan_heston(&invalid, &config),
        Err(ComputeError::InvalidInput(_))
    ));
    invalid = base.clone();
    invalid.params.pop();
    assert!(matches!(
        plan_heston(&invalid, &config),
        Err(ComputeError::InvalidInput(_))
    ));
    config.quadrature.upper_bound = f64::NAN;
    assert!(matches!(
        run_heston(&base, &config),
        Err(ComputeError::InvalidInput(_))
    ));
}

#[test]
fn insufficient_budgets_reject_unavailable_target_without_dispatch() {
    let request = request(3);
    let mut config = config(384);
    config.target = ExecutionTarget::Cuda { index: 999 };
    config.budget.compute_bytes = 1;
    assert!(matches!(
        run_heston(&request, &config),
        Err(ComputeError::BudgetExceeded(_))
    ));
    config.budget = MemoryBudget {
        host_working_bytes: 1,
        ..MemoryBudget::default()
    };
    assert!(matches!(
        run_heston(&request, &config),
        Err(ComputeError::BudgetExceeded(_))
    ));
    config.budget = MemoryBudget {
        max_batch_rows: 127,
        ..MemoryBudget::default()
    };
    assert!(matches!(
        run_heston(&request, &config),
        Err(ComputeError::BudgetExceeded(_))
    ));
    config.budget = MemoryBudget {
        headroom_percent: 100,
        ..MemoryBudget::default()
    };
    assert!(matches!(
        plan_heston(&request, &config),
        Err(ComputeError::InvalidInput(_))
    ));
}

#[test]
fn chunk_tails_preserve_row_order_prices_risk_and_signed_totals() {
    let request = request(7);
    for precision in precisions() {
        let mut config = config(128 * 7);
        config.precision = precision;
        let full = run_heston(&request, &config).unwrap();
        for chunk_rows in [1, 2, 4] {
            config.budget.max_batch_rows = 128 * chunk_rows;
            let chunked = run_heston(&request, &config).unwrap();
            assert_eq!(chunked.completed_chunks, 7_usize.div_ceil(chunk_rows));
            assert_eq!(chunked.plan.chunk_rows, chunk_rows);
            assert!(chunked.job_seconds.is_finite() && chunked.job_seconds >= 0.0);
            for (k, ((name, a), (_, b))) in metrics(&chunked.values)
                .into_iter()
                .zip(metrics(&full.values))
                .enumerate()
            {
                assert_eq!(a.len(), 7);
                for (i, (a, b)) in a.iter().zip(b).enumerate() {
                    let tolerance = if precision == Precision::F32 {
                        2e-4 + 2e-5 * b.abs()
                    } else {
                        1e-10 + 1e-10 * b.abs()
                    };
                    assert!(
                        (a - b).abs() < tolerance,
                        "{precision:?} {name} row{i}, chunk{chunk_rows}: {a} vs {b}"
                    );
                }
                let weighted = a
                    .iter()
                    .zip(&request.positions)
                    .map(|(v, p)| v * p.quantity as f64 * p.multiplier as f64)
                    .sum::<f64>();
                assert_eq!(chunked.parallel_sensitivities[k].metric, name);
                assert!(
                    (chunked.parallel_sensitivities[k].value - weighted).abs()
                        < 1e-9 + 1e-12 * weighted.abs()
                );
            }
        }
    }
}

#[test]
fn large_offsetting_positions_keep_residual_through_different_chunk_shapes() {
    let mut request = request(3);
    request.options[1] = request.options[0].clone();
    request.options[1].row_id = "residual".into();
    request.options[2] = request.options[0].clone();
    request.options[2].row_id = "huge-short".into();
    request.params[1] = request.params[0];
    request.params[2] = request.params[0];
    for (p, o) in request.positions.iter_mut().zip(&request.options) {
        p.row_id = o.row_id.clone();
        p.multiplier = 1;
    }
    request.positions[0].quantity = 1_i64 << 60;
    request.positions[1].quantity = 1;
    request.positions[2].quantity = -(1_i64 << 60);
    for precision in precisions() {
        for chunk_rows in [1, 2, 3] {
            let mut config = config(128 * chunk_rows);
            config.precision = precision;
            let run = run_heston(&request, &config).unwrap();
            for ((name, raw), total) in metrics(&run.values)
                .into_iter()
                .zip(&run.parallel_sensitivities)
            {
                assert_eq!(name, total.metric);
                assert!(
                    (total.value - raw[1]).abs() < 1e-10,
                    "{precision:?} {name} chunk{chunk_rows} residual {} vs {}",
                    total.value,
                    raw[1]
                );
            }
        }
    }
}
