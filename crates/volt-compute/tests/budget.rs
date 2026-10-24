use volt_compute::{ExecutionTarget, Precision, budget::MemoryBudget, chunked::value_rows};

#[test]
fn planner_accounts_for_host_outputs_and_shared_memory() {
    let rows = volt_compute::benchmark::fixture(1000);
    let budget = MemoryBudget {
        compute_bytes: 100_000,
        compute_reserve_bytes: 1000,
        host_working_bytes: 100_000,
        max_batch_rows: 1000,
        headroom_percent: 10,
    };
    let cuda = budget
        .plan(&rows, ExecutionTarget::Cuda { index: 0 }, Precision::F32)
        .unwrap();
    let metal = budget
        .plan(&rows, ExecutionTarget::Metal, Precision::F32)
        .unwrap();
    let cpu = budget
        .plan(&rows, ExecutionTarget::Cpu, Precision::F32)
        .unwrap();
    assert!(cuda.chunk_rows < rows.len());
    assert!(metal.chunk_rows < cuda.chunk_rows);
    assert_eq!(metal.chunk_rows, cpu.chunk_rows);
    for plan in [cuda, metal, cpu] {
        assert!(plan.estimated_compute_bytes <= plan.compute_admission_limit_bytes);
        assert!(plan.estimated_host_working_bytes <= plan.host_admission_limit_bytes);
        assert_eq!(plan.chunks, rows.len().div_ceil(plan.chunk_rows));
    }
    let f64 = budget
        .plan(&rows, ExecutionTarget::Cuda { index: 0 }, Precision::F64)
        .unwrap();
    assert!(f64.chunk_rows < 1000);
}

#[test]
fn impossible_or_invalid_budgets_fail_before_allocation() {
    let rows = volt_compute::benchmark::fixture(10);
    let budget = MemoryBudget {
        compute_bytes: 1,
        ..MemoryBudget::default()
    };
    assert!(
        budget
            .plan(&rows, ExecutionTarget::Cpu, Precision::F32)
            .is_err()
    );
    let budget = MemoryBudget {
        host_working_bytes: 1,
        ..MemoryBudget::default()
    };
    assert!(
        budget
            .plan(&rows, ExecutionTarget::Cpu, Precision::F32)
            .is_err()
    );
    let budget = MemoryBudget {
        headroom_percent: 100,
        ..MemoryBudget::default()
    };
    assert!(
        budget
            .plan(&rows, ExecutionTarget::Cpu, Precision::F32)
            .is_err()
    );
    let budget = MemoryBudget {
        compute_bytes: u64::MAX,
        host_working_bytes: u64::MAX,
        ..MemoryBudget::default()
    };
    assert!(
        budget
            .plan(&rows, ExecutionTarget::Cpu, Precision::F64)
            .is_ok()
    );
}

#[test]
fn chunked_execution_preserves_tail_row_order_and_signed_values() {
    let target = std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap();
    let rows = volt_compute::benchmark::fixture(257);
    let budget = MemoryBudget {
        max_batch_rows: 31,
        ..MemoryBudget::default()
    };
    let result = value_rows(&rows, target, Precision::F32, &budget).unwrap();
    assert_eq!(result.prices.len(), 257);
    assert_eq!(result.stats.completed_chunks, 9);
    let mut expected_total = 0.0;
    let mut gross = 0.0;
    for (row, observed) in rows.iter().zip(&result.prices) {
        let expected = volt_compute::reference_price(row).unwrap();
        assert!((observed - expected).abs() <= 0.0002 + 0.0001 * expected.abs());
        let signed = expected * row.quantity as f64 * row.multiplier as f64;
        expected_total += signed;
        gross += signed.abs();
    }
    assert!((result.portfolio_value - expected_total).abs() <= 0.0001 * gross);
    let single = value_rows(&rows, target, Precision::F32, &MemoryBudget::default()).unwrap();
    for (chunked, unchunked) in result.prices.iter().zip(single.prices) {
        assert!((chunked - unchunked).abs() < 0.0002);
    }
}

#[test]
fn duplicate_ids_across_chunks_and_late_invalid_inputs_are_rejected() {
    let mut rows = volt_compute::benchmark::fixture(50);
    rows[49].row_id = rows[0].row_id.clone();
    let budget = MemoryBudget {
        max_batch_rows: 7,
        ..MemoryBudget::default()
    };
    assert!(value_rows(&rows, ExecutionTarget::Cpu, Precision::F32, &budget).is_err());
    rows[49].row_id = "last-row".into();
    rows[49].rate = f64::NAN;
    assert!(value_rows(&rows, ExecutionTarget::Cpu, Precision::F32, &budget).is_err());
}

#[test]
fn reverse_graph_model_reserves_tape_and_all_gradient_outputs() {
    use volt_compute::budget::WorkloadMemoryModel;
    let rows = volt_compute::benchmark::fixture(1000);
    let budget = MemoryBudget {
        compute_bytes: 500_000,
        compute_reserve_bytes: 1000,
        host_working_bytes: 500_000,
        max_batch_rows: 1000,
        headroom_percent: 10,
    };
    let valuation = budget
        .plan(&rows, ExecutionTarget::Cpu, Precision::F32)
        .unwrap();
    let aad = budget
        .plan_for_workload(
            &rows,
            ExecutionTarget::Cpu,
            Precision::F32,
            WorkloadMemoryModel::BLACK_SCHOLES_AAD,
        )
        .unwrap();
    assert!(aad.chunk_rows < valuation.chunk_rows);
    assert_eq!(aad.estimated_compute_bytes_per_row, 1024);
    assert_eq!(
        aad.host_global_bytes - valuation.host_global_bytes,
        40 * rows.len() as u64
    );
    assert!(aad.estimated_compute_bytes <= aad.compute_admission_limit_bytes);
    assert!(aad.estimated_host_working_bytes <= aad.host_admission_limit_bytes);
    let overflow = WorkloadMemoryModel {
        scalar_slots_per_row: u64::MAX,
        ..WorkloadMemoryModel::BLACK_SCHOLES_AAD
    };
    assert!(
        budget
            .plan_for_workload(&rows, ExecutionTarget::Cpu, Precision::F64, overflow)
            .is_err()
    );
}
