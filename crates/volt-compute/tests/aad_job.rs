use volt_compute::{
    ExecutionTarget, Precision,
    aad_job::{differentiate_job, plan_aad_rows},
    budget::MemoryBudget,
};
fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}
#[test]
fn chunking_preserves_order_and_all_greeks() {
    let rows = volt_compute::benchmark::fixture(67);
    let small = MemoryBudget {
        max_batch_rows: 9,
        ..MemoryBudget::default()
    };
    let chunked = differentiate_job(&rows, target(), Precision::F32, &small).unwrap();
    let full =
        differentiate_job(&rows, target(), Precision::F32, &MemoryBudget::default()).unwrap();
    assert_eq!(chunked.completed_chunks, 8);
    assert_eq!(chunked.values.prices.len(), rows.len());
    let left = [
        &chunked.values.prices,
        &chunked.values.delta,
        &chunked.values.vega,
        &chunked.values.rho,
        &chunked.values.theta,
        &chunked.values.dividend_sensitivity,
    ];
    let right = [
        &full.values.prices,
        &full.values.delta,
        &full.values.vega,
        &full.values.rho,
        &full.values.theta,
        &full.values.dividend_sensitivity,
    ];
    for (left, right) in left.into_iter().zip(right) {
        for (a, b) in left.iter().zip(right) {
            assert!((a - b).abs() <= 0.0002 + 0.0002 * b.abs(), "{a} vs {b}");
        }
    }
}
#[test]
fn planning_rejects_global_duplicates_and_impossible_output_storage_without_device() {
    let mut rows = volt_compute::benchmark::fixture(10);
    rows[9].row_id = rows[0].row_id.clone();
    let budget = MemoryBudget {
        max_batch_rows: 1,
        ..MemoryBudget::default()
    };
    assert!(
        plan_aad_rows(
            &rows,
            ExecutionTarget::Cuda { index: 999 },
            Precision::F32,
            &budget
        )
        .is_err()
    );
    rows[9].row_id = "last".into();
    let impossible = MemoryBudget {
        host_working_bytes: 1,
        ..budget
    };
    assert!(
        plan_aad_rows(
            &rows,
            ExecutionTarget::Cuda { index: 999 },
            Precision::F32,
            &impossible
        )
        .is_err()
    );
}
#[test]
fn offsetting_positions_preserve_residual_sensitivities() {
    let mut row = volt_compute::benchmark::fixture(1).remove(0);
    row.spot = 100.0;
    row.strike = 100.0;
    row.maturity_years = 1.0;
    row.volatility = 0.2;
    row.rate = 0.05;
    row.dividend_yield = 0.0;
    row.multiplier = 100;
    let mut rows = vec![row.clone(), row.clone(), row];
    for (i, row) in rows.iter_mut().enumerate() {
        row.row_id = format!("offset-{i}");
    }
    rows[0].quantity = 1_i64 << 40;
    rows[1].quantity = -(1_i64 << 40);
    rows[2].quantity = 1;
    for cap in [1, 2, 3] {
        let result = differentiate_job(
            &rows,
            target(),
            Precision::F32,
            &MemoryBudget {
                max_batch_rows: cap,
                ..MemoryBudget::default()
            },
        )
        .unwrap();
        let totals = result.parallel_sensitivities;
        let observed = [
            totals.option_position_value,
            totals.delta,
            totals.vega,
            totals.rho,
            totals.theta,
            totals.dividend_sensitivity,
        ];
        let raw = [
            result.values.prices[2],
            result.values.delta[2],
            result.values.vega[2],
            result.values.rho[2],
            result.values.theta[2],
            result.values.dividend_sensitivity[2],
        ];
        for (total, raw) in observed.into_iter().zip(raw) {
            assert!(
                (total - 100.0 * raw).abs() < 1e-8,
                "cap {cap}: {total} vs {}",
                100.0 * raw
            );
        }
    }
}

#[test]
fn optional_gamma_is_budgeted_ordered_and_independently_validated() {
    use volt_compute::{
        aad_job::{differentiate_job_with_gamma, plan_aad_rows_with_gamma},
        second_order::analytic_gamma,
    };
    let rows = volt_compute::benchmark::fixture(67);
    let budget = MemoryBudget {
        max_batch_rows: 9,
        ..MemoryBudget::default()
    };
    let plan = plan_aad_rows_with_gamma(&rows, target(), Precision::F32, &budget, true).unwrap();
    assert_eq!(plan.estimated_compute_bytes_per_row, 384 * 4);
    let result =
        differentiate_job_with_gamma(&rows, target(), Precision::F32, &budget, true).unwrap();
    assert_eq!(result.completed_chunks, 8);
    let gammas = result.gamma.unwrap();
    let mut expected_total = 0.0;
    for (row, gamma) in rows.iter().zip(gammas) {
        let expected = analytic_gamma(row).unwrap();
        assert!((gamma - expected).abs() <= 1e-6 + 0.0002 * expected.abs());
        expected_total += expected * row.quantity as f64 * row.multiplier as f64;
    }
    assert!((result.parallel_gamma.unwrap() - expected_total).abs() < 0.002);
}
