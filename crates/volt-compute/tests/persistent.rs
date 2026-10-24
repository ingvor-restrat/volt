use volt_compute::{
    ExecutionTarget, Precision, budget::MemoryBudget, persistent::PreparedValuation,
};

fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}

#[test]
fn prepared_jobs_reuse_inputs_across_distinct_device_scenarios() {
    let rows = volt_compute::benchmark::fixture(259);
    let original_hash = volt_compute::job::input_hash(&rows).unwrap();
    let job =
        PreparedValuation::new(&rows, target(), Precision::F32, &MemoryBudget::default()).unwrap();
    let mut base_prices = None;
    for scale in [1.0, 0.8, 1.2, 1.0] {
        let result = job.evaluate(scale).unwrap();
        assert_eq!(result.stats.packing_and_upload_seconds, 0.0);
        assert!(
            volt_compute::job::validate_scaled_result(&rows, &result, Precision::F32, scale)
                .unwrap()
                .passed
        );
        if scale == 1.0 {
            if let Some(prices) = &base_prices {
                assert_eq!(&result.prices, prices);
            }
            base_prices = Some(result.prices);
        }
    }
    assert_eq!(job.row_ids()[258], rows[258].row_id);
    assert_eq!(volt_compute::job::input_hash(&rows).unwrap(), original_hash);
}

#[test]
fn insufficient_residency_and_invalid_shocks_fail_explicitly() {
    let rows = volt_compute::benchmark::fixture(100);
    assert!(
        PreparedValuation::new(
            &rows,
            target(),
            Precision::F32,
            &MemoryBudget {
                max_batch_rows: 99,
                ..MemoryBudget::default()
            }
        )
        .is_err()
    );
    let job =
        PreparedValuation::new(&rows, target(), Precision::F32, &MemoryBudget::default()).unwrap();
    for scale in [f64::NAN, f64::INFINITY, 0.0, -1.0, f64::MIN_POSITIVE] {
        assert!(job.evaluate(scale).is_err());
    }
    assert!(job.evaluate(1.0).is_ok());
}

#[test]
fn persistent_f64_prices_remain_accurate_after_shocks() {
    let rows = volt_compute::benchmark::fixture(259);
    let prepared =
        PreparedValuation::new(&rows, target(), Precision::F64, &MemoryBudget::default());
    if target() == ExecutionTarget::Metal {
        assert!(prepared.is_err());
        return;
    }
    let prepared = prepared.unwrap();
    for scale in [0.9, 1.0, 1.1] {
        let result = prepared.evaluate(scale).unwrap();
        assert!(
            volt_compute::job::validate_scaled_result(&rows, &result, Precision::F64, scale)
                .unwrap()
                .passed
        );
    }
}

#[test]
fn scaled_spot_underflow_and_overflow_are_rejected_before_a_bad_frame() {
    for (spot, bad_scale) in [(1e-20, 1e-30), (1e20, 1e20)] {
        let mut rows = volt_compute::benchmark::fixture(1);
        rows[0].spot = spot;
        let job = PreparedValuation::new(&rows, target(), Precision::F32, &MemoryBudget::default())
            .unwrap();
        let base = job.evaluate(1.0).unwrap();
        assert!(job.evaluate(bad_scale).is_err());
        assert_eq!(job.evaluate(1.0).unwrap().prices, base.prices);
    }
}
