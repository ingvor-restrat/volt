use volt_compute::{
    ExecutionTarget, OptionRow, OptionSide, Precision, budget::MemoryBudget, chunked::value_rows,
};

fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}

fn row(id: &str, quantity: i64) -> OptionRow {
    OptionRow {
        row_id: id.into(),
        contract_id: format!("contract-{id}"),
        side: OptionSide::Call,
        spot: 100.0,
        strike: 100.0,
        maturity_years: 1.0,
        volatility: 0.2,
        rate: 0.05,
        dividend_yield: 0.0,
        quantity,
        multiplier: 100,
    }
}

#[test]
fn memory_chunking_preserves_small_net_exposure_between_large_offsetting_legs() {
    let large = 1_i64 << 40;
    let rows = [row("long", large), row("small", 1), row("short", -large)];
    let mut prior_total = None::<f64>;
    for cap in [1, 2, 3] {
        let budget = MemoryBudget {
            max_batch_rows: cap,
            ..Default::default()
        };
        let result = value_rows(&rows, target(), Precision::F32, &budget).unwrap();
        assert_eq!(result.stats.completed_chunks, 3_usize.div_ceil(cap));
        // Equal large marks cancel before scaling, providing an independent
        // expression for the expected value without repeating the summation code.
        let expected =
            (result.prices[0] - result.prices[2]) * large as f64 * 100.0 + result.prices[1] * 100.0;
        assert!(expected.abs() > 1000.0);
        assert!(
            (result.portfolio_value - expected).abs() <= 1e-10 + 1e-12 * expected.abs(),
            "cap {cap}: {} versus {expected}",
            result.portfolio_value
        );
        if let Some(prior) = prior_total {
            assert!(
                (result.portfolio_value - prior).abs() <= 0.0002,
                "cap {cap}: prior {prior}, current {}, prices {:?}",
                result.portfolio_value,
                result.prices
            );
        }
        prior_total = Some(result.portfolio_value);
    }
}

#[test]
fn overflowing_f32_device_reduction_does_not_discard_finite_marks() {
    let mut rows = [
        row("long", i64::MAX),
        row("small", 1),
        row("short", -i64::MAX),
    ];
    for row in &mut rows {
        row.spot = 1e12;
        row.multiplier = u32::MAX;
    }
    for cap in [1, 2, 3] {
        let result = value_rows(
            &rows,
            target(),
            Precision::F32,
            &MemoryBudget {
                max_batch_rows: cap,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.prices.iter().all(|price| price.is_finite()));
        let expected = result.prices[1] * u32::MAX as f64;
        assert!(result.portfolio_value.is_finite());
        assert!(
            (result.portfolio_value - expected).abs() <= 1e-12 * expected.abs(),
            "cap {cap}: {} versus {expected}",
            result.portfolio_value
        );
    }
}
