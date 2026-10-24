use volt_compute::{
    ExecutionTarget, OptionSide, Precision,
    budget::MemoryBudget,
    chunked::{ChunkedStats, ChunkedValuation, value_rows},
    job::validate_result,
};

#[test]
fn scalar_reference_retains_small_net_between_offsetting_positions() {
    let mut rows = volt_compute::benchmark::fixture(3);
    for row in &mut rows {
        row.side = OptionSide::Call;
        row.spot = 100.0;
        row.strike = 100.0;
        row.maturity_years = 1.0;
        row.volatility = 0.2;
        row.rate = 0.05;
        row.dividend_yield = 0.0;
    }
    rows[0].quantity = 1_i64 << 40;
    rows[1].quantity = 1;
    rows[2].quantity = -(1_i64 << 40);
    let result = value_rows(
        &rows,
        ExecutionTarget::Cpu,
        Precision::F64,
        &MemoryBudget::default(),
    )
    .unwrap();
    let summary = validate_result(&rows, &result, Precision::F64).unwrap();
    let expected = 100.0 * 10.450583572185565;
    assert!((summary.reference_portfolio_value - expected).abs() < 1e-10);
    assert!(summary.passed);
}

#[test]
fn infinite_reference_gross_cannot_become_an_infinite_acceptance_tolerance() {
    let mut rows = volt_compute::benchmark::fixture(6);
    for (index, row) in rows.iter_mut().enumerate() {
        row.side = OptionSide::Call;
        row.spot = 4e306;
        row.dividend_yield = 0.0;
        row.quantity = if index % 2 == 0 { 10 } else { -10 };
        row.multiplier = 1;
    }
    let prices = rows
        .iter()
        .map(|row| volt_compute::reference_price(row).unwrap())
        .collect();
    let plan = MemoryBudget::default()
        .plan(&rows, ExecutionTarget::Cpu, Precision::F64)
        .unwrap();
    let result = ChunkedValuation {
        prices,
        portfolio_value: 0.0,
        device_reduced_value: None,
        plan,
        stats: ChunkedStats::default(),
    };
    assert!(validate_result(&rows, &result, Precision::F64).is_err());
}
