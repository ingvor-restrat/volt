use volt_compute::{
    ExecutionTarget, OptionRow, OptionSide, Precision, ResidentOptions, Session, reference_price,
};

#[test]
fn matrix_exchange_preserves_layout_and_device_computation() {
    let target = std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap();
    let session = Session::new(target, Precision::F32, 10).unwrap();
    let a =
        volt_compute::ResidentMatrix::upload(&session, 2, 3, vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let b = volt_compute::ResidentMatrix::upload(&session, 3, 2, vec![7., 8., 9., 10., 11., 12.])
        .unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), [2, 2]);
    assert_eq!(c.download().unwrap(), [58., 64., 139., 154.]);
    assert!(a.matmul(&a).is_err());
    assert!(volt_compute::ResidentMatrix::upload(&session, usize::MAX, 2, vec![]).is_err());
    assert!(volt_compute::ResidentMatrix::upload(&session, 1, 1, vec![f64::NAN]).is_err());
}

fn row(side: OptionSide) -> OptionRow {
    OptionRow {
        row_id: "row".into(),
        contract_id: "contract".into(),
        side,
        spot: 100.0,
        strike: 100.0,
        maturity_years: 1.0,
        volatility: 0.2,
        rate: 0.05,
        dividend_yield: 0.0,
        quantity: -2,
        multiplier: 100,
    }
}

#[test]
fn scalar_matches_known_prices_and_put_call_parity() {
    let call = reference_price(&row(OptionSide::Call)).unwrap();
    let put = reference_price(&row(OptionSide::Put)).unwrap();
    assert!((call - 10.450583572185565).abs() < 1e-12);
    assert!((put - 5.573526022256971).abs() < 1e-12);
    assert!((call - put - (100.0 - 100.0 * (-0.05_f64).exp())).abs() < 1e-12);
}

#[test]
fn f64_device_math_retains_double_precision_accuracy() {
    let target = std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap();
    if target == ExecutionTarget::Metal {
        assert!(Session::new(target, Precision::F64, 257).is_err());
        return;
    }
    let rows = volt_compute::benchmark::fixture(257);
    let session = Session::new(target, Precision::F64, rows.len()).unwrap();
    let resident = ResidentOptions::upload(&session, &rows).unwrap();
    for (row, actual) in rows
        .iter()
        .zip(session.read_vector(&resident.prices()).unwrap())
    {
        let expected = reference_price(row).unwrap();
        assert!(
            (actual - expected).abs() < 1e-10 + 1e-10 * expected.abs(),
            "{}: {actual} vs {expected}",
            row.row_id
        );
    }
}

#[test]
fn rejects_ambiguous_identity_and_invalid_domains() {
    let session = Session::new(ExecutionTarget::Cpu, Precision::F32, 10).unwrap();
    let mut invalid = row(OptionSide::Call);
    invalid.volatility = f64::NAN;
    assert!(ResidentOptions::upload(&session, &[invalid]).is_err());
    let mut invalid = row(OptionSide::Call);
    invalid.maturity_years = 0.0;
    assert!(ResidentOptions::upload(&session, &[invalid]).is_err());
    assert!(
        ResidentOptions::upload(&session, &[row(OptionSide::Call), row(OptionSide::Put)]).is_err()
    );
    assert!(Session::new(ExecutionTarget::Metal, Precision::F64, 10).is_err());
    assert!(ResidentOptions::upload(&session, &[]).is_err());
}

#[test]
fn resident_marks_and_signed_portfolio_match_independent_reference() {
    let target = std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap();
    let rows = volt_compute::benchmark::fixture(257);
    let session = Session::new(target, Precision::F32, rows.len()).unwrap();
    let resident = ResidentOptions::upload(&session, &rows).unwrap();
    for _ in 0..2 {
        let prices = resident.prices();
        let value = resident.portfolio_value(prices.clone());
        let actual = session.read_vector(&prices).unwrap();
        let mut expected_value = 0.0;
        let mut gross = 0.0;
        for (r, actual) in rows.iter().zip(actual) {
            let expected = reference_price(r).unwrap();
            assert!(
                (actual - expected).abs() <= 0.0002 + 0.0001 * expected.abs(),
                "{}: {actual} vs {expected}",
                r.row_id
            );
            let amount = expected * r.quantity as f64 * r.multiplier as f64;
            expected_value += amount;
            gross += amount.abs();
        }
        assert!((session.read_vector(&value).unwrap()[0] - expected_value).abs() <= 0.0001 * gross);
    }
    assert_eq!(resident.row_ids()[0], rows[0].row_id);
}
