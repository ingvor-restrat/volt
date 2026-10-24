use volt_compute::{
    ExecutionTarget, OptionRow, OptionSide, Precision, Session,
    aad::{AnalyticGreeks, analytic_greeks, differentiate_rows},
    reference_price,
};

fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}

fn row(side: OptionSide) -> OptionRow {
    OptionRow {
        row_id: "known-option".into(),
        contract_id: "known-contract".into(),
        side,
        spot: 100.0,
        strike: 100.0,
        maturity_years: 1.0,
        volatility: 0.2,
        rate: 0.05,
        dividend_yield: 0.0,
        quantity: -3,
        multiplier: 100,
    }
}

#[test]
fn analytic_values_have_declared_annual_fraction_units_and_theta_sign() {
    let call = analytic_greeks(&row(OptionSide::Call)).unwrap();
    for (actual, expected) in [
        (call.price, 10.450583572185565),
        (call.delta, 0.6368306511756191),
        (call.vega, 37.52403469169379),
        (call.rho, 53.232481545376345),
        (call.theta, -6.414027546438197),
        (call.dividend_sensitivity, -63.68306511756191),
    ] {
        assert!((actual - expected).abs() < 1e-11, "{actual} vs {expected}");
    }
    let put = analytic_greeks(&row(OptionSide::Put)).unwrap();
    assert!((put.price - 5.573526022256971).abs() < 1e-11);
    assert!((put.delta + 0.3631693488243809).abs() < 1e-11);
    assert!((put.rho + 41.89046090469506).abs() < 1e-11);
    assert!((put.theta + 1.657880423934626).abs() < 1e-11);
    assert!((put.dividend_sensitivity - 36.31693488243809).abs() < 1e-11);
}

#[test]
fn analytic_sensitivities_match_independent_central_repricing() {
    for side in [OptionSide::Call, OptionSide::Put] {
        let mut input = row(side);
        input.spot = 110.0;
        input.maturity_years = 0.5;
        input.rate = -0.005;
        input.dividend_yield = 0.02;
        input.volatility = 0.23;
        let analytic = analytic_greeks(&input).unwrap();
        type Perturbation = (fn(&mut OptionRow, f64), f64, f64);
        let derivatives: [Perturbation; 5] = [
            (|r, dx| r.spot += dx, 0.0001, analytic.delta),
            (|r, dx| r.volatility += dx, 0.00001, analytic.vega),
            (|r, dx| r.rate += dx, 0.00001, analytic.rho),
            (|r, dx| r.maturity_years += dx, 0.00001, -analytic.theta),
            (
                |r, dx| r.dividend_yield += dx,
                0.00001,
                analytic.dividend_sensitivity,
            ),
        ];
        for (perturb, step, expected) in derivatives {
            let mut upper = input.clone();
            let mut lower = input.clone();
            perturb(&mut upper, step);
            perturb(&mut lower, -step);
            let difference = (reference_price(&upper).unwrap() - reference_price(&lower).unwrap())
                / (2.0 * step);
            assert!((difference - expected).abs() < 0.000001 + 0.00000001 * expected.abs());
        }
    }
}

#[test]
fn reverse_aad_matches_analytic_calls_puts_and_dividend_parameters() {
    let target = target();
    let precisions = if target == ExecutionTarget::Metal {
        vec![Precision::F32]
    } else {
        vec![Precision::F32, Precision::F64]
    };
    let rows = volt_compute::benchmark::fixture(257);
    for precision in precisions {
        let session = Session::new(target, precision, rows.len()).unwrap();
        for _ in 0..2 {
            let result = differentiate_rows(&session, &rows).unwrap();
            assert_eq!(result.prices.len(), rows.len());
            for (index, input) in rows.iter().enumerate() {
                let expected = analytic_greeks(input).unwrap();
                let actual = AnalyticGreeks {
                    price: result.prices[index],
                    delta: result.delta[index],
                    vega: result.vega[index],
                    rho: result.rho[index],
                    theta: result.theta[index],
                    dividend_sensitivity: result.dividend_sensitivity[index],
                };
                for (name, actual, expected) in [
                    ("price", actual.price, expected.price),
                    ("delta", actual.delta, expected.delta),
                    ("vega", actual.vega, expected.vega),
                    ("rho", actual.rho, expected.rho),
                    ("theta", actual.theta, expected.theta),
                    (
                        "dividend sensitivity",
                        actual.dividend_sensitivity,
                        expected.dividend_sensitivity,
                    ),
                ] {
                    let tolerance = match precision {
                        Precision::F32 => 0.0002 + 0.0002 * expected.abs(),
                        Precision::F64 => 1e-9 + 1e-10 * expected.abs(),
                    };
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{} {name}: {actual} vs {expected} at {precision:?}",
                        input.row_id
                    );
                }
            }
        }
    }
}

#[test]
fn raw_aad_is_per_unit_and_independent_of_signed_position_quantities() {
    let mut call_long = row(OptionSide::Call);
    call_long.row_id = "long".into();
    call_long.quantity = 10;
    let mut call_short = call_long.clone();
    call_short.row_id = "short".into();
    call_short.quantity = -1_000_000;
    call_short.multiplier = 1;
    let session = Session::new(target(), Precision::F32, 2).unwrap();
    let result = differentiate_rows(&session, &[call_long, call_short]).unwrap();
    for values in [
        &result.prices,
        &result.delta,
        &result.vega,
        &result.rho,
        &result.theta,
        &result.dividend_sensitivity,
    ] {
        assert_eq!(values[0], values[1]);
    }
    assert!(result.delta[0] > 0.0);
    assert!(result.vega[0] > 0.0);
}

#[test]
fn f64_normal_density_path_preserves_tight_vega_and_theta_accuracy() {
    let session = Session::new(target(), Precision::F64, 2);
    if target() == ExecutionTarget::Metal {
        assert!(session.is_err());
        return;
    }
    let session = session.unwrap();
    let mut call = row(OptionSide::Call);
    call.row_id = "density-call".into();
    let mut put = row(OptionSide::Put);
    put.row_id = "density-put".into();
    let result = differentiate_rows(&session, &[call, put]).unwrap();
    // d1=0.35. Recover the normal density from dP/dsigma divided by S*sqrt(T).
    // This catches loss of f64 precision inside a CDF backward rule even when
    // the forward price and delta remain accurate.
    for index in 0..2 {
        assert!((result.vega[index] / 100.0 - 0.3752403469169379).abs() < 1e-12);
    }
    assert!((result.theta[0] + 6.414027546438197).abs() < 1e-10);
    assert!((result.theta[1] + 1.657880423934626).abs() < 1e-10);
}

#[test]
fn rejects_bad_batches_without_consuming_a_later_fresh_graph() {
    let session = Session::new(target(), Precision::F32, 2).unwrap();
    assert!(differentiate_rows(&session, &[]).is_err());
    let base = row(OptionSide::Call);
    assert!(differentiate_rows(&session, &[base.clone(), base.clone()]).is_err());
    let mut invalid = base.clone();
    invalid.volatility = 0.0;
    assert!(differentiate_rows(&session, &[invalid]).is_err());
    let mut invalid = base.clone();
    invalid.rate = f64::NAN;
    assert!(differentiate_rows(&session, &[invalid]).is_err());
    assert!(differentiate_rows(&session, &volt_compute::benchmark::fixture(3)).is_err());
    assert!(differentiate_rows(&session, &[base]).is_ok());
}
