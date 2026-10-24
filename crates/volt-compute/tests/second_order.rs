use volt_compute::{
    ExecutionTarget, OptionRow, OptionSide, Precision, Session,
    aad::{analytic_greeks, differentiate_rows},
    second_order::spot_second_order,
};

fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}

fn precisions() -> Vec<Precision> {
    if target() == ExecutionTarget::Metal {
        eprintln!("SKIP second-order f64: Metal has no native f64 arithmetic");
        assert!(Session::new(target(), Precision::F64, 1).is_err());
        vec![Precision::F32]
    } else {
        vec![Precision::F32, Precision::F64]
    }
}

// Deliberately independent of the jet's chain rules and tensor pricing graph.
fn scalar_gamma(row: &OptionRow) -> f64 {
    let root_time = row.maturity_years.sqrt();
    let d1 = ((row.spot / row.strike).ln()
        + (row.rate - row.dividend_yield + row.volatility.powi(2) / 2.0) * row.maturity_years)
        / (row.volatility * root_time);
    (-row.dividend_yield * row.maturity_years - d1 * d1 / 2.0).exp()
        / ((2.0 * std::f64::consts::PI).sqrt() * row.spot * row.volatility * root_time)
}

#[test]
fn forward_jet_matches_independent_prices_delta_gamma_across_shapes() {
    for precision in precisions() {
        for length in [1, 2, 17, 257] {
            let mut rows = volt_compute::benchmark::fixture(length);
            // Include a narrow, near-expiry peak where gamma is appreciable.
            rows[0].spot = 100.0;
            rows[0].strike = 100.0;
            rows[0].maturity_years = 0.002;
            rows[0].volatility = 0.1;
            let session = Session::new(target(), precision, length).unwrap();
            let result = spot_second_order(&session, &rows).unwrap();
            assert_eq!(result.prices.len(), length);
            assert_eq!(result.delta.len(), length);
            assert_eq!(result.gamma.len(), length);
            for (i, row) in rows.iter().enumerate() {
                let expected = analytic_greeks(row).unwrap();
                for (name, actual, expected) in [
                    ("price", result.prices[i], expected.price),
                    ("delta", result.delta[i], expected.delta),
                    ("gamma", result.gamma[i], scalar_gamma(row)),
                ] {
                    let tolerance = match precision {
                        Precision::F32 => 3e-5 + 3e-4 * expected.abs(),
                        Precision::F64 => 1e-10 + 1e-9 * expected.abs(),
                    };
                    assert!(
                        (actual - expected).abs() < tolerance,
                        "row {} {name}, shape {length}, {precision:?}: {actual} vs {expected}",
                        row.row_id
                    );
                }
            }
        }
    }
}

#[test]
fn gamma_agrees_with_central_difference_of_reverse_aad_delta() {
    for precision in precisions() {
        let mut rows = volt_compute::benchmark::fixture(17);
        for row in &mut rows {
            row.spot = 100.0;
            row.strike = 100.0;
            row.maturity_years = 0.7;
            row.volatility = 0.23;
            row.dividend_yield = 0.017;
        }
        let step = match precision {
            // Exactly representable spot bump reduces subtraction noise in the
            // f32 delta control while keeping central truncation below tolerance.
            Precision::F32 => 0.25,
            Precision::F64 => 0.0001,
        };
        let mut upper = rows.clone();
        let mut lower = rows.clone();
        for row in &mut upper {
            row.spot += step;
        }
        for row in &mut lower {
            row.spot -= step;
        }
        let session = Session::new(target(), precision, rows.len()).unwrap();
        let result = spot_second_order(&session, &rows).unwrap();
        let upper_delta = differentiate_rows(&session, &upper).unwrap().delta;
        let lower_delta = differentiate_rows(&session, &lower).unwrap().delta;
        for i in 0..rows.len() {
            let finite_difference = (upper_delta[i] - lower_delta[i]) / (2.0 * step);
            let tolerance = match precision {
                Precision::F32 => 2e-6,
                Precision::F64 => 1e-10,
            };
            assert!(
                (result.gamma[i] - finite_difference).abs() < tolerance,
                "{precision:?} gamma {} vs finite difference {finite_difference}",
                result.gamma[i]
            );
        }
    }
}

#[test]
fn call_put_gamma_and_signed_positions_preserve_per_unit_semantics() {
    let mut rows = volt_compute::benchmark::fixture(3);
    rows[1] = rows[0].clone();
    rows[1].row_id = "same-call-short".into();
    rows[1].quantity = -100_000;
    rows[1].multiplier = 1;
    rows[0].side = OptionSide::Call;
    rows[1].side = OptionSide::Call;
    rows[2] = rows[0].clone();
    rows[2].row_id = "same-put".into();
    rows[2].side = OptionSide::Put;
    for precision in precisions() {
        let session = Session::new(target(), precision, 3).unwrap();
        let result = spot_second_order(&session, &rows).unwrap();
        assert_eq!(result.prices[0], result.prices[1]);
        assert_eq!(result.delta[0], result.delta[1]);
        assert_eq!(result.gamma[0], result.gamma[1]);
        let tolerance = if precision == Precision::F32 {
            2e-6
        } else {
            1e-12
        };
        assert!((result.gamma[0] - result.gamma[2]).abs() < tolerance);
        assert!(result.gamma[0] > 0.0);
    }
}

#[test]
fn invalid_batches_are_rejected_before_a_subsequent_valid_run() {
    let session = Session::new(target(), Precision::F32, 2).unwrap();
    let rows = volt_compute::benchmark::fixture(3);
    assert!(spot_second_order(&session, &[]).is_err());
    assert!(spot_second_order(&session, &rows).is_err());
    assert!(spot_second_order(&session, &[rows[0].clone(), rows[0].clone()]).is_err());
    let mut invalid = rows[0].clone();
    invalid.maturity_years = 0.0;
    assert!(spot_second_order(&session, &[invalid]).is_err());
    assert!(spot_second_order(&session, &rows[..2]).is_ok());
}
