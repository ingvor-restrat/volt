use volt_compute::{
    ExecutionTarget, OptionRow, OptionSide, Precision, Session,
    aad::analytic_greeks,
    monte_carlo::{GreekMoments, OnlineMoments, monte_carlo_rows, normal_at},
    second_order::analytic_gamma,
};

fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}
fn precisions() -> Vec<Precision> {
    if target() == ExecutionTarget::Metal {
        eprintln!("SKIP Monte Carlo f64: Metal has no native f64 arithmetic");
        assert!(Session::new(target(), Precision::F64, 1).is_err());
        vec![Precision::F32]
    } else {
        vec![Precision::F32, Precision::F64]
    }
}
fn rows() -> Vec<OptionRow> {
    (0..4)
        .map(|i| OptionRow {
            row_id: format!("row{i}"),
            contract_id: format!("contract{i}"),
            side: if i % 2 == 0 {
                OptionSide::Call
            } else {
                OptionSide::Put
            },
            spot: 100.0,
            strike: 95.0 + i as f64 * 3.0,
            maturity_years: 0.7,
            volatility: 0.24,
            rate: if i < 2 { 0.03 } else { -0.01 },
            dividend_yield: 0.02,
            quantity: if i % 2 == 0 { 2 } else { -1 },
            multiplier: 100,
        })
        .collect()
}
fn fields(m: &GreekMoments) -> [&OnlineMoments; 7] {
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
fn scalar_payoff(r: &OptionRow, z: f64) -> f64 {
    let terminal = r.spot
        * ((r.rate - r.dividend_yield - 0.5 * r.volatility * r.volatility) * r.maturity_years
            + r.volatility * r.maturity_years.sqrt() * z)
            .exp();
    let sign = match r.side {
        OptionSide::Call => 1.0,
        OptionSide::Put => -1.0,
    };
    (sign * (terminal - r.strike)).max(0.0) * (-r.rate * r.maturity_years).exp()
}

#[test]
fn analytic_expectations_fit_pair_uncertainty_for_independent_seeds() {
    let rows = rows();
    for precision in precisions() {
        let pairs = 8192;
        let session = Session::new(target(), precision, rows.len() * pairs * 2).unwrap();
        for seed in [7, 1301, 987654] {
            let draws = (0..pairs)
                .map(|i| normal_at(seed, i as u64))
                .collect::<Vec<_>>();
            let result = monte_carlo_rows(&session, &rows, &draws).unwrap();
            for (row, result) in rows.iter().zip(result.rows.iter()) {
                let a = analytic_greeks(row).unwrap();
                let expected = [
                    a.price,
                    a.delta,
                    a.vega,
                    a.rho,
                    a.theta,
                    a.dividend_sensitivity,
                    analytic_gamma(row).unwrap(),
                ];
                for (k, (moment, expected)) in fields(&result.moments)
                    .into_iter()
                    .zip(expected)
                    .enumerate()
                {
                    let estimate = moment.estimate();
                    let error = estimate.standard_error.unwrap();
                    assert_eq!(estimate.samples, pairs as u64);
                    assert!(error > 0.0);
                    assert!(
                        (estimate.mean - expected).abs() < 6.0 * error + 1e-4,
                        "{precision:?} seed{seed} {} Greek{k}: {} vs {expected}, SE{error}",
                        row.row_id,
                        estimate.mean
                    );
                }
            }
        }
    }
}

#[test]
fn pathwise_first_derivatives_match_fixed_draw_repricing() {
    let rows = rows();
    let draws = [-1.7, -0.83, -0.31, 0.12, 0.61, 1.31, 2.1];
    for precision in precisions() {
        let session = Session::new(target(), precision, rows.len() * draws.len() * 2).unwrap();
        let result = monte_carlo_rows(&session, &rows, &draws).unwrap();
        for (r, result) in rows.iter().zip(&result.rows) {
            type Bump = fn(&mut OptionRow, f64);
            let cases: [(Bump, f64); 5] = [
                (|r, h| r.spot += h, 1.0),
                (|r, h| r.volatility += h, 1.0),
                (|r, h| r.rate += h, 1.0),
                (|r, h| r.maturity_years += h, -1.0),
                (|r, h| r.dividend_yield += h, 1.0),
            ];
            let actual = [
                result.moments.delta.mean,
                result.moments.vega.mean,
                result.moments.rho.mean,
                result.moments.theta.mean,
                result.moments.dividend_sensitivity.mean,
            ];
            let price = |r: &OptionRow| {
                draws
                    .iter()
                    .map(|z| 0.5 * (scalar_payoff(r, *z) + scalar_payoff(r, -*z)))
                    .sum::<f64>()
                    / draws.len() as f64
            };
            for ((bump, sign), actual) in cases.into_iter().zip(actual) {
                let h = 1e-5;
                let mut high = r.clone();
                bump(&mut high, h);
                let mut low = r.clone();
                bump(&mut low, -h);
                let expected = sign * (price(&high) - price(&low)) / (2.0 * h);
                let tolerance = if precision == Precision::F32 {
                    2e-5 + 2e-5 * expected.abs()
                } else {
                    1e-7
                };
                assert!(
                    (actual - expected).abs() < tolerance,
                    "{precision:?}: {actual} vs FD{expected}"
                );
            }
        }
    }
}

#[test]
fn pair_errors_gamma_weights_and_portfolio_covariance_are_explicit() {
    let mut rows = rows();
    rows.truncate(2);
    rows[1] = rows[0].clone();
    rows[1].row_id = "cancel".into();
    rows[1].quantity = -rows[0].quantity;
    let draws = [-1.7, -0.3, 0.7, 1.2, 2.4];
    for precision in precisions() {
        let session = Session::new(target(), precision, rows.len() * draws.len() * 2).unwrap();
        let result = monte_carlo_rows(&session, &rows, &draws).unwrap();
        let mut expected_price = OnlineMoments::default();
        let mut expected_gamma = OnlineMoments::default();
        let r = &rows[0];
        let sigma_t = r.volatility * r.maturity_years.sqrt();
        let score = |z: f64| {
            (z * z - 1.0) / (r.spot * r.spot * sigma_t * sigma_t) - z / (r.spot * r.spot * sigma_t)
        };
        for z in draws {
            expected_price.add(0.5 * (scalar_payoff(r, z) + scalar_payoff(r, -z)));
            expected_gamma
                .add(0.5 * (scalar_payoff(r, z) * score(z) + scalar_payoff(r, -z) * score(-z)));
        }
        for (actual, expected) in [
            (&result.rows[0].moments.price, expected_price),
            (&result.rows[0].moments.gamma, expected_gamma),
        ] {
            let tolerance = if precision == Precision::F32 {
                2e-4
            } else {
                1e-10
            };
            assert!((actual.mean - expected.mean).abs() < tolerance);
            assert!(
                (actual.estimate().standard_error.unwrap()
                    - expected.estimate().standard_error.unwrap())
                .abs()
                    < tolerance
            );
        }
        for moment in fields(&result.portfolio) {
            assert_eq!(moment.mean, 0.0);
            assert_eq!(moment.m2, 0.0);
        }
    }
}

#[test]
fn indexed_stream_and_merge_preserve_pairs_across_chunks() {
    let draws = (0..37).map(|i| normal_at(771, i)).collect::<Vec<_>>();
    assert!(draws.iter().all(|z| z.is_finite()));
    assert_eq!(draws[23], normal_at(771, 23));
    let rows = rows();
    for precision in precisions() {
        let session = Session::new(target(), precision, rows.len() * draws.len() * 2).unwrap();
        let full = monte_carlo_rows(&session, &rows, &draws).unwrap();
        let mut combined = GreekMoments::default();
        for chunk in draws.chunks(7) {
            combined.merge(&monte_carlo_rows(&session, &rows, chunk).unwrap().portfolio);
        }
        for (a, b) in fields(&combined).into_iter().zip(fields(&full.portfolio)) {
            assert_eq!(a.count, b.count);
            let tolerance = if precision == Precision::F32 {
                2e-4
            } else {
                1e-10
            };
            assert!((a.mean - b.mean).abs() < tolerance * (1.0 + b.mean.abs()));
            assert!((a.m2 - b.m2).abs() < tolerance * (1.0 + b.m2.abs()));
        }
    }
    let mut one = OnlineMoments::default();
    one.add(2.0);
    assert!(one.estimate().standard_error.is_none());
    one.add(4.0);
    assert_eq!(one.estimate().standard_error, Some(1.0));
}

#[test]
fn bad_draws_and_expanded_caps_fail_without_poisoning_device() {
    let rows = rows();
    let session = Session::new(target(), Precision::F32, 16).unwrap();
    assert!(monte_carlo_rows(&session, &rows, &[]).is_err());
    assert!(monte_carlo_rows(&session, &rows, &[0.1, 0.2, 0.3]).is_err());
    assert!(monte_carlo_rows(&session, &rows, &[f64::NAN]).is_err());
    assert!(monte_carlo_rows(&session, &[rows[0].clone(), rows[0].clone()], &[0.1]).is_err());
    assert!(monte_carlo_rows(&session, &rows, &[0.1, 0.3]).is_ok());
}
