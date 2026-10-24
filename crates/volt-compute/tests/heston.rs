use volt_compute::heston::{
    HestonOption, HestonParams, QuadratureConfig, differentiate_rows, price_rows, validate_inputs,
};
use volt_compute::{ExecutionTarget, OptionRow, OptionSide, Precision, Session, reference_price};

fn target() -> ExecutionTarget {
    std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap()
}
fn precisions() -> Vec<Precision> {
    if target() == ExecutionTarget::Metal {
        vec![Precision::F32]
    } else {
        vec![Precision::F32, Precision::F64]
    }
}
fn option(id: &str, side: OptionSide) -> HestonOption {
    HestonOption {
        row_id: id.into(),
        side,
        spot: 100.0,
        strike: 100.0,
        maturity_years: 1.0,
        rate: 0.01,
        dividend_yield: 0.02,
    }
}
fn params() -> HestonParams {
    HestonParams {
        initial_variance: 0.04,
        kappa: 4.0,
        long_run_variance: 0.25,
        vol_of_variance: 1.0,
        correlation: -0.5,
    }
}
fn quad() -> QuadratureConfig {
    QuadratureConfig {
        nodes: 256,
        upper_bound: 160.0,
    }
}
fn close(x: f64, y: f64, abs: f64, rel: f64) {
    assert!(
        (x - y).abs() <= abs + rel * y.abs(),
        "{x:.15e} vs {y:.15e}; difference {:.5e}",
        (x - y).abs()
    );
}

/// Independent published fixtures: QuantLib hestonmodel.cpp
/// testAlanLewisReferencePrices at revision
/// e96579fd471b2e9f6bc974c96674d1ccfff248fa. These constants were not generated
/// by this tensor graph. The official test checks relative accuracy 1e-12.
#[test]
fn published_prices_calls_puts_and_nonuniform_row_order() {
    for precision in precisions() {
        let session = Session::new(target(), precision, 4096).unwrap();
        let fixtures = [
            (80.0, OptionSide::Call, 26.774758743998845),
            (100.0, OptionSide::Put, 17.055_270_961_270_11),
            (100.0, OptionSide::Call, 16.070154917028834),
            (80.0, OptionSide::Put, 7.958878113256771),
        ];
        let options: Vec<_> = fixtures
            .iter()
            .enumerate()
            .map(|(i, (strike, side, _))| {
                let mut o = option(&format!("row-{i}"), *side);
                o.strike = *strike;
                o
            })
            .collect();
        let values =
            price_rows(&session, &options, &vec![params(); options.len()], quad()).unwrap();
        for (actual, (_, _, expected)) in values.iter().zip(fixtures) {
            let abs = if precision == Precision::F64 {
                2e-10
            } else {
                6e-5
            };
            close(*actual, expected, abs, 0.0);
        }
    }
}

/// QuantLib1.40 Gatheral adaptive-Lobatto bumped controls, checked against
/// independent exp-sinh integration and half-step Richardson extrapolation.
#[test]
fn reverse_parameter_greeks_match_independent_adaptive_quantlib_controls() {
    for precision in precisions() {
        let session = Session::new(target(), precision, 4096).unwrap();
        let options = [
            option("call", OptionSide::Call),
            option("put", OptionSide::Put),
        ];
        let a = differentiate_rows(&session, &options, &[params(); 2], quad()).unwrap();
        let expected = [
            (a.delta[0], 0.5994713257509925),
            (a.gamma[0], 0.009287538421689749),
            (a.initial_variance_sensitivity[0], 10.179265404511439),
            (a.kappa_sensitivity[0], 0.6898541767442387),
            (a.long_run_variance_sensitivity[0], 32.560133473875844),
            (a.vol_of_variance_sensitivity[0], -1.11317149811738),
            (a.correlation_sensitivity[0], 0.9498327391839941),
            (a.rho[0], 43.87697765795471),
            (a.dividend_sensitivity[0], -59.94713257518766),
        ];
        for (x, y) in expected {
            if precision == Precision::F64 {
                close(x, y, 3e-8, 1e-9)
            } else {
                close(x, y, 1e-4, 8e-6)
            }
        }
        let dd = (-0.02f64).exp();
        let kd = 100.0 * (-0.01f64).exp();
        let tol = if precision == Precision::F64 {
            2e-11
        } else {
            8e-5
        };
        close(a.prices[0] - a.prices[1], 100.0 * dd - kd, tol, 0.0);
        close(a.delta[0] - a.delta[1], dd, tol, 0.0);
        close(a.rho[0] - a.rho[1], kd, tol, 0.0);
        close(
            a.dividend_sensitivity[0] - a.dividend_sensitivity[1],
            -100.0 * dd,
            tol,
            0.0,
        );
        close(a.gamma[0], a.gamma[1], tol, 0.0);
    }
}

#[test]
fn first_derivatives_and_fourier_gamma_match_bumped_same_quadrature() {
    // f64 is necessary for tight finite differences; Metal has no f64 tensors.
    if target() == ExecutionTarget::Metal {
        return;
    }
    let session = Session::new(target(), Precision::F64, 4096).unwrap();
    let o = option("call", OptionSide::Call);
    let p = params();
    let a = differentiate_rows(&session, std::slice::from_ref(&o), &[p], quad()).unwrap();
    type Bump = (fn(&mut HestonOption, &mut HestonParams, f64), f64, f64);
    let cases: [Bump; 9] = [
        (|o, _, h| o.spot += h, 1e-3, a.delta[0]),
        (|o, _, h| o.maturity_years += h, 1e-5, -a.theta[0]),
        (|o, _, h| o.rate += h, 1e-5, a.rho[0]),
        (
            |o, _, h| o.dividend_yield += h,
            1e-5,
            a.dividend_sensitivity[0],
        ),
        (
            |_, p, h| p.initial_variance += h,
            1e-5,
            a.initial_variance_sensitivity[0],
        ),
        (|_, p, h| p.kappa += h, 1e-4, a.kappa_sensitivity[0]),
        (
            |_, p, h| p.long_run_variance += h,
            1e-5,
            a.long_run_variance_sensitivity[0],
        ),
        (
            |_, p, h| p.vol_of_variance += h,
            1e-5,
            a.vol_of_variance_sensitivity[0],
        ),
        (
            |_, p, h| p.correlation += h,
            1e-5,
            a.correlation_sensitivity[0],
        ),
    ];
    for (bump, h, expected) in cases {
        let (mut up, mut down) = (o.clone(), o.clone());
        let (mut pu, mut pd) = (p, p);
        bump(&mut up, &mut pu, h);
        bump(&mut down, &mut pd, -h);
        let upper = price_rows(&session, &[up], &[pu], quad()).unwrap()[0];
        let lower = price_rows(&session, &[down], &[pd], quad()).unwrap()[0];
        close((upper - lower) / (2.0 * h), expected, 4e-7, 2e-8);
    }
    let mut up = o.clone();
    let mut down = o.clone();
    let h = 0.02;
    up.spot += h;
    down.spot -= h;
    let upper = price_rows(&session, &[up], &[p], quad()).unwrap()[0];
    let lower = price_rows(&session, &[down], &[p], quad()).unwrap()[0];
    close(
        (upper + lower - 2.0 * a.prices[0]) / (h * h),
        a.gamma[0],
        3e-9,
        1e-7,
    );
}

#[test]
fn paper_parameters_require_independent_cutoff_and_node_refinement() {
    for precision in precisions() {
        let session = Session::new(target(), precision, 4096).unwrap();
        let mut o = option("paper", OptionSide::Call);
        o.rate = 0.0;
        o.dividend_yield = 0.0;
        o.maturity_years = 30.0 / 365.0;
        let p = HestonParams {
            initial_variance: 0.04,
            kappa: 1.0,
            long_run_variance: 0.04,
            vol_of_variance: 2.0,
            correlation: -0.7,
        };
        // Independent refinement directions; not just repeated evaluation.
        let configurations = [
            QuadratureConfig {
                nodes: 1024,
                upper_bound: 2000.0,
            },
            QuadratureConfig {
                nodes: 2048,
                upper_bound: 2000.0,
            },
            QuadratureConfig {
                nodes: 2048,
                upper_bound: 4000.0,
            },
        ];
        let mut last = None;
        for q in configurations {
            let a = differentiate_rows(&session, std::slice::from_ref(&o), &[p], q).unwrap();
            let abs = if precision == Precision::F64 {
                3e-9
            } else {
                2e-4
            };
            for (x, y) in [
                (a.prices[0], 1.691833925110386),
                (a.delta[0], 0.6960070233948596),
                (a.gamma[0], 0.11217017223716684),
                (a.initial_variance_sensitivity[0], 28.449951907626616),
                (a.rho[0], 5.581550828511404),
            ] {
                close(
                    x,
                    y,
                    abs,
                    if precision == Precision::F64 {
                        1e-9
                    } else {
                        1e-5
                    },
                );
            }
            if let Some(previous) = last {
                close(a.prices[0], previous, abs, 0.0);
            }
            last = Some(a.prices[0]);
        }
    }
}

#[test]
fn vanishing_vol_of_variance_converges_to_independent_deterministic_variance_bs() {
    if target() == ExecutionTarget::Metal {
        return;
    }
    let session = Session::new(target(), Precision::F64, 4096).unwrap();
    let o = option("limit", OptionSide::Call);
    let mut p = HestonParams {
        initial_variance: 0.09,
        kappa: 2.0,
        long_run_variance: 0.04,
        vol_of_variance: 0.01,
        correlation: 0.0,
    };
    let total_variance = p.long_run_variance * o.maturity_years
        + (p.initial_variance - p.long_run_variance) * (-(-p.kappa * o.maturity_years).exp_m1())
            / p.kappa;
    let bs = OptionRow {
        row_id: "reference".into(),
        contract_id: "reference".into(),
        side: o.side,
        spot: o.spot,
        strike: o.strike,
        maturity_years: o.maturity_years,
        volatility: (total_variance / o.maturity_years).sqrt(),
        rate: o.rate,
        dividend_yield: o.dividend_yield,
        quantity: 1,
        multiplier: 1,
    };
    let expected = reference_price(&bs).unwrap();
    let mut previous = f64::INFINITY;
    for xi in [0.01, 0.001, 0.0001] {
        p.vol_of_variance = xi;
        let actual = price_rows(&session, std::slice::from_ref(&o), &[p], quad()).unwrap()[0];
        let error = (actual - expected).abs();
        assert!(
            error < previous * 0.02,
            "{xi}: error {error}, previous {previous}"
        );
        previous = error;
    }
    assert!(previous < 1e-7, "limit error {previous}");
}

#[test]
fn validation_rejects_identity_shape_domain_and_allows_fresh_recovery() {
    let precision = Precision::F32;
    let o = option("a", OptionSide::Call);
    let p = params();
    assert!(validate_inputs(&[], &[], quad(), precision).is_err());
    assert!(validate_inputs(&[o.clone(), o.clone()], &[p, p], quad(), precision).is_err());
    assert!(validate_inputs(std::slice::from_ref(&o), &[], quad(), precision).is_err());
    for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut bad = p;
        bad.vol_of_variance = value;
        assert!(validate_inputs(std::slice::from_ref(&o), &[bad], quad(), precision).is_err());
    }
    let mut bad = p;
    bad.correlation = 1.0;
    assert!(validate_inputs(std::slice::from_ref(&o), &[bad], quad(), precision).is_err());
    let mut bad = o.clone();
    bad.spot = f64::MAX;
    assert!(validate_inputs(&[bad], &[p], quad(), precision).is_err());
    for q in [
        QuadratureConfig {
            nodes: 15,
            upper_bound: 100.0,
        },
        QuadratureConfig {
            nodes: 2049,
            upper_bound: 100.0,
        },
        QuadratureConfig {
            nodes: 256,
            upper_bound: f64::NAN,
        },
    ] {
        assert!(validate_inputs(std::slice::from_ref(&o), &[p], q, precision).is_err());
    }
    let small = Session::new(target(), precision, 255).unwrap();
    assert!(price_rows(&small, std::slice::from_ref(&o), &[p], quad()).is_err());
    let good = Session::new(target(), precision, 4096).unwrap();
    for _ in 0..2 {
        let result = differentiate_rows(&good, std::slice::from_ref(&o), &[p], quad()).unwrap();
        close(result.prices[0], 16.070154917028834, 6e-5, 0.0);
    }
}

/// Positive correlation with kappa < rho*xi probes the complex-log branch;
/// neither the Feller condition nor kappa>rho*xi is a supported-domain shortcut.
/// Independent QuantLib adaptive Lobatto and exp-sinh controls agree to 2e-13.
#[test]
fn positive_correlation_low_kappa_and_long_maturity_avoid_log_branch_trap() {
    for precision in precisions() {
        let session = Session::new(target(), precision, 4096).unwrap();
        let p = HestonParams {
            initial_variance: 0.04,
            kappa: 0.3,
            long_run_variance: 0.04,
            vol_of_variance: 1.5,
            correlation: 0.8,
        };
        for (time, price, delta, gamma, v0) in [
            (
                1.0,
                3.8187291781568242,
                0.1934099121836219,
                0.03500489518197073,
                66.95624539161335,
            ),
            (
                10.0,
                10.282129834723825,
                0.16092183435187185,
                0.0041131683505094925,
                62.20600634314187,
            ),
        ] {
            let mut o = option("positive-corr", OptionSide::Call);
            o.maturity_years = time;
            let q = QuadratureConfig {
                nodes: 2048,
                upper_bound: 4000.0,
            };
            let a = differentiate_rows(&session, &[o], &[p], q).unwrap();
            for (x, y) in [
                (a.prices[0], price),
                (a.delta[0], delta),
                (a.gamma[0], gamma),
                (a.initial_variance_sensitivity[0], v0),
            ] {
                if precision == Precision::F64 {
                    close(x, y, 3e-8, 1e-9);
                } else {
                    close(x, y, 2e-4, 1e-5);
                }
            }
        }
    }
}

/// QuantLib's HestonModel rejects exact V0=0, so the independent boundary
/// controls are extrapolations of strictly positive-V0 prices, checked with
/// adaptive Lobatto and exp-sinh. First-variance-control uncertainty is 3e-6
/// (paper case) / 2e-7 (ordinary case), and gamma uncertainty is a few e-6 /
/// 1e-7; these are separate from the tighter internal quadrature checks below.
#[test]
fn zero_current_variance_matches_independent_right_limits_and_refines_tail() {
    for precision in precisions() {
        let session = Session::new(target(), precision, 4096).unwrap();
        let ordinary = option("ordinary-zero", OptionSide::Call);
        let mut ordinary_params = params();
        ordinary_params.initial_variance = 0.0;
        let mut paper = option("paper-zero", OptionSide::Call);
        paper.maturity_years = 30.0 / 365.0;
        paper.rate = 0.0;
        paper.dividend_yield = 0.0;
        let paper_params = HestonParams {
            initial_variance: 0.0,
            kappa: 1.0,
            long_run_variance: 0.04,
            vol_of_variance: 2.0,
            correlation: -0.7,
        };
        let configurations = [
            QuadratureConfig {
                nodes: 1024,
                upper_bound: 20000.0,
            },
            QuadratureConfig {
                nodes: 2048,
                upper_bound: 20000.0,
            },
            QuadratureConfig {
                nodes: 2048,
                upper_bound: 40000.0,
            },
        ];
        for (o, p, price, delta, gamma, v0) in [
            (
                ordinary,
                ordinary_params,
                15.6580294201670,
                0.5975699110186,
                0.009545653,
                10.42968647,
            ),
            (
                paper,
                paper_params,
                0.1929794545818,
                0.728777145409575,
                1.41709809,
                57.9092034,
            ),
        ] {
            let mut previous: Option<[f64; 4]> = None;
            for q in configurations {
                let result =
                    differentiate_rows(&session, std::slice::from_ref(&o), &[p], q).unwrap();
                let actual = [
                    result.prices[0],
                    result.delta[0],
                    result.gamma[0],
                    result.initial_variance_sensitivity[0],
                ];
                if precision == Precision::F64 {
                    close(actual[0], price, 3e-10, 0.0);
                    close(actual[1], delta, 3e-9, 0.0);
                    close(actual[2], gamma, 3e-6, 0.0);
                    close(actual[3], v0, 3e-6, 0.0);
                    if let Some(previous) = previous {
                        for (x, y) in actual.into_iter().zip(previous) {
                            close(x, y, 3e-9, 1e-10);
                        }
                    }
                } else {
                    for (x, y) in actual.into_iter().zip([price, delta, gamma, v0]) {
                        close(x, y, 2e-4, 1e-5);
                    }
                    if let Some(previous) = previous {
                        for (x, y) in actual.into_iter().zip(previous) {
                            close(x, y, 2e-4, 1e-5);
                        }
                    }
                }
                previous = Some(actual);
            }
            let mut negative = p;
            negative.initial_variance = -f64::MIN_POSITIVE;
            assert!(
                validate_inputs(std::slice::from_ref(&o), &[negative], quad(), precision).is_err()
            );
        }
    }
}

#[test]
fn zero_variance_right_derivative_and_near_zero_risk_are_continuous() {
    if target() == ExecutionTarget::Metal {
        return;
    }
    let session = Session::new(target(), Precision::F64, 4096).unwrap();
    let mut o = option("paper-zero", OptionSide::Call);
    o.maturity_years = 30.0 / 365.0;
    o.rate = 0.0;
    o.dividend_yield = 0.0;
    let p = HestonParams {
        initial_variance: 0.0,
        kappa: 1.0,
        long_run_variance: 0.04,
        vol_of_variance: 2.0,
        correlation: -0.7,
    };
    let q = QuadratureConfig {
        nodes: 2048,
        upper_bound: 40000.0,
    };
    let zero = differentiate_rows(&session, std::slice::from_ref(&o), &[p], q).unwrap();
    let at = |variance: f64| {
        let mut p = p;
        p.initial_variance = variance;
        price_rows(&session, std::slice::from_ref(&o), &[p], q).unwrap()[0]
    };
    let h = 1e-6;
    let coarse = (at(h) - zero.prices[0]) / h;
    let fine = (at(h * 0.5) - zero.prices[0]) / (h * 0.5);
    close(
        2.0 * fine - coarse,
        zero.initial_variance_sensitivity[0],
        3e-6,
        1e-8,
    );
    let mut near_params = p;
    near_params.initial_variance = 1e-8;
    let near = differentiate_rows(&session, &[o], &[near_params], q).unwrap();
    close(
        near.prices[0],
        zero.prices[0] + 1e-8 * zero.initial_variance_sensitivity[0],
        1e-10,
        0.0,
    );
    close(
        near.initial_variance_sensitivity[0],
        zero.initial_variance_sensitivity[0],
        1e-4,
        0.0,
    );
    close(near.delta[0], zero.delta[0], 1e-6, 0.0);
    close(near.gamma[0], zero.gamma[0], 1e-4, 0.0);
}

/// Independent QuantLib adaptive-Lobatto right-domain extrapolations (the model
/// rejects exact V0=0), saved in heston-one-day-boundary-controls.json. The spot
/// derivatives are at V0=1e-10, so their positive-variance shift is included in
/// the 3e-8 delta bound. Right variance controls are noisy at the 1e-4..5e-4
/// level; price/node/cutoff convergence remains checked independently and tightly.
#[test]
fn one_day_zero_variance_near_atm_resolves_extended_tail_and_right_sensitivity() {
    for precision in precisions() {
        let session = Session::new(target(), precision, 4096).unwrap();
        let mut o = option("one-day-zero", OptionSide::Call);
        o.maturity_years = 1.0 / 365.0;
        o.rate = 0.0;
        o.dividend_yield = 0.0;
        let p = HestonParams {
            initial_variance: 0.0,
            kappa: 1.0,
            long_run_variance: 0.04,
            vol_of_variance: 2.0,
            correlation: -0.7,
        };
        for (spot, price, delta, cv, cv_uncertainty) in [
            (99.9, 0.0002119979366, 0.00389210064, 5.47632, 5e-4),
            (100.0, 0.0065178659884, 0.7285337304, 59.448183, 1e-4),
            (100.1, 0.1019144476667, 0.9855145440, 30.67944, 5e-4),
        ] {
            o.spot = spot;
            let mut previous: Option<[f64; 4]> = None;
            for (nodes, upper_bound) in [(1024, 1e6), (2048, 1e6), (2048, 2e6)] {
                let a = differentiate_rows(
                    &session,
                    std::slice::from_ref(&o),
                    &[p],
                    QuadratureConfig { nodes, upper_bound },
                )
                .unwrap();
                if precision == Precision::F64 {
                    close(a.prices[0], price, 3e-10, 0.0);
                    close(a.delta[0], delta, 3e-8, 0.0);
                    close(a.initial_variance_sensitivity[0], cv, cv_uncertainty, 0.0);
                } else {
                    close(a.prices[0], price, 6e-5, 0.0);
                    close(a.delta[0], delta, 2e-5, 0.0);
                    close(a.initial_variance_sensitivity[0], cv, 1e-3, 1e-5);
                }
                let actual = [
                    a.prices[0],
                    a.delta[0],
                    a.gamma[0],
                    a.initial_variance_sensitivity[0],
                ];
                if let Some(previous) = previous {
                    for (x, y) in actual.into_iter().zip(previous) {
                        if precision == Precision::F64 {
                            close(x, y, 3e-9, 1e-10);
                        } else {
                            close(x, y, 6e-5, 1e-5);
                        }
                    }
                }
                previous = Some(actual);
            }
        }
    }
}

/// A longer tail is not a global accuracy certificate: for far-moneyness the
/// oscillatory integrand is underresolved by this fixed node budget. This is an
/// explicit stress fixture for consumers' mandatory convergence rejection, not
/// an assertion that any of the returned Greeks is an acceptable reference.
#[test]
fn oscillatory_far_moneyness_exposes_failed_refinement_at_fixed_node_cap() {
    if target() == ExecutionTarget::Metal {
        return;
    }
    let session = Session::new(target(), Precision::F64, 4096).unwrap();
    let mut o = option("one-day-oscillation", OptionSide::Call);
    o.spot = 99.0;
    o.maturity_years = 1.0 / 365.0;
    o.rate = 0.0;
    o.dividend_yield = 0.0;
    let p = HestonParams {
        initial_variance: 0.0,
        kappa: 1.0,
        long_run_variance: 0.04,
        vol_of_variance: 2.0,
        correlation: -0.7,
    };
    let at = |nodes, upper_bound| {
        differentiate_rows(
            &session,
            std::slice::from_ref(&o),
            &[p],
            QuadratureConfig { nodes, upper_bound },
        )
        .unwrap()
    };
    let coarse = at(1024, 1e6);
    let fine = at(2048, 1e6);
    let tail = at(2048, 2e6);
    assert!(
        (coarse.delta[0] - fine.delta[0]).abs() > 1e-4,
        "fixture must expose insufficient nodes before certifying this range"
    );
    assert!(
        (fine.delta[0] - tail.delta[0]).abs() > 1e-4,
        "fixture must expose insufficient oscillatory resolution on cutoff refinement"
    );
}
