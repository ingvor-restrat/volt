use volt_compute::{OptionSide, volatility_surface::*};
fn input(side: OptionSide) -> EuropeanInputs {
    EuropeanInputs {
        spot: 100.0,
        strike: 100.0,
        maturity_years: 0.5,
        rate: 0.03,
        dividend_yield: 0.01,
        side,
    }
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "{a:e} != {b:e}; diff={:e}",
        (a - b).abs()
    );
}
// Separate direct call/put scalar formula, not the module's OTM/parity path.
fn reference(i: &EuropeanInputs, v: f64) -> f64 {
    let z = v * i.maturity_years.sqrt();
    let d1 = ((i.spot / i.strike).ln()
        + (i.rate - i.dividend_yield + 0.5 * v * v) * i.maturity_years)
        / z;
    let d2 = d1 - z;
    let n = |x: f64| 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2);
    let a = i.spot * (-i.dividend_yield * i.maturity_years).exp();
    let b = i.strike * (-i.rate * i.maturity_years).exp();
    match i.side {
        OptionSide::Call => a * n(d1) - b * n(d2),
        OptionSide::Put => b * n(-d2) - a * n(-d1),
    }
}
fn chain() -> ChainConfig {
    ChainConfig {
        cutoff_ns: 100,
        max_age_ns: 20,
        max_quotes: 1000,
        max_input_bytes: 1_000_000,
        iv: IvConfig::default(),
    }
}
fn row(id: usize, k: f64, t: f64, v: f64, side: OptionSide) -> QuoteRow {
    let i = EuropeanInputs {
        strike: k,
        maturity_years: t,
        side,
        ..input(side)
    };
    let p = reference(&i, v);
    QuoteRow {
        quote_id: format!("quote-{id}"),
        contract_id: format!("contract-{id}"),
        source_id: "example-source".into(),
        underlying_id: "example-stock".into(),
        timestamp_ns: 99,
        input: i,
        bid: p - 0.00001,
        ask: p + 0.00001,
    }
}
fn grid(v: f64) -> Vec<QuoteRow> {
    let mut rows = Vec::new();
    for t in [0.5, 1.0] {
        for k in [80.0, 100.0, 120.0] {
            for side in [OptionSide::Call, OptionSide::Put] {
                rows.push(row(rows.len(), k, t, v, side));
            }
        }
    }
    rows
}
fn fitted(rows: &[QuoteRow]) -> VolatilitySurface {
    fit_surface(
        &process_chain(rows, &chain()).unwrap(),
        &SurfaceConfig::default(),
    )
    .unwrap()
}
#[test]
fn known_european_roots_reprice_and_obey_carry_parity() {
    let c = input(OptionSide::Call);
    let (price, vega) = european_price_and_vega(&c, 0.2).unwrap();
    close(price, 6.090127223710589, 2e-14);
    close(vega, 27.789494747628723, 2e-14);
    for side in [OptionSide::Call, OptionSide::Put] {
        for v in [0.05, 0.2, 0.7, 2.0] {
            let i = EuropeanInputs { side, ..c };
            let p = reference(&i, v);
            let root = implied_volatility(&i, p, &IvConfig::default()).unwrap();
            assert_eq!(root.status, IvStatus::Converged);
            close(root.volatility.unwrap(), v, 2e-10);
            assert!(root.price_residual.unwrap().abs() < 1e-9);
        }
    }
    let put = european_price_and_vega(
        &EuropeanInputs {
            side: OptionSide::Put,
            ..c
        },
        0.2,
    )
    .unwrap()
    .0;
    close(
        price - put,
        100.0 * (-0.005f64).exp() - 100.0 * (-0.015f64).exp(),
        1e-14,
    );
}
#[test]
fn boundaries_expiry_malformed_and_nonconvergence_are_explicit() {
    let i = input(OptionSide::Call);
    let cfg = IvConfig::default();
    let (lo, hi) = i.bounds().unwrap();
    for (p, s) in [
        (lo, IvStatus::LowerBoundary),
        (hi, IvStatus::UpperBoundary),
        (lo - 1e-12, IvStatus::OutsideBounds),
        (hi + 1.0, IvStatus::OutsideBounds),
    ] {
        assert_eq!(implied_volatility(&i, p, &cfg).unwrap().status, s);
    }
    let expired = EuropeanInputs {
        maturity_years: 0.0,
        ..i
    };
    assert_eq!(
        implied_volatility(&expired, 0.0, &cfg).unwrap().status,
        IvStatus::Expired
    );
    assert_eq!(
        implied_volatility(&expired, 1.0, &cfg).unwrap().status,
        IvStatus::OutsideBounds
    );
    assert!(
        implied_volatility(
            &EuropeanInputs {
                spot: f64::NAN,
                ..i
            },
            1.0,
            &cfg
        )
        .is_err()
    );
    assert!(implied_volatility(&i, f64::NAN, &cfg).is_err());
    assert!(
        implied_volatility(
            &i,
            6.0,
            &IvConfig {
                max_iterations: 0,
                ..cfg.clone()
            }
        )
        .is_err()
    );
    assert!(implied_volatility(&EuropeanInputs { rate: -1e308, ..i }, 6.0, &cfg).is_err());
    let p = reference(&i, 0.7);
    assert_eq!(
        implied_volatility(
            &i,
            p,
            &IvConfig {
                initial_upper_volatility: 0.1,
                maximum_volatility: 0.2,
                ..cfg.clone()
            }
        )
        .unwrap()
        .status,
        IvStatus::BracketLimit
    );
    assert_eq!(
        implied_volatility(
            &i,
            p,
            &IvConfig {
                max_iterations: 1,
                ..cfg
            }
        )
        .unwrap()
        .status,
        IvStatus::MaxIterations
    );
}
#[test]
fn low_vega_identifiability_is_not_merely_price_convergence() {
    let i = EuropeanInputs {
        strike: 180.0,
        maturity_years: 0.1,
        ..input(OptionSide::Call)
    };
    let p = reference(&i, 0.5);
    let cfg = IvConfig {
        minimum_vega: 1.0,
        price_absolute_tolerance: 1e-14,
        ..IvConfig::default()
    };
    let root = implied_volatility(&i, p, &cfg).unwrap();
    assert_eq!(root.status, IvStatus::LowVega);
    close(root.volatility.unwrap(), 0.5, 1e-9);
    assert!(root.vega.unwrap() < 1.0);
    // A premium rounded all the way to intrinsic identifies a boundary, not .2.
    assert_eq!(
        implied_volatility(&i, 0.0, &cfg).unwrap().status,
        IvStatus::LowerBoundary
    );
}
#[test]
fn bid_mid_ask_statuses_freshness_and_identity_caps_are_preserved() {
    let good = row(0, 100.0, 0.5, 0.2, OptionSide::Call);
    let mut rows = vec![good.clone(); 5];
    for (n, q) in rows.iter_mut().enumerate() {
        q.quote_id = format!("q{n}");
    }
    rows[1].timestamp_ns = 79;
    rows[2].timestamp_ns = 101;
    rows[3].bid = 8.0;
    rows[3].ask = 7.0;
    rows[4].bid = 0.0;
    let result = process_chain(&rows, &chain()).unwrap();
    assert!(result[0].eligible);
    assert!(
        result[0].bid_iv.as_ref().unwrap().volatility
            < result[0].ask_iv.as_ref().unwrap().volatility
    );
    assert!(result[1].issues.contains(&QuoteIssue::Stale));
    assert!(result[2].issues.contains(&QuoteIssue::Future));
    assert!(result[3].issues.contains(&QuoteIssue::CrossedOrNegative));
    assert!(result[4].issues.contains(&QuoteIssue::IneligibleBid));
    assert_eq!(result[0].quote.source_id, "example-source");
    assert!(process_chain(&[good.clone(), good], &chain()).is_err());
    assert!(
        process_chain(
            &rows,
            &ChainConfig {
                max_quotes: 2,
                ..chain()
            }
        )
        .is_err()
    );
    assert!(
        process_chain(
            &rows,
            &ChainConfig {
                max_input_bytes: 1,
                ..chain()
            }
        )
        .is_err()
    );
}
#[test]
fn flat_surface_reprices_duplicates_interpolates_variance_and_carry_without_extrapolation() {
    let rows = grid(0.2);
    let surface = fitted(&rows);
    assert_eq!(surface.expiry_count(), 2);
    assert_eq!(surface.diagnostics.knots, 6);
    assert!(!surface.diagnostics.arbitrage_free_guaranteed);
    assert!(
        surface.diagnostics.issues.is_empty(),
        "{:?}",
        surface.diagnostics.issues
    );
    assert!(
        surface
            .diagnostics
            .residuals
            .iter()
            .all(|r| r.price_residual.abs() < 1e-8)
    );
    let p = surface.query(101.0, 0.75).unwrap();
    close(p.total_variance, 0.03, 1e-10);
    close(p.volatility, 0.2, 1e-9);
    close(p.rate, 0.03, 1e-15);
    close(p.dividend_yield, 0.01, 1e-15);
    assert!(surface.query(101.0, 0.25).is_err());
    assert!(surface.query(50.0, 0.75).is_err());
    assert!(surface.query(f64::NAN, 0.75).is_err());
    let encoded = serde_json::to_string(&surface).unwrap();
    assert!(encoded.contains("arbitrage_free_guaranteed\":false"));
}
#[test]
fn integrated_carry_is_interpolated_and_inconsistent_expiry_carry_is_rejected() {
    let mut rows = grid(0.2);
    for q in &mut rows {
        if q.input.maturity_years == 1.0 {
            q.input.rate = 0.05;
            q.input.dividend_yield = 0.02;
            let p = reference(&q.input, 0.2);
            q.bid = p - 1e-5;
            q.ask = p + 1e-5;
        }
    }
    let surface = fitted(&rows);
    let p = surface.query(100.0, 0.75).unwrap();
    close(p.rate, (0.03 * 0.5 + 0.05) / 2.0 / 0.75, 1e-15);
    close(p.dividend_yield, (0.01 * 0.5 + 0.02) / 2.0 / 0.75, 1e-15);
    rows[0].input.rate = 0.031;
    let p = reference(&rows[0].input, 0.2);
    rows[0].bid = p - 1e-5;
    rows[0].ask = p + 1e-5;
    assert!(
        fit_surface(
            &process_chain(&rows, &chain()).unwrap(),
            &SurfaceConfig::default()
        )
        .is_err()
    );
}
#[test]
fn unsafe_observed_smiles_and_calendars_are_flags_not_silent_guarantees() {
    let mut rows = Vec::new();
    for (t, v) in [(0.5, 0.4), (1.0, 0.1)] {
        for k in [80.0, 100.0, 120.0] {
            rows.push(row(rows.len(), k, t, v, OptionSide::Call));
        }
    }
    let surface = fitted(&rows);
    assert!(
        surface
            .diagnostics
            .issues
            .iter()
            .any(|i| i.kind == "calendar_total_variance")
    );
    let mut rows = grid(0.2);
    for q in &mut rows {
        if q.input.maturity_years == 0.5
            && q.input.strike == 100.0
            && matches!(q.input.side, OptionSide::Put)
        {
            let p = reference(&q.input, 0.3);
            q.bid = p - 1e-5;
            q.ask = p + 1e-5;
        }
    }
    let surface = fitted(&rows);
    assert!(
        surface
            .diagnostics
            .issues
            .iter()
            .any(|i| i.kind == "put_call_parity")
    );
    let mut rows: Vec<_> = [80.0, 100.0, 120.0]
        .into_iter()
        .enumerate()
        .map(|(n, k)| row(n, k, 0.5, 0.2, OptionSide::Call))
        .collect();
    for (q, p) in rows.iter_mut().zip([22.0, 30.0, 1.0]) {
        q.bid = p - 1e-5;
        q.ask = p + 1e-5;
    }
    let surface = fitted(&rows);
    assert!(
        surface
            .diagnostics
            .issues
            .iter()
            .any(|i| i.kind == "strike_convexity" || i.kind == "normalized_call_slope")
    );
}
#[test]
fn greedy_work_caps_and_mixed_underlying_fail_before_diagnostics() {
    let rows = grid(0.2);
    let processed = process_chain(&rows, &chain()).unwrap();
    for cfg in [
        SurfaceConfig {
            max_quotes: 1,
            ..SurfaceConfig::default()
        },
        SurfaceConfig {
            max_input_bytes: 1,
            ..SurfaceConfig::default()
        },
        SurfaceConfig {
            max_knots: 1,
            ..SurfaceConfig::default()
        },
        SurfaceConfig {
            max_diagnostic_checks: 1,
            ..SurfaceConfig::default()
        },
    ] {
        assert!(fit_surface(&processed, &cfg).is_err());
    }
    let mut mixed = processed.clone();
    mixed[0].quote.underlying_id = "other".into();
    assert!(fit_surface(&mixed, &SurfaceConfig::default()).is_err());
    let mut mixed = processed.clone();
    mixed[0].quote.input.spot = 101.0;
    assert!(fit_surface(&mixed, &SurfaceConfig::default()).is_err());
    let mut forged = processed.clone();
    forged[0].quote.bid = f64::NAN;
    assert!(fit_surface(&forged, &SurfaceConfig::default()).is_err());
    assert!(fit_surface(&[], &SurfaceConfig::default()).is_err());
}
#[test]
fn fixed_and_sticky_greeks_agree_for_flat_smile_and_sticky_is_frozen_surface() {
    let surface = fitted(&grid(0.2));
    let fixed = surface
        .greeks(
            101.0,
            0.75,
            OptionSide::Call,
            GreekConvention::FixedVolatility,
        )
        .unwrap();
    let sticky = surface
        .greeks(
            101.0,
            0.75,
            OptionSide::Call,
            GreekConvention::StickyLogMoneynessFiniteBump {
                relative_spot_bump: 1e-4,
            },
        )
        .unwrap();
    close(sticky.spot_delta, fixed.spot_delta, 1e-7);
    close(sticky.spot_gamma, fixed.spot_gamma, 1e-7);
    let mut rows = Vec::new();
    // w(k)=.02-.04k: a declining smile, not a refit after spot moves.
    for (n, k) in [-0.2f64, 0.0, 0.2].into_iter().enumerate() {
        let strike = 100.0 * (0.01 + k).exp();
        rows.push(row(
            n,
            strike,
            0.5,
            ((0.02 - 0.04 * k) / 0.5).sqrt(),
            OptionSide::Call,
        ));
    }
    let surface = fitted(&rows);
    let strike = 100.0 * (0.01 + 0.05f64).exp();
    let fixed = surface
        .greeks(
            strike,
            0.5,
            OptionSide::Call,
            GreekConvention::FixedVolatility,
        )
        .unwrap();
    let sticky = surface
        .greeks(
            strike,
            0.5,
            OptionSide::Call,
            GreekConvention::StickyLogMoneynessFiniteBump {
                relative_spot_bump: 1e-4,
            },
        )
        .unwrap();
    let sigma_s = 0.04 / (2.0 * 0.5 * 100.0 * fixed.point.volatility);
    close(
        sticky.spot_delta,
        fixed.spot_delta + fixed.fixed_volatility_partials.vega * sigma_s,
        2e-7,
    );
    assert!(sticky.spot_delta > fixed.spot_delta);
    let plus = surface.query_at_spot(101.0, strike, 0.5).unwrap();
    assert!(plus.volatility > fixed.point.volatility);
    assert!(
        surface
            .greeks(
                strike,
                0.5,
                OptionSide::Call,
                GreekConvention::StickyLogMoneynessFiniteBump {
                    relative_spot_bump: 0.0
                }
            )
            .is_err()
    );
    // Surface wings are unsupported; a Greek bump cannot silently extrapolate.
    assert!(
        surface
            .greeks(
                rows[0].input.strike,
                0.5,
                OptionSide::Call,
                GreekConvention::StickyLogMoneynessFiniteBump {
                    relative_spot_bump: 1e-4
                }
            )
            .is_err()
    );
}

#[test]
fn source_metadata_status_serialization_and_input_order_are_reproducible() {
    let rows = grid(0.2);
    let a = fitted(&rows);
    let mut reversed = rows.clone();
    reversed.reverse();
    let b = fitted(&reversed);
    assert_eq!(
        serde_json::to_string(&a).unwrap(),
        serde_json::to_string(&b).unwrap()
    );
    assert_eq!(a.source_observations.len(), rows.len());
    assert!(
        a.source_observations
            .iter()
            .all(|o| o.timestamp_ns == 99 && o.source_id == "example-source")
    );
    let bad = implied_volatility(&input(OptionSide::Call), -1.0, &IvConfig::default()).unwrap();
    let value = serde_json::to_value(bad).unwrap();
    assert_eq!(value["status"], "outside_bounds");
    assert!(value["volatility"].is_null());
    let mut old = rows[0].clone();
    old.timestamp_ns = i64::MIN;
    let result = process_chain(
        &[old],
        &ChainConfig {
            cutoff_ns: i64::MAX,
            ..chain()
        },
    )
    .unwrap();
    assert!(result[0].issues.contains(&QuoteIssue::Stale));
    // A zero-variance numerical boundary is reported, not an invented finite IV.
    let deep = EuropeanInputs {
        strike: 1e6,
        ..input(OptionSide::Call)
    };
    let (price, vega) = european_price_and_vega(&deep, 0.01).unwrap();
    assert_eq!((price, vega), (0.0, 0.0));
    assert_eq!(
        implied_volatility(&deep, price, &IvConfig::default())
            .unwrap()
            .status,
        IvStatus::LowerBoundary
    );
}
