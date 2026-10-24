use volt_compute::{
    ComputeError, OptionSide,
    heston::{HestonOption, HestonParams},
    heston_adaptive::{AdaptiveHestonConfig, AdaptiveHestonResult, adaptive_greeks},
};
fn option(spot: f64) -> HestonOption {
    HestonOption {
        row_id: "independent-adaptive-control".into(),
        side: OptionSide::Call,
        spot,
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
fn tight() -> AdaptiveHestonConfig {
    AdaptiveHestonConfig {
        price_absolute_tolerance: 1e-9,
        delta_absolute_tolerance: 1e-9,
        variance_absolute_tolerance: 1e-7,
        ..Default::default()
    }
}
fn close(x: f64, y: f64, t: f64) {
    assert!(
        (x - y).abs() <= t,
        "{x:.16e} vs {y:.16e}, error{}",
        (x - y).abs()
    );
}
fn certified(a: &AdaptiveHestonResult, c: AdaptiveHestonConfig) {
    assert!(a.converged);
    assert!(a.error_estimates.price <= c.price_absolute_tolerance);
    assert!(a.error_estimates.delta <= c.delta_absolute_tolerance);
    assert!(a.error_estimates.initial_variance_sensitivity <= c.variance_absolute_tolerance);
    assert!(a.evaluations <= c.max_evaluations);
    assert!(a.panels <= c.max_panels);
    assert_eq!(a.evaluations, a.panels * 15);
    assert!(a.cutoff <= c.max_cutoff && a.cutoff >= 4.0 * c.initial_cutoff);
}

/// Published QuantLib Lewis fixture and independently bumped adaptive controls.
#[test]
fn published_price_and_parameter_risk_controls_are_resolved_tightly() {
    let c = tight();
    let a = adaptive_greeks(&option(100.0), &params(), c).unwrap();
    certified(&a, c);
    close(a.price, 16.070154917028834, 1e-8);
    close(a.delta, 0.5994713257509925, 1e-8);
    close(a.initial_variance_sensitivity, 10.179265404511439, 1e-8);
}

#[test]
fn calls_puts_and_positive_correlation_log_branch_match_independent_controls() {
    let c = tight();
    let mut o = option(100.0);
    let call = adaptive_greeks(&o, &params(), c).unwrap();
    o.side = OptionSide::Put;
    let put = adaptive_greeks(&o, &params(), c).unwrap();
    close(put.price, 17.05527096127011, 1e-8);
    close(
        call.price - put.price,
        100.0 * ((-0.02f64).exp() - (-0.01f64).exp()),
        1e-12,
    );
    close(call.delta - put.delta, (-0.02f64).exp(), 1e-12);
    close(
        call.initial_variance_sensitivity,
        put.initial_variance_sensitivity,
        0.0,
    );
    let p = HestonParams {
        initial_variance: 0.04,
        kappa: 0.3,
        long_run_variance: 0.04,
        vol_of_variance: 1.5,
        correlation: 0.8,
    };
    o.side = OptionSide::Call;
    let result = adaptive_greeks(&o, &p, c).unwrap();
    certified(&result, c);
    close(result.price, 3.8187291781568242, 1e-8);
    close(result.delta, 0.1934099121836219, 1e-8);
    close(result.initial_variance_sensitivity, 66.95624539161335, 2e-7);
}

/// QuantLib1.40 AngledContour+adaptive Lobatto with optimal-control-variate
/// crosscheck, using strictly positive V0 nodes then extrapolating to zero.
/// Ordinary Gatheral/exp-sinh is NOT a reliable far-OTM control in this case.
#[test]
fn one_day_zero_variance_resolves_far_oscillations_and_near_atm_right_limits() {
    let c = AdaptiveHestonConfig::default();
    let p = HestonParams {
        initial_variance: 0.0,
        kappa: 1.0,
        long_run_variance: 0.04,
        vol_of_variance: 2.0,
        correlation: -0.7,
    };
    for (spot, price, delta, cv, cv_tol) in [
        (90.0, 0.0, 0.0, 0.0, 1e-7),
        (99.0, 7.4577784e-10, 9.693e-9, 9.465e-5, 3e-8),
        (99.9, 0.00021199793678, 0.00389209299, 5.4763164, 1e-6),
        (100.0, 0.0065178659884, 0.7285337304, 59.44818436, 1e-5),
        (100.1, 0.1019144476669, 0.98551455828, 30.6794365, 1e-6),
        (101.0, 1.000015308378956, 0.999927566656, 0.803976543, 3e-8),
        (110.0, 10.0, 1.0, 0.0, 1e-7),
    ] {
        let mut o = option(spot);
        o.maturity_years = 1.0 / 365.0;
        o.rate = 0.0;
        o.dividend_yield = 0.0;
        let a = adaptive_greeks(&o, &p, c).unwrap();
        eprintln!(
            "S{spot}: P{} D{} Cv{}, eval{} panels{} U{} err{:?}",
            a.price,
            a.delta,
            a.initial_variance_sensitivity,
            a.evaluations,
            a.panels,
            a.cutoff,
            a.error_estimates
        );
        certified(&a, c);
        close(a.price, price, 2e-9);
        close(a.delta, delta, 3e-8);
        close(a.initial_variance_sensitivity, cv, cv_tol);
    }
}

#[test]
fn independent_initial_panel_scale_and_tighter_tolerances_converge() {
    let mut o = option(99.9);
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
    let coarse = adaptive_greeks(&o, &p, AdaptiveHestonConfig::default()).unwrap();
    let c = AdaptiveHestonConfig {
        initial_cutoff: 37.0,
        price_absolute_tolerance: 1e-9,
        delta_absolute_tolerance: 1e-9,
        variance_absolute_tolerance: 1e-7,
        max_evaluations: 4_000_000,
        max_panels: 262_144,
        ..Default::default()
    };
    let fine = adaptive_greeks(&o, &p, c).unwrap();
    certified(&fine, c);
    close(coarse.price, fine.price, 1e-9);
    close(coarse.delta, fine.delta, 1e-8);
    close(
        coarse.initial_variance_sensitivity,
        fine.initial_variance_sensitivity,
        1e-7,
    );
}

#[test]
fn insufficient_work_tail_and_invalid_domain_return_errors_not_proxy_results() {
    let o = option(100.0);
    let p = params();
    let c = AdaptiveHestonConfig {
        max_evaluations: 45,
        ..Default::default()
    };
    assert!(matches!(
        adaptive_greeks(&o, &p, c),
        Err(ComputeError::BudgetExceeded(_))
    ));
    let c = AdaptiveHestonConfig {
        max_panels: 3,
        ..Default::default()
    };
    assert!(matches!(
        adaptive_greeks(&o, &p, c),
        Err(ComputeError::BudgetExceeded(_))
    ));
    let mut day = o.clone();
    day.maturity_years = 1.0 / 365.0;
    let mut zero = p;
    zero.initial_variance = 0.0;
    zero.kappa = 1.0;
    zero.long_run_variance = 0.04;
    zero.vol_of_variance = 2.0;
    zero.correlation = -0.7;
    let c = AdaptiveHestonConfig {
        initial_cutoff: 1.0,
        max_cutoff: 4.0,
        ..Default::default()
    };
    assert!(adaptive_greeks(&day, &zero, c).is_err());
    let mut bad = p;
    bad.initial_variance = -1e-100;
    assert!(adaptive_greeks(&o, &bad, Default::default()).is_err());
    let c = AdaptiveHestonConfig {
        delta_absolute_tolerance: f64::NAN,
        ..Default::default()
    };
    assert!(c.validate().is_err());
    let c = AdaptiveHestonConfig {
        initial_cutoff: 100.0,
        max_cutoff: 200.0,
        ..Default::default()
    };
    assert!(c.validate().is_err());
}

#[test]
fn independent_queries_are_deterministic_across_threads() {
    let o = option(100.0);
    let p = params();
    let c = tight();
    let reference = adaptive_greeks(&o, &p, c).unwrap();
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let o = o.clone();
            std::thread::spawn(move || adaptive_greeks(&o, &p, c).unwrap())
        })
        .collect();
    for t in threads {
        let a = t.join().unwrap();
        assert_eq!(a.price.to_bits(), reference.price.to_bits());
        assert_eq!(a.delta.to_bits(), reference.delta.to_bits());
        assert_eq!(
            a.initial_variance_sensitivity.to_bits(),
            reference.initial_variance_sensitivity.to_bits()
        );
        assert_eq!(a.evaluations, reference.evaluations);
    }
}

#[test]
fn phase_resolved_zero_variance_atm_work_stays_bounded_without_weakening_accuracy() {
    use volt_compute::heston_adaptive::{AdaptiveTailMethod, analytic_tail_bounds};
    let mut o = option(100.0);
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
    let c = AdaptiveHestonConfig {
        max_evaluations: 10000,
        max_panels: 650,
        ..Default::default()
    };
    let a = adaptive_greeks(&o, &p, c).unwrap();
    certified(&a, c);
    assert_eq!(
        a.tail_method,
        AdaptiveTailMethod::AnalyticCharacteristicBound
    );
    close(a.price, 0.0065178659884, 2e-9);
    close(a.delta, 0.7285337304, 3e-8);
    close(a.initial_variance_sensitivity, 59.44818436, 1e-5);
    assert!(analytic_tail_bounds(&p, o.maturity_years, 64.0).is_none());
    let at = analytic_tail_bounds(&p, o.maturity_years, a.cutoff).unwrap();
    let farther = analytic_tail_bounds(&p, o.maturity_years, a.cutoff * 2.0).unwrap();
    assert!(at.iter().zip(farther).all(|(a, b)| *a >= b && b > 0.0));
}

#[test]
fn ineligible_tail_domain_keeps_the_original_successive_shell_rule() {
    use volt_compute::heston_adaptive::{AdaptiveTailMethod, analytic_tail_bounds};
    let o = option(100.0);
    let p = HestonParams {
        initial_variance: 0.04,
        kappa: 0.3,
        long_run_variance: 0.04,
        vol_of_variance: 1.5,
        correlation: 0.8,
    };
    assert!(analytic_tail_bounds(&p, o.maturity_years, 10000.0).is_none());
    let a = adaptive_greeks(&o, &p, tight()).unwrap();
    assert_eq!(a.tail_method, AdaptiveTailMethod::ObservedEnvelope);
    close(a.price, 3.8187291781568242, 1e-8);
    close(a.delta, 0.1934099121836219, 1e-8);
}
