use volt_compute::risk_objective::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a:0.16e} != {b:0.16e}");
}
fn boot() -> BootstrapConfig {
    BootstrapConfig {
        replicates: 200,
        confidence: 0.95,
        seed: 76543,
    }
}

#[test]
fn cvar_fractional_tail_mass_is_not_a_rounded_sample_count() {
    let pnl = [-10., -4., 1.];
    let weights = [1., 2., 7.];
    let risk = weighted_risk(&pnl, Some(&weights), RiskObjective::Cvar { alpha: 0.8 }).unwrap();
    close(risk.value, 7., 1e-13);
    assert_eq!(risk.cvar_boundary_loss, Some(4.));
    close(risk.cvar_tail_probability.unwrap(), 0.2, 1e-15);
    let quarter = weighted_risk(&pnl, Some(&weights), RiskObjective::Cvar { alpha: 0.75 }).unwrap();
    close(quarter.value, 6.4, 1e-13);
    let average = weighted_risk(&pnl, Some(&weights), RiskObjective::Cvar { alpha: 0. }).unwrap();
    close(average.value, 0.1 * 10. + 0.2 * 4. - 0.7, 1e-14);
}

#[test]
fn cvar_tail_handles_zero_mass_gains_ties_and_less_than_one_sample() {
    let x = weighted_risk(
        &[-100., 1., 2.],
        Some(&[0., 1., 1.]),
        RiskObjective::Cvar { alpha: 0.5 },
    )
    .unwrap();
    assert_eq!(x.value, -1.);
    assert_eq!(x.positive_weight_observations, 2);
    let x = weighted_risk(
        &[-5., -5., 0., 2., 8.],
        None,
        RiskObjective::Cvar { alpha: 0.99 },
    )
    .unwrap();
    close(x.value, 5., 1e-14);
    assert_eq!(x.cvar_boundary_loss, Some(5.));
}

#[test]
fn constant_risks_sign_and_cash_translation_are_explicit() {
    for objective in [
        RiskObjective::Cvar { alpha: 0.95 },
        RiskObjective::Entropic { risk_aversion: 2. },
    ] {
        close(
            weighted_risk(&[7.; 5], None, objective).unwrap().value,
            -7.,
            1e-13,
        );
        let a = weighted_risk(&[-3., 1., 4.], Some(&[2., 1., 3.]), objective).unwrap();
        let b = weighted_risk(&[2., 6., 9.], Some(&[2., 1., 3.]), objective).unwrap();
        close(b.value, a.value - 5., 1e-13);
    }
    let a = weighted_risk(&[-1., 1.], None, RiskObjective::SquaredLoss).unwrap();
    let b = weighted_risk(&[1., 3.], None, RiskObjective::SquaredLoss).unwrap();
    assert_eq!(a.value, 1.);
    assert_eq!(b.value, 5.);
}

#[test]
fn stable_entropy_matches_weighted_known_distribution_and_tilt_ess() {
    let risk = weighted_risk(
        &[-1., 2.],
        Some(&[1., 3.]),
        RiskObjective::Entropic { risk_aversion: 1. },
    )
    .unwrap();
    let a = 0.25 * 1_f64.exp();
    let b = 0.75 * (-2_f64).exp();
    close(risk.value, (a + b).ln(), 1e-14);
    close(
        risk.entropic_tilt_ess.unwrap(),
        (a + b).powi(2) / (a * a + b * b),
        1e-14,
    );
    close(
        risk.probability_weight_ess,
        1. / (0.25_f64.powi(2) + 0.75_f64.powi(2)),
        1e-14,
    );
    let extreme = weighted_risk(
        &[-1000., 1000.],
        None,
        RiskObjective::Entropic {
            risk_aversion: 1000.,
        },
    )
    .unwrap();
    close(extreme.value, 1000. - 2_f64.ln() / 1000., 1e-12);
    close(extreme.entropic_tilt_ess.unwrap(), 1., 1e-14);
}

#[test]
fn small_risk_aversion_retains_mean_limit_and_weight_scale_is_irrelevant() {
    let subnormal = weighted_risk(
        &[-1.0, 0.0],
        None,
        RiskObjective::Entropic {
            risk_aversion: f64::from_bits(1),
        },
    )
    .unwrap();
    close(subnormal.value, 0.5, 1e-15);
    let risk = weighted_risk(
        &[-1., 3.],
        None,
        RiskObjective::Entropic {
            risk_aversion: 1e-12,
        },
    )
    .unwrap();
    close(risk.value, -1., 1e-10);
    for objective in [
        RiskObjective::Cvar { alpha: 0.6 },
        RiskObjective::Entropic { risk_aversion: 0.7 },
        RiskObjective::SquaredLoss,
    ] {
        let a = weighted_risk(&[-2., 3.], Some(&[1., 1.]), objective).unwrap();
        let b = weighted_risk(&[-2., 3.], Some(&[f64::MAX, f64::MAX]), objective).unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn rejects_invalid_distributions_parameters_and_unrepresentable_arithmetic() {
    let objective = RiskObjective::Cvar { alpha: 0.5 };
    for (pnl, w) in [
        (vec![], None),
        (vec![f64::NAN], None),
        (vec![1.], Some(vec![0.])),
        (vec![1.], Some(vec![-1.])),
        (vec![1.], Some(vec![f64::INFINITY])),
        (vec![1., 2.], Some(vec![1.])),
    ] {
        assert!(weighted_risk(&pnl, w.as_deref(), objective).is_err());
    }
    for bad in [
        RiskObjective::Cvar { alpha: 1. },
        RiskObjective::Cvar { alpha: -0.1 },
        RiskObjective::Cvar { alpha: f64::NAN },
        RiskObjective::Entropic { risk_aversion: 0. },
        RiskObjective::Entropic {
            risk_aversion: f64::INFINITY,
        },
    ] {
        assert!(weighted_risk(&[1.], None, bad).is_err());
    }
    assert!(weighted_risk(&[1e300], None, RiskObjective::SquaredLoss).is_err());
    assert!(
        weighted_risk(
            &[-1000., 0.],
            Some(&[1e-300, 1e300]),
            RiskObjective::Entropic { risk_aversion: 1. }
        )
        .is_err()
    );
    // Entropy remains finite when a naive exp(1e6) would overflow.
    assert!(
        weighted_risk(
            &[-1000., 1000.],
            None,
            RiskObjective::Entropic {
                risk_aversion: 1000.
            }
        )
        .unwrap()
        .value
        .is_finite()
    );
}

#[test]
fn indifference_price_subtracts_no_liability_optimized_value_and_rejects_square() {
    close(
        indifference_price(RiskObjective::Cvar { alpha: 0.5 }, 1.94, 0.1).unwrap(),
        1.84,
        1e-14,
    );
    close(
        indifference_price(RiskObjective::Entropic { risk_aversion: 1. }, 1.94, -0.1).unwrap(),
        2.04,
        1e-14,
    );
    assert!(indifference_price(RiskObjective::SquaredLoss, 2., 1.).is_err());
    assert!(indifference_price(RiskObjective::Cvar { alpha: 0.5 }, f64::NAN, 1.).is_err());
}

#[test]
fn paired_bootstrap_identical_paths_has_zero_difference_and_preserves_covariance() {
    let pnl = [-2., 0., 2., 4.];
    let report =
        paired_path_comparison(&pnl, &pnl, RiskObjective::Cvar { alpha: 0.75 }, boot()).unwrap();
    assert_eq!(report.risk_difference_candidate_minus_baseline, 0.);
    assert_eq!(
        report.risk_difference_interval,
        PercentileInterval {
            lower: 0.,
            upper: 0.
        }
    );
    assert_eq!(report.mean_pnl_difference_standard_error, 0.);
    close(report.paired_sample_covariance, 20. / 3., 1e-14);
}

#[test]
fn paired_common_shock_cancels_in_uncertainty_while_unpaired_changes_do_not() {
    let baseline = [-2., 0., 2., 4.];
    let candidate = [0., 2., 4., 6.];
    let objective = RiskObjective::Entropic { risk_aversion: 1. };
    let paired = paired_path_comparison(&candidate, &baseline, objective, boot()).unwrap();
    close(paired.risk_difference_candidate_minus_baseline, -2., 1e-14);
    close(paired.risk_difference_interval.lower, -2., 1e-14);
    close(paired.risk_difference_interval.upper, -2., 1e-14);
    assert_eq!(paired.mean_pnl_difference_standard_error, 0.);
    let reversed = [4., 2., 0., -2.];
    let mismatch = paired_path_comparison(&candidate, &reversed, objective, boot()).unwrap();
    assert!(mismatch.mean_pnl_difference_standard_error > 1.);
    assert!(mismatch.paired_sample_covariance < 0.);
}

#[test]
fn nonlinear_tail_risk_is_recomputed_for_each_seeded_resampled_pair() {
    let candidate = [-20., 0., 0., 0.];
    let baseline = [-10., 0., 0., 0.];
    let objective = RiskObjective::Cvar { alpha: 0.75 };
    let a = paired_path_comparison(&candidate, &baseline, objective, boot()).unwrap();
    let b = paired_path_comparison(&candidate, &baseline, objective, boot()).unwrap();
    assert_eq!(a, b);
    close(a.risk_difference_candidate_minus_baseline, 10., 1e-14);
    close(a.risk_difference_interval.lower, 0., 1e-14);
    close(a.risk_difference_interval.upper, 10., 1e-14);
    let serialized = serde_json::to_string(&a).unwrap();
    assert_eq!(
        serde_json::from_str::<PairedPathComparison>(&serialized).unwrap(),
        a
    );
}

#[test]
fn bootstrap_contract_rejects_misalignment_nonfinite_and_invalid_settings() {
    assert!(paired_path_comparison(&[1.], &[1.], RiskObjective::SquaredLoss, boot()).is_err());
    assert!(paired_path_comparison(&[1., 2.], &[1.], RiskObjective::SquaredLoss, boot()).is_err());
    assert!(
        paired_path_comparison(
            &[1., f64::NAN],
            &[1., 2.],
            RiskObjective::SquaredLoss,
            boot()
        )
        .is_err()
    );
    let mut bad = boot();
    bad.replicates = 1;
    assert!(paired_path_comparison(&[1., 2.], &[1., 2.], RiskObjective::SquaredLoss, bad).is_err());
    let mut bad = boot();
    bad.confidence = 1.;
    assert!(paired_path_comparison(&[1., 2.], &[1., 2.], RiskObjective::SquaredLoss, bad).is_err());
}
