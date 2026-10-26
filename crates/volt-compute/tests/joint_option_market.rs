use volt_compute::{
    ExecutionTarget, Precision,
    hedge_policy::HedgePolicyConfig,
    joint_option_market::*,
    native_learner::{LearnerConfig, Objective, Split},
};
fn fixture() -> (
    JointParameters,
    JointOptionConfig,
    JointState,
    InnovationModel,
) {
    let mut r = [[0.; 6]; 14];
    for i in 0..5 {
        r[i + 1][i] = 0.98;
        r[7][i] = 0.1;
    }
    r[0] = [0.004, 0.0004, -0.001, 0.0001, 0.0002, 2.];
    r[6][1] = 0.01;
    r[8][0] = 0.4;
    r[8][5] = 1.;
    r[9] = [0.7; 6];
    r[10] = [0.02; 6];
    r[11] = [0.1; 6];
    r[12] = [0.2; 6];
    r[13] = [3.; 6];
    let provenance = JointProvenance {
        calibration_id: "synthetic".into(),
        calibration_sha256: "a".repeat(64),
        initial_state_id: "synthetic".into(),
        initial_state_sha256: "b".repeat(64),
        innovation_table_id: "synthetic-linear".into(),
        innovation_table_sha256: "c".repeat(64),
        decision_origin_ns: None,
        calibration_known_ns: None,
        initial_state_known_ns: None,
        synthetic: true,
        availability_note: "Invented values for tests; no historical calibration".into(),
    };
    let mut c = JointOptionConfig::paper_case(731802, provenance);
    c.steps = 3;
    let state = JointState {
        stock: 100.,
        betas: [0.2, 0.02, -0.1, 0.015, 0.01],
        lag_betas: [0.19, 0.018, -0.09, 0.014, 0.009],
        conditional_variances: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        previous_innovations: [0.2, -0.3, 0.4, -0.5, 0.6, -0.7],
    };
    let mut correlation = [[0.; 6]; 6];
    for (i, row) in correlation.iter_mut().enumerate() {
        row[i] = 1.;
    }
    (
        JointParameters { rows: r },
        c,
        state,
        InnovationModel {
            original_correlation: correlation,
            probabilities: vec![0.0001, 0.5, 0.9995],
            quantiles: vec![[-3.; 6], [0.; 6], [3.; 6]],
        },
    )
}
fn range(first: u64, n: usize) -> ScenarioRange {
    ScenarioRange {
        split: Split::Train,
        first_path: first,
        paths: n,
    }
}
fn near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} vs {b}");
}
#[test]
fn transition_matches_independent_scalar_equations() {
    let (p, c, s, _) = fixture();
    let e = [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6];
    let (n, ret) = advance(&p, &c, &s, e).unwrap();
    let atm = s.betas[0] + s.betas[1] * (-(1_f64 / 12. / 0.25).sqrt()).exp();
    let mut h = [0.; 6];
    for (i, v) in h.iter_mut().enumerate() {
        let level = if i == 0 || i == 5 {
            (p.rows[8][i] * atm).powi(2)
        } else {
            (p.rows[7][i] / 252_f64.sqrt()).powi(2)
        };
        *v = level
            + p.rows[9][i] * (s.conditional_variances[i] - level)
            + p.rows[10][i]
                * s.conditional_variances[i]
                * (s.previous_innovations[i].powi(2)
                    - 1.
                    - 2. * s.previous_innovations[i] * p.rows[11][i]);
        near(n.conditional_variances[i], *v, 1e-16);
    }
    for i in 0..5 {
        let ar = (0..5).map(|j| p.rows[j + 1][i] * s.betas[j]).sum::<f64>();
        let expected = p.rows[0][i]
            + ar
            + if i == 1 {
                p.rows[6][1] * s.lag_betas[1]
            } else {
                0.
            }
            + (h[i] * c.dt).sqrt() * e[i];
        near(n.betas[i], expected, 1e-15);
    }
    let psi = |z: f64| {
        let phi = p.rows[13][5];
        let zeta = p.rows[12][5];
        phi * phi / (phi * phi + zeta * zeta)
            * (-zeta * z + phi * phi - phi * (phi * phi + zeta * zeta - (zeta + z).powi(2)).sqrt())
    };
    let z = (h[5] * c.dt).sqrt();
    near(
        ret,
        (c.rate - c.dividend_yield) * c.dt + psi(-2. * z) - psi(-z) + z * e[5],
        1e-15,
    );
    near(n.stock, 100. * ret.exp(), 1e-12);
}
#[test]
fn surface_prices_and_zero_moneyness_control() {
    let (_, c, s, _) = fixture();
    let vol = surface_volatility(s.betas, 0., 0.25, None).unwrap();
    near(vol, 0.2 + 0.02 * (-1_f64).exp(), 1e-15);
    for m in [-0.5_f64, -0.1, 0., 0.1, 0.5] {
        let b = s.betas;
        let neg = m.min(0.);
        let expected = b[0]
            + b[1] * (-1_f64).exp()
            + b[2]
                * if m < 0. {
                    (f64::exp(2. * m) - 1.) / (f64::exp(2. * m) + 1.)
                } else {
                    m
                }
            + b[3] * (1. - (-m * m).exp()) * (0.25_f64 / 5.).ln()
            + if m < 0. {
                b[4] * (1. - (3. * neg).powi(3).exp()) * (0.25_f64 / 5.).ln()
            } else {
                0.
            };
        near(
            surface_volatility(b, m, 0.25, None).unwrap(),
            expected,
            1e-14,
        );
    }
    let call = option_value(&c, &s, 100., 0.25, volt_compute::OptionSide::Call).unwrap();
    let put = option_value(&c, &s, 100., 0.25, volt_compute::OptionSide::Put).unwrap();
    near(
        call.0 - put.0,
        100. * (-c.dividend_yield * 0.25).exp() - 100. * (-c.rate * 0.25).exp(),
        1e-12,
    );
}
#[test]
fn seeded_paths_and_identity_are_stable_across_chunks() {
    let (p, c, s, m) = fixture();
    let all = generate(&p, &c, &s, &m, range(10, 4)).unwrap();
    let a = generate(&p, &c, &s, &m, range(10, 2)).unwrap();
    let b = generate(&p, &c, &s, &m, range(12, 2)).unwrap();
    assert_eq!(all.dataset_id, a.dataset_id);
    assert_eq!(all.dataset_id, b.dataset_id);
    assert_eq!(all.states, [a.states, b.states].concat());
    assert_eq!(all.batch.marks, [a.batch.marks, b.batch.marks].concat());
    assert_eq!(
        all.batch.features,
        [a.batch.features, b.batch.features].concat()
    );
    assert_eq!(all.innovations, [a.innovations, b.innovations].concat());
}
#[test]
fn changing_future_innovation_cannot_change_earlier_features() {
    let (p, c, s, _) = fixture();
    let e = vec![[0.1; 6]; 3];
    let a = generate_with_innovations(&p, &c, &s, range(0, 1), e.clone()).unwrap();
    let mut future = e;
    future[2] = [-0.4; 6];
    let b = generate_with_innovations(&p, &c, &s, range(0, 1), future).unwrap();
    assert_eq!(a.batch.features, b.batch.features);
    assert_ne!(a.batch.payoffs, b.batch.payoffs);
    assert_ne!(a.content_sha256, b.content_sha256);
    assert_ne!(a.dataset_id, b.dataset_id);
}
#[test]
fn funding_costs_shapes_and_native_learner_upload() {
    let (p, c, s, m) = fixture();
    let x = generate(&p, &c, &s, &m, range(0, 2)).unwrap();
    assert_eq!(x.batch.marks.len(), 2 * 4 * 2);
    assert_eq!(x.batch.features.len(), 2 * 3 * 17);
    assert_eq!(x.economics.trading_costs, [0.0005, 0.02]);
    assert_eq!(x.economics.liquidation_costs, [0., 0.02]);
    for t in 0..3 {
        near(x.economics.cash_growth[t], (c.rate * c.dt).exp(), 0.);
        near(
            x.economics.income_growth[t * 2],
            (c.dividend_yield * c.dt).exp(),
            0.,
        );
        assert_eq!(x.economics.income_growth[t * 2 + 1], 1.);
    }
    let mut policy = HedgePolicyConfig::paper_heston(2);
    policy.time_steps = 3;
    policy.feature_count = 17;
    policy.batch_norm.enabled = false;
    policy.hidden_width = 4;
    let cfg = LearnerConfig {
        target: ExecutionTarget::Cpu,
        precision: Precision::F64,
        policy,
        objective: Objective::SquaredLoss,
        premium: x.economics.premium,
        trading_costs: x.economics.trading_costs.to_vec(),
        liquidation_costs: x.economics.liquidation_costs.to_vec(),
        cash_growth: x.economics.cash_growth.clone(),
        income_growth: x.economics.income_growth.clone(),
        ..Default::default()
    };
    // The wire body is the exact public EpisodeBatch type consumed by native_learner.
    let payload = serde_json::to_value(&x.batch).unwrap();
    let decoded: volt_compute::native_learner::EpisodeBatch =
        serde_json::from_value(payload).unwrap();
    assert_eq!(decoded.features, x.batch.features);
    assert_eq!(cfg.policy.feature_count, FEATURE_NAMES.len());
    assert!(x.economics.premium > 0.);
    use volt_compute::native_learner::{Command, NativeLearner};
    let mut learner = NativeLearner::new(cfg).unwrap();
    learner
        .handle(Command::RegisterDataset {
            dataset_id: x.dataset_id.clone(),
            partitions: dataset_partitions(&[range(0, 2)]).unwrap(),
        })
        .unwrap();
    learner
        .handle(Command::Upload {
            batch_id: "joint".into(),
            split: Split::Train,
            batch: decoded,
        })
        .unwrap();
    let evaluation = learner
        .handle(Command::Evaluate {
            batch_id: "joint".into(),
            retain_holdings: true,
        })
        .unwrap();
    for path in 0..2 {
        let mut value = x.economics.premium;
        let mut previous = [0_f64; 2];
        let mut fees = 0.;
        for t in 0..3 {
            let target = [
                evaluation["holdings"][t][path * 2].as_f64().unwrap(),
                evaluation["holdings"][t][path * 2 + 1].as_f64().unwrap(),
            ];
            let at = (path * 4 + t) * 2;
            let next = at + 2;
            let charge = (0..2)
                .map(|i| {
                    (target[i] - previous[i]).abs()
                        * x.batch.marks[at + i]
                        * x.economics.trading_costs[i]
                })
                .sum::<f64>();
            value = (value
                - (0..2)
                    .map(|i| target[i] * x.batch.marks[at + i])
                    .sum::<f64>()
                - charge)
                * x.economics.cash_growth[t]
                + (0..2)
                    .map(|i| {
                        target[i] * x.batch.marks[next + i] * x.economics.income_growth[t * 2 + i]
                    })
                    .sum::<f64>();
            fees += charge;
            previous = target;
        }
        let terminal_fee = (0..2)
            .map(|i| {
                previous[i].abs()
                    * x.batch.marks[(path * 4 + 3) * 2 + i]
                    * x.economics.liquidation_costs[i]
            })
            .sum::<f64>();
        value -= terminal_fee;
        fees += terminal_fee;
        near(
            value - x.batch.payoffs[path],
            evaluation["pnl"][path].as_f64().unwrap(),
            1e-10,
        );
        near(fees, evaluation["costs"][path].as_f64().unwrap(), 1e-12);
    }
    let updated = learner
        .handle(Command::Update {
            batch_id: "joint".into(),
            expected_version: 0,
        })
        .unwrap();
    assert_eq!(updated["version"], 1);
    assert!(updated["loss_before_update"].as_f64().unwrap().is_finite());
}
#[test]
fn invalid_paths_ranges_and_budgets_fail_closed() {
    let (p, mut c, s, m) = fixture();
    c.max_working_bytes = 1;
    assert!(generate(&p, &c, &s, &m, range(0, 1)).is_err());
    c.max_working_bytes = 64 * 1024 * 1024;
    assert!(plan(&c, &range(u64::MAX, 2)).is_err());
    assert!(plan(&c, &range(0, usize::MAX)).is_err());
    assert!(
        validate_ranges(&[
            range(0, 3),
            ScenarioRange {
                split: Split::Test,
                first_path: 2,
                paths: 3
            }
        ])
        .is_err()
    );
    let mut bad = p.clone();
    bad.rows[10][0] = 100.;
    assert!(advance(&bad, &c, &s, [0.; 6]).is_err());
    assert!(return_cumulant(&p, 100.).is_err());
    let mut bad_m = m.clone();
    bad_m.original_correlation[0][1] = 1.;
    assert!(bad_m.validate().is_err());
    assert!(generate_with_innovations(&p, &c, &s, range(0, 1), vec![]).is_err());
}
#[test]
fn zero_variance_channels_and_expiry_payoff() {
    let (mut p, mut c, mut s, _) = fixture();
    p.rows[8] = [0.; 6];
    p.rows[7] = [0.; 6];
    p.rows[9] = [0.; 6];
    p.rows[10] = [0.; 6];
    s.conditional_variances = [0.; 6];
    c.steps = 3;
    c.straddle_days = 3;
    let x = generate_with_innovations(&p, &c, &s, range(0, 1), vec![[0.; 6]; 3]).unwrap();
    near(x.batch.payoffs[0], (x.states[3].stock - 100.).abs(), 1e-14);
}
#[test]
fn copula_permutation_and_quantile_clipping() {
    let (_, _, _, m) = fixture();
    let e = m
        .from_standard_normals([100., -100., 0., 0., 0., 0.])
        .unwrap();
    near(e[0], -3., 0.);
    near(e[5], 3., 1e-12);
    near(e[1], 0., 0.);
}
#[test]
fn author_gamma_feature_is_explicit_and_price_derivative_stays_canonical() {
    let (p, mut c, s, _) = fixture();
    let e = vec![[0.; 6]; 3];
    let author = generate_with_innovations(&p, &c, &s, range(0, 1), e.clone()).unwrap();
    c.gamma_feature = GammaFeatureConvention::FixedVolatilityDerivative;
    let canonical = generate_with_innovations(&p, &c, &s, range(0, 1), e).unwrap();
    assert_eq!(author.batch.marks, canonical.batch.marks);
    assert_eq!(author.batch.payoffs, canonical.batch.payoffs);
    let tau = c.straddle_days as f64 * c.dt;
    near(
        author.batch.features[15],
        canonical.batch.features[15] * (c.dividend_yield * (tau - tau.sqrt())).exp(),
        1e-14,
    );
    assert!((author.batch.features[15] - canonical.batch.features[15]).abs() > 1e-5);
    assert_ne!(author.dataset_id, canonical.dataset_id);
}
#[test]
fn source_availability_is_required_for_real_data_and_rejects_lookahead() {
    let (_, mut c, _, _) = fixture();
    c.provenance.synthetic = false;
    assert!(c.validate().is_err());
    c.provenance.decision_origin_ns = Some(100);
    c.provenance.calibration_known_ns = Some(99);
    c.provenance.initial_state_known_ns = Some(100);
    assert!(c.validate().is_ok());
    c.provenance.calibration_known_ns = Some(101);
    assert!(c.validate().is_err());
}
#[test]
fn initial_history_reconstructs_source_innovation_equations() {
    let (p, c, s, _) = fixture();
    let previous2 = s.lag_betas;
    let previous = s.betas;
    let innovation = [0.1, -0.2, 0.3, -0.4, 0.5, 0.6];
    let mut last = [0.; 5];
    for i in 0..5 {
        last[i] = p.rows[0][i]
            + (0..5).map(|j| p.rows[j + 1][i] * previous[j]).sum::<f64>()
            + if i == 1 {
                p.rows[6][1] * previous2[1]
            } else {
                0.
            }
            + (s.conditional_variances[i] * c.dt).sqrt() * innovation[i];
    }
    let z = (s.conditional_variances[5] * c.dt).sqrt();
    let excess = return_cumulant(&p, -p.rows[0][5] * z).unwrap()
        - return_cumulant(&p, (1. - p.rows[0][5]) * z).unwrap()
        + z * innovation[5];
    let out = state_from_history(
        &p,
        &c,
        100.,
        [previous2, previous, last],
        s.conditional_variances,
        excess,
    )
    .unwrap();
    for (a, b) in out.previous_innovations.iter().zip(innovation) {
        near(*a, b, 1e-13);
    }
}
#[test]
fn cli_synthetic_example_and_output_alias_guard() {
    use std::{fs, process::Command};
    let exe = env!("CARGO_BIN_EXE_joint_option_scenarios");
    let example = Command::new(exe).arg("--example").output().unwrap();
    assert!(example.status.success());
    let dir = std::env::temp_dir().join(format!(
        "volt-joint-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&dir).unwrap();
    let input = dir.join("input.json");
    let output = dir.join("output.json");
    fs::write(&input, &example.stdout).unwrap();
    let result = Command::new(exe)
        .args([
            "--input",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let x: JointOptionScenarios = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(x.batch.paths, 8);
    assert!(x.config.provenance.synthetic);
    assert_eq!(x.feature_names.len(), 17);
    assert_eq!(
        x.config.gamma_feature,
        GammaFeatureConvention::AuthorSqrtTimeDividend
    );
    let guard = Command::new(exe)
        .args([
            "--input",
            input.to_str().unwrap(),
            "--output",
            input.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!guard.status.success());
    assert_eq!(fs::read(&input).unwrap(), example.stdout);
    fs::remove_dir_all(dir).unwrap();
}
