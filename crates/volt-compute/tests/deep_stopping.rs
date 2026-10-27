use volt_compute::{ComputeError, ExecutionTarget, Precision, deep_stopping::*};
fn tiny(algorithm: StoppingAlgorithm) -> StoppingTrainingConfig {
    let mut c = StoppingTrainingConfig {
        algorithm,
        width: 4,
        training_paths: 256,
        validation_paths: 128,
        batch_rows: 32,
        updates_per_date: 20,
        validation_every: 10,
        ..Default::default()
    };
    c.spec.times = vec![0., 0.5, 1.];
    c.spec.exercise = vec![false, true, true];
    c
}
fn zero_model(spec: GbmPutSpec) -> StoppingCheckpoint {
    let mut c = tiny(StoppingAlgorithm::ContinuationRegression2020);
    c.spec = spec;
    let n = c.spec.times.len();
    let network = StoppingNetwork {
        width: 4,
        parameters: vec![
            vec![0.; 8],
            vec![0.; 4],
            vec![0.; 16],
            vec![0.; 4],
            vec![0.; 4],
            vec![0.],
        ],
    };
    StoppingCheckpoint {
        schema: SCHEMA.into(),
        config: c,
        dates: (0..n)
            .map(|i| {
                if i > 0 && i + 1 < n {
                    Some(StoppingDateModel {
                        continuation: network.clone(),
                        decision: None,
                        selected_update: 0,
                        completed_updates: 0,
                        training_parameter_change_l2: 0.,
                        validation_lower_mean: 0.,
                    })
                } else {
                    None
                }
            })
            .collect(),
        stop_initial: false,
        initial_training_continuation: 0.,
    }
}
#[test]
fn exact_gbm_paths_are_chunk_invariant_and_risk_neutral() {
    let spec = GbmPutSpec::monthly_put();
    let all = simulate_gbm_paths(&spec, 20000, 811, 0).unwrap();
    assert_eq!(
        simulate_gbm_paths(&spec, 7, 811, 13).unwrap(),
        all[13 * 13..20 * 13]
    );
    let terminal: Vec<_> = all.chunks(13).map(|p| p[12]).collect();
    let mean = terminal.iter().sum::<f64>() / 20000.;
    let expected = spec.spot * ((spec.rate - spec.dividend_yield) * spec.times[12]).exp();
    let se = (terminal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (19999. * 20000.)).sqrt();
    assert!((mean - expected).abs() < 4.5 * se);
}
#[test]
fn deterministic_initial_exercise_and_prefix_adaptation() {
    let mut s = GbmPutSpec::monthly_put();
    s.spot = 80.;
    s.volatility = 0.;
    s.exercise[0] = true;
    let mut m = zero_model(s);
    m.stop_initial = true;
    m.initial_training_continuation = 17.;
    let mut path = vec![80.; 13];
    path[12] = f64::NAN;
    let x = stop_path(&m, &path).unwrap();
    assert_eq!(x.step, 0);
    assert_eq!(x.amount, 20.);
    let e = StoppingEvaluationConfig {
        lower_paths: 16,
        upper_paths: 16,
        inner_paths: 3,
        ..Default::default()
    };
    let r = evaluate_bounds(&m, &e).unwrap();
    assert_eq!(r.lower.mean, 20.);
    assert_eq!(r.upper.mean, 20.);
    assert!(r.mean_martingale_increment.iter().all(|x| x.abs() < 1e-12));
}
#[test]
fn one_date_dual_matches_independent_european_price_expectation() {
    let mut s = GbmPutSpec::monthly_put();
    s.times = vec![0., 1.];
    s.exercise = vec![false, true];
    let m = zero_model(s);
    let e = StoppingEvaluationConfig {
        lower_paths: 32768,
        upper_paths: 4096,
        inner_paths: 32,
        ..Default::default()
    };
    let r = evaluate_bounds(&m, &e).unwrap();
    let expected = 5.573526022256971;
    assert!((r.lower.mean - expected).abs() < 4.5 * r.lower.standard_error + 0.001);
    assert!((r.upper.mean - expected).abs() < 4.5 * r.upper.standard_error + 0.001);
    assert!(
        r.hoeffding_joint_95_enclosure[0] < expected
            && r.hoeffding_joint_95_enclosure[1] > expected
    );
}
#[test]
fn both_backward_algorithms_replay_serialized_decisions_and_train_real_parameters() {
    for a in [
        StoppingAlgorithm::DirectStopping2019,
        StoppingAlgorithm::ContinuationRegression2020,
    ] {
        let c = tiny(a);
        let m = train(&c).unwrap();
        m.validate().unwrap();
        assert_eq!(
            m.dates[1].as_ref().unwrap().completed_updates,
            c.updates_per_date
        );
        assert!(m.dates[1].as_ref().unwrap().training_parameter_change_l2 > 0.);
        let raw = serde_json::to_vec(&m).unwrap();
        let restored: StoppingCheckpoint = serde_json::from_slice(&raw).unwrap();
        let paths = simulate_gbm_paths(&c.spec, 20, 9003, 0).unwrap();
        for p in paths.chunks(3) {
            assert_eq!(
                stop_path(&m, p).unwrap().step,
                stop_path(&restored, p).unwrap().step
            );
        }
        let e = StoppingEvaluationConfig {
            lower_paths: 128,
            upper_paths: 64,
            inner_paths: 8,
            ..Default::default()
        };
        let b = evaluate_bounds(&restored, &e).unwrap();
        assert!(b.lower.mean.is_finite() && b.upper.mean.is_finite());
    }
}
#[test]
fn invalid_configs_and_budgets_fail_before_unavailable_device() {
    let mut c = tiny(StoppingAlgorithm::ContinuationRegression2020);
    c.target = ExecutionTarget::Cuda { index: 999 };
    c.training_seed = c.validation_seed;
    assert!(matches!(train(&c), Err(ComputeError::InvalidInput(_))));
    c.training_seed = 77;
    c.max_working_bytes = 1;
    assert!(matches!(train(&c), Err(ComputeError::BudgetExceeded(_))));
    let m = zero_model(GbmPutSpec::monthly_put());
    let e = StoppingEvaluationConfig {
        lower_seed: m.config.training_seed,
        ..Default::default()
    };
    assert!(evaluate_bounds(&m, &e).is_err());
    let e = StoppingEvaluationConfig {
        max_proxy_evaluations: 1,
        ..Default::default()
    };
    assert!(matches!(
        evaluate_bounds(&m, &e),
        Err(ComputeError::BudgetExceeded(_))
    ));
}
#[test]
fn oversized_finite_financial_scales_are_rejected_before_statistics() {
    let mut s = GbmPutSpec::monthly_put();
    s.strike = 1e308;
    assert!(s.validate().is_err());
    let mut m = zero_model(GbmPutSpec::monthly_put());
    m.dates[1].as_mut().unwrap().continuation.parameters[4][0] = 1e308;
    assert!(evaluate_bounds(&m, &StoppingEvaluationConfig::default()).is_err());
}

#[test]
fn metal_backward_training_when_available() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let mut c = tiny(StoppingAlgorithm::ContinuationRegression2020);
        c.target = ExecutionTarget::Metal;
        c.precision = Precision::F32;
        train(&c).unwrap();
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = Precision::F32;
    }
}
#[test]
fn interval_regression_is_separate_and_stops_with_holder() {
    let holder = train(&tiny(StoppingAlgorithm::ContinuationRegression2020)).unwrap();
    let mut c = tiny(StoppingAlgorithm::ContinuationRegression2020);
    c.training_seed = 7001;
    c.validation_seed = 7003;
    c.initialization_seed = 7007;
    let h = train_interval_hedge(&holder, &c, holder.initial_training_continuation).unwrap();
    assert_eq!(h.networks.len(), 2);
    assert!(
        h.validation_interval_mse
            .iter()
            .all(|x| x.is_finite() && *x >= 0.)
    );
    let r = evaluate_interval_hedge(&h, 256, 7011, 0).unwrap();
    assert!(r.mse.is_finite());
    assert_eq!(r.exercise_counts.iter().sum::<usize>(), 256);
    let replay: IntervalHedgeCheckpoint =
        serde_json::from_slice(&serde_json::to_vec(&h).unwrap()).unwrap();
    assert_eq!(
        r.errors,
        evaluate_interval_hedge(&replay, 256, 7011, 0)
            .unwrap()
            .errors
    );
}

#[test]
fn interval_evaluation_refuses_unadmitted_path_allocation() {
    let holder = train(&tiny(StoppingAlgorithm::ContinuationRegression2020)).unwrap();
    let mut c = tiny(StoppingAlgorithm::ContinuationRegression2020);
    c.training_seed = 8101;
    c.validation_seed = 8103;
    c.max_working_bytes = 32 * 1024 * 1024;
    let h = train_interval_hedge(&holder, &c, holder.initial_training_continuation).unwrap();
    assert!(matches!(
        evaluate_interval_hedge(&h, 1_000_000, 8107, 0),
        Err(ComputeError::BudgetExceeded(_))
    ));
}
#[test]
fn native_cli_bounds_input_and_refuses_output_overwrite() {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    let bin = env!("CARGO_BIN_EXE_deep_stopping");
    let mut child = Command::new(bin)
        .arg("serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let _ = input.write_all(&vec![b'a'; 16 * 1024 * 1024 + 1]);
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success() && output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("exceeds16MiB"));
    let dir = std::env::temp_dir().join(format!(
        "volt-stopping-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("example.json");
    assert!(
        Command::new(bin)
            .args(["example", "--output"])
            .arg(&path)
            .output()
            .unwrap()
            .status
            .success()
    );
    let before = std::fs::read(&path).unwrap();
    assert!(
        !Command::new(bin)
            .args(["example", "--output"])
            .arg(&path)
            .output()
            .unwrap()
            .status
            .success()
    );
    assert_eq!(before, std::fs::read(&path).unwrap());
    std::fs::remove_dir_all(dir).unwrap();
}
