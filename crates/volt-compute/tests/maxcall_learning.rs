use serde_json::Value;
use volt_compute::{
    ComputeError, ExecutionTarget, Precision, hedge_policy::AdamEpsilonConvention,
    maxcall_learning::*,
};
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/maxcall_bn_reference.json")).unwrap()
}
fn nums(v: &Value) -> Vec<f64> {
    serde_json::from_value(v.clone()).unwrap()
}
fn flat(v: &Value) -> Vec<f64> {
    if let Some(x) = v.as_f64() {
        vec![x]
    } else {
        v.as_array().unwrap().iter().flat_map(flat).collect()
    }
}
fn small() -> MaxCallTrainingConfig {
    let mut c = MaxCallTrainingConfig::default();
    c.market.spots = vec![100.; 2];
    c.market.volatilities = vec![0.2; 2];
    c.market.dividend_yields = vec![0.1; 2];
    c.market.correlation = vec![vec![1., 0.], vec![0., 1.]];
    c.hidden_width = 3;
    c.batch_rows = 4;
    c.training_paths = 4;
    c.validation_paths = 4;
    c.validation_every = 1;
    c.final_date_schedule = vec![LearningRateStage {
        updates: 2,
        learning_rate: 0.001,
    }];
    c.earlier_date_schedule = c.final_date_schedule.clone();
    c.precision = Precision::F64;
    c
}
fn imported(c: &MaxCallTrainingConfig, f: &Value) -> MaxCallNetworkSnapshot {
    let mut s = MaxCallNetworkSnapshot::initialized(3, 3, 1, c.normalization.clone(), 7).unwrap();
    for (p, v) in s
        .parameters
        .iter_mut()
        .zip(f["parameters"].as_array().unwrap())
    {
        p.values = nums(&v["values"]);
    }
    for (p, v) in s
        .running
        .iter_mut()
        .zip(f["running_state_before"].as_array().unwrap())
    {
        p.mean = nums(&v["mean"]);
        p.variance = nums(&v["variance"]);
    }
    s
}
fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "{i}: {x} vs {y},gap{} tol{tol}",
            (x - y).abs()
        );
    }
}
#[test]
fn actual_keras_known_weights_bn_all_gradients_running_state_and_adam() {
    let f = fixture();
    for precision in [Precision::F64, Precision::F32] {
        let case = &f["cases"][if precision == Precision::F64 { 0 } else { 1 }];
        let mut c = small();
        c.precision = precision;
        let s = imported(&c, case);
        let mut learner = MaxCallDateLearner::new(&c, 1, Some(&s)).unwrap();
        let x = flat(&case["inputs"]);
        let y = flat(&case["discounted_targets"]);
        learner.upload_training_features(&x, &y).unwrap();
        let tol = if precision == Precision::F64 {
            2e-10
        } else {
            4e-5
        };
        close(
            &learner.predict_training().unwrap(),
            &flat(&case["inference_before"]),
            tol,
        );
        let before = serde_json::to_value(learner.snapshot().unwrap()).unwrap();
        let analysis = learner.analyze_rows(&[0, 1, 2, 3]).unwrap();
        assert_eq!(
            before,
            serde_json::to_value(learner.snapshot().unwrap()).unwrap()
        );
        close(
            &analysis.training_prediction,
            &flat(&case["training_prediction"]),
            tol,
        );
        close(
            &[analysis.loss],
            &[case["training_mse"].as_f64().unwrap()],
            tol,
        );
        for (a, b) in analysis
            .gradients
            .iter()
            .zip(case["gradients"].as_array().unwrap())
        {
            close(&a.values, &nums(&b["values"]), tol);
        }
        for (a, b) in analysis
            .proposed_running
            .iter()
            .zip(case["native_exact_running_state_after"].as_array().unwrap())
        {
            close(&a.mean, &nums(&b["mean"]), tol);
            close(&a.variance, &nums(&b["variance"]), tol);
        }
        let update = learner
            .update_rows(&[0, 1, 2, 3], 0.001, StepTimingMode::SynchronizedStages)
            .unwrap();
        assert_eq!(update.completed_updates, 1);
        let actual_gradients = learner.last_gradient_snapshot().unwrap();
        let after = learner.snapshot().unwrap();
        let moments = learner.optimizer_snapshot().unwrap();
        for (i, ((before, after), grad)) in s
            .parameters
            .iter()
            .zip(&after.parameters)
            .zip(&actual_gradients)
            .enumerate()
        {
            let expected = before
                .values
                .iter()
                .zip(&grad.values)
                .map(|(p, g)| p - 0.001 * g / (g.abs() + 1e-8))
                .collect::<Vec<_>>();
            close(
                &after.values,
                &expected,
                if precision == Precision::F64 {
                    1e-11
                } else {
                    4e-5
                },
            );
            close(
                &moments.first[i].values,
                &grad.values.iter().map(|g| 0.1 * g).collect::<Vec<_>>(),
                tol,
            );
            close(
                &moments.second[i].values,
                &grad
                    .values
                    .iter()
                    .map(|g| 0.001 * g * g)
                    .collect::<Vec<_>>(),
                tol,
            );
            if i != 1 && i != 3 && i != 7 {
                close(
                    &after.values,
                    &nums(&case["parameters_after_canonical_Adam"][i]),
                    tol,
                );
            }
        }
    }
}
#[test]
fn bn_inference_is_batch_independent_and_snapshot_restore_is_exact() {
    let c = small();
    let f = fixture();
    let case = &f["cases"][0];
    let s = imported(&c, case);
    let x = flat(&case["inputs"]);
    let y = flat(&case["discounted_targets"]);
    let mut a = MaxCallDateLearner::new(&c, 1, Some(&s)).unwrap();
    a.upload_training_features(&x, &y).unwrap();
    a.update_rows(&[0, 1, 2, 3], 0.001, StepTimingMode::EndToEnd)
        .unwrap();
    let snapshot = a.snapshot().unwrap();
    let p = a.predict_training().unwrap();
    let mut b = MaxCallDateLearner::new(&c, 1, Some(&snapshot)).unwrap();
    b.upload_training_features(&x[..6], &y[..2]).unwrap();
    close(&b.predict_training().unwrap(), &p[..2], 1e-11);
    let opt = b.optimizer_snapshot().unwrap();
    assert_eq!(opt.completed_updates, 0);
    assert!(opt.first.iter().flat_map(|p| &p.values).all(|x| *x == 0.));
    assert_eq!(snapshot.running[0].batches, 1);
}
#[test]
fn two_backward_dates_save_final_schedule_and_resume_completed_suffix() {
    let mut c = small();
    c.market.times = vec![0., 1. / 3., 2. / 3., 1.];
    c.market.exercise = vec![false, true, true, true];
    c.training_paths = 32;
    c.validation_paths = 16;
    c.batch_rows = 8;
    c.final_date_schedule[0].updates = 4;
    c.earlier_date_schedule = c.final_date_schedule.clone();
    let mut archive = None;
    let result = train_with_progress(&c, None, |p, a| {
        if p.date_complete && archive.is_none() {
            archive = a.cloned();
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(
        result.final_checkpoint.basis,
        "final_scheduled_update_each_date"
    );
    assert_eq!(result.dates[0].completed_updates, 4);
    assert_eq!(archive.as_ref().unwrap().next_date, 1);
    let resumed = train_with_progress(&c, archive.as_ref(), |_, _| Ok(())).unwrap();
    assert_eq!(
        serde_json::to_value(&result.diagnostic_validation_checkpoint).unwrap(),
        serde_json::to_value(&resumed.diagnostic_validation_checkpoint).unwrap()
    );
    assert_eq!(
        serde_json::to_value(result.final_checkpoint).unwrap(),
        serde_json::to_value(resumed.final_checkpoint).unwrap()
    );
}
#[test]
fn preflight_rejects_budget_shape_before_unavailable_device_and_preserves_state_on_error() {
    let mut c = small();
    c.target = ExecutionTarget::Cuda { index: 999 };
    c.compute_budget_bytes = 1;
    assert!(matches!(
        MaxCallDateLearner::new(&c, 1, None),
        Err(ComputeError::BudgetExceeded(_))
    ));
    let c = small();
    let mut a = MaxCallDateLearner::new(&c, 1, None).unwrap();
    let before = serde_json::to_value(a.snapshot().unwrap()).unwrap();
    assert!(
        a.update_rows(&[0, 1], 0.001, StepTimingMode::EndToEnd)
            .is_err()
    );
    assert_eq!(before, serde_json::to_value(a.snapshot().unwrap()).unwrap());
}
#[test]
fn source_dimensions_and_large_batch_resource_plan() {
    let c = MaxCallTrainingConfig {
        training_paths: 1_048_576,
        validation_paths: 65_536,
        ..Default::default()
    };
    let p = c.validate().unwrap();
    assert!(p.modeled_compute_bytes < 4 * 1024 * 1024 * 1024);
    assert!(p.modeled_host_bytes < 8 * 1024 * 1024 * 1024);
    assert_eq!(
        c.final_date_schedule
            .iter()
            .map(|s| s.updates)
            .sum::<usize>(),
        6000
    );
    assert_eq!(
        c.earlier_date_schedule
            .iter()
            .map(|s| s.updates)
            .sum::<usize>(),
        3500
    );
}
#[test]
fn tensorflow_epsilon_convention_is_explicit() {
    let mut c = small();
    c.epsilon_convention = AdamEpsilonConvention::TensorflowUncorrected;
    let f = fixture();
    let s = imported(&c, &f["cases"][0]);
    let mut a = MaxCallDateLearner::new(&c, 1, Some(&s)).unwrap();
    a.upload_training_features(
        &flat(&f["cases"][0]["inputs"]),
        &flat(&f["cases"][0]["discounted_targets"]),
    )
    .unwrap();
    a.update_rows(&[0, 1, 2, 3], 0.001, StepTimingMode::EndToEnd)
        .unwrap();
    assert_eq!(a.completed_updates(), 1);
}

#[test]
fn malformed_partial_refuses_terminal_nextdate_before_device() {
    let mut c = small();
    c.market.times = vec![0., 0.5, 1.];
    c.market.exercise = vec![false, true, true];
    let mut archive = None;
    train_with_progress(&c, None, |_, p| {
        if p.is_some() {
            archive = p.cloned();
        }
        Ok(())
    })
    .unwrap();
    let mut p = archive.unwrap();
    p.next_date = 2;
    p.config.target = ExecutionTarget::Cuda { index: 999 };
    assert!(matches!(
        train_with_progress(&p.config, Some(&p), |_, _| Ok(())),
        Err(ComputeError::InvalidInput(_))
    ));
}
#[test]
fn optional_zero_translations_preserve_warm_function_and_are_idempotent() {
    let f = fixture();
    let c = small();
    let raw = imported(&c, &f["cases"][0]);
    let converted = raw.canonicalized_translations().unwrap();
    let x = flat(&f["cases"][0]["inputs"]);
    for row in x.chunks(3) {
        close(
            &raw.predict_features(row).unwrap(),
            &converted.predict_features(row).unwrap(),
            1e-12,
        );
    }
    assert_eq!(
        serde_json::to_value(&converted).unwrap(),
        serde_json::to_value(converted.canonicalized_translations().unwrap()).unwrap()
    );
    assert!(
        !serde_json::to_string(&raw)
            .unwrap()
            .contains("canonical_zero_translations")
    );
    let mut invalid = converted;
    invalid.parameters[1].values[0] = 0.1;
    assert!(invalid.validate().is_err());
}
#[test]
fn canonical_reference_full_gradients_parameters_and_function_cpu_metal() {
    let f: Value =
        serde_json::from_str(include_str!("fixtures/maxcall_bn_canonical_reference.json")).unwrap();
    let cases = vec![
        (ExecutionTarget::Cpu, Precision::F64),
        (ExecutionTarget::Cpu, Precision::F32),
    ];
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let cases = cases
        .into_iter()
        .chain([(ExecutionTarget::Metal, Precision::F32)])
        .collect::<Vec<_>>();
    for (target, precision) in cases {
        let case = &f["cases"][if precision == Precision::F64 { 0 } else { 1 }];
        let mut c = small();
        c.target = target;
        c.precision = precision;
        c.normalization.canonical_zero_translations = true;
        let snapshot = imported(&c, case);
        let mut learner = MaxCallDateLearner::new(&c, 1, Some(&snapshot)).unwrap();
        learner
            .upload_training_features(&flat(&case["inputs"]), &flat(&case["discounted_targets"]))
            .unwrap();
        let tol = if precision == Precision::F64 {
            2e-10
        } else {
            4e-5
        };
        close(
            &learner.predict_training().unwrap(),
            &flat(&case["inference_before"]),
            tol,
        );
        let analysis = learner.analyze_rows(&[0, 1, 2, 3]).unwrap();
        close(
            &analysis.training_prediction,
            &flat(&case["training_prediction"]),
            tol,
        );
        for (a, b) in analysis
            .gradients
            .iter()
            .zip(case["gradients"].as_array().unwrap())
        {
            close(&a.values, &nums(&b["values"]), tol);
        }
        learner
            .update_rows(&[0, 1, 2, 3], 0.001, StepTimingMode::EndToEnd)
            .unwrap();
        let final_snapshot = learner.snapshot().unwrap();
        for (i, p) in final_snapshot.parameters.iter().enumerate() {
            close(
                &p.values,
                &nums(&case["parameters_after_canonical_Adam"][i]),
                tol,
            );
        }
        close(
            &learner.predict_training().unwrap(),
            &flat(&case["inference_after_canonical_Adam"]),
            if precision == Precision::F64 {
                2e-9
            } else {
                5e-5
            },
        );
        for i in [1, 3, 7] {
            assert!(final_snapshot.parameters[i].values.iter().all(|v| *v == 0.));
        }
    }
}
