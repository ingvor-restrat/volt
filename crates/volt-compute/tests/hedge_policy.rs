use burn::tensor::{DType, Tensor, TensorCreationOptions, TensorData};
use volt_compute::hedge_policy::{HedgePolicy, HedgePolicyConfig, PolicySnapshot, strict_matmul};
use volt_compute::{ExecutionTarget, Precision, Session};
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
fn tensor(s: &Session, values: Vec<f64>, shape: [usize; 2]) -> Tensor<2> {
    Tensor::from_data(
        TensorData::new(values, shape),
        TensorCreationOptions::new(s.device().clone()).with_dtype(
            if s.precision() == Precision::F64 {
                DType::F64
            } else {
                DType::F32
            },
        ),
    )
}
fn read(t: Tensor<2>) -> Vec<f64> {
    t.without_autodiff().try_to_vec_as::<f64>().unwrap()
}
fn close(a: f64, b: f64, abs: f64, rel: f64) {
    assert!(
        (a - b).abs() <= abs + rel * b.abs(),
        "{a:.16e} vs {b:.16e}, error {}",
        (a - b).abs()
    );
}
fn config(batch: usize) -> HedgePolicyConfig {
    let mut c = HedgePolicyConfig::paper_black_scholes(batch);
    c.time_steps = 2;
    c.hidden_width = 3;
    c.batch_norm.enabled = false;
    c
}
fn controlled(s: &Session, c: HedgePolicyConfig) -> PolicySnapshot {
    let mut p = HedgePolicy::new(s, c, 17).unwrap().snapshot().unwrap();
    for (i, x) in p.parameters.iter_mut().enumerate() {
        x.values.fill(match i % 10 {
            0 | 4 | 8 => 0.2,
            2 | 6 => 1.0,
            _ => 0.3,
        });
    }
    p
}
fn inputs(s: &Session) -> (Tensor<2>, Tensor<2>) {
    (
        tensor(s, vec![1.1, 1.3, 1.9], [3, 1]),
        tensor(s, vec![0.2, 0.4, 0.3], [3, 1]),
    )
}
fn recurrent_loss(s: &Session, p: &mut HedgePolicy, train: bool) -> Tensor<2> {
    let (x, prev) = inputs(s);
    let first = p.forward_at(0, x.clone(), prev, train).unwrap();
    let last = p.forward_at(1, x, first, train).unwrap();
    last.square() * 0.5
}

#[test]
fn strict_native_dtype_rectangular_products_and_gradients_match_independent_scalar() {
    for precision in precisions() {
        let s = Session::new(target(), precision, 4096).unwrap();
        let (b, k, o) = (7, 19, 13);
        let a: Vec<_> = (0..b * k)
            .map(|i| ((i * 29 % 113) as f64 - 51.0) / 127.0)
            .collect();
        let w: Vec<_> = (0..k * o)
            .map(|i| ((i * 37 % 97) as f64 - 47.0) / 127.0)
            .collect();
        let left = tensor(&s, a.clone(), [b, k]).autodiff().require_grad();
        let right = tensor(&s, w.clone(), [k, o]).autodiff().require_grad();
        let product = strict_matmul(left.clone(), right.clone(), b * k * o).unwrap();
        assert_eq!(
            product.dtype(),
            if precision == Precision::F64 {
                DType::F64
            } else {
                DType::F32
            }
        );
        let actual = read(product.clone());
        for row in 0..b {
            for col in 0..o {
                let expected = (0..k).map(|j| a[row * k + j] * w[j * o + col]).sum::<f64>();
                close(
                    actual[row * o + col],
                    expected,
                    if precision == Precision::F64 {
                        1e-13
                    } else {
                        2e-7
                    },
                    if precision == Precision::F64 {
                        1e-12
                    } else {
                        2e-6
                    },
                );
            }
        }
        let grads = product.sum().backward();
        let gradient = read(left.grad(&grads).unwrap());
        for row in 0..b {
            for j in 0..k {
                close(
                    gradient[row * k + j],
                    (0..o).map(|col| w[j * o + col]).sum(),
                    if precision == Precision::F64 {
                        1e-13
                    } else {
                        1e-6
                    },
                    0.0,
                );
            }
        }
        assert!(
            strict_matmul(tensor(&s, a, [b, k]), tensor(&s, w, [k, o]), b * k * o - 1).is_err()
        );
    }
}

#[test]
fn gradients_flow_through_previous_decisions_and_match_finite_differences() {
    for precision in precisions() {
        let s = Session::new(target(), precision, 4096).unwrap();
        let snapshot = controlled(&s, config(8));
        let mut p = HedgePolicy::from_snapshot(&s, snapshot.clone()).unwrap();
        let leaves = p.parameters();
        let loss = recurrent_loss(&s, &mut p, true).sum();
        let grads = loss.backward();
        let actual = read(leaves[0].grad(&grads).unwrap())[0];
        assert!(actual.abs() > 0.01);
        let h = if precision == Precision::F64 {
            1e-5
        } else {
            1e-3
        };
        let at = |offset: f64| {
            let mut snap = snapshot.clone();
            snap.parameters[0].values[0] += offset;
            let mut p = HedgePolicy::from_snapshot(&s, snap).unwrap();
            recurrent_loss(&s, &mut p, false)
                .sum()
                .try_to_vec_as::<f64>()
                .unwrap()[0]
        };
        let finite = (at(h) - at(-h)) / (2.0 * h);
        close(
            actual,
            finite,
            if precision == Precision::F64 {
                2e-9
            } else {
                2e-4
            },
            if precision == Precision::F64 {
                1e-8
            } else {
                2e-3
            },
        );
        // Both time nets retain gradients; the first net only enters through holdings.
        assert!(leaves[10].grad(&grads).is_some());
    }
}

#[test]
fn adam_first_step_matches_independent_scalar_formula_and_rejects_invalid_steps() {
    for precision in precisions() {
        let s = Session::new(target(), precision, 4096).unwrap();
        let mut c = config(8);
        c.time_steps = 1;
        c.hidden_width = 1;
        c.include_previous_holdings = false;
        let snap = controlled(&s, c);
        let mut p = HedgePolicy::from_snapshot(&s, snap).unwrap();
        let before = p.snapshot().unwrap();
        let output = p
            .forward_at(
                0,
                tensor(&s, vec![1.2, 1.4], [2, 1]),
                tensor(&s, vec![0.0; 2], [2, 1]),
                true,
            )
            .unwrap();
        let mut grads = (output.square() * 0.5).sum().backward();
        let gradient = read(p.parameters()[9].grad(&grads).unwrap())[0];
        assert!(p.backward_update(&mut grads, 0.0).is_err());
        let stats = p.backward_update(&mut grads, 0.005).unwrap();
        assert!(stats.gradient_l2_norm.is_finite());
        let after = p.snapshot().unwrap();
        let expected = before.parameters[9].values[0]
            - 0.005 * gradient / (gradient.abs() + before.config.adam.epsilon);
        close(
            after.parameters[9].values[0],
            expected,
            if precision == Precision::F64 {
                1e-14
            } else {
                3e-8
            },
            0.0,
        );
        close(
            after.parameters[9].first_moment[0],
            gradient * (1.0 - before.config.adam.beta1),
            if precision == Precision::F64 {
                1e-14
            } else {
                1e-7
            },
            0.0,
        );
        close(
            after.parameters[9].second_moment[0],
            gradient * gradient * (1.0 - before.config.adam.beta2),
            if precision == Precision::F64 {
                1e-14
            } else {
                1e-7
            },
            0.0,
        );
        assert_eq!(after.parameters[9].adam_step, 1);
        assert_eq!(after.parameters[2].adam_step, 0);
    }
}

fn training_step(s: &Session, p: &mut HedgePolicy) {
    let mut previous = tensor(s, vec![0.0; 4], [4, 1]);
    let features = tensor(s, vec![1.0, 2.0, 3.0, 4.0], [4, 1]);
    for step in 0..p.config().time_steps {
        previous = p
            .forward_at(step, features.clone(), previous, true)
            .unwrap();
    }
    let mut gradients = (previous - 0.4).square().sum().backward();
    p.backward_update(&mut gradients, 0.005).unwrap();
}
#[test]
fn checkpoint_roundtrip_inference_and_optimizer_resume_are_exact() {
    for precision in precisions() {
        let s = Session::new(target(), precision, 4096).unwrap();
        let mut c = config(8);
        c.batch_norm.enabled = true;
        let mut original = HedgePolicy::new(&s, c, 1234).unwrap();
        for _ in 0..3 {
            training_step(&s, &mut original);
        }
        let snapshot = original.snapshot().unwrap();
        let serialized = serde_json::to_vec(&snapshot).unwrap();
        let decoded: PolicySnapshot = serde_json::from_slice(&serialized).unwrap();
        let mut resumed = HedgePolicy::from_snapshot(&s, decoded).unwrap();
        let (x, prev) = inputs(&s);
        let one = original
            .forward_at(0, x.clone(), prev.clone(), false)
            .unwrap();
        let two = resumed.forward_at(0, x, prev, false).unwrap();
        assert!(!one.is_autodiff());
        assert_eq!(read(one), read(two));
        training_step(&s, &mut original);
        training_step(&s, &mut resumed);
        assert_eq!(
            serde_json::to_value(original.snapshot().unwrap()).unwrap(),
            serde_json::to_value(resumed.snapshot().unwrap()).unwrap()
        );
        let path = std::env::temp_dir().join(format!(
            "volt-policy-{}-{precision:?}.json",
            std::process::id()
        ));
        original.save(&path).unwrap();
        let loaded = HedgePolicy::load(&s, &path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(
            serde_json::to_value(original.snapshot().unwrap()).unwrap(),
            serde_json::to_value(loaded.snapshot().unwrap()).unwrap()
        );
    }
}

#[test]
fn batch_norm_training_and_inference_have_explicit_different_state_semantics() {
    for precision in precisions() {
        let s = Session::new(target(), precision, 4096).unwrap();
        let mut c = config(8);
        c.hidden_width = 2;
        c.batch_norm.enabled = true;
        let mut p = HedgePolicy::from_snapshot(&s, controlled(&s, c)).unwrap();
        let output = p
            .forward_at(
                0,
                tensor(&s, vec![1.0, 2.0, 3.0, 4.0], [4, 1]),
                tensor(&s, vec![0.5; 4], [4, 1]),
                true,
            )
            .unwrap();
        assert!(output.is_autodiff());
        let state = p.snapshot().unwrap();
        for mean in &state.batch_norm[0].running_mean {
            close(
                *mean,
                0.09,
                if precision == Precision::F64 {
                    1e-14
                } else {
                    1e-7
                },
                0.0,
            );
        }
        for variance in &state.batch_norm[0].running_variance {
            close(
                *variance,
                0.9066666666666666,
                if precision == Precision::F64 {
                    1e-14
                } else {
                    1e-7
                },
                0.0,
            );
        }
        let before = serde_json::to_value(&state).unwrap();
        let alone = p
            .forward_at(
                0,
                tensor(&s, vec![1.0], [1, 1]),
                tensor(&s, vec![0.5], [1, 1]),
                false,
            )
            .unwrap();
        let together = p
            .forward_at(
                0,
                tensor(&s, vec![1.0, 7.0], [2, 1]),
                tensor(&s, vec![0.5, 3.0], [2, 1]),
                false,
            )
            .unwrap();
        assert_eq!(read(alone)[0], read(together)[0]);
        assert_eq!(before, serde_json::to_value(p.snapshot().unwrap()).unwrap());
        assert!(
            p.forward_at(
                0,
                tensor(&s, vec![1.0], [1, 1]),
                tensor(&s, vec![0.5], [1, 1]),
                true
            )
            .is_err()
        );
    }
}

#[test]
fn signed_actions_scaling_and_nonrecurrent_ablation_are_real_model_choices() {
    let s = Session::new(target(), Precision::F32, 4096).unwrap();
    let mut c = config(8);
    c.include_previous_holdings = false;
    c.action_scales = Some(vec![1000.0]);
    let mut snap = controlled(&s, c);
    assert_eq!(snap.parameters[0].shape, [1, 3]);
    snap.parameters[8].values.fill(0.0);
    snap.parameters[9].values.fill(-2.0);
    let mut p = HedgePolicy::from_snapshot(&s, snap).unwrap();
    let one = p
        .forward_at(
            0,
            tensor(&s, vec![1.0], [1, 1]),
            tensor(&s, vec![1000.0], [1, 1]),
            false,
        )
        .unwrap();
    let two = p
        .forward_at(
            0,
            tensor(&s, vec![1.0], [1, 1]),
            tensor(&s, vec![-1000.0], [1, 1]),
            false,
        )
        .unwrap();
    assert_eq!(read(one), vec![-2000.0]);
    assert_eq!(read(two), vec![-2000.0]);
}

#[test]
fn invalid_inputs_and_failed_gradient_update_do_not_poison_state() {
    let s = Session::new(target(), Precision::F32, 4096).unwrap();
    let mut c = config(8);
    c.batch_norm.enabled = true;
    let mut p = HedgePolicy::new(&s, c, 37).unwrap();
    let before = serde_json::to_value(p.snapshot().unwrap()).unwrap();
    assert!(
        p.forward_at(
            0,
            tensor(&s, vec![f64::NAN, 1.0], [2, 1]),
            tensor(&s, vec![0.0; 2], [2, 1]),
            true
        )
        .is_err()
    );
    assert_eq!(before, serde_json::to_value(p.snapshot().unwrap()).unwrap());
    assert!(
        p.forward_at(
            2,
            tensor(&s, vec![1.0; 2], [2, 1]),
            tensor(&s, vec![0.0; 2], [2, 1]),
            true
        )
        .is_err()
    );
    let out = p
        .forward_at(
            0,
            tensor(&s, vec![1.0, 2.0], [2, 1]),
            tensor(&s, vec![0.0; 2], [2, 1]),
            true,
        )
        .unwrap();
    let before = serde_json::to_value(p.snapshot().unwrap()).unwrap();
    let mut grads = (out.sum() * 1e40).backward();
    assert!(p.backward_update(&mut grads, 0.005).is_err());
    assert_eq!(before, serde_json::to_value(p.snapshot().unwrap()).unwrap());
    training_step(&s, &mut p);
    let mut bad = p.snapshot().unwrap();
    bad.parameters[0].second_moment[0] = -1.0;
    assert!(HedgePolicy::from_snapshot(&s, bad).is_err());
    let mut bad = p.snapshot().unwrap();
    bad.parameters[0].shape = [99, 99];
    assert!(HedgePolicy::from_snapshot(&s, bad).is_err());
}

#[test]
fn default_paper_profile_has_thirty_distinct_nets_and_two_signed_hedges() {
    let s = Session::new(target(), Precision::F32, 4096).unwrap();
    let c = HedgePolicyConfig::paper_heston(4096);
    assert_eq!(
        (c.time_steps, c.hidden_width, c.input_width(), c.hedge_count),
        (30, 17, 4, 2)
    );
    assert!(c.action_scales.is_none() && c.feature_affine.is_none());
    let p = HedgePolicy::new(&s, c, 123).unwrap();
    assert_eq!(p.parameters().len(), 300);
    let snap = p.snapshot().unwrap();
    assert_ne!(snap.parameters[0].values, snap.parameters[10].values);
}

#[test]
fn full_thirty_step_two_instrument_policy_backpropagates_and_refreshes_all_leaves() {
    let s = Session::new(target(), Precision::F32, 4096).unwrap();
    let mut p = HedgePolicy::new(&s, HedgePolicyConfig::paper_heston(16), 777).unwrap();
    let mut previous = tensor(&s, vec![0.0; 8], [4, 2]);
    let mut loss = None;
    for step in 0..30 {
        let features = tensor(
            &s,
            vec![
                4.5 + step as f64 * 0.001,
                0.03,
                4.6,
                0.05,
                4.4,
                0.02,
                4.7,
                0.06,
            ],
            [4, 2],
        );
        previous = p.forward_at(step, features, previous, true).unwrap();
        let term = previous.clone().square().sum() * (1.0 / 30.0);
        loss = Some(match loss {
            Some(total) => total + term,
            None => term,
        });
    }
    let mut gradients = loss.unwrap().backward();
    let stats = p.backward_update(&mut gradients, 0.005).unwrap();
    assert_eq!(stats.updated_parameters, 300);
    assert!(stats.gradient_l2_norm.is_finite() && stats.gradient_l2_norm > 0.0);
    assert!(
        p.snapshot()
            .unwrap()
            .batch_norm
            .iter()
            .all(|n| n.batches_tracked == 1)
    );
    // A second completely fresh graph succeeds after the update, while recurrent
    // holdings from the consumed previous graph are deliberately not reused.
    let input = tensor(&s, vec![4.5, 0.03, 4.6, 0.05], [2, 2]);
    let next = p
        .forward_at(0, input, tensor(&s, vec![0.0; 4], [2, 2]), true)
        .unwrap();
    let mut gradients = next.square().sum().backward();
    p.backward_update(&mut gradients, 0.005).unwrap();
}

#[test]
fn batch_norm_training_graph_gradients_match_finite_difference() {
    if target() == ExecutionTarget::Metal {
        return;
    }
    let s = Session::new(target(), Precision::F64, 4096).unwrap();
    let mut c = config(8);
    c.batch_norm.enabled = true;
    let snap = HedgePolicy::new(&s, c, 908).unwrap().snapshot().unwrap();
    let mut p = HedgePolicy::from_snapshot(&s, snap.clone()).unwrap();
    let leaves = p.parameters();
    let grads = recurrent_loss(&s, &mut p, true).sum().backward();
    let mut nonzero = 0;
    for (parameter, element) in [(0, 0), (2, 0), (4, 1), (12, 0), (18, 1)] {
        let gradient = read(leaves[parameter].grad(&grads).unwrap())[element];
        if gradient.abs() > 1e-6 {
            nonzero += 1;
        }
        let at = |offset: f64| {
            let mut snap = snap.clone();
            snap.parameters[parameter].values[element] += offset;
            let mut p = HedgePolicy::from_snapshot(&s, snap).unwrap();
            recurrent_loss(&s, &mut p, true)
                .sum()
                .try_to_vec_as::<f64>()
                .unwrap()[0]
        };
        let h = 1e-5;
        close(gradient, (at(h) - at(-h)) / (2.0 * h), 1e-7, 2e-6);
    }
    assert!(
        nonzero >= 3,
        "finite-difference checks must exercise nonzero BatchNorm gradients"
    );
}

#[test]
fn malformed_snapshot_and_precision_preflight_need_no_session_or_device() {
    use volt_compute::hedge_policy::{
        BatchNormSnapshot, CHECKPOINT_SCHEMA, ParameterSnapshot, validate_policy_config,
        validate_policy_snapshot,
    };
    let mut c = HedgePolicyConfig::paper_black_scholes(4);
    c.time_steps = 1;
    c.hidden_width = 1;
    let parameters = (0..10)
        .map(|i| {
            let shape = if i == 0 { [2, 1] } else { [1, 1] };
            let n = shape[0] * shape[1];
            ParameterSnapshot {
                shape,
                values: vec![0.0; n],
                first_moment: vec![0.0; n],
                second_moment: vec![0.0; n],
                adam_step: 0,
            }
        })
        .collect();
    let snapshot = PolicySnapshot {
        schema: CHECKPOINT_SCHEMA.into(),
        config: c.clone(),
        precision: Precision::F32,
        updates: 0,
        parameters,
        batch_norm: vec![
            BatchNormSnapshot {
                running_mean: vec![0.0],
                running_variance: vec![1.0],
                batches_tracked: 0
            };
            2
        ],
    };
    validate_policy_snapshot(&snapshot).unwrap();
    let mut bad = snapshot.clone();
    bad.schema = "unknown".into();
    assert!(validate_policy_snapshot(&bad).is_err());
    let mut bad = snapshot.clone();
    bad.parameters[0].values[0] = f64::MAX;
    assert!(validate_policy_snapshot(&bad).is_err());
    let mut bad = snapshot.clone();
    bad.parameters[0].second_moment[0] = -1.0;
    assert!(validate_policy_snapshot(&bad).is_err());
    let mut bad = snapshot.clone();
    bad.parameters[0].first_moment[0] = 1.0;
    assert!(validate_policy_snapshot(&bad).is_err());
    let mut bad = snapshot.clone();
    bad.parameters[0].values.pop();
    assert!(validate_policy_snapshot(&bad).is_err());
    let mut bad = snapshot;
    bad.batch_norm[0].running_variance[0] = f64::NAN;
    assert!(validate_policy_snapshot(&bad).is_err());
    c.adam.epsilon = 1e-100;
    assert!(validate_policy_config(&c, Precision::F32).is_err());
    assert!(validate_policy_config(&c, Precision::F64).is_ok());
}

#[test]
fn frozen_weight_calibration_refreshes_exact_unbiased_stats_and_replays_recurrence() {
    for precision in precisions() {
        let s = Session::new(target(), precision, 4096).unwrap();
        let mut c = config(8);
        c.batch_norm.enabled = true;
        let mut p = HedgePolicy::new(&s, c, 1234).unwrap();
        for _ in 0..3 {
            training_step(&s, &mut p);
        }
        let before = p.snapshot().unwrap();
        let features = tensor(&s, vec![2.0, 4.0, 1.0, 7.0], [4, 1])
            .autodiff()
            .require_grad();
        let mut previous = tensor(&s, vec![0.0; 4], [4, 1]).autodiff().require_grad();
        let mut calibrated = Vec::new();
        for step in 0..2 {
            previous = p.calibrate_at(step, features.clone(), previous).unwrap();
            assert!(!previous.is_autodiff());
            calibrated.push(read(previous.clone()));
        }
        let after = p.snapshot().unwrap();
        assert_eq!(after.updates, before.updates);
        assert_eq!(
            serde_json::to_value(&after.parameters).unwrap(),
            serde_json::to_value(&before.parameters).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&after.config).unwrap(),
            serde_json::to_value(&before.config).unwrap()
        );
        for (a, b) in after.batch_norm.iter().zip(&before.batch_norm) {
            assert_eq!(a.batches_tracked, b.batches_tracked);
        }
        assert_ne!(
            after.batch_norm[0].running_mean,
            before.batch_norm[0].running_mean
        );
        // First layer receives calibration X=[2,4,1,7] and previous holdings zero;
        // mean(X)=3.5, unbiased var(X)=7. Stored moments have weight 1, not EMA.
        for column in 0..3 {
            let weight = before.parameters[0].values[column];
            let bias = before.parameters[1].values[column];
            let tolerance = if precision == Precision::F64 {
                1e-12
            } else {
                3e-6
            };
            close(
                after.batch_norm[0].running_mean[column],
                3.5 * weight + bias,
                tolerance,
                0.0,
            );
            close(
                after.batch_norm[0].running_variance[column],
                7.0 * weight * weight,
                tolerance,
                1e-6,
            );
        }
        let mut previous = tensor(&s, vec![0.0; 4], [4, 1]);
        for (step, expected) in calibrated.into_iter().enumerate() {
            previous = p
                .forward_at(step, features.clone(), previous, false)
                .unwrap();
            assert_eq!(read(previous.clone()), expected);
        }
        assert_eq!(
            serde_json::to_value(&after).unwrap(),
            serde_json::to_value(p.snapshot().unwrap()).unwrap()
        );
        assert!(
            p.calibrate_at(
                0,
                tensor(&s, vec![f64::NAN, 2.0, 3.0, 4.0], [4, 1]),
                tensor(&s, vec![0.0; 4], [4, 1])
            )
            .is_err()
        );
        assert_eq!(
            serde_json::to_value(after).unwrap(),
            serde_json::to_value(p.snapshot().unwrap()).unwrap()
        );
    }
}

#[test]
fn deterministic_calibration_state_has_zero_variance_and_preserves_training_counters() {
    for precision in precisions() {
        let s = Session::new(target(), precision, 4096).unwrap();
        let mut c = config(8);
        c.batch_norm.enabled = true;
        let mut snap = controlled(&s, c);
        for n in &mut snap.batch_norm {
            n.batches_tracked = u64::MAX;
        }
        let mut p = HedgePolicy::from_snapshot(&s, snap).unwrap();
        let features = tensor(&s, vec![4.5; 4], [4, 1]);
        let previous = tensor(&s, vec![0.0; 4], [4, 1]);
        let calibrated = p
            .calibrate_at(0, features.clone(), previous.clone())
            .unwrap();
        assert!(!calibrated.is_autodiff());
        let state = p.snapshot().unwrap();
        assert!(
            state.batch_norm[0]
                .running_variance
                .iter()
                .all(|x| *x == 0.0)
        );
        assert!(
            state
                .batch_norm
                .iter()
                .all(|n| n.batches_tracked == u64::MAX)
        );
        let evaluated = p
            .forward_at(0, features.clone(), previous.clone(), false)
            .unwrap();
        assert_eq!(read(calibrated), read(evaluated));
        assert!(p.forward_at(0, features, previous, true).is_err());
    }
}

#[test]
fn calibration_without_batch_norm_is_pure_inference_and_allows_single_rows() {
    for precision in precisions() {
        let s = Session::new(target(), precision, 4096).unwrap();
        let mut p = HedgePolicy::from_snapshot(&s, controlled(&s, config(8))).unwrap();
        let before = serde_json::to_value(p.snapshot().unwrap()).unwrap();
        let features = tensor(&s, vec![1.3], [1, 1]).autodiff().require_grad();
        let previous = tensor(&s, vec![-0.4], [1, 1]).autodiff().require_grad();
        let calibrated = p
            .calibrate_at(0, features.clone(), previous.clone())
            .unwrap();
        assert!(!calibrated.is_autodiff());
        let evaluated = p.forward_at(0, features, previous, false).unwrap();
        assert_eq!(read(calibrated), read(evaluated));
        assert_eq!(before, serde_json::to_value(p.snapshot().unwrap()).unwrap());
    }
}

#[test]
fn previous_holding_units_preserve_forward_values_and_chain_rule() {
    for precision in precisions() {
        let s = Session::new(target(), precision, 4096).unwrap();
        let reference = controlled(&s, config(8));
        let mut scaled = reference.clone();
        scaled.config.previous_holding_scales = Some(vec![1000.0]);
        let mut raw = HedgePolicy::from_snapshot(&s, reference).unwrap();
        let mut normalized = HedgePolicy::from_snapshot(&s, scaled).unwrap();
        let previous = tensor(&s, vec![0.2, 0.4, 0.3], [3, 1])
            .autodiff()
            .require_grad();
        let units = tensor(&s, vec![200.0, 400.0, 300.0], [3, 1])
            .autodiff()
            .require_grad();
        let features = tensor(&s, vec![1.1, 1.3, 1.9], [3, 1]);
        let a = raw
            .forward_at(0, features.clone(), previous.clone(), true)
            .unwrap();
        let b = normalized
            .forward_at(0, features, units.clone(), true)
            .unwrap();
        for (x, y) in read(a.clone()).into_iter().zip(read(b.clone())) {
            close(x, y, 1e-6, 1e-6);
        }
        let mut ga = a.sum().backward();
        let mut gb = b.sum().backward();
        let da = read(previous.grad_remove(&mut ga).unwrap());
        let db = read(units.grad_remove(&mut gb).unwrap());
        assert!(da.iter().any(|x| x.abs() > 0.0));
        for (x, y) in da.into_iter().zip(db) {
            close(x, 1000.0 * y, 1e-6, 1e-6);
        }
        let mut invalid = config(8);
        invalid.previous_holding_scales = Some(vec![0.0]);
        assert!(HedgePolicy::new(&s, invalid, 17).is_err());
    }
}
