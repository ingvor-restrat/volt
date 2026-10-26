//! Own numerical fixture executed with actual Keras 2.15 single-step LSTM
//! layers. It checks reset semantics, wealth/holdings feedback and shared AD.
use burn::tensor::{DType, Tensor, TensorCreationOptions, TensorData};
use serde_json::{Value, json};
use volt_compute::{
    ExecutionTarget, Precision, Session,
    hedge_policy::{
        AdamEpsilonConvention, HedgePolicy, HedgePolicyConfig, PolicyForwardContext,
        RecurrentActivation, SharedLstmResetConfig, SmallTradeGateConfig,
    },
    native_learner::{Command, LearnerConfig, NativeLearner},
};
fn fixture() -> Value {
    serde_json::from_str(KERAS_FIXTURE).unwrap()
}
fn numbers(v: &Value) -> Vec<f64> {
    serde_json::from_value(v.clone()).unwrap()
}
fn config(dropout: f64) -> HedgePolicyConfig {
    let mut c = HedgePolicyConfig::paper_heston(32);
    c.time_steps = 3;
    c.hidden_width = 3;
    c.batch_norm.enabled = false;
    c.include_wealth = true;
    c.shared_lstm_reset = Some(SharedLstmResetConfig {
        dropout_probability: dropout,
        dropout_seed: 713,
        recurrent_activation: RecurrentActivation::Sigmoid,
    });
    c
}
fn t2(s: &Session, x: Vec<f64>, shape: [usize; 2]) -> Tensor<2> {
    Tensor::from_data(
        TensorData::new(x, shape),
        TensorCreationOptions::new(s.device().clone()).with_dtype(
            if s.precision() == Precision::F64 {
                DType::F64
            } else {
                DType::F32
            },
        ),
    )
}
fn t1(s: &Session, x: Vec<f64>) -> Tensor<1> {
    let n = x.len();
    t2(s, x, [n, 1]).reshape([n])
}
fn read<const D: usize>(x: Tensor<D>) -> Vec<f64> {
    x.without_autodiff().try_to_vec_as::<f64>().unwrap()
}
fn policy(s: &Session, f: &Value, dropout: f64) -> HedgePolicy {
    let p = HedgePolicy::new(s, config(dropout), 37).unwrap();
    let mut snap = p.snapshot().unwrap();
    for (i, x) in f["parameters"].as_array().unwrap().iter().enumerate() {
        snap.parameters[i].values = numbers(&x["values"]);
    }
    HedgePolicy::from_snapshot(s, snap).unwrap()
}
fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "at{i}: {x} vs{y},gap{}",
            (x - y).abs()
        );
    }
}
fn rollout(
    s: &Session,
    p: &mut HedgePolicy,
    f: &Value,
    detach_wealth: bool,
) -> (Tensor<1>, Vec<Tensor<2>>) {
    let n = 4;
    let marks = numbers(&f["marks"]);
    let features = numbers(&f["features"]);
    let nf = f["feature_count"].as_u64().unwrap() as usize;
    let mark = |day: usize| {
        t2(
            s,
            (0..n)
                .flat_map(|r| {
                    marks[(r * 4 + day) * 2..(r * 4 + day + 1) * 2]
                        .iter()
                        .copied()
                })
                .collect(),
            [n, 2],
        )
    };
    let fees = t2(s, numbers(&f["trading_costs"]), [1, 2]);
    let terminal = t2(s, numbers(&f["liquidation_costs"]), [1, 2]);
    let mut value = t1(s, vec![f["premium"].as_f64().unwrap(); n]);
    let mut prev = t2(s, vec![0.; n * 2], [n, 2]);
    let mut holdings = Vec::new();
    for day in 0..3 {
        let x = t2(
            s,
            (0..n)
                .flat_map(|r| {
                    features[(r * 3 + day) * nf..(r * 3 + day + 1) * nf]
                        .iter()
                        .copied()
                })
                .collect(),
            [n, nf],
        );
        let wealth = if detach_wealth {
            value.clone().without_autodiff()
        } else {
            value.clone()
        };
        let signal = f.get("force_trade_signal").map(|v| {
            let raw = numbers(v);
            t2(
                s,
                (0..n)
                    .flat_map(|r| {
                        raw[(r * 3 + day) * 2..(r * 3 + day + 1) * 2]
                            .iter()
                            .copied()
                    })
                    .collect(),
                [n, 2],
            )
        });
        let target = p
            .forward_at_with_context(
                day,
                x,
                prev.clone(),
                PolicyForwardContext {
                    wealth: Some(wealth),
                    force_trade_signal: signal,
                    sharpen_gate: f.get("sharpen_gate").and_then(Value::as_bool),
                },
                f.get("train_mode").and_then(Value::as_bool).unwrap_or(true),
            )
            .unwrap();
        let charge = ((target.clone() - prev).abs() * mark(day) * fees.clone())
            .sum_dim(1)
            .reshape([n]);
        let cash = value - (target.clone() * mark(day)).sum_dim(1).reshape([n]) - charge;
        let income = t2(
            s,
            numbers(&f["income_growth"])[day * 2..day * 2 + 2].to_vec(),
            [1, 2],
        );
        value = cash * f["cash_growth"][day].as_f64().unwrap()
            + (target.clone() * mark(day + 1) * income)
                .sum_dim(1)
                .reshape([n]);
        prev = target.clone();
        holdings.push(target);
    }
    (
        value
            - (prev.abs() * mark(3) * terminal).sum_dim(1).reshape([n])
            - t1(s, numbers(&f["payoffs"])),
        holdings,
    )
}
#[test]
fn actual_keras_reset_forward_finance_all_shared_gradients_and_adam() {
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
        let s = Session::new(target, precision, 32).unwrap();
        let f = fixture();
        let mut p = policy(&s, &f, 0.);
        let (pnl, h) = rollout(&s, &mut p, &f, false);
        let tol = if precision == Precision::F64 {
            2e-10
        } else {
            5e-5
        };
        for (day, x) in h.iter().enumerate() {
            close(&read(x.clone()), &numbers(&f["holdings_by_time"][day]), tol);
        }
        close(&read(pnl.clone()), &numbers(&f["pnl"]), tol);
        let loss = pnl.square().mean();
        close(&read(loss.clone()), &[f["loss"].as_f64().unwrap()], tol);
        let mut g = loss.backward();
        for (i, x) in p.parameters().iter().enumerate() {
            close(
                &read(x.grad(&g).unwrap()),
                &numbers(&f["gradients"][i]),
                tol,
            );
        }
        p.backward_update(&mut g, 0.001).unwrap();
        for (i, x) in p.snapshot().unwrap().parameters.iter().enumerate() {
            close(&x.values, &numbers(&f["after_first_adam"][i]), tol);
        }
    }
}
#[test]
fn wealth_feedback_gradient_is_connected_and_reset_is_pure() {
    let s = Session::new(ExecutionTarget::Cpu, Precision::F64, 32).unwrap();
    let f = fixture();
    // Amplify the readout so the wealth-feedback path is separately observable,
    // rather than relying on a tiny difference through the small parity fixture.
    let mut snap = policy(&s, &f, 0.).snapshot().unwrap();
    for i in [6, 8, 10] {
        snap.parameters[i].values.fill(0.6);
    }
    let mut p = HedgePolicy::from_snapshot(&s, snap.clone()).unwrap();
    let (pnl, _) = rollout(&s, &mut p, &f, false);
    let g = pnl.square().mean().backward();
    let connected = read(p.parameters()[0].grad(&g).unwrap());
    let mut q = HedgePolicy::from_snapshot(&s, snap).unwrap();
    let (pnl, _) = rollout(&s, &mut q, &f, true);
    let g = pnl.square().mean().backward();
    let detached = read(q.parameters()[0].grad(&g).unwrap());
    assert!(
        connected
            .iter()
            .zip(detached)
            .any(|(a, b)| (a - b).abs() > 1e-7)
    );
    let before = p.snapshot().unwrap();
    let x = t2(&s, vec![0.2; 8], [4, 2]);
    let prev = t2(&s, vec![0.1; 8], [4, 2]);
    let a = p
        .forward_at_with_wealth(
            0,
            x.clone(),
            prev.clone(),
            Some(t1(&s, vec![2.5; 4])),
            false,
        )
        .unwrap();
    let b = p
        .forward_at_with_wealth(2, x, prev, Some(t1(&s, vec![2.5; 4])), false)
        .unwrap();
    assert_eq!(read(a), read(b));
    assert_eq!(
        serde_json::to_value(before).unwrap(),
        serde_json::to_value(p.snapshot().unwrap()).unwrap()
    );
    assert!(
        p.forward_at(
            0,
            t2(&s, vec![0.; 8], [4, 2]),
            t2(&s, vec![0.; 8], [4, 2]),
            false
        )
        .is_err()
    );
}
#[test]
fn dropout_changes_training_and_is_stateless_between_updates() {
    let s = Session::new(ExecutionTarget::Cpu, Precision::F64, 32).unwrap();
    let f = fixture();
    let mut p = policy(&s, &f, 0.5);
    let before = serde_json::to_value(p.snapshot().unwrap()).unwrap();
    let (a, _) = rollout(&s, &mut p, &f, false);
    let a = read(a);
    let (b, _) = rollout(&s, &mut p, &f, false);
    assert_eq!(a, read(b));
    assert!(
        a.iter()
            .zip(numbers(&f["pnl"]))
            .any(|(x, y)| (x - y).abs() > 1e-5)
    );
    assert_eq!(before, serde_json::to_value(p.snapshot().unwrap()).unwrap());
}
fn learner(dropout: f64) -> NativeLearner {
    let c = LearnerConfig {
        policy: config(dropout),
        policy_engine: "volt_shared_lstm_reset".into(),
        cash_growth: vec![1.; 3],
        income_growth: vec![1.; 6],
        trading_costs: vec![0.001; 2],
        liquidation_costs: vec![0.; 2],
        ..LearnerConfig::default()
    };
    NativeLearner::new(c).unwrap()
}
fn command(l: &mut NativeLearner, x: Value) -> Value {
    l.handle(serde_json::from_value::<Command>(x).unwrap())
        .unwrap()
}
fn upload(l: &mut NativeLearner) {
    let f = fixture();
    command(
        l,
        json!({"op":"register_dataset","dataset_id":"keras","partitions":[{"split":"train","first_path":0,"paths":4}]}),
    );
    command(
        l,
        json!({"op":"upload","batch_id":"train","split":"train","batch":{"paths":4,"provenance":{"dataset_id":"keras","first_path":0},"marks":f["marks"],"features":f["features"],"payoffs":f["payoffs"]}}),
    );
}
#[test]
fn native_protocol_dropout_exact_resume_and_inference_wealth_validation() {
    let mut a = learner(0.5);
    upload(&mut a);
    command(
        &mut a,
        json!({"op":"update","batch_id":"train","expected_version":0}),
    );
    let saved = command(&mut a, json!({"op":"checkpoint"}));
    let mut b = NativeLearner::restore(serde_json::from_value(saved.clone()).unwrap()).unwrap();
    let f = fixture();
    command(
        &mut b,
        json!({"op":"upload","batch_id":"train","split":"train","batch":{"paths":4,"provenance":{"dataset_id":"keras","first_path":0},"marks":f["marks"],"features":f["features"],"payoffs":f["payoffs"]}}),
    );
    command(
        &mut a,
        json!({"op":"update","batch_id":"train","expected_version":1}),
    );
    command(
        &mut b,
        json!({"op":"update","batch_id":"train","expected_version":1}),
    );
    assert_eq!(
        command(&mut a, json!({"op":"checkpoint"})),
        command(&mut b, json!({"op":"checkpoint"}))
    );
    assert!(a.handle(serde_json::from_value(json!({"op":"infer","step":0,"paths":1,"features":[0.,0.],"previous_holdings":[0.,0.]})).unwrap()).is_err());
    let before = command(&mut a, json!({"op":"checkpoint"}));
    let req = json!({"op":"infer","step":1,"paths":1,"features":[0.2,0.3],"previous_holdings":[0.1,0.2],"wealth":[2.5]});
    assert_eq!(command(&mut a, req.clone()), command(&mut a, req));
    assert_eq!(before, command(&mut a, json!({"op":"checkpoint"})));
}
#[test]
fn legacy_mlp_serialization_and_new_config_failures() {
    let mut c = HedgePolicyConfig::paper_heston(4);
    let v = serde_json::to_value(&c).unwrap();
    assert!(v.get("include_wealth").is_none());
    assert!(v.get("shared_lstm_reset").is_none());
    assert!(v["adam"].get("epsilon_convention").is_none());
    c.shared_lstm_reset = Some(SharedLstmResetConfig {
        dropout_probability: 0.5,
        dropout_seed: 1,
        recurrent_activation: RecurrentActivation::Sigmoid,
    });
    assert!(c.validate().is_err());
    c.batch_norm.enabled = false;
    c.shared_lstm_reset.as_mut().unwrap().dropout_probability = 1.;
    assert!(c.validate().is_err());
    let cfg = LearnerConfig {
        policy: config(0.),
        ..LearnerConfig::default()
    };
    assert!(NativeLearner::new(cfg).is_err());
}
#[test]
fn standalone_planner_counts_shared_parameters_and_refuses_insufficient_budget() {
    use volt_compute::{budget::MemoryBudget, hedge_application::plan_hedging};
    let mut legacy = config(0.);
    legacy.shared_lstm_reset = None;
    legacy.include_wealth = false;
    let budget = MemoryBudget::default();
    let old = plan_hedging(
        &legacy,
        4,
        Precision::F64,
        ExecutionTarget::Cpu,
        &budget,
        true,
    )
    .unwrap();
    // Golden pre-extension MLP admission receipt for this fixed3date fixture.
    assert_eq!(
        (
            old.parameter_count,
            old.estimated_compute_bytes,
            old.estimated_host_working_bytes
        ),
        (141, 33_747_104, 33_766_784)
    );
    let shared = config(0.5);
    let plan = plan_hedging(
        &shared,
        4,
        Precision::F64,
        ExecutionTarget::Cpu,
        &budget,
        true,
    )
    .unwrap();
    let session = Session::new(ExecutionTarget::Cpu, Precision::F64, 32).unwrap();
    let policy = HedgePolicy::new(&session, shared.clone(), 1).unwrap();
    assert_eq!(plan.parameter_count, policy.parameter_count() as u64);
    assert_eq!(plan.parameter_count, 224);
    let tight = MemoryBudget {
        compute_bytes: 42_220_000,
        host_working_bytes: 42_220_000,
        ..budget
    };
    assert!(
        plan_hedging(
            &legacy,
            4,
            Precision::F64,
            ExecutionTarget::Cpu,
            &tight,
            true
        )
        .is_ok()
    );
    assert!(
        plan_hedging(
            &shared,
            4,
            Precision::F64,
            ExecutionTarget::Cpu,
            &tight,
            true
        )
        .is_err()
    );
}
#[test]
fn source_b1000_profile_admits_cpu_gemm_with_explicit_budget() {
    use volt_compute::{budget::MemoryBudget, hedge_application::plan_hedging};
    let mut c = config(0.5);
    c.time_steps = 63;
    c.feature_count = 17;
    c.hidden_width = 56;
    c.max_batch_rows = 1000;
    let budget = MemoryBudget {
        compute_bytes: 32 * 1024u64.pow(3),
        host_working_bytes: 32 * 1024u64.pow(3),
        max_batch_rows: 1000,
        ..MemoryBudget::default()
    };
    let p = plan_hedging(
        &c,
        1000,
        Precision::F32,
        ExecutionTarget::Cpu,
        &budget,
        true,
    )
    .unwrap();
    assert!(p.estimated_compute_bytes * 2 < p.compute_admission_limit_bytes);
    assert!(p.estimated_host_working_bytes * 2 < p.host_admission_limit_bytes);
    assert_eq!(p.parameter_count, 49_058);
}
#[test]
fn actual_tensorflow_small_trade_rule_all_thirteen_gradients_and_update() {
    let f = fixture();
    let rule: Value = serde_json::from_str(SMALL_TRADE_FIXTURE).unwrap();
    let mut data = f.clone();
    data["force_trade_signal"] = rule["force_trade_signal"].clone();
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
        let s = Session::new(target, precision, 32).unwrap();
        let mut c = config(0.);
        c.small_trade_gate =
            Some(serde_json::from_value(rule["small_trade_rule"].clone()).unwrap());
        let mut snap = HedgePolicy::new(&s, c, 37).unwrap().snapshot().unwrap();
        for (i, x) in f["parameters"].as_array().unwrap().iter().enumerate() {
            snap.parameters[i].values = numbers(&x["values"]);
        }
        let original = snap.clone();
        let mut p = HedgePolicy::from_snapshot(&s, snap).unwrap();
        let (pnl, h) = rollout(&s, &mut p, &data, false);
        let tol = if precision == Precision::F64 {
            2e-10
        } else {
            5e-5
        };
        for (day, x) in h.iter().enumerate() {
            close(
                &read(x.clone()),
                &numbers(&rule["holdings_by_time"][day]),
                tol,
            );
        }
        close(&read(pnl.clone()), &numbers(&rule["pnl"]), tol);
        let loss = pnl.square().mean();
        close(&read(loss.clone()), &[rule["loss"].as_f64().unwrap()], tol);
        let mut g = loss.backward();
        for (i, x) in p.parameters().iter().enumerate() {
            close(
                &read(x.grad(&g).unwrap()),
                &numbers(&rule["gradients"][i]),
                tol,
            );
        }
        p.backward_update(&mut g, 0.001).unwrap();
        for (i, x) in p.snapshot().unwrap().parameters.iter().enumerate() {
            close(&x.values, &numbers(&rule["after_first_adam"][i]), tol);
        }
        let mut hard_data = data.clone();
        hard_data["train_mode"] = json!(false);
        hard_data["sharpen_gate"] = json!(true);
        let mut hp = HedgePolicy::from_snapshot(&s, original.clone()).unwrap();
        let (hard_pnl, hard_h) = rollout(&s, &mut hp, &hard_data, false);
        close(&read(hard_pnl), &numbers(&rule["hard_pnl"]), tol);
        for (day, x) in hard_h.iter().enumerate() {
            close(
                &read(x.clone()),
                &numbers(&rule["hard_holdings_by_time"][day]),
                tol,
            );
        }
        let mut tf_snapshot = original;
        tf_snapshot.config.adam.epsilon_convention = AdamEpsilonConvention::TensorflowUncorrected;
        let mut tfp = HedgePolicy::from_snapshot(&s, tf_snapshot).unwrap();
        let (pnl, _) = rollout(&s, &mut tfp, &data, false);
        let mut g = pnl.square().mean().backward();
        tfp.backward_update(&mut g, 0.001).unwrap();
        for (i, x) in tfp.snapshot().unwrap().parameters.iter().enumerate() {
            // TensorFlow legacy beta powers originate as float32 scalar constants,
            // including for float64 variables; native powers use f64 before tensor casting.
            close(
                &x.values,
                &numbers(&rule["after_first_tensorflow_adam"][i]),
                if precision == Precision::F64 {
                    1e-8
                } else {
                    tol
                },
            );
        }
        // Same static observations, separate source soft-validation and sharpened-inference phases.
        let x = t2(&s, vec![0.1; 8], [4, 2]);
        let prev = t2(&s, vec![0.; 8], [4, 2]);
        let soft = p
            .forward_at_with_context(
                0,
                x.clone(),
                prev.clone(),
                PolicyForwardContext {
                    wealth: Some(t1(&s, vec![2.5; 4])),
                    force_trade_signal: None,
                    sharpen_gate: Some(false),
                },
                false,
            )
            .unwrap();
        let hard = p
            .forward_at_with_context(
                0,
                x,
                prev,
                PolicyForwardContext {
                    wealth: Some(t1(&s, vec![2.5; 4])),
                    force_trade_signal: None,
                    sharpen_gate: Some(true),
                },
                false,
            )
            .unwrap();
        assert!(
            read(soft)
                .iter()
                .zip(read(hard))
                .any(|(a, b)| (a - b).abs() > 1e-7)
        );
        let theta = p.parameters()[12].clone();
        let mut g = theta.sum().backward();
        p.backward_update(&mut g, 1.).unwrap();
        assert_eq!(p.snapshot().unwrap().parameters[12].values, [0.]);
    }
}
#[test]
fn boolean_tracking_diagnostic_has_zero_gradient_and_rule_context_restores() {
    let mut c = LearnerConfig {
        policy: config(0.),
        policy_engine: "volt_shared_lstm_reset".into(),
        premium: 1.,
        cash_growth: vec![1.; 3],
        income_growth: vec![1.; 6],
        trading_costs: vec![0.; 2],
        liquidation_costs: vec![0.; 2],
        ..LearnerConfig::default()
    };
    c.policy.feature_count = 17;
    c.policy.small_trade_gate = Some(SmallTradeGateConfig::default());
    c.tracking_penalty = Some(volt_compute::native_learner::TrackingPenaltyConfig {
        liability_feature: 13,
        threshold_multiple: 1.,
        coefficient: 2.,
    });
    let mut original = NativeLearner::new(c).unwrap();
    let mut checkpoint = command(&mut original, json!({"op":"checkpoint"}));
    for x in checkpoint["policy"]["parameters"].as_array_mut().unwrap() {
        for v in x["values"].as_array_mut().unwrap() {
            *v = json!(0.);
        }
    }
    checkpoint["policy"]["parameters"][12]["values"] = json!([0.001]);
    let mut l = NativeLearner::restore(serde_json::from_value(checkpoint).unwrap()).unwrap();
    command(
        &mut l,
        json!({"op":"register_dataset","dataset_id":"tracking","partitions":[{"split":"train","first_path":0,"paths":4}]}),
    );
    let mut features = vec![0.; 4 * 3 * 17];
    for path in 0..4 {
        features[(path * 3) * 17 + 13] = 1.;
    }
    for (path, day, value) in [(0, 1, 3.), (2, 1, 4.), (3, 2, 3.)] {
        features[(path * 3 + day) * 17 + 13] = value;
    }
    let batch = json!({"paths":4,"provenance":{"dataset_id":"tracking","first_path":0},"marks":([100.,3.].repeat(16)),"features":features,"payoffs":vec![0.;4],"force_trade_signal":vec![0.;24]});
    command(
        &mut l,
        json!({"op":"upload","batch_id":"train","split":"train","batch":batch}),
    );
    let eval = command(
        &mut l,
        json!({"op":"evaluate","batch_id":"train","sharpen_gate":false}),
    );
    assert_eq!(eval["loss"], 1.);
    assert_eq!(eval["tracking_violation_fraction"], 0.75);
    assert_eq!(eval["tracking_penalty"], 1.5);
    assert_eq!(eval["source_loss"], 2.5);
    let before = command(&mut l, json!({"op":"checkpoint"}));
    let update = command(
        &mut l,
        json!({"op":"update","batch_id":"train","expected_version":0}),
    );
    assert_eq!(update["gradient_l2_norm"], 0.);
    assert_eq!(update["source_loss_before_update"], 2.5);
    let after = command(&mut l, json!({"op":"checkpoint"}));
    for (a, b) in before["policy"]["parameters"]
        .as_array()
        .unwrap()
        .iter()
        .zip(after["policy"]["parameters"].as_array().unwrap())
    {
        assert_eq!(a["values"], b["values"]);
    }
    let mut restored = NativeLearner::restore(serde_json::from_value(after).unwrap()).unwrap();
    let infer = json!({"op":"infer","step":0,"paths":1,"features":vec![0.;17],"previous_holdings":[0.,0.],"wealth":[1.],"force_trade_signal":[0.,0.],"sharpen_gate":false});
    assert_eq!(
        command(&mut l, infer.clone()),
        command(&mut restored, infer)
    );
}
// Generated by actual Keras2.15 on2026-10-26 with explicit weights and dropout
// disabled. Mathematical data only; no author GPL implementation is embedded.
const KERAS_FIXTURE: &str = r#"{
  "schema": "volt.test.actual_keras_shared_lstm_reset.v1",
  "tensorflow": "2.15.0",
  "keras": "2.15.0",
  "profile": "actualKeras twoLSTM(reset1step)+twoReLUDense+linear; wealth+previousholdings attached; dropoutdisabled",
  "dates": 3,
  "paths": 4,
  "width": 3,
  "hedge_count": 2,
  "feature_count": 2,
  "parameters": [
    {
      "shape": [
        5,
        12
      ],
      "values": [
        0.08,
        0.11365883939231586,
        0.11637189707302728,
        0.08564480032239469,
        0.04972790018768287,
        0.04164302901347446,
        0.06882338007204297,
        0.10627946394875157,
        0.11957432986493527,
        0.09648473940967027,
        0.05823915556442521,
        0.04000039173797186,
        0.0585370832799826,
        0.09680668147306563,
        0.11962429422779482,
        0.10601151360628468,
        0.0684838673333974,
        0.04154410032481773,
        0.049960510129132954,
        0.08599508838651809,
        0.1165178100291051,
        0.11346622554144224,
        0.07964594762838384,
        0.04615118383299317,
        0.043776865519735045,
        0.07470592999608908,
        0.1105023380191841,
        0.11825503713618013,
        0.09083623153231477,
        0.0534546446314813,
        0.040478735036285525,
        0.0638384941870774,
        0.10205706724966762,
        0.11999647440429069,
        0.10116330744480095,
        0.06287269322015396,
        0.04032884586227537,
        0.05425847466572002,
        0.09185474314837541,
        0.11855181545136351,
        0.10980452641917396,
        0.07365509324781164,
        0.04333913808337465,
        0.04672901029485607,
        0.08070807700421655,
        0.11403614098136475,
        0.11607153390595237,
        0.08494292490980897,
        0.04926981354705333,
        0.041849893889621126,
        0.06950500585184285,
        0.106809167033735,
        0.11946510368161942,
        0.09583700600727336,
        0.05764843804593535,
        0.040009793065655205,
        0.05913795991652353,
        0.097446590209913,
        0.11971490592338149,
        0.10546952028556551
      ]
    },
    {
      "shape": [
        3,
        12
      ],
      "values": [
        0.10576870748950765,
        0.11966659241809874,
        0.0970951952093532,
        0.058806554363660266,
        0.040003069697435965,
        0.05797257829609449,
        0.09619399682466394,
        0.11952672935508002,
        0.10651876920328733,
        0.06912957494356231,
        0.041734599349192485,
        0.04952065664323867,
        0.08532928165679769,
        0.11623818969233848,
        0.11382987324571738,
        0.08031852735143749,
        0.04651432887921013,
        0.04349670200835262,
        0.07404003896743205,
        0.11006293661408593,
        0.11844610898008466,
        0.09148210605310914,
        0.053961507773350126,
        0.04038057916411374,
        0.06322556335707075,
        0.10149288724025887,
        0.11999974971428083,
        0.10173102676928991,
        0.0634828980303784,
        0.04042051666981859,
        0.05374732975288839,
        0.09121072679076078,
        0.11836703331812322,
        0.11024886635144251,
        0.07432003116103994,
        0.043613333126658875
      ]
    },
    {
      "shape": [
        1,
        12
      ],
      "values": [
        0.15,
        0.15,
        0.15,
        1.0,
        1.0,
        1.0,
        0.22,
        0.22,
        0.22,
        0.12,
        0.12,
        0.12
      ]
    },
    {
      "shape": [
        3,
        12
      ],
      "values": [
        0.11452837466595496,
        0.08166322649733164,
        0.04726891555742359,
        0.0429674127068907,
        0.07271349982911617,
        0.10915876160503504,
        0.11879559243380346,
        0.09276393449397409,
        0.05499717404428472,
        0.04021789647184043,
        0.06201410141861594,
        0.10034645857489495,
        0.11997237554991672,
        0.10284787478639955,
        0.06471714331263964,
        0.040637399796734266,
        0.052747449377780004,
        0.0899133683193184,
        0.11796498214591579,
        0.11111176647204371,
        0.07565453618303682,
        0.04419250528721273,
        0.04565171989560282,
        0.07869058482676619,
        0.11293332002952326,
        0.11689731267692344,
        0.08693808620983569,
        0.050600015278048895,
        0.04129215471464779,
        0.06757210859622512,
        0.1052782085202754,
        0.1197436401072158,
        0.09766895226676896,
        0.05934951119680292,
        0.04001603429788789,
        0.0574436310699895
      ]
    },
    {
      "shape": [
        3,
        12
      ],
      "values": [
        0.0933995260062362,
        0.055525684362291244,
        0.040153415646566375,
        0.06141591282344971,
        0.09976453404554433,
        0.1199417338149842,
        0.10339668771567047,
        0.06534083482992287,
        0.04076255079734034,
        0.0522589966089151,
        0.08926039300406156,
        0.11774782677776419,
        0.11153008269501266,
        0.07632372599089274,
        0.044497318656739816,
        0.04531191282057675,
        0.07801857436486531,
        0.11254694950028421,
        0.11715180936308961,
        0.08759946703181751,
        0.0510602097582302,
        0.04112806217024547,
        0.06693459495581111,
        0.10475340088480158,
        0.11981404419646237,
        0.09826983888576771,
        0.059928427959177036,
        0.04004072780212203,
        0.056891398222170736,
        0.09498801054597848,
        0.11930471509456558,
        0.10748484584818978,
        0.07039553608184902,
        0.04213652614860158,
        0.048689019457973826,
        0.08402868387970003
      ]
    },
    {
      "shape": [
        1,
        12
      ],
      "values": [
        0.15,
        0.15,
        0.15,
        1.0,
        1.0,
        1.0,
        0.22,
        0.22,
        0.22,
        0.12,
        0.12,
        0.12
      ]
    },
    {
      "shape": [
        3,
        3
      ],
      "values": [
        0.04513696910345649,
        0.04466181377119386,
        0.07667642388730012,
        0.1117467145539661,
        0.11762922226719093,
        0.08891559656400991,
        0.05200501249625831,
        0.040832890833947304,
        0.06567082871052685
      ]
    },
    {
      "shape": [
        1,
        3
      ],
      "values": [
        0.18,
        0.18,
        0.18
      ]
    },
    {
      "shape": [
        3,
        3
      ],
      "values": [
        0.05474933448510713,
        0.09246165454053512,
        0.11871678688125945,
        0.10937588391496454,
        0.07302692875108081,
        0.043088983135487736,
        0.047086856201251644,
        0.08134492188884547,
        0.11436647259425983
      ]
    },
    {
      "shape": [
        1,
        3
      ],
      "values": [
        0.18,
        0.18,
        0.18
      ]
    },
    {
      "shape": [
        3,
        2
      ],
      "values": [
        0.10627946394875157,
        0.11957432986493527,
        0.09648473940967027,
        0.05823915556442521,
        0.04000039173797186,
        0.0585370832799826
      ]
    },
    {
      "shape": [
        1,
        2
      ],
      "values": [
        0.04,
        0.04
      ]
    }
  ],
  "marks": [
    100.0,
    3.0,
    101.0,
    3.2,
    102.0,
    3.4,
    103.0,
    3.6,
    100.0,
    3.0,
    99.0,
    2.8,
    98.0,
    2.6,
    97.0,
    2.4,
    100.0,
    3.0,
    102.0,
    3.3,
    100.0,
    3.1,
    101.0,
    3.2,
    100.0,
    3.0,
    98.0,
    2.7,
    100.0,
    2.9,
    99.0,
    2.8
  ],
  "features": [
    0.1,
    -0.2,
    0.3,
    0.1,
    0.5,
    0.2,
    -0.2,
    0.1,
    -0.3,
    -0.2,
    -0.5,
    -0.3,
    0.2,
    0.3,
    0.4,
    0.2,
    0.1,
    0.3,
    -0.1,
    -0.3,
    -0.4,
    -0.2,
    -0.2,
    -0.1
  ],
  "payoffs": [
    2.0,
    0.0,
    1.0,
    3.0
  ],
  "premium": 2.5,
  "cash_growth": [
    1.000082195158659,
    1.000082195158659,
    1.000082195158659
  ],
  "income_growth": [
    1.0000273976355822,
    1.0,
    1.0000273976355822,
    1.0,
    1.0000273976355822,
    1.0
  ],
  "trading_costs": [
    0.001,
    0.002
  ],
  "liquidation_costs": [
    0.0005,
    0.001
  ],
  "holdings_by_time": [
    [
      0.09491078584367732,
      0.09346039939672252,
      0.0949094489607174,
      0.09345909755923706,
      0.0949344577602854,
      0.09348345032145618,
      0.09489845331931726,
      0.09344839038751379
    ],
    [
      0.09494002753418612,
      0.09348887399943932,
      0.09489489227447784,
      0.09344492276273601,
      0.094951943552494,
      0.09350047742972871,
      0.09488728451989045,
      0.0934375145872664
    ],
    [
      0.09495664523216733,
      0.09350505577939205,
      0.094879066961497,
      0.0934295126049703,
      0.09493556048603749,
      0.09348452412198577,
      0.09490639552987201,
      0.09345612424524699
    ]
  ],
  "pnl": [
    0.8246143271677906,
    2.1433383438663034,
    1.5974343896086247,
    -0.6296576856405567
  ],
  "loss": 2.055538368762109,
  "gradients": [
    [
      2.1175554035670518e-05,
      2.601033175279897e-05,
      3.1993105034785194e-05,
      0.0,
      0.0,
      0.0,
      0.00016029932682699947,
      0.0001800273768473294,
      0.00017784583349942518,
      1.9946518342794873e-05,
      2.5552870011440168e-05,
      3.470129620829966e-05,
      1.2388830581926559e-05,
      1.533076863713587e-05,
      1.8443966450681957e-05,
      0.0,
      0.0,
      0.0,
      8.656440194909188e-05,
      9.553296247560715e-05,
      9.22981362786543e-05,
      1.1542768966692509e-05,
      1.530643448987313e-05,
      2.0768122810770202e-05,
      -3.362146904624822e-05,
      -3.771083692919232e-05,
      -5.009789998517413e-05,
      0.0,
      0.0,
      0.0,
      -0.0003115459964478586,
      -0.00036719281615251077,
      -0.00037805247109230756,
      -3.180695620353512e-05,
      -3.376696693149009e-05,
      -4.852407513056862e-05,
      -4.05623323961911e-06,
      -4.942773540756078e-06,
      -5.998237986628437e-06,
      0.0,
      0.0,
      0.0,
      -2.946239432701885e-05,
      -3.286791921093018e-05,
      -3.215475494764477e-05,
      -3.789303292542135e-06,
      -4.881706894735938e-06,
      -6.641956843276727e-06,
      -3.994250972842655e-06,
      -4.8672442259293234e-06,
      -5.9065806324549706e-06,
      0.0,
      0.0,
      0.0,
      -2.9012191156970826e-05,
      -3.236567856995574e-05,
      -3.166341308905731e-05,
      -3.731399990709047e-06,
      -4.807110580866068e-06,
      -6.540462547546653e-06
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      -1.4866975318300674e-05,
      -1.6712400309292982e-05,
      -2.224571266371459e-05,
      0.0,
      0.0,
      0.0,
      -0.00013890004192586817,
      -0.00016380100638056823,
      -0.0001690180948384993,
      -1.4118561741279682e-05,
      -1.4978817929301483e-05,
      -2.14267997771604e-05
    ],
    [
      -8.487547020217133e-06,
      -1.658039381964439e-05,
      -7.958405450947499e-06,
      0.0,
      0.0,
      0.0,
      -7.514594340706052e-05,
      -0.000147133413061731,
      -7.200044312225318e-05,
      -8.758216748751095e-06,
      -1.708940790807235e-05,
      -8.16184102033069e-06,
      -7.9764315407789e-06,
      -1.55875190212298e-05,
      -7.5070479878316694e-06,
      0.0,
      0.0,
      0.0,
      -7.179315627051322e-05,
      -0.00014059182240678907,
      -6.874806379077452e-05,
      -8.213249184646794e-06,
      -1.6036449942383204e-05,
      -7.697355832538332e-06,
      -1.067456427793626e-05,
      -2.085637822614209e-05,
      -1.0027181318643724e-05,
      0.0,
      0.0,
      0.0,
      -9.526919298232048e-05,
      -0.0001865488361704928,
      -9.12553486221584e-05,
      -1.1003585996046317e-05,
      -2.1477469216261864e-05,
      -1.028247699094478e-05
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      -0.00013390696499948672,
      -0.0002614485786934334,
      -0.00012487800124510486,
      0.0,
      0.0,
      0.0,
      -0.0011570300169540556,
      -0.002264823202016956,
      -0.0011095468651058879,
      -0.00013859978242881818,
      -0.000270188910482715,
      -0.00012810901750235477
    ],
    [
      -0.0018460879581313687,
      -0.0019330740441690247,
      -0.001653132369200025,
      -0.0018370408889112158,
      -0.0019236022586691555,
      -0.0016450338396611823,
      -0.0017896427620706628,
      -0.0018739798960571382,
      -0.0016026068007777812
    ],
    [
      -0.026848856201814437,
      -0.02811458451698503,
      -0.0240437556217789
    ],
    [
      -0.029054508638922928,
      -0.024505351039814518,
      -0.011438603850060514,
      -0.02899844743965104,
      -0.024458010443506737,
      -0.011416548209361412,
      -0.029281370649253076,
      -0.02469709614394039,
      -0.011527809648887999
    ],
    [
      -0.1496064008950113,
      -0.12616018172742643,
      -0.05890511664068558
    ],
    [
      -0.2634440552793017,
      -0.04241439373354207,
      -0.27177001661943034,
      -0.04375158771369434,
      -0.2786362263521014,
      -0.04485429264245322
    ],
    [
      -1.1916974865131933,
      -0.19195951894726465
    ]
  ],
  "after_first_adam": [
    [
      0.0790004719753159,
      0.11265922366216374,
      0.11537220949723587,
      0.08564480032239469,
      0.04972790018768287,
      0.04164302901347446,
      0.06782344240435238,
      0.10527951944563618,
      0.11857438604311593,
      0.09548524045483059,
      0.05723954671184519,
      0.03900067978299536,
      0.05753788976538603,
      0.0958073332876126,
      0.11862483607273228,
      0.10601151360628468,
      0.0684838673333974,
      0.04154410032481773,
      0.04896062558996841,
      0.08499519300465569,
      0.11551791831512238,
      0.11246709109290394,
      0.07864660047858352,
      0.0451516650640457,
      0.04477656822474009,
      0.07570566493640318,
      0.11150213849605818,
      0.11825503713618013,
      0.09083623153231477,
      0.0534546446314813,
      0.04147870298661598,
      0.06483846700149165,
      0.10305704084633853,
      0.12099616015192978,
      0.10216301143064048,
      0.06387248722550867,
      0.04132638661535929,
      0.055256455629438185,
      0.0928530788036228,
      0.11855181545136351,
      0.10980452641917396,
      0.07365509324781164,
      0.04433879882810912,
      0.04772870618488039,
      0.0817077661503922,
      0.11503350895031801,
      0.11706948966401626,
      0.08594142163054297,
      0.050267316232433454,
      0.042847843585679654,
      0.07050331572297645,
      0.106809167033735,
      0.11946510368161942,
      0.09583700600727336,
      0.05864809352727746,
      0.041009484237315284,
      0.06013764423976353,
      0.09844391744383763,
      0.12071283002419947,
      0.10646799371342691
    ],
    [
      0.10576870748950765,
      0.11966659241809874,
      0.0970951952093532,
      0.058806554363660266,
      0.040003069697435965,
      0.05797257829609449,
      0.09619399682466394,
      0.11952672935508002,
      0.10651876920328733,
      0.06912957494356231,
      0.041734599349192485,
      0.04952065664323867,
      0.08532928165679769,
      0.11623818969233848,
      0.11382987324571738,
      0.08031852735143749,
      0.04651432887921013,
      0.04349670200835262,
      0.07404003896743205,
      0.11006293661408593,
      0.11844610898008466,
      0.09148210605310914,
      0.053961507773350126,
      0.04038057916411374,
      0.06322556335707075,
      0.10149288724025887,
      0.11999974971428083,
      0.10173102676928991,
      0.0634828980303784,
      0.04042051666981859,
      0.05374732975288839,
      0.09121072679076078,
      0.11836703331812322,
      0.11024886635144251,
      0.07432003116103994,
      0.043613333126658875
    ],
    [
      0.15099932786349557,
      0.15099940204334583,
      0.15099955072174587,
      1.0,
      1.0,
      1.0,
      0.2209999280579947,
      0.22099993900114248,
      0.22099994088535133,
      0.12099929225678342,
      0.1209993328791555,
      0.12099953355693344
    ],
    [
      0.11552719789563086,
      0.08266262378247646,
      0.04826766064057097,
      0.0429674127068907,
      0.07271349982911617,
      0.10915876160503504,
      0.11979545942376232,
      0.09376386658012256,
      0.05599703522213485,
      0.0412167560291947,
      0.06301351664675212,
      0.10134523490000905,
      0.12097112346563078,
      0.10384723370213791,
      0.0657158130420682,
      0.040637399796734266,
      0.052747449377780004,
      0.0899133683193184,
      0.11896484292286638,
      0.1121116953962465,
      0.07665439079210438,
      0.04519128926250493,
      0.04665109674826002,
      0.079689287404031,
      0.11393238414135425,
      0.11789683348146318,
      0.08793708995523726,
      0.050600015278048895,
      0.04129215471464779,
      0.06757210859622512,
      0.10627810361238303,
      0.12074358655197295,
      0.09866884274294506,
      0.060348603268933064,
      0.04101556895484921,
      0.05844265952777197
    ],
    [
      0.0933995260062362,
      0.055525684362291244,
      0.040153415646566375,
      0.06141591282344971,
      0.09976453404554433,
      0.1199417338149842,
      0.10339668771567047,
      0.06534083482992287,
      0.04076255079734034,
      0.0522589966089151,
      0.08926039300406156,
      0.11774782677776419,
      0.11153008269501266,
      0.07632372599089274,
      0.044497318656739816,
      0.04531191282057675,
      0.07801857436486531,
      0.11254694950028421,
      0.11715180936308961,
      0.08759946703181751,
      0.0510602097582302,
      0.04112806217024547,
      0.06693459495581111,
      0.10475340088480158,
      0.11981404419646237,
      0.09826983888576771,
      0.059928427959177036,
      0.04004072780212203,
      0.056891398222170736,
      0.09498801054597848,
      0.11930471509456558,
      0.10748484584818978,
      0.07039553608184902,
      0.04213652614860158,
      0.048689019457973826,
      0.08402868387970003
    ],
    [
      0.1509999253738753,
      0.15099996180027336,
      0.15099991997523474,
      1.0,
      1.0,
      1.0,
      0.22099999140469798,
      0.22099999563213263,
      0.22099999103483198,
      0.1209999279020495,
      0.12099996303748571,
      0.12099992199456637
    ],
    [
      0.04613696373408817,
      0.045661808645576944,
      0.07767641788567256,
      0.11274670915792091,
      0.11862921711610182,
      0.08991559053260259,
      0.053005006956043405,
      0.041832885545202085,
      0.06667082251818902
    ],
    [
      0.18099999967503982,
      0.180999999691808,
      0.18099999963158653
    ],
    [
      0.05574933418842181,
      0.09346165417995596,
      0.11971678605451942,
      0.11037588361761383,
      0.07402692838971178,
      0.04408898230705878,
      0.04808685590723293,
      0.08234492153143455,
      0.11536647177428493
    ],
    [
      0.18099999998065497,
      0.18099999996823263,
      0.18099999987773183
    ],
    [
      0.10727946395829005,
      0.12057432967666223,
      0.09748473942037167,
      0.0592391553833581,
      0.04100039174957999,
      0.05953708310453456
    ],
    [
      0.041000000039106005,
      0.0409999999954028
    ]
  ],
  "learning_rate": 0.001,
  "first_step_optimizer": {
    "actual": "tf.keras.optimizers.legacy.Adam",
    "epsilon_hat": 3.1622776601683806e-10,
    "mapping": "first-step epsilonhat=canonicalepsilon*sqrt(1-beta2); not a wholetrajectory epsilonhat equivalence"
  }
}
"#;

const SMALL_TRADE_FIXTURE: &str = r#"{
  "small_trade_rule": {
    "sharpness": 10.0,
    "force_threshold": 0.1,
    "initial_trade_threshold": 0.08,
    "inference_sharpen": true
  },
  "force_trade_signal": [
    0.0,
    0.14,
    0.0,
    0.0,
    0.0,
    0.2,
    0.0,
    0.0,
    0.0,
    0.1,
    0.0,
    0.0,
    0.0,
    0.02,
    0.0,
    0.3,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.0,
    0.0
  ],
  "holdings_by_time": [
    [
      0.07091693809285825,
      0.06983321546897413,
      0.07091546611065457,
      0.06983177690176674,
      0.07094300328885998,
      0.069858688616047,
      0.07090335961239427,
      0.06981994539521719
    ],
    [
      0.08100283949990225,
      0.07976486447576508,
      0.0809790769227673,
      0.08163821332774598,
      0.08102082511594248,
      0.09067954071496265,
      0.0809698393564006,
      0.08708618409898906
    ],
    [
      0.08619454487677078,
      0.08980808286775599,
      0.08608724908209572,
      0.08597148993105999,
      0.08584694830856117,
      0.09165204525494682,
      0.08591809054722077,
      0.089347929486066
    ]
  ],
  "pnl": [
    0.7713032345434598,
    2.200293541012537,
    1.563133086097189,
    -0.5926002235883718
  ],
  "loss": 2.0576901040218503,
  "gradients": [
    [
      2.2205520326343598e-05,
      2.7679058605284122e-05,
      3.342851351617417e-05,
      0.0,
      0.0,
      0.0,
      0.0001568888860328783,
      0.0001734307338114808,
      0.00016774621077225,
      2.070480316501517e-05,
      2.7558827082871334e-05,
      3.744412797638371e-05,
      4.5920000010765884e-06,
      5.662773162927417e-06,
      6.725271929424983e-06,
      0.0,
      0.0,
      0.0,
      3.0683416982779297e-05,
      3.34917480658378e-05,
      3.190763617500305e-05,
      4.245420509267829e-06,
      5.706968363064594e-06,
      7.755959821868763e-06,
      -4.6502229792456815e-05,
      -5.6073849772511747e-05,
      -7.203696675238684e-05,
      0.0,
      0.0,
      0.0,
      -0.00039201933032202805,
      -0.00045093237059851826,
      -0.000452505973515876,
      -4.3658719451038656e-05,
      -5.2701683198084785e-05,
      -7.436907575416677e-05,
      -1.7197487116037388e-06,
      -2.0947863207267528e-06,
      -2.575780354116091e-06,
      0.0,
      0.0,
      0.0,
      -1.302490804428527e-05,
      -1.4646619341237157e-05,
      -1.4470337650427383e-05,
      -1.615249053626607e-06,
      -2.0521757536012516e-06,
      -2.7977647480550433e-06,
      -1.6355931343487708e-06,
      -1.988956835118024e-06,
      -2.4506012693467864e-06,
      0.0,
      0.0,
      0.0,
      -1.248348446198979e-05,
      -1.4059159121258339e-05,
      -1.3918275109790329e-05,
      -1.5377069044580263e-06,
      -1.9456278042812193e-06,
      -2.652875157555614e-06
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      -1.965039855676595e-05,
      -2.3690380580886533e-05,
      -3.040760926763687e-05,
      0.0,
      0.0,
      0.0,
      -0.0001654467201438643,
      -0.00019026297687323821,
      -0.0001909767629411906,
      -1.8460668840563015e-05,
      -2.2288839339397692e-05,
      -3.140102652236399e-05
    ],
    [
      -1.1886891961972113e-05,
      -2.3210911356167165e-05,
      -1.1068556947984091e-05,
      0.0,
      0.0,
      0.0,
      -0.00010190371297833565,
      -0.00019944711313638722,
      -9.77363081258096e-05,
      -1.2312986163589181e-05,
      -2.4005236131060144e-05,
      -1.1356914397999235e-05,
      -1.2634540860810492e-05,
      -2.4674170137476408e-05,
      -1.1777654697141504e-05,
      0.0,
      0.0,
      0.0,
      -0.00010884804606878912,
      -0.00021305005458853324,
      -0.0001043762962547298,
      -1.307899583196715e-05,
      -2.5504596015182604e-05,
      -1.2083836871718573e-05,
      -1.630181051193322e-05,
      -3.183292152867791e-05,
      -1.5183851143560822e-05,
      0.0,
      0.0,
      0.0,
      -0.00013992871028150998,
      -0.000273873781519649,
      -0.00013419928252101948,
      -1.688332013063084e-05,
      -3.291764574904857e-05,
      -1.5579221017238014e-05
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      -0.00015922640701621844,
      -0.0003107956044229071,
      -0.00014780206457989202,
      0.0,
      0.0,
      0.0,
      -0.0013458116062395422,
      -0.0026336297204221882,
      -0.0012914991806818698,
      -0.00016523569527393715,
      -0.00032193226820547306,
      -0.00015167627022598002
    ],
    [
      -0.002203774292371585,
      -0.002307832074778714,
      -0.001973838059152091,
      -0.0021916016751979787,
      -0.002295086747899762,
      -0.001962939328200461,
      -0.002123142972341145,
      -0.002223407263069672,
      -0.001901645198603452
    ],
    [
      -0.03122164936125968,
      -0.03269668088952989,
      -0.027965564847440005
    ],
    [
      -0.03395373675724627,
      -0.028322570093525103,
      -0.013452105639977804,
      -0.0338873877287036,
      -0.02826715089613021,
      -0.01342583878099379,
      -0.03422455209850381,
      -0.028548997228385626,
      -0.013559258284360689
    ],
    [
      -0.1745193164807864,
      -0.14554711175781612,
      -0.06915033988098021
    ],
    [
      -0.29940217377195677,
      -0.05661841033076661,
      -0.30888065255456515,
      -0.05840643925462108,
      -0.31669753212004087,
      -0.05988096220504415
    ],
    [
      -1.353881043319726,
      -0.256155020738874
    ],
    [
      0.13704031971566888
    ]
  ],
  "after_first_adam": [
    [
      0.07900045009117465,
      0.11265920050064418,
      0.11537219608382453,
      0.08564480032239469,
      0.04972790018768287,
      0.04164302901347446,
      0.06782344376027429,
      0.10527952155821062,
      0.11857438942813597,
      0.09548522211186336,
      0.057239518247804745,
      0.0390006586854726,
      0.05753925621482031,
      0.09580844424323212,
      0.11862577891095974,
      0.10601151360628468,
      0.0684838673333974,
      0.04154410032481773,
      0.04896083588650869,
      0.0849953868328428,
      0.1155181232900486,
      0.11246857545317339,
      0.0786476967710342,
      0.045152471464674486,
      0.04477665056862075,
      0.07570575173798023,
      0.11150219926743309,
      0.11825503713618013,
      0.09083623153231477,
      0.0534546446314813,
      0.04147870957532207,
      0.0648384720586484,
      0.10305704519835435,
      0.1209962454534351,
      0.10216311777980527,
      0.06387255882036581,
      0.04132306468593992,
      0.055253723606445114,
      0.09285087586639,
      0.11855181545136351,
      0.10980452641917396,
      0.07365509324781164,
      0.04433837095516474,
      0.047728328052353906,
      0.08170738645560736,
      0.11502998808606847,
      0.11706668467495404,
      0.08593936338234674,
      0.050263736719138086,
      0.0428448912955653,
      0.07050094182576952,
      0.106809167033735,
      0.11946510368161942,
      0.09583700600727336,
      0.058647637671034425,
      0.04100908233401393,
      0.060137241995376366,
      0.09844012904411233,
      0.12070979249044184,
      0.10646576496939957
    ],
    [
      0.10576870748950765,
      0.11966659241809874,
      0.0970951952093532,
      0.058806554363660266,
      0.040003069697435965,
      0.05797257829609449,
      0.09619399682466394,
      0.11952672935508002,
      0.10651876920328733,
      0.06912957494356231,
      0.041734599349192485,
      0.04952065664323867,
      0.08532928165679769,
      0.11623818969233848,
      0.11382987324571738,
      0.08031852735143749,
      0.04651432887921013,
      0.04349670200835262,
      0.07404003896743205,
      0.11006293661408593,
      0.11844610898008466,
      0.09148210605310914,
      0.053961507773350126,
      0.04038057916411374,
      0.06322556335707075,
      0.10149288724025887,
      0.11999974971428083,
      0.10173102676928991,
      0.0634828980303784,
      0.04042051666981859,
      0.05374732975288839,
      0.09121072679076078,
      0.11836703331812322,
      0.11024886635144251,
      0.07432003116103994,
      0.043613333126658875
    ],
    [
      0.150999491407513,
      0.15099957811060966,
      0.15099967128843098,
      1.0,
      1.0,
      1.0,
      0.2209999396083395,
      0.2209999474910861,
      0.22099994768750825,
      0.1209994586450772,
      0.12099955159075639,
      0.12099968168589213
    ],
    [
      0.11552753415233558,
      0.08266279589571858,
      0.0482680129544371,
      0.0429674127068907,
      0.07271349982911617,
      0.10915876160503504,
      0.11979549435844236,
      0.0937638844050553,
      0.055997071785464585,
      0.041217085022460125,
      0.06301368506109588,
      0.10134557887047013,
      0.1209715847371601,
      0.10384746971333728,
      0.06571629500946422,
      0.040637399796734266,
      0.052747449377780004,
      0.0899133683193184,
      0.11896489033006272,
      0.11211171958411319,
      0.07665444043189784,
      0.04519174132921062,
      0.04665132800803622,
      0.07968975800146046,
      0.11393270702031547,
      0.11789699868084862,
      0.08793742809208512,
      0.050600015278048895,
      0.04129215471464779,
      0.06757210859622512,
      0.1062781371074541,
      0.12074360364263945,
      0.09866887780328881,
      0.06034891929051431,
      0.0410157306472435,
      0.05844298964445913
    ],
    [
      0.0933995260062362,
      0.055525684362291244,
      0.040153415646566375,
      0.06141591282344971,
      0.09976453404554433,
      0.1199417338149842,
      0.10339668771567047,
      0.06534083482992287,
      0.04076255079734034,
      0.0522589966089151,
      0.08926039300406156,
      0.11774782677776419,
      0.11153008269501266,
      0.07632372599089274,
      0.044497318656739816,
      0.04531191282057675,
      0.07801857436486531,
      0.11254694950028421,
      0.11715180936308961,
      0.08759946703181751,
      0.0510602097582302,
      0.04112806217024547,
      0.06693459495581111,
      0.10475340088480158,
      0.11981404419646237,
      0.09826983888576771,
      0.059928427959177036,
      0.04004072780212203,
      0.056891398222170736,
      0.09498801054597848,
      0.11930471509456558,
      0.10748484584818978,
      0.07039553608184902,
      0.04213652614860158,
      0.048689019457973826,
      0.08402868387970003
    ],
    [
      0.15099993724738137,
      0.15099996787283682,
      0.1509999323935825,
      1.0,
      1.0,
      1.0,
      0.22099999261704398,
      0.22099999625044625,
      0.2209999923045677,
      0.12099993953115644,
      0.12099996898582635,
      0.12099993412152518
    ],
    [
      0.04613696461327533,
      0.04566180948561109,
      0.07767641886851863,
      0.11274671003858183,
      0.11862921795754533,
      0.08991559151709931,
      0.053005007833749,
      0.04183288638383425,
      0.06667082349941335
    ],
    [
      0.1809999997272049,
      0.18099999974165412,
      0.18099999968991262
    ],
    [
      0.05574933423808437,
      0.0934616542349551,
      0.11971678618537458,
      0.11037588366736513,
      0.07402692844480857,
      0.04408898243814849,
      0.048086855956559385,
      0.08234492158606577,
      0.11536647190424958
    ],
    [
      0.18099999999019684,
      0.18099999997879074,
      0.18099999990288407
    ],
    [
      0.10727946396284892,
      0.12057432973581061,
      0.09748473942479255,
      0.05923915544070758,
      0.04100039175389324,
      0.059537083160481015
    ],
    [
      0.04100000004011123,
      0.04100000000845834
    ],
    [
      0.07900000002547425
    ]
  ],
  "hard_holdings_by_time": [
    [
      0.08752221104274754,
      0.08618473366780517,
      0.08752063862219352,
      0.08618319875425486,
      0.08755005200222098,
      0.08621191009124192,
      0.08750770540482744,
      0.08617057423549479
    ],
    [
      0.08879280720139764,
      0.08743585958150832,
      0.08878188224610406,
      0.08981401871517275,
      0.08881711252707304,
      0.09334128612649367,
      0.08877016575270387,
      0.09278183095362497
    ],
    [
      0.08980060745212946,
      0.0929565122857354,
      0.08973313442057892,
      0.09037810546753997,
      0.08971194402172195,
      0.09336218970651791,
      0.0896769053517138,
      0.09288146761982932
    ]
  ],
  "hard_pnl": [
    0.8040298056652961,
    2.1656862424934054,
    1.588437172297878,
    -0.6188693165350423
  ],
  "after_first_tensorflow_adam": [
    [
      0.07901404103789389,
      0.1126701351676052,
      0.1153812682574794,
      0.08564480032239469,
      0.04972790018768287,
      0.04164302901347446,
      0.0678253915991373,
      0.10528128396078283,
      0.11857621143843315,
      0.09549978285776178,
      0.05725050006119253,
      0.03900876634143955,
      0.05760151171880661,
      0.09585957148409156,
      0.11866920360853947,
      0.10601151360628468,
      0.0684838673333974,
      0.04154410032481773,
      0.04897071115969645,
      0.08500444204298807,
      0.1155276235097839,
      0.11253554904905169,
      0.07869844955874934,
      0.04519035900702263,
      0.044770111184037245,
      0.0757003221455467,
      0.1114979674261049,
      0.11825503713618013,
      0.09083623153231477,
      0.0534546446314813,
      0.04147792906503563,
      0.06483779344610387,
      0.10305636894400484,
      0.12098928331633298,
      0.10215734290790009,
      0.06386845910385766,
      0.041173525107197885,
      0.05512731432247187,
      0.09274539717021135,
      0.11855181545136351,
      0.10980452641917396,
      0.07365509324781164,
      0.04431543476910783,
      0.04770787601067047,
      0.08168669076191185,
      0.11487241683595185,
      0.11693801382384042,
      0.08584137377869694,
      0.05010779591821795,
      0.04271271206574178,
      0.07039071275726148,
      0.106809167033735,
      0.11946510368161942,
      0.09583700600727336,
      0.058623732087486446,
      0.04098779511237472,
      0.06011574424112252,
      0.098276018190057,
      0.12057509618980551,
      0.1063630135503622
    ],
    [
      0.10576870748950765,
      0.11966659241809874,
      0.0970951952093532,
      0.058806554363660266,
      0.040003069697435965,
      0.05797257829609449,
      0.09619399682466394,
      0.11952672935508002,
      0.10651876920328733,
      0.06912957494356231,
      0.041734599349192485,
      0.04952065664323867,
      0.08532928165679769,
      0.11623818969233848,
      0.11382987324571738,
      0.08031852735143749,
      0.04651432887921013,
      0.04349670200835262,
      0.07404003896743205,
      0.11006293661408593,
      0.11844610898008466,
      0.09148210605310914,
      0.053961507773350126,
      0.04038057916411374,
      0.06322556335707075,
      0.10149288724025887,
      0.11999974971428083,
      0.10173102676928991,
      0.0634828980303784,
      0.04042051666981859,
      0.05374732975288839,
      0.09121072679076078,
      0.11836703331812322,
      0.11024886635144251,
      0.07432003116103994,
      0.043613333126658875
    ],
    [
      0.15098416212979632,
      0.150986827433302,
      0.15098970739442122,
      1.0,
      1.0,
      1.0,
      0.22099809232434933,
      0.22099834073837224,
      0.22099834692987838,
      0.12098315861789104,
      0.12098601071551458,
      0.12099002977095527
    ],
    [
      0.11550246086756653,
      0.08264978548054558,
      0.048241139079662966,
      0.0429674127068907,
      0.07271349982911617,
      0.10915876160503504,
      0.1197924988597618,
      0.09376235151922102,
      0.05599394898590124,
      0.04119285697471536,
      0.06300109937908523,
      0.10131936825056966,
      0.12094795775966913,
      0.1038352207820419,
      0.06569099544903867,
      0.040637399796734266,
      0.052747449377780004,
      0.0899133683193184,
      0.11896208536829175,
      0.1121102844211974,
      0.07665151567282473,
      0.04516889768334554,
      0.046639472859333356,
      0.07966508260681307,
      0.11391429076846554,
      0.11788747638959947,
      0.0879176844402183,
      0.050600015278048895,
      0.04129215471464779,
      0.06757210859622512,
      0.10627595372837172,
      0.12074248683109222,
      0.09866660143434744,
      0.060331125308740755,
      0.04100651905980791,
      0.05842373675775395
    ],
    [
      0.0933995260062362,
      0.055525684362291244,
      0.040153415646566375,
      0.06141591282344971,
      0.09976453404554433,
      0.1199417338149842,
      0.10339668771567047,
      0.06534083482992287,
      0.04076255079734034,
      0.0522589966089151,
      0.08926039300406156,
      0.11774782677776419,
      0.11153008269501266,
      0.07632372599089274,
      0.044497318656739816,
      0.04531191282057675,
      0.07801857436486531,
      0.11254694950028421,
      0.11715180936308961,
      0.08759946703181751,
      0.0510602097582302,
      0.04112806217024547,
      0.06693459495581111,
      0.10475340088480158,
      0.11981404419646237,
      0.09826983888576771,
      0.059928427959177036,
      0.04004072780212203,
      0.056891398222170736,
      0.09498801054597848,
      0.11930471509456558,
      0.10748484584818978,
      0.07039553608184902,
      0.04213652614860158,
      0.048689019457973826,
      0.08402868387970003
    ],
    [
      0.15099801794526435,
      0.1509989835968777,
      0.1509978650660083,
      1.0,
      1.0,
      1.0,
      0.22099976512938027,
      0.22099987998815582,
      0.22099975525260326,
      0.12099808989262269,
      0.1209990187246405,
      0.12099791948551805
    ],
    [
      0.04613682567689354,
      0.04566167681284291,
      0.07767626374984218,
      0.11274657033063298,
      0.11862908454810893,
      0.08991543553730821,
      0.053004863621743795,
      0.04183274867409851,
      0.06667066249291924
    ],
    [
      0.18099998991905708,
      0.1809999903759727,
      0.1809999887397966
    ],
    [
      0.05574932521914157,
      0.09346164342286342,
      0.11971676342147919,
      0.11037587463076406,
      0.07402691761151947,
      0.04408895962971797,
      0.048086847008981866,
      0.08234491085972571,
      0.11536644932024256
    ],
    [
      0.18099999823549606,
      0.18099999787480486,
      0.18099999547444154
    ],
    [
      0.10727946294004599,
      0.1205743243271814,
      0.09748473843337588,
      0.05923915019765484,
      0.04100039078694716,
      0.05953707804653374
    ],
    [
      0.04099999981392469,
      0.04099999881297382
    ],
    [
      0.07900000226006494
    ]
  ]
}"#;

#[test]
fn actual_pinned_deep_agent_class_hard_sigmoid_full_finance_and_update() {
    let f: Value = serde_json::from_str(ACTUAL_DEEP_AGENT_FIXTURE).unwrap();
    let targets = vec![ExecutionTarget::Cpu];
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let targets = targets
        .into_iter()
        .chain([ExecutionTarget::Metal])
        .collect::<Vec<_>>();
    for target in targets {
        let session = Session::new(target, Precision::F32, 32).unwrap();
        let mut c = config(0.);
        c.feature_count = 17;
        c.shared_lstm_reset.as_mut().unwrap().recurrent_activation =
            RecurrentActivation::HardSigmoid;
        c.adam.epsilon_convention = AdamEpsilonConvention::TensorflowUncorrected;
        c.small_trade_gate = Some(SmallTradeGateConfig {
            sharpness: 10.,
            force_threshold: 0.1,
            initial_trade_threshold: 0.08,
            inference_sharpen: true,
        });
        let mut snapshot = HedgePolicy::new(&session, c, 37)
            .unwrap()
            .snapshot()
            .unwrap();
        for (i, leaf) in snapshot.parameters.iter_mut().enumerate() {
            leaf.values = numbers(&f["parameters"][i]["values"]);
        }
        let mut policy = HedgePolicy::from_snapshot(&session, snapshot).unwrap();
        let (pnl, holdings) = rollout(&session, &mut policy, &f, false);
        for (day, holdings) in holdings.iter().enumerate() {
            close(
                &read(holdings.clone()),
                &numbers(&f["holdings_by_time"][day]),
                5e-5,
            );
        }
        close(&read(pnl.clone()), &numbers(&f["pnl"]), 5e-5);
        let mut gradients = pnl.square().mean().backward();
        for (i, p) in policy.parameters().iter().enumerate() {
            close(
                &read(p.grad(&gradients).unwrap()),
                &numbers(&f["gradients"][i]),
                5e-5,
            );
        }
        policy.backward_update(&mut gradients, 0.001).unwrap();
        for (i, p) in policy.snapshot().unwrap().parameters.iter().enumerate() {
            close(
                &p.values,
                &numbers(&f["after_first_tensorflow_adam"][i]),
                2e-5,
            );
        }
    }
}
const ACTUAL_DEEP_AGENT_FIXTURE: &str = r#"{"schema":"volt.test.actual_keras_shared_lstm_reset.v1","tensorflow":"2.15.0","keras":"2.15.0","profile":"actualKeras twoLSTM(reset1step)+twoReLUDense+linear; wealth+previousholdings attached; dropoutdisabled","dates":3,"paths":4,"width":3,"hedge_count":2,"feature_count":17,"parameters":[{"shape":[20,12],"values":[0.08,0.11365883939231586,0.11637189707302728,0.08564480032239469,0.04972790018768287,0.04164302901347446,0.06882338007204297,0.10627946394875157,0.11957432986493527,0.09648473940967027,0.05823915556442521,0.04000039173797186,0.0585370832799826,0.09680668147306563,0.11962429422779482,0.10601151360628468,0.0684838673333974,0.04154410032481773,0.049960510129132954,0.08599508838651809,0.1165178100291051,0.11346622554144224,0.07964594762838384,0.04615118383299317,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.043776865519735045,0.07470592999608908,0.1105023380191841,0.11825503713618013,0.09083623153231477,0.0534546446314813,0.040478735036285525,0.0638384941870774,0.10205706724966762,0.11999647440429069,0.10116330744480095,0.06287269322015396,0.04032884586227537,0.05425847466572002,0.09185474314837541,0.11855181545136351,0.10980452641917396,0.07365509324781164,0.04333913808337465,0.04672901029485607,0.08070807700421655,0.11403614098136475,0.11607153390595237,0.08494292490980897,0.04926981354705333,0.041849893889621126,0.06950500585184285,0.106809167033735,0.11946510368161942,0.09583700600727336,0.05764843804593535,0.040009793065655205,0.05913795991652353,0.097446590209913,0.11971490592338149,0.10546952028556551]},{"shape":[3,12],"values":[0.10576870748950765,0.11966659241809874,0.0970951952093532,0.058806554363660266,0.040003069697435965,0.05797257829609449,0.09619399682466394,0.11952672935508002,0.10651876920328733,0.06912957494356231,0.041734599349192485,0.04952065664323867,0.08532928165679769,0.11623818969233848,0.11382987324571738,0.08031852735143749,0.04651432887921013,0.04349670200835262,0.07404003896743205,0.11006293661408593,0.11844610898008466,0.09148210605310914,0.053961507773350126,0.04038057916411374,0.06322556335707075,0.10149288724025887,0.11999974971428083,0.10173102676928991,0.0634828980303784,0.04042051666981859,0.05374732975288839,0.09121072679076078,0.11836703331812322,0.11024886635144251,0.07432003116103994,0.043613333126658875]},{"shape":[1,12],"values":[0.15,0.15,0.15,1.0,1.0,1.0,0.22,0.22,0.22,0.12,0.12,0.12]},{"shape":[3,12],"values":[0.11452837466595496,0.08166322649733164,0.04726891555742359,0.0429674127068907,0.07271349982911617,0.10915876160503504,0.11879559243380346,0.09276393449397409,0.05499717404428472,0.04021789647184043,0.06201410141861594,0.10034645857489495,0.11997237554991672,0.10284787478639955,0.06471714331263964,0.040637399796734266,0.052747449377780004,0.0899133683193184,0.11796498214591579,0.11111176647204371,0.07565453618303682,0.04419250528721273,0.04565171989560282,0.07869058482676619,0.11293332002952326,0.11689731267692344,0.08693808620983569,0.050600015278048895,0.04129215471464779,0.06757210859622512,0.1052782085202754,0.1197436401072158,0.09766895226676896,0.05934951119680292,0.04001603429788789,0.0574436310699895]},{"shape":[3,12],"values":[0.0933995260062362,0.055525684362291244,0.040153415646566375,0.06141591282344971,0.09976453404554433,0.1199417338149842,0.10339668771567047,0.06534083482992287,0.04076255079734034,0.0522589966089151,0.08926039300406156,0.11774782677776419,0.11153008269501266,0.07632372599089274,0.044497318656739816,0.04531191282057675,0.07801857436486531,0.11254694950028421,0.11715180936308961,0.08759946703181751,0.0510602097582302,0.04112806217024547,0.06693459495581111,0.10475340088480158,0.11981404419646237,0.09826983888576771,0.059928427959177036,0.04004072780212203,0.056891398222170736,0.09498801054597848,0.11930471509456558,0.10748484584818978,0.07039553608184902,0.04213652614860158,0.048689019457973826,0.08402868387970003]},{"shape":[1,12],"values":[0.15,0.15,0.15,1.0,1.0,1.0,0.22,0.22,0.22,0.12,0.12,0.12]},{"shape":[3,3],"values":[0.04513696910345649,0.04466181377119386,0.07667642388730012,0.1117467145539661,0.11762922226719093,0.08891559656400991,0.05200501249625831,0.040832890833947304,0.06567082871052685]},{"shape":[1,3],"values":[0.18,0.18,0.18]},{"shape":[3,3],"values":[0.05474933448510713,0.09246165454053512,0.11871678688125945,0.10937588391496454,0.07302692875108081,0.043088983135487736,0.047086856201251644,0.08134492188884547,0.11436647259425983]},{"shape":[1,3],"values":[0.18,0.18,0.18]},{"shape":[3,2],"values":[0.10627946394875157,0.11957432986493527,0.09648473940967027,0.05823915556442521,0.04000039173797186,0.0585370832799826]},{"shape":[1,2],"values":[0.04,0.04]},{"shape":[1,1],"values":[0.08]}],"marks":[100.0,3.0,101.0,3.2,102.0,3.4,103.0,3.6,100.0,3.0,99.0,2.8,98.0,2.6,97.0,2.4,100.0,3.0,102.0,3.3,100.0,3.1,101.0,3.2,100.0,3.0,98.0,2.7,100.0,2.9,99.0,2.8],"features":[0.1,-0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,0.3,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,0.5,0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,-0.2,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,-0.3,-0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,-0.5,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,0.2,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,0.4,0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,0.1,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,-0.1,-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,-0.4,-0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,-0.2,-0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0],"payoffs":[2.0,0.0,1.0,3.0],"premium":2.5,"cash_growth":[1.000082195158659,1.000082195158659,1.000082195158659],"income_growth":[1.0000273976355822,1.0,1.0000273976355822,1.0,1.0000273976355822,1.0],"trading_costs":[0.001,0.002],"liquidation_costs":[0,0.002],"holdings_by_time":[[0.0708797425031662,0.06979686766862869,0.07087842375040054,0.06979557871818542,0.07090207934379578,0.06981869041919708,0.07086803764104843,0.06978541612625122],[0.08096537739038467,0.07972830533981323,0.08094493299722672,0.07970835268497467,0.08098076283931732,0.07974331080913544,0.0809369906783104,0.07970059663057327],[0.08615590631961823,0.08483941853046417,0.08611647039651871,0.08480095863342285,0.08615700900554657,0.08484049141407013,0.0861220434308052,0.0848063975572586]],"pnl":[0.7743711471557617,2.2049405574798584,1.569038987159729,-0.5897789001464844],"loss":2.055538368762109,"gradients":[[1.6491641872562468e-05,2.0880183001281694e-05,2.5799785362323746e-05,0.0,0.0,0.0,0.00014036944776307791,0.00015533529222011566,0.00015057993005029857,1.5895930118858814e-05,2.1008410840295255e-05,2.802577000693418e-05,3.4896265788120218e-06,4.397352313390002e-06,5.36616425961256e-06,0.0,0.0,0.0,2.7921185392187908e-05,3.052065949304961e-05,2.9149650799809024e-05,3.356989964231616e-06,4.468028691917425e-06,5.942153620708268e-06,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.3582142350496724e-05,-4.035430902149528e-05,-5.2948351367376745e-05,0.0,0.0,0.0,-0.0003527363878674805,-0.0004068767884746194,-0.00040979820187203586,-3.229347930755466e-05,-3.8741905882488936e-05,-5.407349090091884e-05,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.160662890877575e-05,-3.796746750595048e-05,-4.9898302677320316e-05,0.0,0.0,0.0,-0.00033307058038190007,-0.0003843941376544535,-0.00038716342533007264,-3.038011163880583e-05,-3.6407913285074756e-05,-5.091771890874952e-05,-1.1717673942257534e-06,-1.436746629224217e-06,-1.8068176359520294e-06,0.0,0.0,0.0,-1.0930685675702989e-05,-1.2340877219685353e-05,-1.2253563909325749e-05,-1.1325390687488834e-06,-1.4237655250326497e-06,-1.912826292027603e-06,-1.1538656963239191e-06,-1.4147967704047915e-06,-1.7792143580663833e-06,0.0,0.0,0.0,-1.0763695172499865e-05,-1.215234169649193e-05,-1.2066362614859827e-05,-1.1152367278555175e-06,-1.4020140497450484e-06,-1.883603317764937e-06],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[-1.3432858395390213e-05,-1.6141724699991755e-05,-2.117934127454646e-05,0.0,0.0,0.0,-0.0001410945551469922,-0.00016275071538984776,-0.00016391929239034653,-1.2917393178213388e-05,-1.549676017020829e-05,-2.1629395632771775e-05],[-8.313307262142189e-06,-1.6232499547186308e-05,-7.746210030745715e-06,0.0,0.0,0.0,-8.777419134275988e-05,-0.00017184563330374658,-8.418357174377888e-05,-8.541045644960832e-06,-1.666016578383278e-05,-7.908911356935278e-06,-8.735831215744838e-06,-1.7060439859051257e-05,-8.151047040882986e-06,0.0,0.0,0.0,-9.27766741369851e-05,-0.00018164791981689632,-8.896132931113243e-05,-8.970429917098954e-06,-1.7501995898783207e-05,-8.321737368532922e-06,-1.1310107765893918e-05,-2.2085252567194402e-05,-1.054303447745042e-05,0.0,0.0,0.0,-0.00011963026190642267,-0.00023421707737725228,-0.00011472834012238309,-1.1618049029493704e-05,-2.2663958588964306e-05,-1.0764284525066614e-05],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[-0.00011896862997673452,-0.0002321929787285626,-0.0001104492912418209,0.0,0.0,0.0,-0.0012364286230877042,-0.0024203918874263763,-0.0011865736450999975,-0.00012239848729223013,-0.0002385981206316501,-0.00011278813326498494],[-0.002005833201110363,-0.0021005929447710514,-0.0017966398736461997,-0.0019959150813519955,-0.002090207766741514,-0.001787759130820632,-0.0019374453695490956,-0.0020289854146540165,-0.0017354052979499102],[-0.02950238808989525,-0.030896801501512527,-0.02642674371600151],[-0.03201865032315254,-0.026646994054317474,-0.012701962143182755,-0.03195793926715851,-0.026596417650580406,-0.012677895836532116,-0.032265547662973404,-0.026852892711758614,-0.012799801304936409],[-0.16501478850841522,-0.1373082995414734,-0.06546829640865326],[-0.2814072072505951,-0.05488273873925209,-0.29029327630996704,-0.05661230534315109,-0.2976212799549103,-0.05803854018449783],[-1.273167371749878,-0.24840819835662842],[0.15917962789535522]],"after_first_adam":[[0.0790004719753159,0.11265922366216374,0.11537220949723587,0.08564480032239469,0.04972790018768287,0.04164302901347446,0.06782344240435238,0.10527951944563618,0.11857438604311593,0.09548524045483059,0.05723954671184519,0.03900067978299536,0.05753788976538603,0.0958073332876126,0.11862483607273228,0.10601151360628468,0.0684838673333974,0.04154410032481773,0.04896062558996841,0.08499519300465569,0.11551791831512238,0.11246709109290394,0.07864660047858352,0.0451516650640457,0.04477656822474009,0.07570566493640318,0.11150213849605818,0.11825503713618013,0.09083623153231477,0.0534546446314813,0.04147870298661598,0.06483846700149165,0.10305704084633853,0.12099616015192978,0.10216301143064048,0.06387248722550867,0.04132638661535929,0.055256455629438185,0.0928530788036228,0.11855181545136351,0.10980452641917396,0.07365509324781164,0.04433879882810912,0.04772870618488039,0.0817077661503922,0.11503350895031801,0.11706948966401626,0.08594142163054297,0.050267316232433454,0.042847843585679654,0.07050331572297645,0.106809167033735,0.11946510368161942,0.09583700600727336,0.05864809352727746,0.041009484237315284,0.06013764423976353,0.09844391744383763,0.12071283002419947,0.10646799371342691],[0.10576870748950765,0.11966659241809874,0.0970951952093532,0.058806554363660266,0.040003069697435965,0.05797257829609449,0.09619399682466394,0.11952672935508002,0.10651876920328733,0.06912957494356231,0.041734599349192485,0.04952065664323867,0.08532928165679769,0.11623818969233848,0.11382987324571738,0.08031852735143749,0.04651432887921013,0.04349670200835262,0.07404003896743205,0.11006293661408593,0.11844610898008466,0.09148210605310914,0.053961507773350126,0.04038057916411374,0.06322556335707075,0.10149288724025887,0.11999974971428083,0.10173102676928991,0.0634828980303784,0.04042051666981859,0.05374732975288839,0.09121072679076078,0.11836703331812322,0.11024886635144251,0.07432003116103994,0.043613333126658875],[0.15099932786349557,0.15099940204334583,0.15099955072174587,1.0,1.0,1.0,0.2209999280579947,0.22099993900114248,0.22099994088535133,0.12099929225678342,0.1209993328791555,0.12099953355693344],[0.11552719789563086,0.08266262378247646,0.04826766064057097,0.0429674127068907,0.07271349982911617,0.10915876160503504,0.11979545942376232,0.09376386658012256,0.05599703522213485,0.0412167560291947,0.06301351664675212,0.10134523490000905,0.12097112346563078,0.10384723370213791,0.0657158130420682,0.040637399796734266,0.052747449377780004,0.0899133683193184,0.11896484292286638,0.1121116953962465,0.07665439079210438,0.04519128926250493,0.04665109674826002,0.079689287404031,0.11393238414135425,0.11789683348146318,0.08793708995523726,0.050600015278048895,0.04129215471464779,0.06757210859622512,0.10627810361238303,0.12074358655197295,0.09866884274294506,0.060348603268933064,0.04101556895484921,0.05844265952777197],[0.0933995260062362,0.055525684362291244,0.040153415646566375,0.06141591282344971,0.09976453404554433,0.1199417338149842,0.10339668771567047,0.06534083482992287,0.04076255079734034,0.0522589966089151,0.08926039300406156,0.11774782677776419,0.11153008269501266,0.07632372599089274,0.044497318656739816,0.04531191282057675,0.07801857436486531,0.11254694950028421,0.11715180936308961,0.08759946703181751,0.0510602097582302,0.04112806217024547,0.06693459495581111,0.10475340088480158,0.11981404419646237,0.09826983888576771,0.059928427959177036,0.04004072780212203,0.056891398222170736,0.09498801054597848,0.11930471509456558,0.10748484584818978,0.07039553608184902,0.04213652614860158,0.048689019457973826,0.08402868387970003],[0.1509999253738753,0.15099996180027336,0.15099991997523474,1.0,1.0,1.0,0.22099999140469798,0.22099999563213263,0.22099999103483198,0.1209999279020495,0.12099996303748571,0.12099992199456637],[0.04613696373408817,0.045661808645576944,0.07767641788567256,0.11274670915792091,0.11862921711610182,0.08991559053260259,0.053005006956043405,0.041832885545202085,0.06667082251818902],[0.18099999967503982,0.180999999691808,0.18099999963158653],[0.05574933418842181,0.09346165417995596,0.11971678605451942,0.11037588361761383,0.07402692838971178,0.04408898230705878,0.04808685590723293,0.08234492153143455,0.11536647177428493],[0.18099999998065497,0.18099999996823263,0.18099999987773183],[0.10727946395829005,0.12057432967666223,0.09748473942037167,0.0592391553833581,0.04100039174957999,0.05953708310453456],[0.041000000039106005,0.0409999999954028]],"learning_rate":0.001,"first_step_optimizer":{"actual":"tf.keras.optimizers.legacy.Adam","epsilon_hat":3.1622776601683806e-10,"mapping":"first-step epsilonhat=canonicalepsilon*sqrt(1-beta2); not a wholetrajectory epsilonhat equivalence"},"after_first_tensorflow_adam":[[0.07901881635189056,0.11267375946044922,0.11538400501012802,0.08564480394124985,0.0497279018163681,0.04164302721619606,0.06782563030719757,0.10528149455785751,0.11857642978429794,0.09550424665212631,0.05725398287177086,0.039011549204587936,0.057620175182819366,0.09587377309799194,0.1186799481511116,0.10601150989532471,0.068483866751194,0.041544102132320404,0.04897170886397362,0.08500534296035767,0.11552854627370834,0.11255231499671936,0.07871204614639282,0.04520171135663986,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0009906713385134935,0.0009922245517373085,0.0009940629824995995,0.0,0.0,0.0,0.0009991043480113149,0.0009992234408855438,0.0009992289124056697,0.0009903025347739458,0.0009919037111103535,0.0009941859170794487,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.04476695880293846,0.07569766789674759,0.11149603873491287,0.11825503408908844,0.09083623439073563,0.05345464497804642,0.041477788239717484,0.06483767181634903,0.10305625200271606,0.12098617106676102,0.1021546944975853,0.0638665184378624,0.041116323322057724,0.05507807806134224,0.09270579367876053,0.1185518130660057,0.10980452597141266,0.07365509122610092,0.04431102052330971,0.047704026103019714,0.08168292045593262,0.11481786519289017,0.11688979715108871,0.08580105751752853,0.05005470663309097,0.0426672101020813,0.07035409659147263,0.10680916905403137,0.11946510523557663,0.09583700448274612,0.058619897812604904,0.04098442941904068,0.060112420469522476,0.0982256755232811,0.120530866086483,0.10632576793432236],[0.10576871037483215,0.11966659128665924,0.09709519892930984,0.05880655348300934,0.04000306874513626,0.05797257646918297,0.09619399905204773,0.11952672898769379,0.10651876777410507,0.0691295713186264,0.04173459857702255,0.04952065646648407,0.08532927930355072,0.11623819172382355,0.11382987350225449,0.08031852543354034,0.04651432856917381,0.04349670186638832,0.07404004037380219,0.11006293445825577,0.11844611167907715,0.09148210287094116,0.053961507976055145,0.040380578488111496,0.06322555989027023,0.10149288922548294,0.11999975144863129,0.10173102468252182,0.06348289549350739,0.040420517325401306,0.053747329860925674,0.0912107303738594,0.1183670312166214,0.110248863697052,0.07432003319263458,0.043613333255052567],[0.1509770005941391,0.15098078548908234,0.15098530054092407,1.0,1.0,1.0,0.22099776566028595,0.22099806368350983,0.22099807858467102,0.12097609788179398,0.12098000198602676,0.12098558992147446],[0.11549172550439835,0.08264411985874176,0.04822969436645508,0.04296741262078285,0.07271350175142288,0.10915876179933548,0.11979200690984726,0.0937620997428894,0.05599343031644821,0.04118219390511513,0.06299547851085663,0.10130801051855087,0.12093744426965714,0.10382967442274094,0.06567979604005814,0.04063740000128746,0.0527474507689476,0.08991336822509766,0.11896158754825592,0.11211002618074417,0.07665099203586578,0.04515845328569412,0.046633969992399216,0.07965397834777832,0.1139061227440834,0.11788319796323776,0.08790896087884903,0.05060001462697983,0.04129215329885483,0.06757210940122604,0.10627557337284088,0.12074229121208191,0.09866620600223541,0.06032301485538483,0.04100227355957031,0.058415088802576065],[0.0933995246887207,0.055525682866573334,0.04015341401100159,0.061415914446115494,0.09976453334093094,0.11994173377752304,0.1033966913819313,0.06534083187580109,0.04076255112886429,0.05225899815559387,0.08926039189100266,0.11774782836437225,0.11153008043766022,0.0763237252831459,0.04449731856584549,0.04531191289424896,0.07801857590675354,0.11254695057868958,0.11715181171894073,0.08759946376085281,0.051060210913419724,0.04112806171178818,0.06693459302186966,0.10475339740514755,0.11981404572725296,0.09826984256505966,0.05992842838168144,0.04004072770476341,0.05689139664173126,0.09498801082372665,0.11930471658706665,0.1074848473072052,0.07039553672075272,0.042136527597904205,0.048689018934965134,0.0840286836028099],[0.1509973555803299,0.15099865198135376,0.15099714696407318,1.0,1.0,1.0,0.22099974751472473,0.22099986672401428,0.22099973261356354,0.12099742144346237,0.12099867314100266,0.12099719792604446],[0.04613681137561798,0.04566166177392006,0.07767624408006668,0.11274655163288116,0.11862906813621521,0.08991541713476181,0.05300484970211983,0.04183273762464523,0.06667064130306244],[0.1809999942779541,0.1809999942779541,0.1809999942779541],[0.055749326944351196,0.09346164017915726,0.11971675604581833,0.11037587374448776,0.0740269124507904,0.04408895969390869,0.048086848109960556,0.0823449119925499,0.11536644399166107],[0.1810000091791153,0.1810000091791153,0.1810000091791153],[0.1072794646024704,0.12057432532310486,0.09748474508523941,0.05923914909362793,0.04100038856267929,0.05953707918524742],[0.04099999740719795,0.04099999740719795],[0.07900000363588333]],"actual_author_source_sha256":"e6c0d5fe4ddc9273a50041882435b82381b52988bd50c984a382afdd393d33af"}"#;
