//! Own numerical fixture executed with actual Keras 2.15 single-step LSTM
//! layers. It checks reset semantics, wealth/holdings feedback and shared AD.
use burn::tensor::{DType, Tensor, TensorCreationOptions, TensorData};
use serde_json::{Value, json};
use volt_compute::{
    ExecutionTarget, Precision, Session,
    hedge_policy::{HedgePolicy, HedgePolicyConfig, SharedLstmResetConfig},
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
                    features[(r * 3 + day) * 2..(r * 3 + day + 1) * 2]
                        .iter()
                        .copied()
                })
                .collect(),
            [n, 2],
        );
        let wealth = if detach_wealth {
            value.clone().without_autodiff()
        } else {
            value.clone()
        };
        let target = p
            .forward_at_with_wealth(day, x, prev.clone(), Some(wealth), true)
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
    c.shared_lstm_reset = Some(SharedLstmResetConfig {
        dropout_probability: 0.5,
        dropout_seed: 1,
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
