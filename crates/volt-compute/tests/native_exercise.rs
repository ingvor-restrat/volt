//! Independent stopped-account checks through the public JSON command protocol.
use serde_json::{Value, json};
use volt_compute::{
    ExecutionTarget, Precision,
    native_learner::{Command, LearnerConfig, NativeLearner},
};

fn request(
    learner: &mut NativeLearner,
    command: Value,
) -> Result<Value, volt_compute::ComputeError> {
    learner.handle(serde_json::from_value::<Command>(command).unwrap())
}
fn ok(learner: &mut NativeLearner, command: Value) -> Value {
    request(learner, command).unwrap()
}
fn targets() -> impl Iterator<Item = (ExecutionTarget, Precision)> {
    [
        (ExecutionTarget::Cpu, Precision::F64),
        (ExecutionTarget::Cpu, Precision::F32),
    ]
    .into_iter()
    .chain({
        #[cfg(feature = "metal")]
        {
            vec![(ExecutionTarget::Metal, Precision::F32)]
        }
        #[cfg(not(feature = "metal"))]
        {
            vec![]
        }
    })
}
fn learner(target: ExecutionTarget, precision: Precision) -> NativeLearner {
    let mut cfg = serde_json::to_value(LearnerConfig::default()).unwrap();
    cfg["target"] = json!(target);
    cfg["precision"] = json!(precision);
    cfg["premium"] = json!(7.);
    cfg["cash_growth"] = json!([1.01, 1.02, 1.03]);
    cfg["income_growth"] = json!([1.001, 1., 1.002, 1., 1.003, 1.]);
    cfg["trading_costs"] = json!([0.001, 0.002]);
    cfg["liquidation_costs"] = json!([0.003, 0.004]);
    cfg["policy"]["time_steps"] = json!(3);
    cfg["policy"]["hidden_width"] = json!(4);
    cfg["policy"]["max_batch_rows"] = json!(4);
    let mut l = NativeLearner::new(serde_json::from_value(cfg).unwrap()).unwrap();
    ok(
        &mut l,
        json!({"op":"register_dataset","dataset_id":"independent-stopped-cash-v1","partitions":[{"split":"train","first_path":0,"paths":4}]}),
    );
    let mut cp = ok(&mut l, json!({"op":"checkpoint"}));
    for (i, p) in cp["policy"]["parameters"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        for v in p["values"].as_array_mut().unwrap() {
            *v = json!(0.);
        }
        if i % 10 == 9 {
            p["values"] = json!([[0.375, -1.25], [-0.625, 0.5], [2., -3.]][i / 10]);
        }
    }
    NativeLearner::restore(serde_json::from_value(cp).unwrap()).unwrap()
}
fn batch() -> Value {
    json!({"paths":4,"provenance":{"dataset_id":"independent-stopped-cash-v1","first_path":0},
      "marks":[100.,50.,91.,53.,87.,48.,96.,55., 100.,50.,91.,53.,87.,48.,96.,55., 100.,50.,91.,53.,87.,48.,96.,55., 100.,50.,91.,53.,87.,48.,96.,55.],
      "features":vec![0.;24],"payoffs":vec![0.;4],
      "cash_income":[0.4,0.1,0.2,0.,0.6,0.2, 0.4,0.1,0.2,0.,0.6,0.2, 0.4,0.1,0.2,0.,0.6,0.2, 0.4,0.1,0.2,0.,0.6,0.2],
      "exercise":{"settlement_step":[0,1,2,3],"settlement_amounts":[12.5,-2.,12.5,3.]}})
}
fn upload(l: &mut NativeLearner, id: &str, b: Value) {
    ok(
        l,
        json!({"op":"upload","batch_id":id,"split":"train","batch":b}),
    );
}
fn eval(l: &mut NativeLearner, id: &str) -> Value {
    ok(
        l,
        json!({"op":"evaluate","batch_id":id,"retain_holdings":true}),
    )
}
fn n(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn close(a: f64, b: f64, p: Precision) {
    let tol = if p == Precision::F64 { 1e-10 } else { 3e-4 };
    assert!((a - b).abs() <= tol * (1. + b.abs()), "{a} vs {b}, {p:?}");
}
fn scalar(b: &Value, path: usize) -> (f64, f64) {
    scalar_shift(b, path, [0.; 2])
}
fn scalar_shift(b: &Value, path: usize, shift: [f64; 2]) -> (f64, f64) {
    let tau = b["exercise"]["settlement_step"][path].as_u64().unwrap() as usize;
    let mut cash = b.get("initial_wealth").map_or(7., |w| n(&w[path]));
    let mut previous = [0_f64; 2];
    let mut fees = 0.;
    for t in 0..tau {
        let mut target = [[0.375, -1.25], [-0.625, 0.5], [2., -3.]][t];
        if t == 0 {
            for j in 0..2 {
                target[j] += shift[j];
            }
        }
        for j in 0..2 {
            let mark = n(&b["marks"][(path * 4 + t) * 2 + j]);
            let charge = (target[j] - previous[j]).abs() * mark * [0.001, 0.002][j];
            cash -= (target[j] - previous[j]) * mark + charge;
            fees += charge;
        }
        cash *= [1.01, 1.02, 1.03][t];
        for (j, h) in target.iter().enumerate() {
            cash += h
                * (n(&b["cash_income"][(path * 3 + t) * 2 + j])
                    + n(&b["marks"][(path * 4 + t + 1) * 2 + j])
                        * (if j == 0 { [0.001, 0.002, 0.003][t] } else { 0. }));
        }
        previous = target;
    }
    for j in 0..2 {
        let mark = n(&b["marks"][(path * 4 + tau) * 2 + j]);
        let charge = previous[j].abs() * mark * [0.003, 0.004][j];
        cash += previous[j] * mark - charge;
        fees += charge;
    }
    cash -= n(&b["exercise"]["settlement_amounts"][path]);
    for g in &[1.01, 1.02, 1.03][tau..] {
        cash *= g;
    }
    (cash, fees)
}
#[test]
fn stopped_cash_income_signed_payout_and_liquidation_reconcile() {
    for (target, p) in targets() {
        let mut l = learner(target, p);
        let b = batch();
        upload(&mut l, "mixed", b.clone());
        let r = eval(&mut l, "mixed");
        for path in 0..4 {
            let (profit, fees) = scalar(&b, path);
            close(n(&r["pnl"][path]), profit, p);
            close(n(&r["costs"][path]), fees, p);
        }
    }
}
#[test]
fn prices_and_features_after_exercise_have_zero_financial_effect() {
    for (target, p) in targets() {
        let mut l = learner(target, p);
        let b = batch();
        upload(&mut l, "a", b.clone());
        let a = eval(&mut l, "a");
        let mut changed = b;
        for path in 0..4 {
            let tau = path;
            for t in tau + 1..4 {
                for j in 0..2 {
                    changed["marks"][(path * 4 + t) * 2 + j] = json!(10000. + (t + j) as f64);
                }
            }
            for t in tau..3 {
                for j in 0..2 {
                    changed["features"][(path * 3 + t) * 2 + j] = json!(999.);
                    changed["cash_income"][(path * 3 + t) * 2 + j] = json!(777.);
                }
            }
        }
        upload(&mut l, "b", changed);
        let b = eval(&mut l, "b");
        for path in 0..4 {
            close(n(&a["pnl"][path]), n(&b["pnl"][path]), p);
            close(n(&a["costs"][path]), n(&b["costs"][path]), p);
        }
    }
}
#[test]
fn mature_exercise_matches_legacy_terminal_liability() {
    for (target, p) in targets() {
        let mut l = learner(target, p);
        let mut stopped = batch();
        stopped["exercise"]["settlement_step"] = json!(vec![3; 4]);
        upload(&mut l, "stopped", stopped.clone());
        let a = eval(&mut l, "stopped");
        let payout = stopped["exercise"]["settlement_amounts"].clone();
        stopped.as_object_mut().unwrap().remove("exercise");
        stopped["payoffs"] = payout;
        upload(&mut l, "legacy", stopped);
        let b = eval(&mut l, "legacy");
        for path in 0..4 {
            close(n(&a["pnl"][path]), n(&b["pnl"][path]), p);
            close(n(&a["costs"][path]), n(&b["costs"][path]), p);
        }
    }
}
#[test]
fn exercise_payload_validation_preserves_published_policy() {
    let mut l = learner(ExecutionTarget::Cpu, Precision::F64);
    let before = ok(&mut l, json!({"op":"checkpoint"}));
    let mut invalids = Vec::new();
    let b = batch();
    let mut x = b.clone();
    x["payoffs"][0] = json!(1.);
    invalids.push(x);
    let mut x = b.clone();
    x["exercise"]["settlement_step"][0] = json!(4);
    invalids.push(x);
    let mut x = b.clone();
    x["exercise"]["settlement_step"] = json!([1]);
    invalids.push(x);
    let mut x = b.clone();
    x["exercise"]["settlement_amounts"] = json!([1.]);
    invalids.push(x);
    let mut x = b.clone();
    x["cash_income"] = json!([0.]);
    invalids.push(x);
    let mut x = b;
    x["initial_wealth"] = json!([1.]);
    invalids.push(x);
    for (i, b) in invalids.into_iter().enumerate() {
        assert!(
            request(
                &mut l,
                json!({"op":"upload","batch_id":format!("bad{i}"),"split":"train","batch":b})
            )
            .is_err()
        );
    }
    assert_eq!(before, ok(&mut l, json!({"op":"checkpoint"})));
}
#[test]
fn gradients_stop_at_settlement_and_cash_only_paths_have_no_policy_update() {
    for (target, p) in targets() {
        let mut l = learner(target, p);
        let mut b = batch();
        b["exercise"]["settlement_step"] = json!(vec![1; 4]);
        upload(&mut l, "early", b);
        let before = ok(&mut l, json!({"op":"checkpoint"}));
        let update = ok(
            &mut l,
            json!({"op":"update","batch_id":"early","expected_version":0}),
        );
        assert!(n(&update["gradient_l2_norm"]) > 0.);
        let after = ok(&mut l, json!({"op":"checkpoint"}));
        for i in 10..30 {
            assert_eq!(
                before["policy"]["parameters"][i]["values"],
                after["policy"]["parameters"][i]["values"],
                "post-settlement parameter {i}, {p:?}"
            );
        }
        let mut l = learner(target, p);
        let mut b = batch();
        b["exercise"]["settlement_step"] = json!(vec![0; 4]);
        upload(&mut l, "immediate", b);
        let before = ok(&mut l, json!({"op":"checkpoint"}));
        let update = ok(
            &mut l,
            json!({"op":"update","batch_id":"immediate","expected_version":0}),
        );
        close(n(&update["gradient_l2_norm"]), 0., p);
        let after = ok(&mut l, json!({"op":"checkpoint"}));
        for i in 0..30 {
            assert_eq!(
                before["policy"]["parameters"][i]["values"],
                after["policy"]["parameters"][i]["values"]
            );
        }
    }
}

#[test]
fn future_observations_leave_recurrent_wealth_policy_gradients_unchanged() {
    for (target, precision) in targets() {
        let mut template = learner(target, precision);
        let snapshot = ok(&mut template, json!({"op":"checkpoint"}));
        let mut config = snapshot["config"].clone();
        config["policy"]["include_wealth"] = json!(true);
        let mut a = NativeLearner::new(serde_json::from_value(config.clone()).unwrap()).unwrap();
        let mut b = NativeLearner::new(serde_json::from_value(config).unwrap()).unwrap();
        for learner in [&mut a, &mut b] {
            ok(
                learner,
                json!({"op":"register_dataset","dataset_id":"independent-stopped-cash-v1","partitions":[{"split":"train","first_path":0,"paths":4}]}),
            );
        }
        let mut original = batch();
        original["exercise"]["settlement_step"] = json!(vec![1; 4]);
        let mut changed = original.clone();
        for path in 0..4 {
            for time in 1..3 {
                for asset in 0..2 {
                    changed["features"][(path * 3 + time) * 2 + asset] = json!(91.);
                    changed["cash_income"][(path * 3 + time) * 2 + asset] = json!(83.);
                }
            }
            for time in 2..4 {
                for asset in 0..2 {
                    changed["marks"][(path * 4 + time) * 2 + asset] = json!(2345.);
                }
            }
        }
        upload(&mut a, "original", original);
        upload(&mut b, "changed", changed);
        let ua = ok(
            &mut a,
            json!({"op":"update","batch_id":"original","expected_version":0}),
        );
        let ub = ok(
            &mut b,
            json!({"op":"update","batch_id":"changed","expected_version":0}),
        );
        close(
            n(&ua["loss_before_update"]),
            n(&ub["loss_before_update"]),
            precision,
        );
        close(
            n(&ua["gradient_l2_norm"]),
            n(&ub["gradient_l2_norm"]),
            precision,
        );
        assert!(n(&ua["gradient_l2_norm"]) > 0.);
        let ca = ok(&mut a, json!({"op":"checkpoint"}));
        let cb = ok(&mut b, json!({"op":"checkpoint"}));
        assert_eq!(ca["policy"], cb["policy"]);
    }
}

#[test]
fn per_path_initial_capital_financing_and_gradient_match_independent_account() {
    for (target, precision) in targets() {
        let mut learner = learner(target, precision);
        let plain = batch();
        upload(&mut learner, "plain", plain.clone());
        let base = eval(&mut learner, "plain");
        let mut varying = plain;
        varying["initial_wealth"] = json!([2., -3., 7., 11.]);
        upload(&mut learner, "varying", varying.clone());
        let result = eval(&mut learner, "varying");
        let mut loss = 0.;
        let mut bias_gradient = [0.; 2];
        for path in 0..4 {
            let (profit, fees) = scalar(&varying, path);
            close(n(&result["pnl"][path]), profit, precision);
            close(n(&result["costs"][path]), fees, precision);
            close(
                profit - n(&base["pnl"][path]),
                (n(&varying["initial_wealth"][path]) - 7.) * 1.01 * 1.02 * 1.03,
                precision,
            );
            loss += profit * profit / 4.;
            for j in 0..2 {
                let mut plus = [0.; 2];
                let mut minus = [0.; 2];
                plus[j] = 1e-4;
                minus[j] = -1e-4;
                let derivative = (scalar_shift(&varying, path, plus).0
                    - scalar_shift(&varying, path, minus).0)
                    / 2e-4;
                bias_gradient[j] += 2. * profit * derivative / 4.;
            }
        }
        close(n(&result["loss"]), loss, precision);
        let update = ok(
            &mut learner,
            json!({"op":"update","batch_id":"varying","expected_version":0}),
        );
        close(n(&update["loss_before_update"]), loss, precision);
        let checkpoint = ok(&mut learner, json!({"op":"checkpoint"}));
        for (j, gradient) in bias_gradient.iter().enumerate() {
            let first = n(&checkpoint["policy"]["parameters"][9]["first_moment"][j]);
            let tolerance = if precision == Precision::F64 {
                1e-7
            } else {
                1e-3
            };
            assert!(
                (first - 0.1 * gradient).abs() <= tolerance * (1. + gradient.abs()),
                "first Adam moment {first} vs scalar gradient {gradient}"
            );
        }
    }
}
