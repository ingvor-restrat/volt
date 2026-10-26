use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

struct Worker {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
    serial: u64,
}
impl Worker {
    fn new() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_native_learner"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let reader = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            reader,
            serial: 0,
        }
    }
    fn request(&mut self, mut request: Value) -> Value {
        self.serial += 1;
        request["id"] = json!(self.serial.to_string());
        writeln!(self.child.stdin.as_mut().unwrap(), "{request}").unwrap();
        let mut line = String::new();
        self.reader.read_line(&mut line).unwrap();
        let reply: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(reply["id"], self.serial.to_string());
        reply
    }
    fn ok(&mut self, request: Value) -> Value {
        let r = self.request(request);
        assert_eq!(r["ok"], true, "{r}");
        r["result"].clone()
    }
    fn register_fixture(&mut self) {
        self.ok(
            json!({"op":"register_dataset","dataset_id":"fixture-v1","partitions":[
            {"split":"train","first_path":0,"paths":4},
            {"split":"validation","first_path":4,"paths":4},
            {"split":"test","first_path":8,"paths":4}]}),
        );
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn upload(split: &str, id: &str) -> Value {
    json!({"op":"upload","batch_id":id,"split":split,"batch":{"paths":4,
        "provenance":{"dataset_id":"fixture-v1","first_path":match split {"train"=>0,"validation"=>4,_=>8}},
        "marks":[100.,3.,101.,3.4,102.,3.8, 100.,3.,99.,2.6,98.,2.2, 100.,3.,101.,3.3,100.5,3.1, 100.,3.,99.,2.7,99.5,2.9],
        "features":[0.,0.,1.,0.4, 0.,0.,-1.,-0.4, 0.,0.,1.,0.3, 0.,0.,-1.,-0.3],
        "payoffs":[1.,-1.,0.25,-0.25]}})
}

#[test]
fn persistent_update_split_guard_and_exact_optimizer_resume() {
    let mut worker = Worker::new();
    worker.ok(json!({"op":"configure","config":{}}));
    worker.register_fixture();
    worker.ok(upload("train", "train"));
    worker.ok(upload("validation", "validation"));
    let initial = worker.ok(json!({"op":"evaluate","batch_id":"validation"}));
    assert_eq!(
        worker.request(json!({"op":"update","batch_id":"validation","expected_version":0}))["ok"],
        false
    );
    let first = worker.ok(json!({"op":"update","batch_id":"train","expected_version":0}));
    assert_eq!(first["version"], 1);
    assert_eq!(
        worker.request(json!({"op":"update","batch_id":"train","expected_version":0}))["ok"],
        false
    );
    let checkpoint = worker.ok(json!({"op":"checkpoint"}));
    let mut resumed = Worker::new();
    resumed.ok(json!({"op":"restore","checkpoint":checkpoint}));
    resumed.ok(upload("train", "train"));
    worker.ok(json!({"op":"update","batch_id":"train","expected_version":1}));
    resumed.ok(json!({"op":"update","batch_id":"train","expected_version":1}));
    assert_eq!(
        worker.ok(json!({"op":"checkpoint"})),
        resumed.ok(json!({"op":"checkpoint"}))
    );
    let mut leaked = upload("train", "leak");
    leaked["batch"]["provenance"]["first_path"] = json!(4);
    assert_eq!(resumed.request(leaked)["ok"], false);
    for version in 2..40 {
        worker.ok(json!({"op":"update","batch_id":"train","expected_version":version}));
    }
    let final_eval = worker.ok(json!({"op":"evaluate","batch_id":"validation"}));
    assert!(final_eval["loss"].as_f64().unwrap() < initial["loss"].as_f64().unwrap());
}

#[test]
fn financed_cash_ledger_and_objectives_agree_with_independent_scalar_calculation() {
    for objective in [
        json!({"kind":"squared_loss"}),
        json!({"kind":"semi_squared_loss"}),
        json!({"kind":"cvar","alpha":0.625}),
        json!({"kind":"entropic","risk_aversion":0.3}),
    ] {
        let mut worker = Worker::new();
        worker.ok(json!({"op":"configure","config":{"premium":2.5,"cash_growth":[1.001,1.002],"income_growth":[1.0004,1.,1.0005,1.],"trading_costs":[0.0005,0.01],"liquidation_costs":[0.0007,0.02],"objective":objective}}));
        worker.register_fixture();
        let request = upload("test", "test");
        let batch = request["batch"].clone();
        worker.ok(request);
        let result = worker.ok(json!({"op":"evaluate","batch_id":"test","retain_holdings":true}));
        let mut profits = Vec::new();
        for path in 0..4 {
            let mut cash = 2.5;
            let mut previous = [0.; 2];
            let mut costs = 0.;
            for time in 0..2 {
                for instrument in 0..2 {
                    let target = result["holdings"][time][path * 2 + instrument]
                        .as_f64()
                        .unwrap();
                    let mark = batch["marks"][(path * 3 + time) * 2 + instrument]
                        .as_f64()
                        .unwrap();
                    let charge =
                        (target - previous[instrument]).abs() * mark * [0.0005, 0.01][instrument];
                    cash -= (target - previous[instrument]) * mark + charge;
                    costs += charge;
                    previous[instrument] = target;
                }
                cash *= [1.001, 1.002][time];
                // Dividend income is credited separately from marked inventory.
                let stock_next = batch["marks"][(path * 3 + time + 1) * 2].as_f64().unwrap();
                cash += previous[0] * stock_next * [0.0004, 0.0005][time];
            }
            for instrument in 0..2 {
                let mark = batch["marks"][(path * 3 + 2) * 2 + instrument]
                    .as_f64()
                    .unwrap();
                let charge = previous[instrument].abs() * mark * [0.0007, 0.02][instrument];
                cash += previous[instrument] * mark - charge;
                costs += charge;
            }
            let profit = cash - batch["payoffs"][path].as_f64().unwrap();
            profits.push(profit);
            assert!((result["pnl"][path].as_f64().unwrap() - profit).abs() < 1e-11);
            assert!((result["costs"][path].as_f64().unwrap() - costs).abs() < 1e-11);
        }
        let expected = match objective["kind"].as_str().unwrap() {
            "squared_loss" => profits.iter().map(|p| p * p).sum::<f64>() / 4.,
            "semi_squared_loss" => profits.iter().map(|p| (-p).max(0.).powi(2)).sum::<f64>() / 4.,
            "cvar" => {
                profits.sort_by(f64::total_cmp);
                (-profits[0] - 0.5 * profits[1]) / 1.5
            }
            _ => (profits.iter().map(|p| (-0.3 * p).exp()).sum::<f64>() / 4.).ln() / 0.3,
        };
        assert!(
            (result["loss"].as_f64().unwrap() - expected).abs() < 1e-10,
            "{result}"
        );
    }
}

#[test]
fn episode_gradients_match_parameter_bumps_including_previous_holdings_feedback() {
    let mut worker = Worker::new();
    worker.ok(json!({"op":"configure","config":{"cash_growth":[1.001,1.002],"income_growth":[1.0004,1.,1.0005,1.]}}));
    worker.register_fixture();
    worker.ok(upload("train", "train"));
    let before = worker.ok(json!({"op":"checkpoint"}));
    worker.ok(json!({"op":"update","batch_id":"train","expected_version":0}));
    let after = worker.ok(json!({"op":"checkpoint"}));
    let mut nonzero = 0;
    for (parameter, index) in [(9, 0), (9, 1), (19, 0), (19, 1), (18, 0), (10, 0)] {
        let mut losses = Vec::new();
        for sign in [-1., 1.] {
            let mut bumped = before.clone();
            let original = bumped["policy"]["parameters"][parameter]["values"][index]
                .as_f64()
                .unwrap();
            bumped["policy"]["parameters"][parameter]["values"][index] =
                json!(original + sign * 1e-6);
            let mut probe = Worker::new();
            probe.ok(json!({"op":"restore","checkpoint":bumped}));
            probe.ok(upload("train", "train"));
            losses.push(
                probe.ok(json!({"op":"evaluate","batch_id":"train"}))["loss"]
                    .as_f64()
                    .unwrap(),
            );
        }
        let finite = (losses[1] - losses[0]) / 2e-6;
        let autodiff = after["policy"]["parameters"][parameter]["first_moment"][index]
            .as_f64()
            .unwrap()
            / 0.1;
        if autodiff.abs() > 1e-8 {
            nonzero += 1;
        }
        assert!(
            (finite - autodiff).abs() < 1e-6 * (1. + finite.abs()),
            "parameter {parameter}/{index}: {finite} versus {autodiff}"
        );
    }
    assert!(nonzero >= 4);
}

#[test]
fn admission_and_protocol_rejections_preserve_live_state() {
    let mut worker = Worker::new();
    assert_eq!(
        worker.request(json!({"op":"configure","config":{"policy_engine":"unknown"}}))["ok"],
        false
    );
    worker.ok(json!({"op":"configure","config":{}}));
    worker.register_fixture();
    for i in 0..4 {
        worker.ok(upload("train", &format!("batch-{i}")));
    }
    assert_eq!(worker.request(upload("train", "fifth"))["ok"], false);
    assert_eq!(worker.ok(json!({"op":"status"}))["resident_batches"], 4);
    worker.ok(json!({"op":"drop_batch","batch_id":"batch-0"}));
    worker.ok(upload("train", "fifth"));
    let mut invalid = upload("train", "bad");
    invalid["batch"]["marks"][0] = json!(-1.0);
    assert_eq!(worker.request(invalid)["ok"], false);
    assert_eq!(worker.ok(json!({"op":"status"}))["version"], 0);
    let mut bounded = Worker::new();
    let mut config = worker.ok(json!({"op":"status"}))["config"].clone();
    config["policy"]["max_batch_rows"] = json!(4);
    config["budget"]["compute_bytes"] = json!(67108864);
    config["budget"]["host_working_bytes"] = json!(67108864);
    bounded.ok(json!({"op":"configure","config":config}));
    bounded.register_fixture();
    assert_eq!(bounded.request(upload("train", "over-budget"))["ok"], false);
    assert_eq!(bounded.ok(json!({"op":"status"}))["resident_batches"], 0);
}

#[test]
fn large_datasets_use_compact_immutable_split_ranges() {
    let mut worker = Worker::new();
    worker.ok(json!({"op":"configure","config":{}}));
    let registration = json!({"op":"register_dataset","dataset_id":"study-calibration-sha256","partitions":[
        {"split":"train","first_path":"0","paths":"400000"},
        {"split":"validation","first_path":"400000","paths":"100000"},
        {"split":"test","first_path":"500000","paths":"100000"}]});
    worker.ok(registration.clone());
    assert_eq!(worker.ok(registration.clone())["already_registered"], true);
    let status = worker.ok(json!({"op":"status"}));
    assert_eq!(status["dataset_count"], 1);
    assert!(status["provenance_bytes"].as_u64().unwrap() < 1024);
    let mut batch = upload("train", "last-train-batch");
    batch["batch"]["provenance"] =
        json!({"dataset_id":"study-calibration-sha256","first_path":"399996"});
    worker.ok(batch.clone());
    batch["batch_id"] = json!("crossing");
    batch["batch"]["provenance"]["first_path"] = json!(399998);
    assert_eq!(worker.request(batch.clone())["ok"], false);
    batch["batch"]["provenance"]["first_path"] = json!(400000);
    assert_eq!(worker.request(batch.clone())["ok"], false);
    let mut changed = registration;
    changed["partitions"][1]["split"] = json!("train");
    assert_eq!(worker.request(changed)["ok"], false);
    let checkpoint = worker.ok(json!({"op":"checkpoint"}));
    let mut resumed = Worker::new();
    resumed.ok(json!({"op":"restore","checkpoint":checkpoint}));
    assert_eq!(resumed.request(batch)["ok"], false);
    assert_eq!(
        resumed.ok(json!({"op":"status"}))["provenance_bytes"],
        status["provenance_bytes"]
    );
    assert_eq!(resumed.request(json!({"op":"register_dataset","dataset_id":"overflow","partitions":[{"split":"train","first_path":"18446744073709551615","paths":"1"}]}))["ok"],false);
}
