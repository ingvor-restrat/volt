//! Actual holder-to-writer CLI integration, provenance and atomic publication.
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
use volt_compute::{
    maxcall_learning::{LearningRateStage, MaxCallTrainingConfig, train},
    multi_asset_market::MaxCallEpisodes,
};
#[test]
fn selected_holder_cli_binds_checkpoint_and_preserves_existing_episode_file() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("volt-maxcall-cli-{}-{stamp}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let mut cfg = MaxCallTrainingConfig::default();
    cfg.market.times = vec![0., 0.5, 1.];
    cfg.market.exercise = vec![false, true, true];
    cfg.precision = volt_compute::Precision::F64;
    cfg.hidden_width = 4;
    cfg.training_paths = 32;
    cfg.validation_paths = 16;
    cfg.batch_rows = 8;
    cfg.validation_every = 1;
    cfg.final_date_schedule = vec![LearningRateStage {
        updates: 2,
        learning_rate: 0.001,
    }];
    cfg.earlier_date_schedule = cfg.final_date_schedule.clone();
    let fitted = train(&cfg).unwrap();
    let checkpoint = dir.join("holder.json");
    let bytes = serde_json::to_vec(&fitted.final_checkpoint).unwrap();
    fs::write(&checkpoint, &bytes).unwrap();
    let hash = format!("{:x}", Sha256::digest(&bytes));
    let request = dir.join("request.json");
    let output = dir.join("episodes.json");
    let payload = json!({"paths":8,"seed":3191,"namespace":71,"first_path":99,
        "target":"cpu","precision":"f64","inference_batch_rows":8,"inference_working_bytes":67108864,
        "economics":{"quantity":-2,"multiplier":100,"premium":-5200.,"trading_costs":vec![0.001;5],
        "liquidation_costs":vec![0.001;5],"holder_id":"test-selected-model","holder_sha256":hash,"max_working_bytes":67108864}});
    fs::write(&request, serde_json::to_vec(&payload).unwrap()).unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_maxcall_episodes"))
            .args([
                "--checkpoint",
                checkpoint.to_str().unwrap(),
                "--request",
                request.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap()
    };
    let first = run();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let original = fs::read(&output).unwrap();
    let episode: MaxCallEpisodes = serde_json::from_slice(&original).unwrap();
    assert_eq!(episode.batch.paths, 8);
    assert_eq!(episode.batch.marks.len(), 8 * 3 * 5);
    assert_eq!(episode.premium, -5200.);
    assert_eq!(episode.holder_sha256, hash);
    assert!(
        episode
            .batch
            .exercise
            .as_ref()
            .unwrap()
            .settlement_amounts
            .iter()
            .all(|v| *v <= 0.)
    );
    assert_eq!(episode.cash_growth, vec![1.; 2]);
    assert_eq!(episode.income_growth, vec![1.; 10]);
    assert!(!run().status.success());
    assert_eq!(fs::read(&output).unwrap(), original);
    fs::remove_file(&output).unwrap();
    let mut corrupt = payload;
    corrupt["economics"]["holder_sha256"] = json!("0".repeat(64));
    fs::write(&request, serde_json::to_vec(&corrupt).unwrap()).unwrap();
    let rejected = run();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("SHA-256"));
    assert!(!output.exists());
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn generated_training_request_uses_cpu_f64_reference() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let output = std::env::temp_dir().join(format!(
        "volt-maxcall-default-{}-{stamp}.json",
        std::process::id()
    ));
    let status = Command::new(env!("CARGO_BIN_EXE_maxcall_learning"))
        .args(["example", "--output"])
        .arg(&output)
        .status()
        .unwrap();
    assert!(status.success());
    let request: serde_json::Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(request["training"]["target"], "cpu");
    assert_eq!(request["training"]["precision"], "f64");
    let restored: MaxCallTrainingConfig =
        serde_json::from_value(request["training"].clone()).unwrap();
    assert!(!restored.normalization.canonical_zero_translations);
    fs::remove_file(output).unwrap();
}
