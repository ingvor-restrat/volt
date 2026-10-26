#![cfg(feature = "live-risk")]
use std::{
    io::Write,
    process::{Command, Stdio},
};
use volt::{ExerciseSchedule, TimestampNs};
use volt_compute::{
    OptionSide,
    exercise_pricing::{ExerciseGrid, ExerciseOption, price_and_greeks},
    exercise_publication::*,
};
const NOW: i64 = 1_767_225_600_000_000_000;
fn fixture() -> PublicationInput {
    let option = ExerciseOption {
        row_id: "r".into(),
        contract_id: "ABC-C".into(),
        side: OptionSide::Call,
        spot: 100.,
        strike: 100.,
        valuation_time: TimestampNs(NOW),
        exercise: ExerciseSchedule::American {
            first_exercise: TimestampNs(NOW),
            expiry: TimestampNs(NOW + 31_536_000_000_000_000),
        },
        volatility: 0.2,
        rate: 0.05,
        dividend_yield: 0.,
        cash_dividends: vec![],
    };
    let risk = price_and_greeks(
        &option,
        &ExerciseGrid {
            space_steps: 200,
            time_steps: 200,
            ..Default::default()
        },
    )
    .unwrap();
    PublicationInput {
        schema: "volt.options_risk_snapshot.v1".into(),
        generation: 1,
        as_of_ns: NOW.to_string(),
        currency: "USD".into(),
        source: "synthetic".into(),
        rows: vec![PublicationRow {
            contract_id: "ABC-C".into(),
            underlying: "ABC".into(),
            spot: 100.,
            quantity: -2,
            multiplier: 100,
            position_units: -200.,
            quote_ns: (NOW - 2_000_000_000).to_string(),
            spot_ns: NOW.to_string(),
            calibration_ns: NOW.to_string(),
            calibration_spot_ns: Some((NOW - 3_000_000_000).to_string()),
            risk_asof_ns: NOW.to_string(),
            risk: Some(risk),
        }],
    }
}
fn as_json(x: &PublicationInput) -> serde_json::Value {
    // PublicationInput intentionally consumes worker JSON; tests use its actual wire fields.
    let r = &x.rows[0];
    serde_json::json!({"schema":x.schema,"generation":x.generation,"as_of_ns":x.as_of_ns,"currency":x.currency,"source":x.source,
        "rows":[{"contract_id":r.contract_id,"underlying":r.underlying,"spot":r.spot,"quantity":r.quantity,"multiplier":r.multiplier,
        "position_units":r.position_units,"quote_ns":r.quote_ns,"spot_ns":r.spot_ns,"calibration_ns":r.calibration_ns,"calibration_spot_ns":r.calibration_spot_ns,"risk_asof_ns":r.risk_asof_ns,"risk":r.risk}]})
}
#[test]
fn native_cube_signed_units_and_oldest_lineage() {
    let input = fixture();
    let g = input.rows[0].risk.as_ref().unwrap();
    let out = ExercisePublisher::new().publish(&input).unwrap();
    assert_eq!(
        out.cube.value("value", "ABC-C"),
        Some(-200. * g.valuation.price)
    );
    assert_eq!(
        out.cube.value("delta_dollars", "ABC-C"),
        Some(-200. * g.delta * 100.)
    );
    assert_eq!(out.cube.value("gamma_1pct", "ABC-C"), Some(-100. * g.gamma));
    assert_eq!(
        out.portfolio["vega_per_vol_point"],
        Some(-200. * g.vega_per_vol_point)
    );
    assert_eq!(
        out.cube.values[0].observed_at_ms,
        (NOW - 3_000_000_000) / 1_000_000
    );
}
#[test]
fn retained_underlying_has_explicit_age_and_missing_theta() {
    let mut input = fixture();
    input.as_of_ns = (NOW + 3_000_000_000).to_string();
    input.rows[0].risk.as_mut().unwrap().theta_per_day = None;
    let out = ExercisePublisher::new().publish(&input).unwrap();
    assert_eq!(out.underlyings["ABC"].risk_age_at_publication_seconds, 3.);
    assert_eq!(out.portfolio["theta_per_day"], None);
    assert_eq!(out.cube.value("theta_per_day", "ABC-C"), None);
    assert!(out.portfolio["value"].is_some());
    assert_eq!(
        out.cube
            .statuses
            .iter()
            .find(|s| s.node == "theta_per_day")
            .unwrap()
            .missing,
        1
    );
}
#[test]
fn unavailable_held_risk_nulls_totals_and_zero_position_does_not() {
    let mut input = fixture();
    input.rows[0].risk = None;
    let mut engine = ExercisePublisher::new();
    let out = engine.publish(&input).unwrap();
    assert!(out.portfolio.values().all(Option::is_none));
    assert_eq!(out.unpriced_positions, ["ABC-C"]);
    input.generation = 2;
    input.rows[0].position_units = 0.;
    input.rows[0].quantity = 0;
    let out = engine.publish(&input).unwrap();
    assert!(out.portfolio.values().all(|v| *v == Some(0.)));
}
#[test]
fn rejected_generation_units_identity_and_clock_leave_state_retriable() {
    let good = fixture();
    let mut engine = ExercisePublisher::new();
    engine.publish(&good).unwrap();
    assert!(engine.publish(&good).is_err());
    let mut next = good.clone();
    next.generation = 2;
    next.rows[0].position_units = -20_000.;
    assert!(engine.publish(&next).is_err());
    next = good.clone();
    next.generation = 2;
    next.rows[0].risk.as_mut().unwrap().valuation.contract_id = "other".into();
    assert!(engine.publish(&next).is_err());
    next = good.clone();
    next.generation = 2;
    next.rows[0].spot_ns = (NOW + 1).to_string();
    assert!(engine.publish(&next).is_err());
    next = good.clone();
    next.generation = 2;
    next.currency = "EUR".into();
    assert!(engine.publish(&next).is_err());
    next = good.clone();
    next.generation = 2;
    next.rows.push(next.rows[0].clone());
    assert!(engine.publish(&next).is_err());
    next = good;
    next.generation = 2;
    assert_eq!(engine.publish(&next).unwrap().cube.generation, 2);
}
#[test]
fn persistent_jsonl_process_publishes_native_snapshot_and_recovers_after_bad_line() {
    let mut input = as_json(&fixture());
    let mut process = Command::new(env!("CARGO_BIN_EXE_exercise_cube"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut pipe = process.stdin.take().unwrap();
    writeln!(pipe, "{input}").unwrap();
    writeln!(pipe, "{{\"bad\":true}}").unwrap();
    input["generation"] = 2.into();
    writeln!(pipe, "{input}").unwrap();
    drop(pipe);
    let output = process.wait_with_output().unwrap();
    assert!(output.status.success());
    let lines: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0]["ok"], true);
    assert_eq!(lines[1]["ok"], false);
    assert_eq!(lines[2]["publication"]["cube"]["generation"], 2);
    assert_eq!(lines[0]["publication"]["cube"]["entity_count"], 1);
}

#[test]
fn unknown_calibration_stock_lineage_is_explicit_and_conservative() {
    let mut input = fixture();
    input.rows[0].calibration_spot_ns = None;
    let out = ExercisePublisher::new().publish(&input).unwrap();
    assert!(!out.underlyings["ABC"].calibration_stock_lineage_complete);
    assert_eq!(out.cube.values[0].observed_at_ms, 0);
}
