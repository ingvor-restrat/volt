use std::{
    io::Write,
    process::{Command, Stdio},
    sync::Arc,
};
use volt::{ExerciseSchedule, TimestampNs};
use volt_compute::{
    OptionSide, bermudan_market::*, exercise_pricing::*, joint_option_market::ScenarioRange,
    native_learner::Split,
};
const START: i64 = 1_767_225_600_000_000_000;
const DAY: i64 = 86_400_000_000_000;
fn day(d: i64) -> TimestampNs {
    TimestampNs(START + d * DAY)
}
fn config() -> BermudanMarketConfig {
    let dates = [0, 90, 180, 270, 365].map(day).to_vec();
    BermudanMarketConfig {
        option: ExerciseOption {
            row_id: "put".into(),
            contract_id: "PUT-100".into(),
            side: OptionSide::Put,
            spot: 100.,
            strike: 100.,
            valuation_time: day(0),
            exercise: ExerciseSchedule::Bermudan {
                opportunities: dates[1..].to_vec(),
            },
            volatility: 0.2,
            rate: 0.05,
            dividend_yield: 0.,
            cash_dividends: vec![],
        },
        grid: ExerciseGrid {
            space_steps: 600,
            time_steps: 600,
            ..Default::default()
        },
        dates,
        seed: 873711,
        physical_drift: 0.05,
        quantity: 1,
        multiplier: 1,
        trading_cost: 0.,
        liquidation_cost: 0.,
        max_working_bytes: 64 * 1024 * 1024,
        max_holder_scalar_values: 100_000,
    }
}
fn range(first_path: u64, paths: usize) -> ScenarioRange {
    ScenarioRange {
        split: Split::Test,
        first_path,
        paths,
    }
}
fn cache() -> HolderGridCache {
    HolderGridCache::new(100_000).unwrap()
}
fn close(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() < t, "{a} vs{b};err{}", (a - b).abs());
}
fn crr_put(o: &ExerciseOption, n: usize) -> f64 {
    let dt = 1. / n as f64;
    let u = (o.volatility * dt.sqrt()).exp();
    let d = 1. / u;
    let p = ((o.rate * dt).exp() - d) / (u - d);
    let discount = (-o.rate * dt).exp();
    let mut v = (0..=n)
        .map(|j| (o.strike - o.spot * u.powi(j as i32) * d.powi((n - j) as i32)).max(0.))
        .collect::<Vec<_>>();
    for t in (0..n).rev() {
        let timestamp =
            TimestampNs(START + (365_i128 * i128::from(DAY) * t as i128 / n as i128) as i64);
        let ex = o.exercise.permits_exercise(timestamp);
        for j in 0..=t {
            v[j] = discount * (p * v[j + 1] + (1. - p) * v[j]);
            if ex {
                v[j] = v[j]
                    .max((o.strike - o.spot * u.powi(j as i32) * d.powi((t - j) as i32)).max(0.));
            }
        }
    }
    v[0]
}
#[test]
fn holder_price_refines_to_independent_crr_and_quantlib() {
    let mut c = config();
    let coarse = build_holder_grid(&c.option, &c.grid, &c.dates, 100_000).unwrap();
    c.grid.space_steps = 1200;
    c.grid.time_steps = 1200;
    let fine = build_holder_grid(&c.option, &c.grid, &c.dates, 100_000).unwrap();
    // QuantLib1.40 FdBlackScholesVanillaEngine Spot/CN/2damping,1600x1600.
    // Evaluation2026-01-01, Actual365Fixed, opportunities90/180/270/365days.
    close(fine.price0, 5.955818029587064, 0.001);
    close(fine.price0, crr_put(&c.option, 3650), 0.006);
    close(coarse.price0, fine.price0, 0.004);
    close(
        fine.evaluate(day(0), 100.).unwrap().delta,
        -0.4038654898620221,
        0.0002,
    );
}
#[test]
fn map_at_saved_dates_matches_fresh_rollback_and_exercises_zero_spot() {
    let c = config();
    let holder = build_holder_grid(&c.option, &c.grid, &c.dates, 100_000).unwrap();
    let mut later = c.option.clone();
    later.valuation_time = day(90);
    later.spot = 105.;
    close(
        holder.evaluate(day(90), 105.).unwrap().value,
        price(&later, &c.grid).unwrap().price,
        0.003,
    );
    let zero = holder.evaluate(day(90), 0.).unwrap();
    assert!(zero.exercise_now);
    assert_eq!(zero.immediate_payoff, 100.);
    assert!(zero.continuation_value < 100.);
    assert_eq!(zero.delta, -1.);
    assert!(holder.evaluate(day(89), 100.).is_err());
    assert!(holder.evaluate(day(90), 1e8).is_err());
}
#[test]
fn policy_cache_is_exact_input_bound_and_storage_admitted() {
    let c = config();
    let mut cache = cache();
    let a = cache.get_or_build(&c.option, &c.grid, &c.dates).unwrap();
    let b = cache.get_or_build(&c.option, &c.grid, &c.dates).unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    let mut changed = c.option.clone();
    changed.rate = 0.06;
    let d = cache.get_or_build(&changed, &c.grid, &c.dates).unwrap();
    assert_ne!(a.cache_key, d.cache_key);
    assert!(build_holder_grid(&c.option, &c.grid, &c.dates, 20).is_err());
    let mut omitted = c.dates.clone();
    omitted.remove(1);
    assert!(build_holder_grid(&c.option, &c.grid, &omitted, 100_000).is_err());
    assert!(a.scalar_values() <= 100_000);
}
#[test]
fn gbm_chunks_identical_and_financial_units_do_not_change_physical_paths() {
    let c = config();
    let mut cache = cache();
    let full = generate(&c, range(10, 12), &mut cache).unwrap();
    let first = generate(&c, range(10, 5), &mut cache).unwrap();
    let last = generate(&c, range(15, 7), &mut cache).unwrap();
    assert_eq!(
        [first.pre_dividend_spots, last.pre_dividend_spots].concat(),
        full.pre_dividend_spots
    );
    assert_eq!(
        [first.numerical_pnl, last.numerical_pnl].concat(),
        full.numerical_pnl
    );
    let mut doubled = c.clone();
    doubled.quantity = -2;
    doubled.multiplier = 10;
    let other = generate(&doubled, range(10, 12), &mut cache).unwrap();
    assert_eq!(full.physical_sha256, other.physical_sha256);
    assert_eq!(full.pre_dividend_spots, other.pre_dividend_spots);
    assert_eq!(
        full.batch.exercise.as_ref().unwrap().settlement_step,
        other.batch.exercise.as_ref().unwrap().settlement_step
    );
    for (a, b) in full.numerical_pnl.iter().zip(other.numerical_pnl) {
        close(-20. * a, b, 1e-10);
    }
    assert!(other.batch.payoffs.iter().all(|v| *v == 0.));
}
#[test]
fn early_put_stops_and_bank_finances_after_settlement() {
    let c = config();
    let mut cache = cache();
    let spots = [100., 50., 40., 40., 40.];
    let out = generate_from_paths(&c, range(0, 1), &spots, &mut cache).unwrap();
    let e = out.batch.exercise.as_ref().unwrap();
    assert_eq!(e.settlement_step, [1]);
    assert_eq!(e.settlement_amounts, [50.]);
    assert!(out.numerical_holdings[1..].iter().all(|h| *h == 0.));
    let h = out.numerical_holdings[0];
    let start = out.economics.premium;
    let at_stop = (start - h * 100.) * out.economics.cash_growth[0] + h * 50. - 50.;
    let expected = at_stop * out.economics.cash_growth[1..].iter().product::<f64>();
    close(out.numerical_pnl[0], expected, 1e-12);
}
#[test]
fn immediate_exercise_opens_no_hedge_and_costs_do_not_charge() {
    let mut c = config();
    c.option.spot = 30.;
    c.option.exercise = ExerciseSchedule::Bermudan {
        opportunities: c.dates.clone(),
    };
    c.trading_cost = 0.5;
    c.liquidation_cost = 0.5;
    let out =
        generate_from_paths(&c, range(0, 1), &[30., 40., 50., 60., 70.], &mut cache()).unwrap();
    assert_eq!(out.batch.exercise.as_ref().unwrap().settlement_step, [0]);
    assert!(out.numerical_holdings.iter().all(|h| *h == 0.));
    close(out.numerical_pnl[0], 0., 1e-8);
}
#[test]
fn dividend_call_exercise_and_continuation_use_correct_marks_and_income() {
    let mut c = config();
    c.option.side = OptionSide::Call;
    c.option.spot = 110.;
    c.option.cash_dividends = vec![CashDividend {
        ex_date: day(90),
        amount: 8.,
    }];
    let holder = build_holder_grid(&c.option, &c.grid, &c.dates, 100_000).unwrap();
    // Independent QuantLib1.40 Spot/CN1600 fixture: same four dated opportunities.
    close(holder.price0, 13.020881352130301, 0.002);
    assert!(holder.evaluate(day(90), 150.).unwrap().exercise_now);
    assert!(!holder.evaluate(day(90), 90.).unwrap().exercise_now);
    let out = generate_from_paths(
        &c,
        range(0, 2),
        &[110., 150., 160., 170., 180., 110., 90., 110., 120., 130.],
        &mut cache(),
    )
    .unwrap();
    assert_eq!(out.batch.exercise.as_ref().unwrap().settlement_step, [1, 4]);
    assert_eq!(out.batch.marks[1], 150.);
    assert_eq!(out.batch.cash_income.as_ref().unwrap()[0], 0.);
    assert_eq!(out.batch.marks[6], 82.);
    assert_eq!(out.batch.cash_income.as_ref().unwrap()[4], 8.);
}
#[test]
fn cash_dividend_put_price_agrees_with_independent_quantlib() {
    let mut c = config();
    c.option.spot = 90.;
    c.grid.space_steps = 1200;
    c.grid.time_steps = 1200;
    c.option.cash_dividends = vec![
        CashDividend {
            ex_date: day(90),
            amount: 3.,
        },
        CashDividend {
            ex_date: day(270),
            amount: 2.,
        },
    ];
    let h = build_holder_grid(&c.option, &c.grid, &c.dates, 100_000).unwrap();
    close(h.price0, 13.519042475279601, 0.003);
}
#[test]
fn costs_are_actual_rebalances_and_liquidation_and_limits_fail_before_generation() {
    let mut c = config();
    let path = [100., 50., 40., 40., 40.];
    let free = generate_from_paths(&c, range(0, 1), &path, &mut cache()).unwrap();
    c.trading_cost = 0.001;
    c.liquidation_cost = 0.002;
    let paid = generate_from_paths(&c, range(0, 1), &path, &mut cache()).unwrap();
    let h = free.numerical_holdings[0].abs();
    let loss = (h * 100. * 0.001 * free.economics.cash_growth[0] + h * 50. * 0.002)
        * free.economics.cash_growth[1..].iter().product::<f64>();
    close(free.numerical_pnl[0] - paid.numerical_pnl[0], loss, 1e-11);
    c.max_working_bytes = 1;
    assert!(generate(&c, range(0, 1), &mut cache()).is_err());
    c.max_working_bytes = 64 * 1024 * 1024;
    assert!(generate(&c, range(u64::MAX, 2), &mut cache()).is_err());
    let mut invalid = c.clone();
    invalid.option.dividend_yield = 0.01;
    assert!(generate(&invalid, range(0, 1), &mut cache()).is_err());
}
#[test]
fn cli_runs_actual_producer_and_emits_stopped_episode() {
    let c = config();
    let request = serde_json::json!({"config":c,"range":range(0,2)});
    let mut child = Command::new(env!("CARGO_BIN_EXE_bermudan_market"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.to_string().as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let result: BermudanScenarios = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result.batch.paths, 2);
    assert!(result.batch.exercise.is_some());
    assert!(result.batch.payoffs.iter().all(|v| *v == 0.));
}

#[test]
fn risk_neutral_discounted_settlement_agrees_with_pde_with_declared_sampling_error() {
    let c = config();
    let result = generate(&c, range(10_000, 8192), &mut cache()).unwrap();
    let e = result.batch.exercise.as_ref().unwrap();
    let observations = e
        .settlement_step
        .iter()
        .zip(&e.settlement_amounts)
        .map(|(t, a)| {
            a * (-c.option.rate * (c.dates[*t].0 - c.dates[0].0) as f64 / (365. * DAY as f64)).exp()
        })
        .collect::<Vec<_>>();
    let n = observations.len() as f64;
    let mean = observations.iter().sum::<f64>() / n;
    let se = (observations.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.) / n).sqrt();
    // Predeclared fixed-seed four-SE control, plus the PDE refinement allowance.
    assert!(
        (mean - result.economics.premium).abs() < 4. * se + 0.004,
        "mean{mean},PDE{},SE{se}",
        result.economics.premium
    );
}

fn chosen(steps: Vec<usize>) -> SuppliedStopping {
    SuppliedStopping {
        settlement_steps: steps,
        provenance: HolderPolicyProvenance {
            policy_id: "chosen-immutable-holder".into(),
            policy_sha256: "a".repeat(64),
        },
        premium_override: None,
    }
}
#[test]
fn supplied_numerical_stops_reproduce_all_cash_dividend_financial_arrays() {
    let mut c = config();
    c.option.side = OptionSide::Call;
    c.option.spot = 110.;
    c.option.cash_dividends = vec![CashDividend {
        ex_date: day(90),
        amount: 8.,
    }];
    c.quantity = -2;
    c.multiplier = 10;
    let spots = [110., 150., 160., 170., 180., 110., 90., 110., 120., 130.];
    let normal = generate_from_paths(&c, range(0, 2), &spots, &mut cache()).unwrap();
    let stops = chosen(
        normal
            .batch
            .exercise
            .as_ref()
            .unwrap()
            .settlement_step
            .clone(),
    );
    let supplied =
        generate_from_paths_with_stopping(&c, range(0, 2), &spots, &stops, &mut cache()).unwrap();
    assert_eq!(normal.batch.marks, supplied.batch.marks);
    assert_eq!(normal.batch.features, supplied.batch.features);
    assert_eq!(normal.batch.cash_income, supplied.batch.cash_income);
    assert_eq!(normal.batch.payoffs, supplied.batch.payoffs);
    assert_eq!(
        normal.batch.exercise.as_ref().unwrap().settlement_amounts,
        supplied.batch.exercise.as_ref().unwrap().settlement_amounts
    );
    assert_eq!(normal.numerical_holdings, supplied.numerical_holdings);
    assert_eq!(normal.numerical_pnl, supplied.numerical_pnl);
    assert_eq!(normal.physical_sha256, supplied.physical_sha256);
    assert_ne!(normal.content_sha256, supplied.content_sha256);
    assert_eq!(supplied.holder_policy.policy_sha256, "a".repeat(64));
    assert_eq!(
        supplied.economics.premium_basis,
        "numerical_pde_position_value"
    );
}
#[test]
fn supplied_late_holder_rebuilds_dividend_cashflows_and_explicit_premium_once() {
    let mut c = config();
    c.option.side = OptionSide::Call;
    c.option.spot = 110.;
    c.option.cash_dividends = vec![CashDividend {
        ex_date: day(90),
        amount: 8.,
    }];
    c.quantity = 2;
    c.multiplier = 100;
    let spots = [110., 150., 160., 170., 180.];
    let mut stops = chosen(vec![4]);
    let base =
        generate_from_paths_with_stopping(&c, range(0, 1), &spots, &stops, &mut cache()).unwrap();
    assert_eq!(base.batch.marks[1], 142.);
    assert_eq!(base.batch.cash_income.as_ref().unwrap()[0], 8.);
    assert_eq!(
        base.batch.exercise.as_ref().unwrap().settlement_amounts,
        [16_000.]
    );
    stops.premium_override = Some(base.economics.premium + 7.);
    let funded =
        generate_from_paths_with_stopping(&c, range(0, 1), &spots, &stops, &mut cache()).unwrap();
    assert_eq!(funded.economics.premium, base.economics.premium + 7.);
    assert_eq!(
        funded.economics.premium_basis,
        "caller_supplied_signed_cash"
    );
    assert_eq!(
        funded.economics.numerical_reference_premium,
        base.economics.premium
    );
    close(
        funded.numerical_pnl[0] - base.numerical_pnl[0],
        7. * 0.05_f64.exp(),
        1e-10,
    );
    assert_ne!(funded.content_sha256, base.content_sha256);
}
#[test]
fn malformed_supplied_stops_and_provenance_are_rejected() {
    let mut c = config();
    c.option.exercise = ExerciseSchedule::Bermudan {
        opportunities: vec![day(180), day(365)],
    };
    let spots = [100., 80., 70., 60., 50.];
    for stops in [
        chosen(vec![]),
        chosen(vec![0]),
        chosen(vec![1]),
        chosen(vec![5]),
    ] {
        assert!(
            generate_from_paths_with_stopping(&c, range(0, 1), &spots, &stops, &mut cache())
                .is_err()
        );
    }
    let mut bad = chosen(vec![2]);
    bad.provenance.policy_sha256 = "missing".into();
    assert!(
        generate_from_paths_with_stopping(&c, range(0, 1), &spots, &bad, &mut cache()).is_err()
    );
    bad = chosen(vec![2]);
    bad.premium_override = Some(f64::NAN);
    assert!(
        generate_from_paths_with_stopping(&c, range(0, 1), &spots, &bad, &mut cache()).is_err()
    );
}
#[test]
fn supplied_stop_prefix_is_independent_of_unobserved_tail_and_step_zero_is_validated() {
    let mut c = config();
    let stops = chosen(vec![1]);
    let first = generate_from_paths_with_stopping(
        &c,
        range(0, 1),
        &[100., 50., 40., 40., 40.],
        &stops,
        &mut cache(),
    )
    .unwrap();
    // Complete finite market input remains required, but post-stop states are
    // outside the PDE domain without ever entering holder/delta interpolation.
    let second = generate_from_paths_with_stopping(
        &c,
        range(0, 1),
        &[100., 50., 1e8, 2e8, 3e8],
        &stops,
        &mut cache(),
    )
    .unwrap();
    assert_eq!(
        first.batch.exercise.as_ref().unwrap().settlement_amounts,
        second.batch.exercise.as_ref().unwrap().settlement_amounts
    );
    assert_eq!(first.numerical_pnl, second.numerical_pnl);
    assert_eq!(first.numerical_holdings, second.numerical_holdings);
    c.option.exercise = ExerciseSchedule::Bermudan {
        opportunities: c.dates.clone(),
    };
    let immediate = generate_from_paths_with_stopping(
        &c,
        range(0, 1),
        &[100., 110., 120., 130., 140.],
        &chosen(vec![0]),
        &mut cache(),
    )
    .unwrap();
    assert!(immediate.numerical_holdings.iter().all(|h| *h == 0.));
    close(
        immediate.numerical_pnl[0],
        immediate.economics.premium * 0.05_f64.exp(),
        1e-12,
    );
}

#[test]
fn cli_supplied_holder_dispatch_uses_explicit_cohort_and_capital() {
    let c = config();
    let request = serde_json::json!({"config":c,"range":range(0,1),"pre_dividend_spots":[100.,90.,110.,100.,80.],
        "stopping":SuppliedStopping{premium_override:Some(3.2),..chosen(vec![2])}});
    let mut child = Command::new(env!("CARGO_BIN_EXE_bermudan_market"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.to_string().as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let out: BermudanScenarios = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(out.batch.exercise.as_ref().unwrap().settlement_step, [2]);
    assert_eq!(
        out.batch.exercise.as_ref().unwrap().settlement_amounts,
        [0.]
    );
    assert_eq!(out.economics.premium, 3.2);
    assert_eq!(out.holder_policy.policy_id, "chosen-immutable-holder");
}
