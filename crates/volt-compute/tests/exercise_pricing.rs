use std::io::Write;
use std::process::{Command, Stdio};
use volt::{ExerciseSchedule, TimestampNs};
use volt_compute::aad::analytic_greeks;
use volt_compute::exercise_pricing::*;
use volt_compute::{OptionRow, OptionSide, reference_price};
const DAY: i64 = 86_400_000_000_000;
const START: i64 = 1_767_225_600_000_000_000;
fn date(d: i64) -> TimestampNs {
    TimestampNs(START + d * DAY)
}
fn option(side: OptionSide, kind: &str) -> ExerciseOption {
    ExerciseOption {
        row_id: "row".into(),
        contract_id: "contract".into(),
        side,
        spot: 100.,
        strike: 100.,
        valuation_time: date(0),
        exercise: match kind {
            "european" => ExerciseSchedule::European { expiry: date(365) },
            "bermudan" => ExerciseSchedule::Bermudan {
                opportunities: [91, 182, 273, 365].into_iter().map(date).collect(),
            },
            "delayed" => ExerciseSchedule::American {
                first_exercise: date(90),
                expiry: date(365),
            },
            _ => ExerciseSchedule::American {
                first_exercise: date(0),
                expiry: date(365),
            },
        },
        volatility: 0.2,
        rate: 0.05,
        dividend_yield: 0.,
        cash_dividends: vec![],
    }
}
fn grid(n: usize) -> ExerciseGrid {
    ExerciseGrid {
        space_steps: n,
        time_steps: n,
        ..Default::default()
    }
}
fn close(a: f64, b: f64, t: f64) {
    assert!(
        (a - b).abs() < t,
        "{a} vs {b}, error {} >= {t}",
        (a - b).abs()
    );
}
fn euro_row(o: &ExerciseOption) -> OptionRow {
    OptionRow {
        row_id: o.row_id.clone(),
        contract_id: o.contract_id.clone(),
        side: o.side,
        spot: o.spot,
        strike: o.strike,
        maturity_years: (o.exercise.expiry().0 - o.valuation_time.0) as f64 / (365. * DAY as f64),
        volatility: o.volatility,
        rate: o.rate,
        dividend_yield: o.dividend_yield,
        quantity: 1,
        multiplier: 1,
    }
}

#[test]
fn european_limit_price_all_greek_units_and_grid_refinement() {
    for side in [OptionSide::Call, OptionSide::Put] {
        let mut o = option(side, "european");
        o.dividend_yield = 0.02;
        let row = euro_row(&o);
        let exact = reference_price(&row).unwrap();
        let a = analytic_greeks(&row).unwrap();
        let coarse = price(&o, &grid(200)).unwrap();
        let fine = price_and_greeks(&o, &grid(800)).unwrap();
        close(fine.valuation.price, exact, 0.0003);
        assert!((fine.valuation.price - exact).abs() < (coarse.price - exact).abs());
        close(fine.delta, a.delta, 0.00003);
        close(
            fine.gamma,
            (-0.02f64).exp() * libm::exp(-0.5 * ((0.05f64 - 0.02 + 0.5 * 0.04) / 0.2).powi(2))
                / (std::f64::consts::TAU.sqrt() * 100. * 0.2),
            0.000004,
        );
        close(fine.vega_per_unit_volatility, a.vega, 0.004);
        close(fine.rho_per_unit_rate, a.rho, 0.004);
        close(fine.theta_per_year.unwrap(), a.theta, 0.004);
        close(
            fine.vega_per_vol_point,
            fine.vega_per_unit_volatility * 0.01,
            1e-14,
        );
        close(fine.rho_per_bp, fine.rho_per_unit_rate * 0.0001, 1e-14);
        close(
            fine.theta_per_day.unwrap() * 365.,
            fine.theta_per_year.unwrap(),
            1e-13,
        );
    }
}
// Independent recombining Cox–Ross–Rubinstein tree. Exact day schedules use a
// 3650-step year (10 tree steps/day), separate from the PDE discretization.
fn crr(o: &ExerciseOption, n: usize) -> f64 {
    assert!(o.cash_dividends.is_empty());
    let t = (o.exercise.expiry().0 - o.valuation_time.0) as f64 / (365. * DAY as f64);
    let dt = t / n as f64;
    let u = (o.volatility * dt.sqrt()).exp();
    let d = 1. / u;
    let p = (((o.rate - o.dividend_yield) * dt).exp() - d) / (u - d);
    let disc = (-o.rate * dt).exp();
    let payoff = |s: f64| match o.side {
        OptionSide::Call => (s - o.strike).max(0.),
        OptionSide::Put => (o.strike - s).max(0.),
    };
    let mut v = (0..=n)
        .map(|j| payoff(o.spot * (o.volatility * dt.sqrt() * (2. * j as f64 - n as f64)).exp()))
        .collect::<Vec<_>>();
    for i in (0..n).rev() {
        let stamp = TimestampNs(
            o.valuation_time.0
                + ((o.exercise.expiry().0 - o.valuation_time.0) as i128 * i as i128 / n as i128)
                    as i64,
        );
        for j in 0..=i {
            v[j] = disc * ((1. - p) * v[j] + p * v[j + 1]);
            if o.exercise.permits_exercise(stamp) {
                let s = o.spot * (o.volatility * dt.sqrt() * (2. * j as f64 - i as f64)).exp();
                v[j] = v[j].max(payoff(s));
            }
        }
    }
    v[0]
}
#[test]
fn independent_tree_exercise_ordering_and_delayed_start() {
    let g = grid(1000);
    let e = option(OptionSide::Put, "european");
    let b = option(OptionSide::Put, "bermudan");
    let a = option(OptionSide::Put, "american");
    let ep = price(&e, &g).unwrap().price;
    let bp = price(&b, &g).unwrap().price;
    let ap = price(&a, &g).unwrap();
    assert!(ep < bp && bp < ap.price);
    for o in [&e, &b, &a] {
        close(price(o, &g).unwrap().price, crr(o, 3650), 0.0015);
    }
    assert!(ap.diagnostics.max_scaled_complementarity_residual <= g.complementarity_tolerance);
    assert!(ap.diagnostics.policy_iterations > 0);
    let mut now = a.clone();
    now.spot = 80.;
    let mut delayed = now.clone();
    delayed.exercise = ExerciseSchedule::American {
        first_exercise: date(180),
        expiry: date(365),
    };
    let v0 = price(&now, &g).unwrap().price;
    let vd = price(&delayed, &g).unwrap();
    assert!(v0 > vd.price + 0.3);
    assert!(!vd.exercise_permitted_now);
    close(vd.price, crr(&delayed, 3650), 0.003);
    let call = option(OptionSide::Call, "american");
    close(
        price(&call, &g).unwrap().price,
        reference_price(&euro_row(&call)).unwrap(),
        0.0003,
    );
}
#[test]
fn dates_inserted_exactly_and_single_opportunity_matches_european() {
    let e = option(OptionSide::Put, "european");
    let mut b = e.clone();
    let exact = TimestampNs(START + 123 * DAY + 123_456_789);
    b.exercise = ExerciseSchedule::Bermudan {
        opportunities: vec![exact, date(365)],
    };
    let p = price(&b, &grid(333)).unwrap();
    assert!(p.diagnostics.event_dates.contains(&exact));
    assert!(p.diagnostics.time_nodes > 334);
    b.exercise = ExerciseSchedule::Bermudan {
        opportunities: vec![date(365)],
    };
    close(
        price(&b, &grid(600)).unwrap().price,
        price(&e, &grid(600)).unwrap().price,
        1e-12,
    );
    b.exercise = ExerciseSchedule::American {
        first_exercise: date(365),
        expiry: date(365),
    };
    close(
        price(&b, &grid(600)).unwrap().price,
        price(&e, &grid(600)).unwrap().price,
        1e-12,
    );
}
// Independent QuantLib1.40 FdBlackScholesVanillaEngine Spot model,1600 time and
// space points,CN+2 damping steps,2026-01-01→2027-01-01,Actual365Fixed.
// Primary engine and spot-dividend source:
// https://github.com/lballabio/QuantLib/tree/v1.40/ql/pricingengines/vanilla
// Price/delta/gamma controls were independently repeated at800 points. Greek
// bump controls use full repricing with dSigma=.001 and dRate=.0001.
#[test]
fn quantlib_cash_dividend_calls_puts_and_refinement() {
    let cases = [
        (
            OptionSide::Call,
            110.,
            vec![(90, 8.)],
            13.020881352136062,
            0.7764103043098711,
            0.021417163491663407,
            27.332986144395832,
            32.19864432385755,
        ),
        (
            OptionSide::Put,
            90.,
            vec![(90, 3.), (270, 2.)],
            14.148780886286666,
            -0.7087655668135482,
            0.021909890704356796,
            29.335151660866643,
            -51.057330849690885,
        ),
    ];
    for (side, s, divs, p, d, g, v, r) in cases {
        let mut o = option(side, "american");
        o.spot = s;
        o.cash_dividends = divs
            .iter()
            .map(|(day, amount)| CashDividend {
                ex_date: date(*day),
                amount: *amount,
            })
            .collect();
        let coarse = price_and_greeks(&o, &grid(600)).unwrap();
        let fine = price_and_greeks(&o, &grid(1600)).unwrap();
        eprintln!(
            "cash {side:?} p={} d={} g={} v={} r={}",
            fine.valuation.price,
            fine.delta,
            fine.gamma,
            fine.vega_per_unit_volatility,
            fine.rho_per_unit_rate
        );
        close(fine.valuation.price, p, 0.0007);
        close(fine.delta, d, 0.00005);
        close(fine.gamma, g, 0.000008);
        close(fine.vega_per_unit_volatility, v, 0.04);
        close(fine.rho_per_unit_rate, r, 0.06);
        close(fine.valuation.price, coarse.valuation.price, 0.0012);
        let mut euro = o.clone();
        euro.exercise = ExerciseSchedule::European { expiry: date(365) };
        assert!(price(&euro, &grid(800)).unwrap().price < fine.valuation.price - 0.2);
        let mut domain = grid(1600);
        domain.s_max_multiple = 16.;
        close(
            price(&o, &domain).unwrap().price,
            fine.valuation.price,
            0.0006,
        );
    }
}
#[test]
fn coincident_bermudan_exercise_is_before_dividend_and_theta_event_is_explicit() {
    let mut o = option(OptionSide::Call, "bermudan");
    o.spot = 110.;
    o.cash_dividends = vec![CashDividend {
        ex_date: date(90),
        amount: 8.,
    }];
    o.exercise = ExerciseSchedule::Bermudan {
        opportunities: vec![date(90), date(365)],
    };
    let before = price(&o, &grid(800)).unwrap();
    let mut after = o.clone();
    after.exercise = ExerciseSchedule::Bermudan {
        opportunities: vec![TimestampNs(date(90).0 + 1_000_000_000), date(365)],
    };
    assert!(before.price > price(&after, &grid(800)).unwrap().price + 0.5);
    let mut at = option(OptionSide::Put, "bermudan");
    at.valuation_time = date(91);
    let risk = price_and_greeks(&at, &grid(600)).unwrap();
    assert!(risk.theta_per_year.is_none() && risk.theta_note.is_some());
    assert!(risk.delta.is_finite());
    let mut near = option(OptionSide::Put, "bermudan");
    near.valuation_time = TimestampNs(date(91).0 - 20_000);
    let risk = price_and_greeks(&near, &grid(300)).unwrap();
    assert!(risk.theta_bump_ns.is_none());
}
#[test]
fn iv_roundtrip_intrinsic_weakness_and_outside_bracket() {
    let o = option(OptionSide::Put, "american");
    let g = grid(400);
    let target = price(&o, &g).unwrap().price;
    let iv = implied_volatility(&o, target, &g, &ExerciseIvConfig::default()).unwrap();
    assert_eq!(iv.state, ExerciseIvState::Converged);
    close(iv.volatility.unwrap(), 0.2, 0.000002);
    let call = option(OptionSide::Call, "american");
    assert_eq!(
        implied_volatility(&call, 0., &g, &ExerciseIvConfig::default())
            .unwrap()
            .state,
        ExerciseIvState::OutsideBracket
    );
    let mut deep = o.clone();
    deep.spot = 50.;
    let v = implied_volatility(&deep, 50., &g, &ExerciseIvConfig::default()).unwrap();
    assert_eq!(v.state, ExerciseIvState::WeaklyIdentified);
    assert!(v.volatility.is_none());
    assert_eq!(
        implied_volatility(&deep, 49., &g, &ExerciseIvConfig::default())
            .unwrap()
            .state,
        ExerciseIvState::OutsideBracket
    );
    assert_eq!(
        implied_volatility(&o, 200., &g, &ExerciseIvConfig::default())
            .unwrap()
            .state,
        ExerciseIvState::OutsideBracket
    );
}
#[test]
fn invalid_inputs_solver_failure_and_serde_rejection() {
    let o = option(OptionSide::Put, "american");
    let mut bad = o.clone();
    bad.valuation_time = TimestampNs(i64::MAX);
    assert!(price(&bad, &grid(200)).is_err());
    bad = o.clone();
    bad.volatility = f64::NAN;
    assert!(price(&bad, &grid(200)).is_err());
    bad = o.clone();
    bad.cash_dividends = vec![CashDividend {
        ex_date: date(0),
        amount: 2.,
    }];
    assert!(price(&bad, &grid(200)).is_err());
    bad = o.clone();
    bad.exercise = ExerciseSchedule::Bermudan {
        opportunities: vec![date(5), date(5)],
    };
    assert!(price(&bad, &grid(200)).is_err());
    let mut g = grid(200);
    g.space_steps = usize::MAX;
    assert!(price(&o, &g).is_err());
    g = grid(128);
    g.time_steps = 8;
    g.max_policy_iterations = 2;
    let mut stressed = o.clone();
    stressed.rate = 0.1;
    let refusal = price(&stressed, &g).unwrap_err();
    assert!(
        refusal
            .to_string()
            .contains("complementarity solver did not converge")
    );
    g.max_policy_iterations = 80;
    let recovered = price(&stressed, &g).unwrap();
    assert!(
        recovered.diagnostics.max_scaled_complementarity_residual <= g.complementarity_tolerance
    );
    let mut nested = serde_json::to_value(&o).unwrap();
    nested["exercise"]["ignored_style_typo"] = true.into();
    assert!(serde_json::from_value::<ExerciseOption>(nested).is_err());
    let mut value = serde_json::to_value(&o).unwrap();
    value["unsupported_curve"] = true.into();
    assert!(serde_json::from_value::<ExerciseOption>(value).is_err());
}
#[test]
fn batch_order_identity_isolation_and_cli_json_roundtrip() {
    let o = option(OptionSide::Put, "american");
    let mut b = o.clone();
    b.row_id = "other".into();
    b.contract_id = "other-contract".into();
    b.volatility = -1.;
    let req = ExerciseBatchRequest {
        schema: "volt.compute.exercise_pricing.v1".into(),
        grid: grid(200),
        iv: Default::default(),
        rows: vec![
            ExerciseBatchRow {
                option: o.clone(),
                market_price: None,
            },
            ExerciseBatchRow {
                option: b,
                market_price: None,
            },
        ],
    };
    let result = price_batch(&req).unwrap();
    assert!(result.rows[0].risk.is_some());
    assert!(result.rows[1].error.is_some());
    assert_eq!(result.rows[0].row_id, "row");
    let mut duplicate = req.clone();
    duplicate.rows[1].option = o.clone();
    let result = price_batch(&duplicate).unwrap();
    assert!(result.rows.iter().all(|r| r.error.is_some()));
    let mut conflict = req.clone();
    conflict.rows[1].option = o;
    conflict.rows[1].option.row_id = "distinct-row".into();
    conflict.rows[1].option.strike = 101.;
    assert!(
        price_batch(&conflict)
            .unwrap()
            .rows
            .iter()
            .all(|r| r.error.is_some())
    );
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_exercise_pricer"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    cmd.stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&req).unwrap())
        .unwrap();
    let output = cmd.wait_with_output().unwrap();
    assert!(output.status.success());
    let actual: ExerciseBatchResult = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(actual.execution_policy, result.execution_policy);
    assert!(actual.rows[0].risk.is_some());
}

#[test]
fn immediate_exercise_risk_and_unresolved_expiry_guard() {
    let mut deep = option(OptionSide::Put, "american");
    deep.spot = 50.;
    let r = price_and_greeks(&deep, &grid(400)).unwrap();
    assert!(r.valuation.at_exercise_value);
    close(r.valuation.price, 50., 1e-10);
    close(r.delta, -1., 1e-12);
    close(r.gamma, 0., 1e-12);
    close(r.vega_per_vol_point, 0., 1e-10);
    close(r.rho_per_bp, 0., 1e-10);
    close(r.theta_per_day.unwrap(), 0., 1e-8);
    let mut short = option(OptionSide::Call, "european");
    short.exercise = ExerciseSchedule::European {
        expiry: TimestampNs(START + 1_000_000_000),
    };
    assert!(price(&short, &grid(400)).is_ok());
    assert!(price_and_greeks(&short, &grid(400)).is_err());
    let req = ExerciseBatchRequest {
        schema: "volt.compute.exercise_pricing.v1".into(),
        grid: grid(200),
        iv: ExerciseIvConfig {
            max_iterations: usize::MAX,
            ..Default::default()
        },
        rows: vec![ExerciseBatchRow {
            option: deep,
            market_price: Some(50.),
        }],
    };
    assert!(price_batch(&req).is_err());
}

#[test]
fn calendar_theta_stable_at_aligned_dividend_time_grids() {
    let mut o = option(OptionSide::Call, "american");
    o.spot = 110.;
    o.cash_dividends = vec![CashDividend {
        ex_date: date(90),
        amount: 8.,
    }];
    let a = price_and_greeks(&o, &grid(365)).unwrap();
    let mut half = grid(365);
    half.theta_bump_ns = 30_000_000_000;
    let b = price_and_greeks(&o, &half).unwrap();
    let refined = price_and_greeks(&o, &grid(730)).unwrap();
    close(a.theta_per_year.unwrap(), b.theta_per_year.unwrap(), 0.003);
    close(
        a.theta_per_year.unwrap(),
        refined.theta_per_year.unwrap(),
        0.02,
    );
}

#[test]
fn central_volatility_and_rate_bump_refinement() {
    let mut o = option(OptionSide::Call, "european");
    o.dividend_yield = 0.02;
    let expected = analytic_greeks(&euro_row(&o)).unwrap();
    let coarse = ExerciseGrid {
        volatility_bump: 0.03,
        rate_bump: 0.01,
        ..grid(800)
    };
    let a = price_and_greeks(&o, &coarse).unwrap();
    let b = price_and_greeks(&o, &grid(800)).unwrap();
    assert!(
        (b.vega_per_unit_volatility - expected.vega).abs()
            < (a.vega_per_unit_volatility - expected.vega).abs()
    );
    assert!(
        (b.rho_per_unit_rate - expected.rho).abs() < (a.rho_per_unit_rate - expected.rho).abs()
    );
}

#[test]
fn volatility_tail_domain_enlarges_or_refuses_explicitly() {
    let mut o = option(OptionSide::Call, "european");
    o.volatility = 1.;
    let p = price(&o, &grid(800)).unwrap();
    assert!(p.diagnostics.s_max > 800.);
    close(p.price, reference_price(&euro_row(&o)).unwrap(), 0.001);
    o.volatility = 3.;
    assert!(matches!(
        price(&o, &grid(800)),
        Err(volt_compute::ComputeError::Unsupported(_))
    ));
}
