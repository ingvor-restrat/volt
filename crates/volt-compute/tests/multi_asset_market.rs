use volt_compute::{ComputeError, OptionRow, OptionSide, multi_asset_market::*, reference_price};
fn one() -> MaxCallMarket {
    MaxCallMarket {
        spots: vec![100.],
        volatilities: vec![0.2],
        dividend_yields: vec![0.1],
        correlation: vec![vec![1.]],
        rate: 0.05,
        strike: 100.,
        times: vec![0., 1.],
        exercise: vec![false, true],
    }
}
fn episode(d: usize) -> MaxCallEpisodeConfig {
    MaxCallEpisodeConfig {
        quantity: 1,
        multiplier: 1,
        premium: 10.,
        trading_costs: vec![0.; d],
        liquidation_costs: vec![0.; d],
        holder_id: "fixed-policy".into(),
        holder_sha256: "a".repeat(64),
        max_working_bytes: 64 * 1024 * 1024,
    }
}
fn price(m: &MaxCallMarket) -> f64 {
    reference_price(&OptionRow {
        row_id: "r".into(),
        contract_id: "c".into(),
        side: OptionSide::Call,
        spot: m.spots[0],
        strike: m.strike,
        maturity_years: *m.times.last().unwrap(),
        volatility: m.volatilities[0],
        rate: m.rate,
        dividend_yield: m.dividend_yields[0],
        quantity: 1,
        multiplier: 1,
    })
    .unwrap()
}
fn close(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() < t, "{a} vs{b} error{}", (a - b).abs());
}
#[test]
fn semidefinite_perfect_correlation_and_invalid_inputs() {
    let f = factor_correlation(&[vec![1., 1.], vec![1., 1.]]).unwrap();
    assert_eq!(f.rank, 1);
    assert_eq!(f.max_reconstruction_error, 0.);
    assert!(factor_correlation(&[vec![1., -1.], vec![-1., 1.]]).is_ok());
    for x in [
        vec![vec![1., 0.5], vec![0.4, 1.]],
        vec![vec![1., 0.9, 0.9], vec![0.9, 1., -0.9], vec![0.9, -0.9, 1.]],
        vec![vec![0.9]],
    ] {
        assert!(factor_correlation(&x).is_err());
    }
    let mut m = one();
    m.spots = vec![];
    assert!(m.validate().is_err());
    m = one();
    m.times[1] = 0.;
    assert!(m.validate().is_err());
}
#[test]
fn exact_chunk_identity_and_conditional_domain_separation() {
    let m = MaxCallMarket::paper_five();
    let a = generate_paths(&m, 20, 731, 100, 8, 1 << 20).unwrap();
    let b = generate_paths(&m, 7, 731, 100, 8, 1 << 20).unwrap();
    let c = generate_paths(&m, 13, 731, 107, 8, 1 << 20).unwrap();
    assert_eq!(a.values, [b.values, c.values].concat());
    let key = RandomKey {
        seed: 731,
        namespace: 8,
        path_id: 100,
        branch: 0,
    };
    let tail = simulate_tail(&m, 0, &m.spots, key, 1 << 20).unwrap();
    assert_ne!(tail, &a.values[..m.times.len() * m.assets()]);
    assert_eq!(tail, simulate_tail(&m, 0, &m.spots, key, 1 << 20).unwrap());
    assert_ne!(
        tail,
        simulate_tail(&m, 0, &m.spots, RandomKey { branch: 1, ..key }, 1 << 20).unwrap()
    );
    assert!(generate_paths(&m, 20, 0, u64::MAX, 0, 1 << 20).is_err());
    assert!(generate_paths(&m, 20, 0, 0, 0, 1).is_err());
    // Native learner's host allowance can exceed this tiny cohort's actual size.
    assert!(generate_paths(&m, 1, 0, 0, 0, 8_589_934_592).is_ok());
}
#[test]
fn correlated_log_moments_and_discounted_stock_martingales() {
    let mut m = one();
    m.spots = vec![100., 80.];
    m.volatilities = vec![0.2, 0.3];
    m.dividend_yields = vec![0.1, 0.02];
    m.correlation = vec![vec![1., 0.6], vec![0.6, 1.]];
    let p = generate_paths(&m, 65536, 71319, 0, 31, 64 << 20).unwrap();
    let mut cross = 0.;
    let mut variances = [0.; 2];
    let mut means = [0.; 2];
    for j in 0..p.paths {
        let s = p.state(j, 1).unwrap();
        let mut z = [0.; 2];
        for i in 0..2 {
            z[i] = ((s[i] / m.spots[i]).ln()
                - (m.rate - m.dividend_yields[i] - 0.5 * m.volatilities[i].powi(2)))
                / m.volatilities[i];
            variances[i] += z[i] * z[i];
            means[i] += s[i] * ((m.dividend_yields[i] - m.rate) * m.times[1]).exp();
        }
        cross += z[0] * z[1];
    }
    close(cross / p.paths as f64, 0.6, 0.025);
    for i in 0..2 {
        close(variances[i] / p.paths as f64, 1., 0.025);
        let se =
            m.spots[i] * (m.volatilities[i].powi(2).exp() - 1.).sqrt() / (p.paths as f64).sqrt();
        close(means[i] / p.paths as f64, m.spots[i], 4. * se);
    }
}
#[test]
fn one_asset_european_limit_and_maxcall_elementary_bounds() {
    let m = one();
    let p = generate_paths(&m, 65536, 57319, 0, 91, 64 << 20).unwrap();
    let rewards = (0..p.paths)
        .map(|j| m.payoff(1, p.state(j, 1).unwrap()).unwrap())
        .collect::<Vec<_>>();
    let mean = rewards.iter().sum::<f64>() / p.paths as f64;
    let se = (rewards.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
        / (p.paths - 1) as f64
        / p.paths as f64)
        .sqrt();
    close(mean, price(&m), 4. * se);
    let mut multi = MaxCallMarket::paper_five();
    multi.times = vec![0., 1.];
    multi.exercise = vec![false, true];
    let paths = generate_paths(&multi, 8192, 18, 0, 98, 64 << 20).unwrap();
    for j in 0..paths.paths {
        let s = paths.state(j, 1).unwrap();
        let payoff = multi.intrinsic(s).unwrap();
        let sum = s.iter().map(|x| (x - 100.).max(0.)).sum::<f64>();
        assert!(payoff <= sum + 1e-12);
        for x in s {
            assert!(payoff >= (*x - 100.).max(0.));
        }
    }
}
#[test]
fn perfect_correlation_reduces_identical_assets_exactly() {
    let mut m = MaxCallMarket::paper_five();
    m.correlation = vec![vec![1.; 5]; 5];
    let p = generate_paths(&m, 8, 1, 0, 1, 1 << 20).unwrap();
    for j in 0..p.paths {
        for t in 0..p.dates {
            let s = p.state(j, t).unwrap();
            assert!(s.iter().all(|v| *v == s[0]));
        }
    }
}
#[test]
fn discounted_reinvested_episode_units_fees_and_settlement() {
    let mut m = one();
    m.spots = vec![120., 80.];
    m.volatilities = vec![0., 0.];
    m.dividend_yields = vec![0.1, 0.02];
    m.correlation = vec![vec![1., 0.], vec![0., 1.]];
    m.times = vec![0., 1., 2.];
    m.exercise = vec![true; 3];
    let p = generate_paths(&m, 1, 0, 0, 0, 1 << 20).unwrap();
    let mut c = episode(2);
    c.quantity = 2;
    c.multiplier = 100;
    c.premium = 50.;
    c.trading_costs = vec![0.001, 0.002];
    c.liquidation_costs = vec![0.003, 0.004];
    let e = assemble_episodes(&m, &p, &[1], &c).unwrap();
    let amount = 200. * m.payoff(1, p.state(0, 1).unwrap()).unwrap();
    assert_eq!(
        e.batch.exercise.as_ref().unwrap().settlement_amounts,
        [amount]
    );
    for t in 0..3 {
        close(e.batch.marks[t * 2], 120., 1e-12);
        close(e.batch.marks[t * 2 + 1], 80., 1e-12);
    }
    let holdings = [2., -3., 999., 999.];
    let pnl = evaluate_episode_holdings(&e, &holdings).unwrap()[0];
    let fees = 2. * 120. * (0.001 + 0.003) + 3. * 80. * (0.002 + 0.004);
    close(pnl, 50. - amount - fees, 1e-10);
    assert_eq!(e.cash_growth, vec![1.; 2]);
    assert!(e.batch.payoffs.iter().all(|x| *x == 0.));
    let initial = assemble_episodes(&m, &p, &[0], &c).unwrap();
    close(
        evaluate_episode_holdings(&initial, &holdings).unwrap()[0],
        50. - 200. * 20.,
        1e-12,
    );
    let mut forged = p.clone();
    forged.values[2] += 1.;
    assert!(assemble_episodes(&m, &forged, &[1], &c).is_err());
    assert!(assemble_episodes(&m, &p, &[3], &c).is_err());
}
#[test]
fn deterministic_policy_tail_bounds_and_cbar0_regression() {
    let mut m = one();
    m.spots[0] = 120.;
    m.volatilities[0] = 0.;
    m.strike = 100.;
    m.times = vec![0., 0.5, 1.];
    m.exercise = vec![true; 3];
    let c = PolicyTailConfig {
        lower_paths: 8,
        upper_paths: 8,
        inner_samples: 4,
        outer_batch_rows: 4,
        ..Default::default()
    };
    let mut never = |_: usize, _: &[f64]| Ok::<bool, ComputeError>(false);
    let r = evaluate_policy_tail(&m, &mut never, &c).unwrap();
    let terminal = 120. * ((-0.05_f64).exp());
    let expected = (terminal - 100.) * (-0.05_f64).exp();
    close(r.lower.mean, expected, 1e-12);
    close(r.upper.mean, 20., 1e-12);
    assert!(
        r.mean_noisy_martingale_increment
            .iter()
            .all(|x| x.abs() < 1e-12)
    );
    assert_eq!(r.estimator_id, "becker_policy_tail_nested_v1");
    m.exercise = vec![false, false, true];
    let r = evaluate_policy_tail(&m, &mut never, &c).unwrap();
    close(r.upper.mean, expected, 1e-12);
    assert!(r.upper.mean > 0.);
}
#[test]
fn european_policy_tail_mc_gate_and_batch_callback() {
    struct Batch {
        calls: usize,
    }
    impl MaxCallStoppingPolicy for Batch {
        fn should_stop(&mut self, _: usize, _: &[f64]) -> Result<bool, ComputeError> {
            panic!("batch override required")
        }
        fn should_stop_batch(
            &mut self,
            _: usize,
            s: &[f64],
            d: usize,
        ) -> Result<Vec<bool>, ComputeError> {
            self.calls += 1;
            Ok(vec![false; s.len() / d])
        }
    }
    let m = one();
    let c = PolicyTailConfig {
        lower_paths: 4096,
        upper_paths: 512,
        inner_samples: 128,
        outer_batch_rows: 64,
        ..Default::default()
    };
    let mut policy = Batch { calls: 0 };
    let r = evaluate_policy_tail(&m, &mut policy, &c).unwrap();
    close(r.lower.mean, price(&m), 4. * r.lower.standard_error);
    close(r.upper.mean, price(&m), 4. * r.upper.standard_error);
    assert!(r.confidence_scope.contains("unbounded"));
    let mut invalid = c;
    invalid.max_transition_asset_steps = 1;
    assert!(evaluate_policy_tail(&m, &mut policy, &invalid).is_err());
}

#[test]
fn hundred_assets_are_real_dynamic_dimensions_and_admission_is_enforced() {
    let mut m = MaxCallMarket::paper_five();
    let d = 100;
    m.spots = vec![100.; d];
    m.volatilities = vec![0.2; d];
    m.dividend_yields = vec![0.1; d];
    m.correlation = (0..d)
        .map(|i| (0..d).map(|j| if i == j { 1. } else { 0. }).collect())
        .collect();
    let paths = generate_paths(&m, 4, 791, 0, 12, 4 << 20).unwrap();
    assert_eq!(paths.assets, 100);
    assert_eq!(paths.values.len(), 4 * 10 * 100);
    let episodes = assemble_episodes(&m, &paths, &[9; 4], &episode(100)).unwrap();
    assert_eq!(episodes.batch.features.len(), 4 * 9 * 101);
    assert_eq!(episodes.physical_cohort_sha256, paths.content_sha256);
    assert!(generate_paths(&m, 1_000_000, 791, 0, 12, 4 << 20).is_err());
}
#[test]
fn selected_policy_batch_callback_and_outer_chunk_invariance() {
    struct Batch {
        calls: usize,
    }
    impl MaxCallStoppingPolicy for Batch {
        fn should_stop(&mut self, _: usize, _: &[f64]) -> Result<bool, ComputeError> {
            panic!("batch callback override expected")
        }
        fn should_stop_batch(
            &mut self,
            _: usize,
            s: &[f64],
            d: usize,
        ) -> Result<Vec<bool>, ComputeError> {
            self.calls += 1;
            Ok(vec![false; s.len() / d])
        }
    }
    let mut m = one();
    m.times = vec![0., 0.5, 1.];
    m.exercise = vec![false, true, true];
    m.dividend_yields[0] = 0.;
    let mut c = PolicyTailConfig {
        lower_paths: 256,
        upper_paths: 32,
        inner_samples: 32,
        outer_batch_rows: 16,
        ..Default::default()
    };
    let mut p = Batch { calls: 0 };
    let a = evaluate_policy_tail(&m, &mut p, &c).unwrap();
    assert!(p.calls > 0);
    c.outer_batch_rows = 7;
    let b = evaluate_policy_tail(&m, &mut p, &c).unwrap();
    assert_eq!(a.lower_samples, b.lower_samples);
    assert_eq!(a.upper_samples, b.upper_samples);
    assert_eq!(a.lower_cohort_sha256, b.lower_cohort_sha256);
    assert_eq!(a.upper_cohort_sha256, b.upper_cohort_sha256);
    c.conditional_outer_batch_rows = 4;
    let grouped = evaluate_policy_tail(&m, &mut p, &c).unwrap();
    assert_eq!(a.lower_samples, grouped.lower_samples);
    assert_eq!(a.upper_samples, grouped.upper_samples);
    assert_eq!(
        a.mean_noisy_martingale_increment,
        grouped.mean_noisy_martingale_increment
    );
}
#[test]
fn state_access_and_episode_prefix_validation_are_safe() {
    let mut m = one();
    m.times = vec![0., 0.5, 1.];
    m.exercise = vec![true; 3];
    let paths = generate_paths(&m, 1, 1, 0, 1, 1 << 20).unwrap();
    let e = assemble_episodes(&m, &paths, &[0], &episode(1)).unwrap();
    assert_eq!(
        evaluate_episode_holdings(&e, &[1e8, -1e8]).unwrap(),
        vec![10.]
    );
    let mut broken = paths.clone();
    broken.dates = usize::MAX;
    broken.assets = usize::MAX;
    assert!(broken.state(usize::MAX - 1, 0).is_err());
    let mut invalid = episode(1);
    invalid.multiplier = 0;
    assert!(assemble_episodes(&m, &paths, &[1], &invalid).is_err());
}

#[test]
fn conditional_branches_use_supplied_state_and_remaining_horizon() {
    let mut m = one();
    m.times = vec![0., 0.5, 1.];
    m.exercise = vec![false, true, true];
    let mut values = Vec::new();
    for branch in 0..8192 {
        let tail = simulate_tail(
            &m,
            1,
            &[125.],
            RandomKey {
                seed: 809,
                namespace: 7,
                path_id: 91,
                branch,
            },
            1 << 20,
        )
        .unwrap();
        assert_eq!(tail[0], 125.);
        values.push(tail[1]);
    }
    let expected = 125. * ((m.rate - m.dividend_yields[0]) * 0.5).exp();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let se = expected * ((m.volatilities[0].powi(2) * 0.5).exp() - 1.).sqrt()
        / (values.len() as f64).sqrt();
    close(mean, expected, 4. * se);
}
#[test]
fn executable_json_roundtrip_publishes_actual_discounted_episodes() {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    let request = serde_json::json!({"market":one(),"paths":2,"seed":391,"first_path":10,"namespace":73,"max_working_bytes":1048576,
        "stopping_steps":[1,1],"episode":episode(1)});
    let mut child = Command::new(env!("CARGO_BIN_EXE_multi_asset_market"))
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
    let result = child.wait_with_output().unwrap();
    assert!(result.status.success());
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    let paths: MaxCallPaths = serde_json::from_value(value["paths"].clone()).unwrap();
    let episodes: MaxCallEpisodes = serde_json::from_value(value["episodes"].clone()).unwrap();
    assert_eq!(
        episodes.batch.exercise.as_ref().unwrap().settlement_amounts[0],
        one().payoff(1, paths.state(0, 1).unwrap()).unwrap()
    );
    assert_eq!(episodes.physical_cohort_sha256, paths.content_sha256);
    assert_eq!(episodes.batch.paths, 2);
}

#[test]
fn source_nine_times_ninety_six_fine_calendar_is_representable() {
    let mut m = one();
    m.times = (0..=864).map(|t| 3. * t as f64 / 864.).collect();
    m.exercise = (0_usize..=864).map(|t| t.is_multiple_of(96)).collect();
    let paths = generate_paths(&m, 2, 8721, 0, 73, 1 << 20).unwrap();
    assert_eq!(paths.dates, 865);
    let episodes = assemble_episodes(&m, &paths, &[864, 96], &episode(1)).unwrap();
    assert_eq!(
        episodes.batch.exercise.as_ref().unwrap().settlement_step,
        [864, 96]
    );
}
