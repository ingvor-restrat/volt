use std::collections::BTreeMap;
use volt::{Currency, PriceMicros, TimestampNs, UnderlyingId};
use volt_compute::{
    ExecutionTarget, Precision,
    budget::MemoryBudget,
    hedge_application::{HedgeCostConfig, HedgeLiability, HedgePathBatch},
    hedge_ledger::{
        HedgeInstrument, HedgeInstrumentKind, HedgeLedger, HedgeLedgerConfig, HedgeMarketSnapshot,
        IntervalAccrual, LiabilityCashflow, ProportionalCost, QuantityMicros,
    },
    heston::{HestonParams, QuadratureConfig},
    market_paths::{
        IntegratedVarianceRule, MarketModel, MarketPathConfig, MarketPathInputs, evolve_host,
        generate_paths,
    },
    model_hedge::*,
};
fn close(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() <= t, "{a:.15e} != {b:.15e}");
}
fn bs(volatility: f64) -> MarketPathConfig {
    MarketPathConfig {
        seed: 42,
        first_path_id: 0,
        paths: 2,
        steps: 1,
        dt: 1.0,
        initial_spot: 100.0,
        rate: 0.0,
        dividend_yield: 0.0,
        model: MarketModel::BlackScholes { volatility },
        integrated_variance_rule: IntegratedVarianceRule::Left,
    }
}
fn lewis() -> MarketPathConfig {
    MarketPathConfig {
        model: MarketModel::Heston {
            params: HestonParams {
                initial_variance: 0.04,
                kappa: 4.0,
                long_run_variance: 0.25,
                vol_of_variance: 1.0,
                correlation: -0.5,
            },
        },
        ..bs(0.2)
    }
}
fn config() -> ModelHedgeConfig {
    ModelHedgeConfig {
        target: ExecutionTarget::Cpu,
        precision: Precision::F64,
        budget: MemoryBudget::default(),
        quadrature: QuadratureConfig {
            nodes: 128,
            upper_bound: 200.0,
        },
        tolerances: BaselineTolerances {
            price_absolute: 1e-6,
            price_relative: 1e-7,
            delta_absolute: 1e-7,
            delta_relative: 1e-7,
            variance_absolute: 1e-5,
            variance_relative: 1e-6,
        },
        max_queries_per_job: 16,
        max_cached_states: 100,
        ..ModelHedgeConfig::default()
    }
}

#[test]
fn old_config_defaults_to_device_fourier_and_explicit_ceiling_is_preflight_checked() {
    let mut json = serde_json::to_value(config()).unwrap();
    let fields = json.as_object_mut().unwrap();
    fields.remove("heston_method");
    fields.remove("adaptive");
    fields.remove("adaptive_threads");
    let old: ModelHedgeConfig = serde_json::from_value(json).unwrap();
    assert_eq!(old.heston_method, HestonBaselineMethod::DeviceFourier);
    let mut large = old;
    large.quadrature.upper_bound = 5_000_000.0;
    large.validate().unwrap();
    large.quadrature.upper_bound = 5_000_001.0;
    assert!(large.validate().is_err());
}

#[test]
fn adaptive_is_explicit_cpu_f64_even_with_unavailable_requested_tensor_target() {
    let market = generate_paths(&lewis()).unwrap();
    let mut config = config();
    config.heston_method = HestonBaselineMethod::AdaptiveScalar;
    config.target = ExecutionTarget::Cuda { index: 999 };
    config.precision = Precision::F32;
    config.max_queries_per_job = 1;
    let result =
        model_hedge_holdings(&[market], HedgeLiability::Call { strike: 100.0 }, &config).unwrap();
    assert!(matches!(
        result.execution_policy,
        BaselineExecutionPolicy::AdaptiveScalarCpuF64
    ));
    assert_eq!(result.validation.device_batches, 0);
    assert_eq!(result.validation.adaptive_converged_queries, 1);
    assert_eq!(result.validation.adaptive_tail_checked_queries, 1);
    assert_eq!(result.validation.exact_cache_hits, 1);
    assert!(result.validation.adaptive_function_evaluations > 0);
    close(result.initial_model_prices[0], 16.844899754126864, 2e-7);
    close(result.holdings[0], 0.6210009625614118, 2e-7);
    let denominator = funded_variance_claim_variance_derivative(4.0, 1.0, 0.0).unwrap();
    close(result.holdings[1] * denominator, 10.325822269681396, 2e-5);
}

#[test]
fn adaptive_resolves_late_zero_variance_and_parallel_results_are_deterministic() {
    let mut c = MarketPathConfig::paper_heston(1, 17);
    c.steps = 1;
    if let MarketModel::Heston { ref mut params } = c.model {
        params.initial_variance = 0.0;
    }
    let market = generate_paths(&c).unwrap();
    let mut config = config();
    config.heston_method = HestonBaselineMethod::AdaptiveScalar;
    let result =
        model_hedge_holdings(&[market], HedgeLiability::Call { strike: 100.0 }, &config).unwrap();
    close(result.initial_model_prices[0], 0.0065178659884, 5e-8);
    close(result.holdings[0], 0.7285337304, 5e-7);
    let denominator = funded_variance_claim_variance_derivative(1.0, 1.0 / 365.0, 0.0).unwrap();
    close(result.holdings[1] * denominator, 59.448183, 5e-5);
    assert!(result.validation.adaptive_max_cutoff > 80_000.0);

    let mut c = lewis();
    c.steps = 2;
    c.dt = 0.5;
    let market = generate_paths(&c).unwrap();
    let serial = model_hedge_holdings(
        std::slice::from_ref(&market),
        HedgeLiability::Call { strike: 100.0 },
        &config,
    )
    .unwrap();
    config.adaptive_threads = 2;
    let parallel =
        model_hedge_holdings(&[market], HedgeLiability::Call { strike: 100.0 }, &config).unwrap();
    assert_eq!(serial.holdings, parallel.holdings);
    assert_eq!(serial.initial_model_prices, parallel.initial_model_prices);
    assert_eq!(
        serial.validation.adaptive_function_evaluations,
        parallel.validation.adaptive_function_evaluations
    );
    assert_eq!(parallel.validation.adaptive_threads_used, 2);
}

#[test]
fn adaptive_work_and_memory_limits_are_errors_not_device_fallbacks() {
    let market = generate_paths(&lewis()).unwrap();
    let mut config = config();
    config.heston_method = HestonBaselineMethod::AdaptiveScalar;
    config.target = ExecutionTarget::Cuda { index: 999 };
    config.precision = Precision::F32;
    config.adaptive.max_evaluations = 45;
    config.adaptive.max_panels = 3;
    assert!(
        model_hedge_holdings(
            std::slice::from_ref(&market),
            HedgeLiability::Call { strike: 100.0 },
            &config
        )
        .is_err()
    );
    config.adaptive = Default::default();
    config.budget.compute_bytes = 1;
    let error = model_hedge_holdings(&[market], HedgeLiability::Call { strike: 100.0 }, &config)
        .unwrap_err();
    assert!(matches!(
        error,
        volt_compute::ComputeError::BudgetExceeded(_)
    ));
    config.adaptive_threads = 65;
    assert!(config.validate().is_err());
}

#[test]
fn bs_model_delta_and_price_match_known_control_without_terminal_decision() {
    let market = generate_paths(&bs(0.2)).unwrap();
    let result =
        model_hedge_holdings(&[market], HedgeLiability::Call { strike: 100.0 }, &config()).unwrap();
    assert_eq!(result.holdings.len(), 2);
    assert_eq!(result.hedge_count, 1);
    for delta in result.holdings {
        close(delta, 0.539827837277029, 1e-14);
    }
    for price in result.initial_model_prices {
        close(price, 7.965567455405804, 1e-12);
    }
    assert_eq!(result.validation.pricing_jobs, 0);
}

#[test]
fn bs_normalized_spread_has_signed_legs_and_no_liability_means_zero_holdings() {
    let market = generate_paths(&bs(0.0)).unwrap();
    let spread = model_hedge_holdings(
        std::slice::from_ref(&market),
        HedgeLiability::NormalizedCallSpread {
            lower: 90.0,
            upper: 110.0,
        },
        &config(),
    )
    .unwrap();
    assert_eq!(spread.holdings, vec![0.05, 0.05]);
    assert_eq!(spread.initial_model_prices, vec![0.5, 0.5]);
    let none = model_hedge_holdings(&[market], HedgeLiability::None, &config()).unwrap();
    assert!(none.holdings.iter().all(|x| *x == 0.0));
    assert!(none.initial_model_prices.iter().all(|x| *x == 0.0));
    let market = generate_paths(&bs(0.0)).unwrap();
    assert!(
        model_hedge_holdings(&[market], HedgeLiability::Call { strike: 100.0 }, &config()).is_err()
    );
}

#[test]
fn heston_is_true_two_instrument_model_hedge_with_independent_quantlib_control() {
    // Independent QuantLib1.40 Gatheral adaptive control, r=q=0, T=1;
    // Lewis parameters with rates changed explicitly, not the published rate fixture.
    let market = generate_paths(&lewis()).unwrap();
    let mut cfg = config();
    cfg.max_queries_per_job = 1;
    let result =
        model_hedge_holdings(&[market], HedgeLiability::Call { strike: 100.0 }, &cfg).unwrap();
    let denominator = (1.0 - (-4.0_f64).exp()) / 4.0;
    for p in 0..2 {
        close(result.initial_model_prices[p], 16.844899754126864, 2e-7);
        close(result.holdings[p * 2], 0.6210009625614118, 2e-7);
        close(
            result.holdings[p * 2 + 1],
            10.325822269681396 / denominator,
            2e-5,
        );
    }
    assert_eq!(result.validation.priced_call_queries, 1);
    assert_eq!(result.validation.exact_cache_hits, 1);
    assert_eq!(result.validation.pricing_jobs, 3);
    assert!(result.validation.max_price_cutoff_difference < 1e-6);
}

#[test]
fn heston_zero_variance_later_state_is_supported_without_terminal_division() {
    let mut c = lewis();
    c.paths = 1;
    c.steps = 2;
    c.dt = 0.5;
    let replay = MarketPathInputs::from_explicit(c, vec![0.04, 0.0, 0.0], vec![0.0, 0.0]).unwrap();
    let market = evolve_host(&replay).unwrap();
    let result =
        model_hedge_holdings(&[market], HedgeLiability::Call { strike: 100.0 }, &config()).unwrap();
    assert_eq!(result.holdings.len(), 4);
    assert!(result.holdings.iter().all(|x| x.is_finite()));
    assert_eq!(result.validation.pricing_jobs, 3);
}

#[test]
fn very_late_paper_boundary_either_meets_independent_control_or_refuses_resolution() {
    let mut c = MarketPathConfig::paper_heston(1, 17);
    c.steps = 1;
    if let MarketModel::Heston { ref mut params } = c.model {
        params.initial_variance = 0.0;
    }
    let market = generate_paths(&c).unwrap();
    let config = ModelHedgeConfig::default();
    match model_hedge_holdings(&[market], HedgeLiability::Call { strike: 100.0 }, &config) {
        Ok(result) => {
            // Independent adaptive QuantLib positive-domain right limits at T=1/365.
            close(result.initial_model_prices[0], 0.0065178659884, 1e-6);
            close(result.holdings[0], 0.7285337304, 5e-5);
            let denominator =
                funded_variance_claim_variance_derivative(1.0, 1.0 / 365.0, 0.0).unwrap();
            close(result.holdings[1] * denominator, 59.44819, 0.005);
        }
        Err(error) => {
            assert!(
                error.to_string().contains("refinement"),
                "unexpected late-boundary error: {error}"
            );
            eprintln!(
                "Declared fixed quadrature refuses very-late zero-variance baseline: {error}"
            );
        }
    }
}

#[test]
fn funded_claim_derivative_uses_integrated_units_discount_and_expm1() {
    close(
        funded_variance_claim_variance_derivative(4.0, 1.0, 0.03).unwrap(),
        (-0.03_f64).exp() * (1.0 - (-4.0_f64).exp()) / 4.0,
        1e-15,
    );
    close(
        funded_variance_claim_variance_derivative(1.0, 1e-12, 0.0).unwrap(),
        1e-12,
        1e-24,
    );
    assert_eq!(
        funded_variance_claim_variance_derivative(1.0, 0.0, 0.0).unwrap(),
        0.0
    );
    assert!(funded_variance_claim_variance_derivative(0.0, 1.0, 0.0).is_err());
    assert!(funded_variance_claim_variance_derivative(1.0, -1.0, 0.0).is_err());
}

fn batch() -> HedgePathBatch {
    HedgePathBatch {
        paths: 1,
        steps: 2,
        hedge_count: 2,
        feature_count: 2,
        marks: vec![100.0, 4.0, 110.0, 5.0, 120.0, 6.0],
        features: vec![100_f64.ln(), 0.04, 110_f64.ln(), 0.05],
        payoffs: vec![3.0],
        source_hash: "independent-accounting-fixture".into(),
    }
}

#[test]
fn scalar_pnl_costs_and_total_turnover_reconcile_with_integer_ledger() {
    let holdings = [0.5, -2.0, 0.25, 1.0];
    let result = scalar_hedge_pnl(
        &batch(),
        &holdings,
        HedgeCostConfig {
            proportional: 0.01,
            terminal_proportional: 0.02,
        },
        10.0,
    )
    .unwrap();
    close(result.pnl[0], 11.775, 1e-13);
    close(result.costs[0], 1.725, 1e-14);
    close(result.turnover[0], 136.5, 1e-14);
    close(result.terminal_turnover[0], 36.0, 1e-14);
    let instrument = |id: &str, kind| HedgeInstrument {
        instrument_id: id.into(),
        currency: Currency("USD".into()),
        kind,
        quantity_step: QuantityMicros(1),
        multiplier: 1,
    };
    let cfg = HedgeLedgerConfig {
        currency: Currency("USD".into()),
        terminal_time: TimestampNs(30),
        instruments: vec![
            instrument(
                "S",
                HedgeInstrumentKind::Stock {
                    underlying: UnderlyingId("X".into()),
                },
            ),
            instrument(
                "V",
                HedgeInstrumentKind::FundedVarianceClaim {
                    underlying: UnderlyingId("X".into()),
                    maturity: TimestampNs(30),
                },
            ),
        ],
    };
    let snapshot = |time, s, v| HedgeMarketSnapshot {
        snapshot_id: format!("snapshot-{time}"),
        as_of: TimestampNs(time),
        currency: Currency("USD".into()),
        prices: BTreeMap::from([("S".into(), PriceMicros(s)), ("V".into(), PriceMicros(v))]),
    };
    let target = |s, v| {
        BTreeMap::from([
            ("S".into(), QuantityMicros(s)),
            ("V".into(), QuantityMicros(v)),
        ])
    };
    let mut ledger =
        HedgeLedger::new(cfg, snapshot(10, 100_000_000, 4_000_000), 10_000_000).unwrap();
    ledger
        .rebalance(
            snapshot(10, 100_000_000, 4_000_000),
            target(500_000, -2_000_000),
            IntervalAccrual::zero_rate(TimestampNs(10), TimestampNs(10)),
            ProportionalCost {
                parts_per_million: 10000,
            },
        )
        .unwrap();
    ledger
        .rebalance(
            snapshot(20, 110_000_000, 5_000_000),
            target(250_000, 1_000_000),
            IntervalAccrual::zero_rate(TimestampNs(10), TimestampNs(20)),
            ProportionalCost {
                parts_per_million: 10000,
            },
        )
        .unwrap();
    ledger
        .settle(
            snapshot(30, 120_000_000, 6_000_000),
            LiabilityCashflow {
                settlement_id: "liability".into(),
                currency: Currency("USD".into()),
                amount_micros: 3_000_000,
            },
            IntervalAccrual::zero_rate(TimestampNs(20), TimestampNs(30)),
            ProportionalCost {
                parts_per_million: 20000,
            },
        )
        .unwrap();
    close(
        result.pnl[0],
        ledger.state().cash_picos as f64 / 1e12,
        1e-13,
    );
}

#[test]
fn frictionless_stock_replication_and_no_hedge_have_common_premium() {
    let batch = HedgePathBatch {
        paths: 1,
        steps: 2,
        hedge_count: 1,
        feature_count: 1,
        marks: vec![100.0, 110.0, 120.0],
        features: vec![100_f64.ln(), 110_f64.ln()],
        payoffs: vec![40.0],
        source_hash: "stock-replication".into(),
    };
    let hedge = scalar_hedge_pnl(&batch, &[1.0, 1.0], HedgeCostConfig::default(), 20.0).unwrap();
    assert_eq!(hedge.pnl, vec![0.0]);
    assert_eq!(hedge.turnover, vec![220.0]);
    let unhedged = no_hedge_pnl(&batch, 20.0).unwrap();
    assert_eq!(unhedged.pnl, vec![-20.0]);
    assert_eq!(unhedged.costs, vec![0.0]);
}

#[test]
fn rejects_unresolved_quadrature_budgets_or_unsupported_models_without_proxy() {
    let market = generate_paths(&lewis()).unwrap();
    let mut low = config();
    low.quadrature = QuadratureConfig {
        nodes: 16,
        upper_bound: 1.0,
    };
    assert!(
        model_hedge_holdings(
            std::slice::from_ref(&market),
            HedgeLiability::Call { strike: 100.0 },
            &low
        )
        .is_err()
    );
    let mut low = config();
    low.budget.host_working_bytes = 1;
    assert!(
        model_hedge_holdings(
            std::slice::from_ref(&market),
            HedgeLiability::Call { strike: 100.0 },
            &low
        )
        .is_err()
    );
    let mut bad = lewis();
    bad.rate = 0.01;
    let rate_market = generate_paths(&bad).unwrap();
    assert!(
        model_hedge_holdings(
            &[rate_market],
            HedgeLiability::Call { strike: 100.0 },
            &config()
        )
        .is_err()
    );
    let mut low = config();
    low.max_queries_per_job = 257;
    assert!(model_hedge_holdings(&[market], HedgeLiability::Call { strike: 100.0 }, &low).is_err());
    assert!(scalar_hedge_pnl(&batch(), &[0.0], HedgeCostConfig::default(), 0.0).is_err());
    assert!(
        scalar_hedge_pnl(
            &batch(),
            &[0.0; 4],
            HedgeCostConfig {
                proportional: -0.1,
                terminal_proportional: 0.0
            },
            0.0
        )
        .is_err()
    );
}

struct CacheTemp(std::path::PathBuf);
impl CacheTemp {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "volt-baseline-cache-{}-{nonce}",
            std::process::id()
        ));
        Self(p)
    }
}
impl Drop for CacheTemp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn disk_cache(t: &CacheTemp) -> BaselineCacheConfig {
    BaselineCacheConfig {
        directory: t.0.clone(),
        mode: BaselineCacheMode::ReadWrite,
        max_entry_bytes: 64 * 1024 * 1024,
    }
}

#[test]
fn scalar_heston_cache_preserves_exact_outputs_and_reuses_across_operational_settings() {
    let temp = CacheTemp::new();
    let mut cache = disk_cache(&temp);
    let market = generate_paths(&lewis()).unwrap();
    let mut c = config();
    c.heston_method = HestonBaselineMethod::AdaptiveScalar;
    let (a, first) = model_hedge_holdings_cached(
        std::slice::from_ref(&market),
        HedgeLiability::Call { strike: 100.0 },
        &c,
        Some(&cache),
    )
    .unwrap();
    assert!(first.written && !first.hit && first.actual_function_evaluations > 0);
    assert!(first.bytes_written > 0);
    cache.mode = BaselineCacheMode::ReadOnly;
    c.adaptive_threads = 2;
    c.max_cached_states = 1;
    c.target = ExecutionTarget::Cuda { index: 99 };
    c.precision = Precision::F32;
    let (b, second) = model_hedge_holdings_cached(
        &[market],
        HedgeLiability::Call { strike: 100.0 },
        &c,
        Some(&cache),
    )
    .unwrap();
    assert!(second.hit && !second.written);
    assert_eq!(second.actual_function_evaluations, 0);
    assert_eq!(first.key, second.key);
    assert_eq!(first.payload_sha256, second.payload_sha256);
    assert_eq!(a.holdings, b.holdings);
    assert_eq!(a.initial_model_prices, b.initial_model_prices);
    assert_eq!(
        a.validation.adaptive_function_evaluations,
        b.validation.adaptive_function_evaluations
    );
}

#[test]
fn checked_cache_rejects_changed_cohort_numerics_corruption_and_forged_shape() {
    use sha2::{Digest, Sha256};
    let temp = CacheTemp::new();
    let mut cache = disk_cache(&temp);
    let market = generate_paths(&bs(0.2)).unwrap();
    let c = config();
    let liability = HedgeLiability::Call { strike: 100.0 };
    let (_, access) =
        model_hedge_holdings_cached(std::slice::from_ref(&market), liability, &c, Some(&cache))
            .unwrap();
    cache.mode = BaselineCacheMode::ReadOnly;
    assert!(
        model_hedge_holdings_cached(
            std::slice::from_ref(&market),
            HedgeLiability::Call { strike: 101.0 },
            &c,
            Some(&cache)
        )
        .is_err()
    );
    let mut other = bs(0.2);
    other.seed += 1;
    let other = generate_paths(&other).unwrap();
    assert!(model_hedge_holdings_cached(&[other], liability, &c, Some(&cache)).is_err());
    let path = cache
        .directory
        .join(format!("{}.baseline-v1", access.key.unwrap()));
    let original = std::fs::read(&path).unwrap();
    let mut corrupt = original.clone();
    let position = corrupt.iter().position(|b| *b == b'\n').unwrap() + 10;
    corrupt[position] ^= 1;
    std::fs::write(&path, &corrupt).unwrap();
    assert!(
        model_hedge_holdings_cached(std::slice::from_ref(&market), liability, &c, Some(&cache))
            .is_err()
    );
    let mut lines = original.splitn(4, |b| *b == b'\n');
    let header = lines.next().unwrap();
    let mut payload: serde_json::Value = serde_json::from_slice(lines.next().unwrap()).unwrap();
    payload["paths"] = serde_json::json!(999);
    let payload = serde_json::to_vec(&payload).unwrap();
    let footer = serde_json::to_vec(
        &serde_json::json!({"payload_sha256":format!("{:x}",Sha256::digest(&payload))}),
    )
    .unwrap();
    let forged = [header, &payload, &footer, b""].join(&b'\n');
    std::fs::write(&path, forged).unwrap();
    assert!(model_hedge_holdings_cached(&[market], liability, &c, Some(&cache)).is_err());
}

#[test]
fn cache_working_admission_and_read_only_miss_leave_existing_files_intact() {
    let temp = CacheTemp::new();
    let cache = BaselineCacheConfig {
        mode: BaselineCacheMode::ReadOnly,
        ..disk_cache(&temp)
    };
    let market = generate_paths(&bs(0.2)).unwrap();
    assert!(
        model_hedge_holdings_cached(
            std::slice::from_ref(&market),
            HedgeLiability::Call { strike: 100.0 },
            &config(),
            Some(&cache)
        )
        .is_err()
    );
    assert!(!temp.0.exists());
    let mut c = config();
    c.budget.host_working_bytes = 1024;
    assert!(
        model_hedge_holdings_cached(
            &[market],
            HedgeLiability::Call { strike: 100.0 },
            &c,
            Some(&cache)
        )
        .is_err()
    );
}

#[test]
fn component_cache_reuses_single_market_calls_inside_joint_systems_and_spreads() {
    let temp = CacheTemp::new();
    let mut cache = disk_cache(&temp);
    let first = generate_paths(&bs(0.2)).unwrap();
    let mut other = bs(0.2);
    other.seed = 43;
    let second = generate_paths(&other).unwrap();
    let c = config();
    let _ = model_hedge_holdings_cached(
        std::slice::from_ref(&first),
        HedgeLiability::Call { strike: 100.0 },
        &c,
        Some(&cache),
    )
    .unwrap();
    let (joint, access) = model_hedge_holdings_cached(
        &[first.clone(), second.clone()],
        HedgeLiability::Call { strike: 100.0 },
        &c,
        Some(&cache),
    )
    .unwrap();
    assert_eq!(access.entry_hits, 1);
    assert_eq!(access.entry_misses, 1);
    let direct = model_hedge_holdings(
        &[first.clone(), second],
        HedgeLiability::Call { strike: 100.0 },
        &c,
    )
    .unwrap();
    assert_eq!(joint.holdings, direct.holdings);
    for (a, b) in joint
        .initial_model_prices
        .iter()
        .zip(direct.initial_model_prices)
    {
        close(*a, b, 1e-12);
    }
    let liability = HedgeLiability::NormalizedCallSpread {
        lower: 100.0,
        upper: 101.0,
    };
    let (spread, access) =
        model_hedge_holdings_cached(std::slice::from_ref(&first), liability, &c, Some(&cache))
            .unwrap();
    assert_eq!(access.entry_hits, 1);
    assert_eq!(access.entry_misses, 1);
    let direct = model_hedge_holdings(std::slice::from_ref(&first), liability, &c).unwrap();
    for (a, b) in spread.holdings.iter().zip(direct.holdings) {
        close(*a, b, 1e-12);
    }
    cache.mode = BaselineCacheMode::ReadOnly;
    let (_, access) = model_hedge_holdings_cached(&[first], liability, &c, Some(&cache)).unwrap();
    assert_eq!(access.entry_hits, 2);
    assert_eq!(access.actual_function_evaluations, 0);
}

#[test]
fn cached_heston_components_preserve_stock_and_variance_claim_columns() {
    let temp = CacheTemp::new();
    let cache = disk_cache(&temp);
    let mut c = config();
    c.heston_method = HestonBaselineMethod::AdaptiveScalar;
    let first = generate_paths(&lewis()).unwrap();
    let mut second = lewis();
    second.seed = 43;
    let second = generate_paths(&second).unwrap();
    let _ = model_hedge_holdings_cached(
        std::slice::from_ref(&first),
        HedgeLiability::Call { strike: 100.0 },
        &c,
        Some(&cache),
    )
    .unwrap();
    let markets = [first, second];
    let liability = HedgeLiability::Call { strike: 100.0 };
    let (cached, access) =
        model_hedge_holdings_cached(&markets, liability, &c, Some(&cache)).unwrap();
    assert_eq!(access.entry_hits, 1);
    assert_eq!(access.entry_misses, 1);
    assert!(matches!(
        cached.execution_policy,
        BaselineExecutionPolicy::AdaptiveScalarCpuF64
    ));
    let direct = model_hedge_holdings(&markets, liability, &c).unwrap();
    assert_eq!(cached.hedge_count, 4);
    for (a, b) in cached.holdings.iter().zip(direct.holdings) {
        close(*a, b, 1e-12);
    }
    let liability = HedgeLiability::NormalizedCallSpread {
        lower: 100.0,
        upper: 101.0,
    };
    let (cached, access) =
        model_hedge_holdings_cached(&markets, liability, &c, Some(&cache)).unwrap();
    assert_eq!(access.entry_hits, 2);
    assert_eq!(access.entry_misses, 2);
    assert!(
        cached
            .validation
            .composed_error_bounds
            .iter()
            .any(|v| *v > 0.0)
    );
    let direct = model_hedge_holdings(&markets, liability, &c).unwrap();
    for (a, b) in cached.holdings.iter().zip(direct.holdings) {
        close(*a, b, 1e-11);
    }
}
