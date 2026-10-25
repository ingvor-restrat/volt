#![cfg(feature = "live-risk")]
use volt_compute::{
    OptionSide,
    live_risk::*,
    volatility_surface::{EuropeanInputs, GreekConvention, QuoteRow},
};
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "{a:e} != {b:e}; diff={:e}",
        (a - b).abs()
    );
}
fn price(i: &EuropeanInputs, v: f64) -> f64 {
    let z = v * i.maturity_years.sqrt();
    let d1 = ((i.spot / i.strike).ln()
        + (i.rate - i.dividend_yield + 0.5 * v * v) * i.maturity_years)
        / z;
    let d2 = d1 - z;
    let n = |x: f64| 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2);
    let a = i.spot * (-i.dividend_yield * i.maturity_years).exp();
    let b = i.strike * (-i.rate * i.maturity_years).exp();
    match i.side {
        OptionSide::Call => a * n(d1) - b * n(d2),
        OptionSide::Put => b * n(-d2) - a * n(-d1),
    }
}
fn frame(generation: u64) -> QuoteFrame {
    let mut quotes = Vec::new();
    for (n, strike) in [80.0, 100.0, 120.0].into_iter().enumerate() {
        let input = EuropeanInputs {
            spot: 100.0,
            strike,
            maturity_years: 0.5,
            rate: 0.03,
            dividend_yield: 0.01,
            side: OptionSide::Call,
        };
        let mid = price(&input, 0.2);
        quotes.push(QuoteRow {
            quote_id: format!("quote-{n}-{generation}"),
            contract_id: format!("contract-{n}"),
            source_id: "independent-example".into(),
            underlying_id: "example-stock".into(),
            timestamp_ns: 10_000_000_000,
            input,
            bid: mid - 1e-5,
            ask: mid + 1e-5,
        });
    }
    QuoteFrame {
        schema: "volt.compute.quote_frame.v1".into(),
        generation,
        cutoff_ns: 10_000_000_000,
        currency: "USD".into(),
        position_revision: 1,
        synthetic: true,
        quotes,
        positions: vec![LivePosition {
            contract_id: "contract-1".into(),
            quantity: -10,
            multiplier: 100,
        }],
    }
}
#[test]
fn independent_european_example_scales_greeks_and_full_repricing_scenarios() {
    let mut engine = LiveRiskEngine::new(LiveRiskConfig::default()).unwrap();
    let snapshot = engine.update(frame(1)).unwrap();
    assert!(snapshot.portfolio.complete);
    assert!(snapshot.surface.is_some());
    assert!(snapshot.surface_error.is_none());
    let row = snapshot
        .rows
        .iter()
        .find(|r| r.quote.quote.contract_id == "contract-1")
        .unwrap();
    let g = row.greeks.as_ref().unwrap();
    assert_eq!(row.pricing_basis, "surface");
    close(g.price, 6.090127223710589, 2e-9);
    close(g.delta, 0.5534572420386372, 2e-9);
    close(g.gamma, 0.027789494747628724, 2e-9);
    close(g.vega_per_vol_point, 0.27789494747628723, 2e-9);
    close(snapshot.portfolio.value.unwrap(), -6090.127223710589, 2e-6);
    close(snapshot.portfolio.delta.unwrap(), -553.4572420386372, 2e-6);
    close(snapshot.portfolio.gamma.unwrap(), -27.789494747628724, 2e-6);
    close(
        snapshot.portfolio.vega_per_vol_point.unwrap(),
        -277.89494747628723,
        2e-6,
    );
    close(
        snapshot.portfolio.theta_per_day.unwrap(),
        17.75920442984,
        2e-6,
    );
    close(
        snapshot.portfolio.rho_per_bp.unwrap(),
        -2.4627798490076565,
        2e-8,
    );
    let i = row.quote.quote.input;
    for scenario in &snapshot.scenarios {
        let expected = -1000.0
            * (price(
                &EuropeanInputs {
                    spot: 100.0 * (1.0 + scenario.spot_change),
                    ..i
                },
                0.2 + scenario.volatility_change,
            ) - price(&i, 0.2));
        close(scenario.pnl.unwrap(), expected, 2e-6);
        assert!(scenario.local_pnl.unwrap().is_finite());
    }
}
#[test]
fn incomplete_book_is_explicit_and_absent_or_duplicate_position_rejects_whole_frame() {
    let mut engine = LiveRiskEngine::new(LiveRiskConfig::default()).unwrap();
    let mut stale = frame(1);
    stale.quotes[1].timestamp_ns = 0;
    let snapshot = engine.update(stale).unwrap();
    assert!(!snapshot.portfolio.complete);
    assert!(snapshot.portfolio.value.is_none());
    assert_eq!(snapshot.portfolio.unpriced_contracts, vec!["contract-1"]);
    assert!(snapshot.rows[1].greeks.is_none());
    assert!(snapshot.rows[1].calculation_error.is_some());
    assert!(
        snapshot
            .scenarios
            .iter()
            .all(|s| s.pnl.is_none() && s.unavailable_contracts == vec!["contract-1"])
    );
    assert!(snapshot.cube.value("delta", "contract-1").is_none());
    let mut absent = frame(2);
    absent.positions[0].contract_id = "unquoted".into();
    assert!(engine.update(absent).is_err());
    let mut duplicate = frame(2);
    duplicate.positions.push(duplicate.positions[0].clone());
    assert!(engine.update(duplicate).is_err());
    // No rejected frame consumed generation2.
    assert!(engine.update(frame(2)).unwrap().portfolio.complete);
}
#[test]
fn publication_generation_is_monotone_and_failed_update_does_not_consume_generation() {
    let mut engine = LiveRiskEngine::new(LiveRiskConfig::default()).unwrap();
    let first = engine.update(frame(10)).unwrap();
    assert!(engine.update(frame(10)).is_err());
    assert!(engine.update(frame(9)).is_err());
    let mut invalid = frame(11);
    invalid.currency = "US".into();
    assert!(engine.update(invalid).is_err());
    let next = engine.update(frame(11)).unwrap();
    assert_eq!((first.generation, first.cube.generation), (10, 10));
    assert_eq!((next.generation, next.cube.generation), (11, 11));
    close(
        first.portfolio.value.unwrap(),
        next.portfolio.value.unwrap(),
        0.0,
    );
}
#[test]
fn cube_cells_and_missing_counts_map_to_one_snapshot_without_mixing_rows() {
    let snapshot = LiveRiskEngine::new(LiveRiskConfig::default())
        .unwrap()
        .update(frame(7))
        .unwrap();
    assert_eq!(snapshot.cube.generation, snapshot.generation);
    assert_eq!(snapshot.cube.observed_at_ms, snapshot.cutoff_ns / 1_000_000);
    assert_eq!(snapshot.cube.entity_count, snapshot.rows.len());
    for row in &snapshot.rows {
        let key = &row.quote.quote.contract_id;
        close(
            snapshot.cube.value("bid", key).unwrap(),
            row.quote.quote.bid,
            0.0,
        );
        close(
            snapshot.cube.value("delta", key).unwrap(),
            row.greeks.as_ref().unwrap().delta,
            0.0,
        );
        close(
            snapshot.cube.value("position_value", key).unwrap(),
            row.position_value.unwrap(),
            0.0,
        );
        assert!(
            snapshot
                .cube
                .values
                .iter()
                .filter(|c| &c.key == key)
                .all(|c| c.observed_at_ms == row.quote.quote.timestamp_ns / 1_000_000)
        );
    }
    assert!(snapshot.cube.statuses.iter().all(|s| s.missing == 0));
}
#[test]
fn position_revision_and_weights_change_identity_and_totals_without_changing_unit_risk() {
    let mut a = LiveRiskEngine::new(LiveRiskConfig::default()).unwrap();
    let original = a.update(frame(1)).unwrap();
    let mut changed = frame(2);
    changed.position_revision = 2;
    changed.positions[0].quantity = 5;
    let altered = a.update(changed).unwrap();
    assert_ne!(original.input_sha256, altered.input_sha256);
    assert_ne!(original.positions_sha256, altered.positions_sha256);
    assert_eq!(altered.position_revision, 2);
    close(
        altered.portfolio.value.unwrap(),
        -0.5 * original.portfolio.value.unwrap(),
        1e-8,
    );
    for (a, b) in original.rows.iter().zip(&altered.rows) {
        close(
            a.greeks.as_ref().unwrap().delta,
            b.greeks.as_ref().unwrap().delta,
            0.0,
        );
    }
}
#[test]
fn sticky_log_moneyness_keeps_wing_failures_visible_and_prices_supported_interior() {
    let config = LiveRiskConfig {
        convention: GreekConvention::StickyLogMoneynessFiniteBump {
            relative_spot_bump: 1e-4,
        },
        ..Default::default()
    };
    let snapshot = LiveRiskEngine::new(config)
        .unwrap()
        .update(frame(1))
        .unwrap();
    assert!(snapshot.portfolio.complete);
    assert!(snapshot.rows[1].greeks.is_some());
    for row in [&snapshot.rows[0], &snapshot.rows[2]] {
        assert!(row.greeks.is_none());
        assert_eq!(row.pricing_basis, "unavailable");
        assert!(
            row.calculation_error
                .as_ref()
                .unwrap()
                .contains("extrapolation")
        );
    }
    // Wing quotes are unpositioned, so their missing Greeks do not poison the
    // supported interior position's complete book or its1% spot scenarios.
    assert!(snapshot.scenarios.iter().all(|s| s.pnl.is_some()));
    let fixed = LiveRiskEngine::new(LiveRiskConfig::default())
        .unwrap()
        .update(frame(1))
        .unwrap();
    close(
        snapshot.portfolio.delta.unwrap(),
        fixed.portfolio.delta.unwrap(),
        1e-4,
    );
}
#[test]
fn failed_fit_uses_labelled_quote_iv_only_for_fixed_volatility() {
    let mut f = frame(1);
    f.quotes[0].input.rate = 0.04;
    let p = price(&f.quotes[0].input, 0.2);
    f.quotes[0].bid = p - 1e-5;
    f.quotes[0].ask = p + 1e-5;
    let snapshot = LiveRiskEngine::new(LiveRiskConfig::default())
        .unwrap()
        .update(f.clone())
        .unwrap();
    assert!(snapshot.surface.is_none());
    assert!(snapshot.surface_error.as_ref().unwrap().contains("carry"));
    assert!(snapshot.rows.iter().all(|r| r.pricing_basis == "quote_iv"));
    assert!(snapshot.portfolio.complete);
    let config = LiveRiskConfig {
        convention: GreekConvention::StickyLogMoneynessFiniteBump {
            relative_spot_bump: 1e-4,
        },
        ..Default::default()
    };
    let snapshot = LiveRiskEngine::new(config).unwrap().update(f).unwrap();
    assert!(!snapshot.portfolio.complete);
    assert!(snapshot.rows.iter().all(|r| r.greeks.is_none()));
}
#[test]
fn normal_synthetic_demo_has_finite_serializable_grid_and_explicit_diagnostics() {
    let f = demo_frame(1, 1_000_000_000).unwrap();
    assert!(f.synthetic);
    assert_eq!(f.quotes.len(), 54);
    let snapshot = LiveRiskEngine::new(LiveRiskConfig::default())
        .unwrap()
        .update(f)
        .unwrap();
    assert!(snapshot.synthetic);
    assert!(snapshot.portfolio.complete);
    assert!(!snapshot.grid.is_empty());
    assert!(snapshot.grid.len() <= 13 * 25);
    assert!(snapshot.grid.iter().all(|p| p.volatility.is_finite()
        && p.volatility > 0.0
        && p.strike.is_finite()
        && p.strike > 0.0));
    assert!(
        !snapshot
            .surface
            .as_ref()
            .unwrap()
            .diagnostics
            .arbitrage_free_guaranteed
    );
    assert!(snapshot.timings.total_ms.is_finite());
    assert!(!snapshot.input_sha256.is_empty());
    let json = serde_json::to_value(snapshot).unwrap();
    assert_eq!(json["cube"]["generation"], 1);
    assert_eq!(json["currency"], "USD");
}

#[test]
fn invalid_common_context_and_convention_are_rejected_and_overflow_never_publishes_infinity() {
    for bump in [0.0, f64::NAN, 0.2] {
        assert!(
            LiveRiskEngine::new(LiveRiskConfig {
                convention: GreekConvention::StickyLogMoneynessFiniteBump {
                    relative_spot_bump: bump
                },
                ..Default::default()
            })
            .is_err()
        );
    }
    let mut invalid = frame(1);
    for q in &mut invalid.quotes {
        q.input.spot = f64::NAN;
    }
    let mut engine = LiveRiskEngine::new(LiveRiskConfig::default()).unwrap();
    assert!(engine.update(invalid).is_err());
    let mut invalid = frame(1);
    for q in &mut invalid.quotes {
        q.underlying_id.clear();
    }
    assert!(engine.update(invalid).is_err());
    let mut huge = frame(1);
    for q in &mut huge.quotes {
        q.input.spot *= 1e305;
        q.input.strike *= 1e305;
        let p = price(&q.input, 0.2);
        assert!(p.is_finite());
        q.bid = p;
        q.ask = p;
    }
    huge.positions[0].quantity = 1_000_000_000;
    huge.positions[0].multiplier = 1_000_000;
    let snapshot = engine.update(huge).unwrap();
    assert!(!snapshot.portfolio.complete);
    let row = snapshot
        .rows
        .iter()
        .find(|r| r.position_units != 0.0)
        .unwrap();
    assert!(row.greeks.is_none());
    assert!(row.calculation_error.is_some());
    assert!(row.position_value.is_none());
    assert!(
        snapshot
            .rows
            .iter()
            .flat_map(|r| [r.position_value, r.position_delta, r.position_vega])
            .flatten()
            .all(f64::is_finite)
    );
    assert!(snapshot.cube.values.iter().all(|c| c.value.is_finite()));
    assert!(serde_json::to_string(&snapshot).is_ok());
}
#[test]
fn stale_cutoff_or_changed_book_without_new_revision_cannot_replace_valid_generation() {
    let mut engine = LiveRiskEngine::new(LiveRiskConfig::default()).unwrap();
    engine.update(frame(1)).unwrap();
    let mut old = frame(2);
    old.cutoff_ns -= 1;
    assert!(engine.update(old).is_err());
    let mut book = frame(2);
    book.positions[0].quantity += 1;
    assert!(engine.update(book).is_err());
    let accepted = engine.update(frame(2)).unwrap();
    assert_eq!(accepted.generation, 2);
    assert!(accepted.portfolio.complete);
}

#[test]
fn zero_scenario_shocks_have_exact_zero_pnl_even_for_large_finite_positions() {
    for volatility in [0.12, 0.2, 0.37, 0.7] {
        let config = LiveRiskConfig {
            relative_spot_scenario: 0.0,
            volatility_scenario: 0.0,
            ..Default::default()
        };
        let mut f = frame(1);
        for q in &mut f.quotes {
            let p = price(&q.input, volatility);
            q.bid = p - 1e-5;
            q.ask = p + 1e-5;
        }
        f.positions[0].quantity = 1_000_000_000;
        f.positions[0].multiplier = 1_000_000;
        let snapshot = LiveRiskEngine::new(config).unwrap().update(f).unwrap();
        assert!(snapshot.portfolio.complete);
        for s in &snapshot.scenarios {
            assert_eq!(s.pnl, Some(0.0), "sigma={volatility}");
            assert_eq!(s.local_pnl, Some(0.0));
        }
    }
}

#[test]
fn declared_byte_cap_handles_raw_and_escaped_identity_growth_without_consuming_generation() {
    let config = LiveRiskConfig {
        max_request_bytes: 16 * 1024,
        max_quotes: 100_000,
        ..Default::default()
    };
    let mut engine = LiveRiskEngine::new(config).unwrap();
    let mut raw = frame(1);
    raw.quotes[0].source_id = "x".repeat(17 * 1024);
    assert!(engine.update(raw).is_err());
    // Raw identities fit the cap, but JSON escaping expands this field6x.
    // The capped streaming serializer must refuse the encoded frame too.
    let mut escaped = frame(1);
    escaped.quotes[0].source_id = "\0".repeat(4000);
    assert!(engine.update(escaped).is_err());
    let snapshot = engine.update(frame(1)).unwrap();
    assert!(snapshot.portfolio.complete);
    // A high admitted quote limit must not invalidate the fit's diagnostic cap
    // when the actual frame is small.
    assert!(snapshot.surface.is_some());
    assert!(snapshot.surface_error.is_none());
}

#[test]
fn rejected_publication_rolls_back_generation_cutoff_and_book_before_same_generation_retry() {
    let mut engine = LiveRiskEngine::new(LiveRiskConfig::default()).unwrap();
    let original = engine.update(frame(1)).unwrap();
    let mut rejected = frame(2);
    rejected.cutoff_ns += 2_000_000_000;
    rejected.position_revision = 3;
    rejected.positions[0].quantity = -8;
    let mut called = false;
    let error = engine
        .update_and_publish(rejected, |candidate| {
            called = true;
            assert_eq!(candidate.generation, 2);
            assert_eq!(candidate.cube.generation, 2);
            assert_eq!(candidate.position_revision, 3);
            Err(volt_compute::ComputeError::Execution(
                "deliberate publication failure".into(),
            ))
        })
        .unwrap_err();
    assert!(called);
    assert!(error.to_string().contains("deliberate publication"));
    // This valid retry has a cutoff and revision older than the failed candidate,
    // but newer than the last successful frame. Every outer guard must roll back.
    let mut retry = frame(2);
    retry.cutoff_ns += 1_000_000_000;
    retry.position_revision = 2;
    retry.positions[0].quantity = -9;
    let snapshot = engine
        .update_and_publish(retry, |candidate| {
            assert_eq!(candidate.position_revision, 2);
            Ok(())
        })
        .unwrap();
    assert_eq!(snapshot.generation, 2);
    assert_eq!(snapshot.cube.generation, 2);
    assert_eq!(snapshot.position_revision, 2);
    close(
        snapshot.portfolio.value.unwrap(),
        0.9 * original.portfolio.value.unwrap(),
        1e-8,
    );
    assert_ne!(snapshot.positions_sha256, original.positions_sha256);
    assert!(engine.update(frame(1)).is_err());
}
#[test]
fn fitted_greek_cube_lineage_uses_older_surface_contributor_without_changing_own_quote_age() {
    let mut f = frame(1);
    f.quotes[0].timestamp_ns -= 1_000_000_000;
    let own_time = f.quotes[1].timestamp_ns;
    let oldest = f.quotes[0].timestamp_ns;
    let snapshot = LiveRiskEngine::new(LiveRiskConfig::default())
        .unwrap()
        .update(f)
        .unwrap();
    assert!(snapshot.portfolio.complete);
    assert_eq!(snapshot.rows[1].pricing_basis, "surface");
    let key = &snapshot.rows[1].quote.quote.contract_id;
    let greek = snapshot
        .cube
        .values
        .iter()
        .find(|c| c.node == "delta" && &c.key == key)
        .unwrap();
    assert_eq!(greek.observed_at_ms, oldest / 1_000_000);
    assert!(greek.observed_at_ms < own_time / 1_000_000);
    assert_eq!(snapshot.rows[1].quote.quote.timestamp_ns, own_time);
    close(snapshot.cube.value("age_seconds", key).unwrap(), 0.0, 0.0);
}
