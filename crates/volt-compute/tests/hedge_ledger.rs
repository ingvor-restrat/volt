use std::collections::BTreeMap;

use volt::{Currency, PriceMicros, TimestampNs, UnderlyingId};
use volt_compute::hedge_ledger::*;

const T0: TimestampNs = TimestampNs(100);
const T1: TimestampNs = TimestampNs(200);
const END: TimestampNs = TimestampNs(300);
const DOLLAR_PICOS: i128 = 1_000_000_000_000;

fn config() -> HedgeLedgerConfig {
    HedgeLedgerConfig {
        currency: Currency("USD".into()),
        terminal_time: END,
        instruments: vec![
            HedgeInstrument {
                instrument_id: "stock".into(),
                currency: Currency("USD".into()),
                kind: HedgeInstrumentKind::Stock {
                    underlying: UnderlyingId("XYZ".into()),
                },
                quantity_step: QuantityMicros(1),
                multiplier: 1,
            },
            HedgeInstrument {
                instrument_id: "variance".into(),
                currency: Currency("USD".into()),
                kind: HedgeInstrumentKind::FundedVarianceClaim {
                    underlying: UnderlyingId("XYZ".into()),
                    maturity: END,
                },
                quantity_step: QuantityMicros(1),
                multiplier: 1,
            },
        ],
    }
}
fn market(id: &str, at: TimestampNs, stock: i64, variance: i64) -> HedgeMarketSnapshot {
    HedgeMarketSnapshot {
        snapshot_id: id.into(),
        as_of: at,
        currency: Currency("USD".into()),
        prices: BTreeMap::from([
            ("stock".into(), PriceMicros(stock)),
            ("variance".into(), PriceMicros(variance)),
        ]),
    }
}
fn targets(stock: i64, variance: i64) -> BTreeMap<String, QuantityMicros> {
    BTreeMap::from([
        ("stock".into(), QuantityMicros(stock)),
        ("variance".into(), QuantityMicros(variance)),
    ])
}
fn ledger(cash_dollars: i128) -> HedgeLedger {
    HedgeLedger::new(
        config(),
        market("inception", T0, 100_000_000, 4_000_000),
        cash_dollars * 1_000_000,
    )
    .unwrap()
}
fn liability(dollars: i128) -> LiabilityCashflow {
    LiabilityCashflow {
        settlement_id: "claim-settlement".into(),
        currency: Currency("USD".into()),
        amount_micros: dollars * 1_000_000,
    }
}

#[test]
fn stock_replication_is_self_financing_and_terminal_liability_settles_once() {
    let mut book = ledger(100);
    book.rebalance(
        market("buy", T0, 100_000_000, 4_000_000),
        targets(1_000_000, 0),
        IntervalAccrual::zero_rate(T0, T0),
        ProportionalCost::ZERO,
    )
    .unwrap();
    assert_eq!(book.state().cash_picos, 0);
    book.rebalance(
        market("hold", T1, 110_000_000, 5_000_000),
        targets(1_000_000, 0),
        IntervalAccrual::zero_rate(T0, T1),
        ProportionalCost::ZERO,
    )
    .unwrap();
    let terminal = book
        .settle(
            market("terminal", END, 120_000_000, 6_000_000),
            liability(120),
            IntervalAccrual::zero_rate(T1, END),
            ProportionalCost::ZERO,
        )
        .unwrap();
    assert_eq!(terminal.after.net_wealth_picos, 0);
    assert_eq!(terminal.after.cumulative_gain_picos, 20 * DOLLAR_PICOS);
    assert!(terminal.after.settled);
    assert_eq!(terminal.trades[0].change, QuantityMicros(-1_000_000));
    assert!(
        book.settle(
            market("duplicate", END, 120_000_000, 6_000_000),
            liability(120),
            IntervalAccrual::zero_rate(END, END),
            ProportionalCost::ZERO
        )
        .is_err()
    );
}

#[test]
fn long_short_fractional_trades_cost_absolute_notional_and_apply_multiplier_once() {
    let mut cfg = config();
    cfg.instruments[0].multiplier = 100;
    let mut book = HedgeLedger::new(cfg, market("initial", T0, 100_000_000, 4_000_000), 0).unwrap();
    let cost = ProportionalCost {
        parts_per_million: 10_000,
    };
    let e = book
        .rebalance(
            market("open", T0, 100_000_000, 4_000_000),
            targets(-500_000, 2_500_000),
            IntervalAccrual::zero_rate(T0, T0),
            cost,
        )
        .unwrap();
    assert_eq!(e.trades[0].notional_picos, -5_000 * DOLLAR_PICOS);
    assert_eq!(
        e.after.cash_picos,
        (5_000 - 10) * DOLLAR_PICOS - 50_100_000_000_000
    );
    assert_eq!(e.after.net_wealth_picos, -50_100_000_000_000);
    let end = book
        .settle(
            market("end", END, 90_000_000, 6_000_000),
            liability(0),
            IntervalAccrual::zero_rate(T0, END),
            cost,
        )
        .unwrap();
    assert_eq!(end.after.cumulative_gain_picos, 505 * DOLLAR_PICOS);
    // Entry notionals 5000+10; exit notionals 4500+15, all charged positively.
    assert_eq!(end.after.cumulative_cost_picos, 95_250_000_000_000);
    assert_eq!(end.after.cash_picos, 409_750_000_000_000);
    assert_eq!(end.after.market_value_picos, 0);
}

#[test]
fn lending_borrowing_and_negative_rates_use_declared_interval_factors() {
    for (initial, lending, borrowing, expected) in [
        (
            100,
            AccrualFactor {
                numerator: 101,
                denominator: 100,
            },
            AccrualFactor {
                numerator: 105,
                denominator: 100,
            },
            101,
        ),
        (
            -100,
            AccrualFactor {
                numerator: 101,
                denominator: 100,
            },
            AccrualFactor {
                numerator: 105,
                denominator: 100,
            },
            -105,
        ),
        (
            100,
            AccrualFactor {
                numerator: 99,
                denominator: 100,
            },
            AccrualFactor::ONE,
            99,
        ),
    ] {
        let mut book = ledger(initial);
        let a = IntervalAccrual {
            source_id: "declared-flat-factor".into(),
            from: T0,
            to: END,
            lending,
            borrowing,
        };
        let e = book
            .settle(
                market("end", END, 100_000_000, 4_000_000),
                liability(0),
                a,
                ProportionalCost::ZERO,
            )
            .unwrap();
        assert_eq!(e.after.cash_picos, expected * DOLLAR_PICOS);
        assert_eq!(e.interest_picos, (expected - initial) * DOLLAR_PICOS);
    }
}

#[test]
fn round_ties_away_from_zero_and_keep_both_fee_and_financing_residuals() {
    let mut book = ledger(0);
    let fee = ProportionalCost {
        parts_per_million: 500_000,
    };
    let e = book
        .rebalance(
            market("one-pico", T0, 1, 0),
            targets(1, 0),
            IntervalAccrual::zero_rate(T0, T0),
            fee,
        )
        .unwrap();
    assert_eq!(e.trades[0].notional_picos, 1);
    assert_eq!(e.trades[0].cost_picos, 1);
    assert_eq!(
        e.trades[0].cost_rounding,
        RationalResidual {
            numerator: -500_000,
            denominator: 1_000_000
        }
    );
    assert_eq!(e.after.cash_picos, -2);
    let a = IntervalAccrual {
        source_id: "three-quarters".into(),
        from: T0,
        to: T1,
        lending: AccrualFactor::ONE,
        borrowing: AccrualFactor {
            numerator: 3,
            denominator: 4,
        },
    };
    let e = book
        .rebalance(
            market("negative-tie", T1, 1, 0),
            targets(1, 0),
            a,
            ProportionalCost::ZERO,
        )
        .unwrap();
    assert_eq!(e.after.cash_picos, -2);
    assert_eq!(
        e.financing_rounding,
        RationalResidual {
            numerator: 2,
            denominator: 4
        }
    );
    // Exact numerator -6 = rounded -2 * 4 + residual 2.
    assert_eq!(
        book.cash_micros().unwrap(),
        (
            0,
            RationalResidual {
                numerator: -2,
                denominator: 1_000_000
            }
        )
    );
}

#[test]
fn explicit_micro_cash_conversion_reports_rounding_including_signed_ties() {
    for qty in [-500_000, 500_000] {
        let mut book = ledger(0);
        book.rebalance(
            market("tiny", T0, 1, 0),
            targets(qty, 0),
            IntervalAccrual::zero_rate(T0, T0),
            ProportionalCost::ZERO,
        )
        .unwrap();
        let (micros, residual) = book.cash_micros().unwrap();
        assert_eq!(micros, -i128::from(qty.signum()));
        assert_eq!(
            micros * PICOS_PER_MICRO + residual.numerator,
            book.state().cash_picos
        );
    }
}

#[test]
fn invalid_events_are_transactional_and_whole_snapshots_are_required() {
    let mut book = ledger(100);
    book.rebalance(
        market("first", T0, 100_000_000, 4_000_000),
        targets(1_000_000, 0),
        IntervalAccrual::zero_rate(T0, T0),
        ProportionalCost::ZERO,
    )
    .unwrap();
    let before = book.state().clone();
    let previous = book.journal().to_vec();
    let mut missing = market("bad", T1, 100_000_000, 4_000_000);
    missing.prices.remove("variance");
    let mut extra = market("bad", T1, 100_000_000, 4_000_000);
    extra.prices.insert("other".into(), PriceMicros(1));
    let mut currency = market("bad", T1, 100_000_000, 4_000_000);
    currency.currency = Currency("EUR".into());
    let bad_mark = market("bad", T1, 100_000_000, -1);
    for snapshot in [
        missing,
        extra,
        currency,
        bad_mark,
        market("first", T1, 100_000_000, 4_000_000),
        market("equal", T0, 100_000_000, 4_000_000),
    ] {
        assert!(
            book.rebalance(
                snapshot,
                targets(0, 0),
                IntervalAccrual::zero_rate(T0, T1),
                ProportionalCost::ZERO
            )
            .is_err()
        );
        assert_eq!(*book.state(), before);
        assert_eq!(book.journal(), previous);
    }
    let mut partial = targets(0, 0);
    partial.remove("stock");
    assert!(
        book.rebalance(
            market("missing-target", T1, 100_000_000, 4_000_000),
            partial,
            IntervalAccrual::zero_rate(T0, T1),
            ProportionalCost::ZERO
        )
        .is_err()
    );
    let wrong_interval = IntervalAccrual::zero_rate(TimestampNs(101), T1);
    assert!(
        book.rebalance(
            market("bad-interval", T1, 100_000_000, 4_000_000),
            targets(0, 0),
            wrong_interval,
            ProportionalCost::ZERO
        )
        .is_err()
    );
    assert_eq!(*book.state(), before);
    assert_eq!(book.journal(), previous);
}

#[test]
fn rejects_invalid_lots_currencies_maturities_prices_and_empty_identities() {
    let mut cases = Vec::new();
    let mut c = config();
    c.instruments[0].quantity_step = QuantityMicros(0);
    cases.push(c);
    let mut c = config();
    c.instruments[0].multiplier = 0;
    cases.push(c);
    let mut c = config();
    c.instruments[0].currency = Currency("EUR".into());
    cases.push(c);
    let mut c = config();
    c.instruments[0].instrument_id = " ".into();
    cases.push(c);
    let mut c = config();
    c.instruments.push(c.instruments[0].clone());
    cases.push(c);
    let mut c = config();
    c.instruments[1].kind = HedgeInstrumentKind::FundedVarianceClaim {
        underlying: UnderlyingId("XYZ".into()),
        maturity: T1,
    };
    cases.push(c);
    for c in cases {
        assert!(HedgeLedger::new(c, market("init", T0, 100_000_000, 4_000_000), 0).is_err());
    }
    let mut c = config();
    c.instruments[0].quantity_step = QuantityMicros(QUANTITY_SCALE);
    let mut book = HedgeLedger::new(c, market("initial", T0, 100_000_000, 4_000_000), 0).unwrap();
    assert!(
        book.rebalance(
            market("fraction", T0, 100_000_000, 4_000_000),
            targets(500_000, 0),
            IntervalAccrual::zero_rate(T0, T0),
            ProportionalCost::ZERO
        )
        .is_err()
    );
}

#[test]
fn overflow_returns_error_without_any_state_or_journal_mutation() {
    assert!(HedgeLedger::new(config(), market("initial", T0, 1, 0), i128::MAX).is_err());
    let mut c = config();
    c.instruments[0].multiplier = u32::MAX;
    let mut book = HedgeLedger::new(c, market("initial", T0, i64::MAX, 0), 0).unwrap();
    let before = book.state().clone();
    assert!(
        book.rebalance(
            market("too-large", T0, i64::MAX, 0),
            targets(i64::MAX, 0),
            IntervalAccrual::zero_rate(T0, T0),
            ProportionalCost::ZERO
        )
        .is_err()
    );
    assert_eq!(*book.state(), before);
    assert!(book.journal().is_empty());
    let mut book = ledger(100);
    let a = IntervalAccrual {
        source_id: "overflow-financing".into(),
        from: T0,
        to: T1,
        lending: AccrualFactor {
            numerator: u64::MAX,
            denominator: 1,
        },
        borrowing: AccrualFactor::ONE,
    };
    // First multiplication fits i128, but a second extreme factor overflows.
    book.rebalance(
        market("large", T1, 1, 0),
        targets(0, 0),
        a,
        ProportionalCost::ZERO,
    )
    .unwrap();
    let before = book.state().clone();
    let a = IntervalAccrual {
        source_id: "overflow".into(),
        from: T1,
        to: END,
        lending: AccrualFactor {
            numerator: u64::MAX,
            denominator: 1,
        },
        borrowing: AccrualFactor::ONE,
    };
    assert!(
        book.settle(
            market("too-large", END, 1, 0),
            liability(0),
            a,
            ProportionalCost::ZERO
        )
        .is_err()
    );
    assert_eq!(*book.state(), before);
    assert_eq!(book.journal().len(), 1);
}

#[test]
fn deterministic_journal_replays_exactly_and_retains_cashflow_receipts() {
    let initial = market("initial", T0, 100_000_000, 4_000_000);
    let mut first = HedgeLedger::new(config(), initial.clone(), 20_000_000).unwrap();
    first
        .rebalance(
            market("open", T0, 100_000_000, 4_000_000),
            targets(123_456, -876_543),
            IntervalAccrual::zero_rate(T0, T0),
            ProportionalCost {
                parts_per_million: 123,
            },
        )
        .unwrap();
    first
        .rebalance(
            market("resize", T1, 105_000_003, 4_765_432),
            targets(-222_222, 333_333),
            IntervalAccrual::zero_rate(T0, T1),
            ProportionalCost {
                parts_per_million: 123,
            },
        )
        .unwrap();
    first
        .settle(
            market("terminal", END, 111_000_007, 5_555_555),
            liability(-7),
            IntervalAccrual::zero_rate(T1, END),
            ProportionalCost::ZERO,
        )
        .unwrap();
    let mut second = HedgeLedger::new(config(), initial, 20_000_000).unwrap();
    for e in first.journal() {
        if let Some(flow) = &e.liability {
            second
                .settle(
                    e.market.clone(),
                    flow.clone(),
                    e.accrual.clone(),
                    e.cost_model,
                )
                .unwrap();
        } else {
            second
                .rebalance(
                    e.market.clone(),
                    e.after.holdings.clone(),
                    e.accrual.clone(),
                    e.cost_model,
                )
                .unwrap();
        }
    }
    assert_eq!(first.state(), second.state());
    assert_eq!(first.journal(), second.journal());
    assert_eq!(first.state().cumulative_liability_picos, -7 * DOLLAR_PICOS);
    let encoded = serde_json::to_string(first.journal()).unwrap();
    let decoded: Vec<HedgeJournalEntry> = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, first.journal());
    // Net P&L is checked independently from terminal cash for every event.
    for e in first.journal() {
        assert_eq!(
            e.after.net_wealth_picos,
            20 * DOLLAR_PICOS + e.after.cumulative_gain_picos + e.after.cumulative_interest_picos
                - e.after.cumulative_cost_picos
                - e.after.cumulative_liability_picos
        );
    }
}

#[test]
fn rejects_wrong_terminal_time_liability_currency_and_nonunit_zero_interval() {
    let mut book = ledger(0);
    assert!(
        book.settle(
            market("early", T1, 1, 0),
            liability(0),
            IntervalAccrual::zero_rate(T0, T1),
            ProportionalCost::ZERO
        )
        .is_err()
    );
    let mut flow = liability(0);
    flow.currency = Currency("EUR".into());
    assert!(
        book.settle(
            market("currency", END, 1, 0),
            flow,
            IntervalAccrual::zero_rate(T0, END),
            ProportionalCost::ZERO
        )
        .is_err()
    );
    let mut a = IntervalAccrual::zero_rate(T0, T0);
    a.lending = AccrualFactor {
        numerator: 2,
        denominator: 1,
    };
    assert!(
        book.rebalance(
            market("same-time-interest", T0, 1, 0),
            targets(0, 0),
            a,
            ProportionalCost::ZERO
        )
        .is_err()
    );
    assert!(
        book.rebalance(
            market("terminal-trade", END, 1, 0),
            targets(0, 0),
            IntervalAccrual::zero_rate(T0, END),
            ProportionalCost::ZERO
        )
        .is_err()
    );
    assert!(book.journal().is_empty());
}

#[test]
fn strict_wire_and_large_decimal_cash_preserve_exact_amounts() {
    let encoded = serde_json::to_string(&config()).unwrap();
    let mut config_value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    config_value["quote_convention"] = serde_json::json!("zero_value_swap");
    assert!(serde_json::from_value::<HedgeLedgerConfig>(config_value).is_err());
    assert!(
        serde_json::from_str::<ProportionalCost>(
            r#"{"parts_per_million":10,"basis":"signed_notional"}"#
        )
        .is_err()
    );
    let large = 12_345_678_901_234_567_890_123_i128;
    let book = HedgeLedger::new(config(), market("initial", T0, 1, 0), large).unwrap();
    let state_json = serde_json::to_value(book.state()).unwrap();
    assert_eq!(
        state_json["cash_picos"].as_str().unwrap(),
        (large * PICOS_PER_MICRO).to_string()
    );
    let decoded: HedgeLedgerState = serde_json::from_value(state_json).unwrap();
    assert_eq!(&decoded, book.state());
    assert!(HedgeLedger::new(config(), market("zero-stock", T0, 0, 0), 0).is_err());
    for large in [i128::MAX / PICOS_PER_MICRO, i128::MIN / PICOS_PER_MICRO] {
        let mut book = HedgeLedger::new(config(), market("initial", T0, 1, 0), large).unwrap();
        book.settle(
            market("end", END, 1, 0),
            liability(0),
            IntervalAccrual::zero_rate(T0, END),
            ProportionalCost::ZERO,
        )
        .unwrap();
        let wire = serde_json::to_value(book.journal()).unwrap();
        let decoded: Vec<HedgeJournalEntry> = serde_json::from_value(wire).unwrap();
        assert_eq!(decoded, book.journal());
    }
    for quantity in [i64::MAX, i64::MIN] {
        let wire = serde_json::to_value(QuantityMicros(quantity)).unwrap();
        assert_eq!(wire.as_str().unwrap(), quantity.to_string());
        assert_eq!(
            serde_json::from_value::<QuantityMicros>(wire).unwrap(),
            QuantityMicros(quantity)
        );
    }
    assert!(serde_json::from_str::<QuantityMicros>(r#""9223372036854775808""#).is_err());
}

#[test]
fn failed_event_does_not_reserve_snapshot_id_or_inception_trade() {
    let mut book = ledger(0);
    let snapshot = market("retry", T0, 1, 0);
    let mut partial = targets(0, 0);
    partial.remove("variance");
    assert!(
        book.rebalance(
            snapshot.clone(),
            partial,
            IntervalAccrual::zero_rate(T0, T0),
            ProportionalCost::ZERO
        )
        .is_err()
    );
    book.rebalance(
        snapshot,
        targets(1, 0),
        IntervalAccrual::zero_rate(T0, T0),
        ProportionalCost::ZERO,
    )
    .unwrap();
    assert_eq!(book.journal().len(), 1);
    assert_eq!(book.state().cash_picos, -1);
}
