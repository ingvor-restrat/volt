use std::collections::BTreeMap;

use volt::{
    BarInterval, BookLevel, CanonicalPortfolio, CogError, CogRuntime, ContractId,
    ContractLifecycle, Currency, Emission, ExerciseSchedule, GraphContext, GraphNodePayload,
    GraphTarget, InMemoryOptionMarketDataSource, MarkBasis, MarketObservation, ObservationOrigin,
    OptionBar, OptionContract, OptionDataSourceManifest, OptionMarketDataSource,
    OptionMarketObservation, OptionObservationPayload, OptionQuote, OptionRight,
    PortfolioGraphSample, Position, PriceAuthority, PriceMicros, Settlement, TimestampNs,
    TransitionAction, TransitionInput, UnderlyingId,
};

const OPEN: TimestampNs = TimestampNs(10);
const EXPIRY: TimestampNs = TimestampNs(20);

fn contract(
    id: &str,
    right: OptionRight,
    strike_dollars: i64,
    exercise: ExerciseSchedule,
    settlement: Settlement,
) -> OptionContract {
    OptionContract::new(
        id,
        "SPY",
        "USD",
        right,
        PriceMicros(strike_dollars * 1_000_000),
        100,
        exercise,
        settlement,
    )
    .unwrap()
}

fn american(
    id: &str,
    right: OptionRight,
    strike_dollars: i64,
    settlement: Settlement,
) -> OptionContract {
    contract(
        id,
        right,
        strike_dollars,
        ExerciseSchedule::American {
            first_exercise: OPEN,
            expiry: EXPIRY,
        },
        settlement,
    )
}

fn observation(event_id: &str, time: i64, spot_dollars: i64) -> MarketObservation {
    MarketObservation {
        event_id: event_id.to_owned(),
        time: TimestampNs(time),
        spot: PriceMicros(spot_dollars * 1_000_000),
    }
}

#[test]
fn canonical_portfolio_is_order_independent_and_cancels_offsets() {
    let call_100 = american("SPY-C-100", OptionRight::Call, 100, Settlement::Physical);
    let call_110 = american("SPY-C-110", OptionRight::Call, 110, Settlement::Physical);

    let first = CanonicalPortfolio::from_positions(vec![
        Position::new(call_110.clone(), -1).unwrap(),
        Position::new(call_100.clone(), 2).unwrap(),
        Position::new(call_100.clone(), -1).unwrap(),
        Position::new(call_110.clone(), 1).unwrap(),
    ])
    .unwrap();
    let second =
        CanonicalPortfolio::from_positions(vec![Position::new(call_100, 1).unwrap()]).unwrap();

    assert_eq!(first, second);
    assert_eq!(
        first.semantic_hash().unwrap(),
        second.semantic_hash().unwrap()
    );
}

#[test]
fn hinge_projection_matches_direct_payoff_for_calls_and_puts() {
    let portfolio = CanonicalPortfolio::from_positions(vec![
        Position::new(
            american("SPY-P-95", OptionRight::Put, 95, Settlement::Physical),
            2,
        )
        .unwrap(),
        Position::new(
            american("SPY-C-100", OptionRight::Call, 100, Settlement::Physical),
            1,
        )
        .unwrap(),
        Position::new(
            american("SPY-C-110", OptionRight::Call, 110, Settlement::Physical),
            -1,
        )
        .unwrap(),
    ])
    .unwrap();
    let projection = portfolio.terminal_payoff_projection().unwrap();

    for spot_dollars in 0..=200 {
        let spot = PriceMicros(spot_dollars * 1_000_000);
        assert_eq!(
            projection.evaluate(spot).unwrap(),
            portfolio.direct_terminal_payoff(spot).unwrap(),
            "payoff differs at spot {spot_dollars}"
        );
    }
}

#[test]
fn terminal_equivalence_does_not_erase_execution_identity() {
    let a = american("VENUE-A-C100", OptionRight::Call, 100, Settlement::Physical);
    let b = american("VENUE-B-C100", OptionRight::Call, 100, Settlement::Physical);
    let portfolio_a =
        CanonicalPortfolio::from_positions(vec![Position::new(a, 1).unwrap()]).unwrap();
    let portfolio_b =
        CanonicalPortfolio::from_positions(vec![Position::new(b, 1).unwrap()]).unwrap();

    assert_ne!(
        portfolio_a.semantic_hash().unwrap(),
        portfolio_b.semantic_hash().unwrap()
    );
    assert_eq!(
        portfolio_a
            .terminal_payoff_projection()
            .unwrap()
            .semantic_hash()
            .unwrap(),
        portfolio_b
            .terminal_payoff_projection()
            .unwrap()
            .semantic_hash()
            .unwrap()
    );
}

#[test]
fn nonempty_option_portfolio_can_have_zero_terminal_payoff_projection() {
    // At strikes 90, 100, 110 choose put weights (1, -2, 1) and
    // opposite call weights. The put quantities sum to zero and their
    // strike-weighted sum is zero, so both the linear and cash terms cancel.
    let positions = [
        ("P90", OptionRight::Put, 90, 1),
        ("P100", OptionRight::Put, 100, -2),
        ("P110", OptionRight::Put, 110, 1),
        ("C90", OptionRight::Call, 90, -1),
        ("C100", OptionRight::Call, 100, 2),
        ("C110", OptionRight::Call, 110, -1),
    ]
    .into_iter()
    .map(|(id, right, strike, quantity)| {
        Position::new(american(id, right, strike, Settlement::Physical), quantity).unwrap()
    })
    .collect::<Vec<_>>();
    let portfolio = CanonicalPortfolio::from_positions(positions).unwrap();
    let projection = portfolio.terminal_payoff_projection().unwrap();

    assert!(!portfolio.positions.is_empty());
    assert_eq!(projection.cash_micros, 0);
    assert_eq!(projection.underlying_units, 0);
    assert!(projection.call_knots.is_empty());
    for spot_dollars in 0..=200 {
        assert_eq!(
            portfolio
                .direct_terminal_payoff(PriceMicros(spot_dollars * 1_000_000))
                .unwrap(),
            0
        );
    }
}

#[test]
fn european_option_rejects_early_exercise_and_settles_at_expiry() {
    let european = contract(
        "SPX-C-100",
        OptionRight::Call,
        100,
        ExerciseSchedule::European { expiry: EXPIRY },
        Settlement::Cash,
    );
    let state =
        volt::PortfolioState::from_positions(vec![Position::new(european, 1).unwrap()]).unwrap();
    let runtime = CogRuntime;
    let action = TransitionAction::Exercise {
        contract_id: ContractId("SPX-C-100".to_owned()),
        quantity: 1,
    };

    let early = runtime.transition(&state, &action, &observation("early", 19, 110));
    assert!(matches!(early, Err(CogError::InvalidAction(_))));

    let settled = runtime
        .transition(&state, &action, &observation("expiry", 20, 110))
        .unwrap();
    assert_eq!(
        settled.state.cash_micros.get(&Currency("USD".to_owned())),
        Some(&1_000_000_000)
    );
    assert_eq!(
        settled.state.positions[&ContractId("SPX-C-100".to_owned())].lifecycle,
        ContractLifecycle::Exercised
    );
}

#[test]
fn american_physical_call_exercise_creates_stock_and_strike_cash_postings() {
    let call = american("SPY-C-100", OptionRight::Call, 100, Settlement::Physical);
    let state =
        volt::PortfolioState::from_positions(vec![Position::new(call, 1).unwrap()]).unwrap();
    let result = CogRuntime
        .transition(
            &state,
            &TransitionAction::Exercise {
                contract_id: ContractId("SPY-C-100".to_owned()),
                quantity: 1,
            },
            &observation("pre_dividend_exercise", 15, 120),
        )
        .unwrap();

    assert_eq!(
        result
            .state
            .underlying_units
            .get(&UnderlyingId("SPY".to_owned())),
        Some(&100)
    );
    assert_eq!(
        result.state.cash_micros.get(&Currency("USD".to_owned())),
        Some(&-10_000_000_000)
    );
    assert!(result
        .journal_entry
        .emissions
        .iter()
        .any(|emission| matches!(emission, Emission::RightConsumed { quantity: 1, .. })));
}

#[test]
fn short_put_assignment_is_external_and_can_be_partial() {
    let put = american("SPY-P-100", OptionRight::Put, 100, Settlement::Physical);
    let state =
        volt::PortfolioState::from_positions(vec![Position::new(put, -2).unwrap()]).unwrap();
    let runtime = CogRuntime;

    let exercise_attempt = runtime.transition(
        &state,
        &TransitionAction::Exercise {
            contract_id: ContractId("SPY-P-100".to_owned()),
            quantity: 1,
        },
        &observation("invalid_holder_action", 15, 90),
    );
    assert!(matches!(exercise_attempt, Err(CogError::InvalidAction(_))));

    let assigned = runtime
        .transition(
            &state,
            &TransitionAction::Assign {
                contract_id: ContractId("SPY-P-100".to_owned()),
                quantity: 1,
            },
            &observation("counterparty_assignment", 15, 90),
        )
        .unwrap();
    let position = &assigned.state.positions[&ContractId("SPY-P-100".to_owned())];
    assert_eq!(position.open_quantity, -1);
    assert_eq!(position.lifecycle, ContractLifecycle::Live);
    assert_eq!(
        assigned
            .state
            .underlying_units
            .get(&UnderlyingId("SPY".to_owned())),
        Some(&100)
    );
    assert_eq!(
        assigned.state.cash_micros.get(&Currency("USD".to_owned())),
        Some(&-10_000_000_000)
    );
}

#[test]
fn replay_reproduces_state_and_journal_hashes() {
    let call = american("SPY-C-100", OptionRight::Call, 100, Settlement::Physical);
    let initial =
        volt::PortfolioState::from_positions(vec![Position::new(call, 1).unwrap()]).unwrap();
    let inputs = vec![
        TransitionInput {
            action: TransitionAction::Hold,
            observation: observation("mark", 12, 105),
        },
        TransitionInput {
            action: TransitionAction::Exercise {
                contract_id: ContractId("SPY-C-100".to_owned()),
                quantity: 1,
            },
            observation: observation("exercise", 15, 120),
        },
    ];
    let runtime = CogRuntime;
    let first = runtime.replay(&initial, &inputs).unwrap();
    let second = runtime.replay(&initial, &inputs).unwrap();

    assert_eq!(first, second);
    assert_eq!(
        first.state.state_hash().unwrap(),
        second.state.state_hash().unwrap()
    );
    assert_eq!(
        first
            .journal
            .iter()
            .map(|entry| entry.entry_hash.as_str())
            .collect::<Vec<_>>(),
        second
            .journal
            .iter()
            .map(|entry| entry.entry_hash.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn expiration_requires_expiry_and_consumes_remaining_rights_without_settlement() {
    let call = american("SPY-C-100", OptionRight::Call, 100, Settlement::Physical);
    let state =
        volt::PortfolioState::from_positions(vec![Position::new(call, 1).unwrap()]).unwrap();
    let action = TransitionAction::Expire {
        contract_id: ContractId("SPY-C-100".to_owned()),
    };
    let runtime = CogRuntime;

    assert!(matches!(
        runtime.transition(&state, &action, &observation("too_early", 19, 90)),
        Err(CogError::InvalidAction(_))
    ));
    let expired = runtime
        .transition(&state, &action, &observation("expiry", 20, 90))
        .unwrap();
    assert_eq!(
        expired.state.positions[&ContractId("SPY-C-100".to_owned())].lifecycle,
        ContractLifecycle::Expired
    );
    assert!(expired.state.cash_micros.is_empty());
    assert!(expired.state.underlying_units.is_empty());
}

#[test]
fn graph_export_is_deterministic_and_keeps_both_equivalence_hashes() {
    let call = american("SPY-C-100", OptionRight::Call, 100, Settlement::Physical);
    let put = american("SPY-P-100", OptionRight::Put, 100, Settlement::Physical);
    let portfolio = CanonicalPortfolio::from_positions(vec![
        Position::new(put.clone(), -1).unwrap(),
        Position::new(call.clone(), 2).unwrap(),
    ])
    .unwrap();
    let context = GraphContext {
        underlying: UnderlyingId("SPY".to_owned()),
        currency: Currency("USD".to_owned()),
        valuation_time: OPEN,
        expiry: EXPIRY,
        reference_spot: PriceMicros(100_000_000),
        marks: BTreeMap::from([
            (call.contract_id, PriceMicros(5_000_000)),
            (put.contract_id, PriceMicros(4_500_000)),
        ]),
    };
    let target = GraphTarget {
        objective_name: "test_score".to_owned(),
        objective: 1.25,
        scenario_spots_micros: vec![90_000_000, 100_000_000, 110_000_000],
        scenario_pnl_micros: vec![-10, 0, 20],
        note: "test only".to_owned(),
    };

    let first =
        PortfolioGraphSample::from_portfolio(&portfolio, &context, Some(target.clone())).unwrap();
    let second =
        PortfolioGraphSample::from_portfolio(&portfolio, &context, Some(target.clone())).unwrap();
    let unlabeled = PortfolioGraphSample::from_portfolio(&portfolio, &context, None).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.sample_id, unlabeled.sample_id);
    let changed_target = GraphTarget {
        objective: -99.0,
        ..target
    };
    assert_eq!(
        first.sample_id,
        PortfolioGraphSample::from_portfolio(&portfolio, &context, Some(changed_target))
            .unwrap()
            .sample_id
    );
    assert_eq!(first.portfolio_hash, portfolio.semantic_hash().unwrap());
    assert_eq!(
        first.terminal_payoff_hash,
        Some(
            portfolio
                .terminal_payoff_projection()
                .unwrap()
                .semantic_hash()
                .unwrap()
        )
    );
    assert_eq!(
        first
            .nodes
            .iter()
            .filter(|node| matches!(node.payload, GraphNodePayload::OptionPosition { .. }))
            .count(),
        2
    );
    assert_eq!(first.edges.len(), 6);
}

#[test]
fn graph_export_supports_zero_portfolio_and_rejects_missing_marks() {
    let empty = CanonicalPortfolio::from_positions(Vec::new()).unwrap();
    let target = GraphTarget {
        objective_name: "test_score".to_owned(),
        objective: 0.0,
        scenario_spots_micros: vec![100_000_000],
        scenario_pnl_micros: vec![0],
        note: "test only".to_owned(),
    };
    let base_context = GraphContext {
        underlying: UnderlyingId("SPY".to_owned()),
        currency: Currency("USD".to_owned()),
        valuation_time: OPEN,
        expiry: EXPIRY,
        reference_spot: PriceMicros(100_000_000),
        marks: BTreeMap::new(),
    };
    let empty_sample =
        PortfolioGraphSample::from_portfolio(&empty, &base_context, Some(target.clone())).unwrap();
    assert_eq!(empty_sample.nodes.len(), 3);
    assert!(empty_sample.edges.is_empty());
    assert_eq!(empty_sample.terminal_payoff_hash, None);

    let call = american("SPY-C-100", OptionRight::Call, 100, Settlement::Physical);
    let nonempty =
        CanonicalPortfolio::from_positions(vec![Position::new(call, 1).unwrap()]).unwrap();
    assert!(matches!(
        PortfolioGraphSample::from_portfolio(&nonempty, &base_context, Some(target)),
        Err(CogError::InvalidDefinition(message)) if message.contains("missing graph mark")
    ));
}

#[test]
fn generated_and_historical_sources_share_one_ordered_interface() {
    fn source(source_id: &str, origin: ObservationOrigin) -> InMemoryOptionMarketDataSource {
        let manifest = OptionDataSourceManifest::new(
            source_id,
            origin,
            "unit_test",
            "fixture-v1",
            PriceAuthority::ProxyOnly,
            vec!["test observations".to_owned()],
        )
        .unwrap();
        let observations = vec![
            OptionMarketObservation::new(
                source_id,
                2,
                ContractId("SPY-C-100".to_owned()),
                "O:SPY-C-100",
                TimestampNs(20),
                OptionObservationPayload::Bar(OptionBar {
                    interval: BarInterval::Minute,
                    open: PriceMicros(5_000_000),
                    high: PriceMicros(5_500_000),
                    low: PriceMicros(4_900_000),
                    close: PriceMicros(5_200_000),
                    volume: 10,
                    transactions: 4,
                }),
            )
            .unwrap(),
            OptionMarketObservation::new(
                source_id,
                1,
                ContractId("SPY-C-100".to_owned()),
                "O:SPY-C-100",
                TimestampNs(10),
                OptionObservationPayload::Bar(OptionBar {
                    interval: BarInterval::Minute,
                    open: PriceMicros(4_500_000),
                    high: PriceMicros(5_000_000),
                    low: PriceMicros(4_400_000),
                    close: PriceMicros(4_900_000),
                    volume: 8,
                    transactions: 3,
                }),
            )
            .unwrap(),
        ];
        InMemoryOptionMarketDataSource::new(manifest, observations).unwrap()
    }

    fn drain(source: &mut dyn OptionMarketDataSource) -> Vec<(TimestampNs, PriceMicros)> {
        let mut rows = Vec::new();
        while let Some(observation) = source.next_observation().unwrap() {
            rows.push((
                observation.event_time,
                observation.proxy_mark().unwrap().price,
            ));
        }
        rows
    }

    let mut historical = source("historical", ObservationOrigin::Historical);
    let mut generated = source("generated", ObservationOrigin::Generated);
    assert_eq!(historical.len(), 2);
    assert_eq!(drain(&mut historical), drain(&mut generated));
    assert_eq!(
        historical.manifest().price_authority,
        PriceAuthority::ProxyOnly
    );
}

#[test]
fn quote_distinguishes_proxy_midpoint_from_executable_sides() {
    let observation = OptionMarketObservation::new(
        "quotes",
        1,
        ContractId("SPY-C-100".to_owned()),
        "O:SPY-C-100",
        TimestampNs(10),
        OptionObservationPayload::Quote(OptionQuote {
            bid: Some(BookLevel {
                price: PriceMicros(4_900_000),
                size: 10,
                exchange: Some(301),
            }),
            ask: Some(BookLevel {
                price: PriceMicros(5_100_000),
                size: 12,
                exchange: Some(302),
            }),
        }),
    )
    .unwrap();

    assert_eq!(
        observation.proxy_mark().unwrap(),
        volt::OptionMark {
            price: PriceMicros(5_000_000),
            basis: MarkBasis::QuoteMidpoint,
        }
    );
    assert_eq!(
        observation.executable_price(volt::OrderSide::Buy),
        Some(PriceMicros(5_100_000))
    );
    assert_eq!(
        observation.executable_price(volt::OrderSide::Sell),
        Some(PriceMicros(4_900_000))
    );
}
