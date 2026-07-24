use std::collections::BTreeMap;

use volt::{
    ContractId, ContractLifecycle, DeterministicFillModel, ExerciseSchedule, MarkBasis,
    OptionContract, OptionRight, Position, PriceMicros, ScenarioAction, ScenarioOptionPrice,
    ScenarioPath, ScenarioPoint, ScenarioSimulator, ScheduledScenarioAction, Settlement,
    SimulationRequest, TimestampNs, TransactionCostModel, UnderlyingId,
};

fn option(id: &str, right: OptionRight, strike_dollars: i64) -> OptionContract {
    OptionContract::new(
        id,
        "SPY",
        "USD",
        right,
        PriceMicros(strike_dollars * 1_000_000),
        100,
        ExerciseSchedule::American {
            first_exercise: TimestampNs(10),
            expiry: TimestampNs(20),
        },
        Settlement::Physical,
    )
    .unwrap()
}

fn option_price(
    id: &str,
    mark_tenths: i64,
    bid_tenths: i64,
    ask_tenths: i64,
) -> (ContractId, ScenarioOptionPrice) {
    let contract_id = ContractId(id.to_owned());
    (
        contract_id.clone(),
        ScenarioOptionPrice {
            contract_id,
            mark: PriceMicros(mark_tenths * 100_000),
            mark_basis: MarkBasis::QuoteMidpoint,
            bid: Some(PriceMicros(bid_tenths * 100_000)),
            ask: Some(PriceMicros(ask_tenths * 100_000)),
        },
    )
}

fn point(
    event_id: &str,
    time: i64,
    spot_dollars: i64,
    prices: impl IntoIterator<Item = (ContractId, ScenarioOptionPrice)>,
) -> ScenarioPoint {
    ScenarioPoint {
        event_id: event_id.to_owned(),
        time: TimestampNs(time),
        spot: PriceMicros(spot_dollars * 1_000_000),
        option_prices: prices.into_iter().collect::<BTreeMap<_, _>>(),
    }
}

#[test]
fn path_simulation_is_deterministic_and_reconciles_closing_fills() {
    let call = option("SPY-C100", OptionRight::Call, 100);
    let put = option("SPY-P100", OptionRight::Put, 100);
    let path = ScenarioPath::new(
        "two-point-close",
        42,
        vec![
            point(
                "open",
                10,
                100,
                [
                    option_price("SPY-C100", 20, 19, 21),
                    option_price("SPY-P100", 30, 29, 31),
                ],
            ),
            point(
                "close",
                15,
                105,
                [
                    option_price("SPY-C100", 70, 69, 71),
                    option_price("SPY-P100", 10, 9, 11),
                ],
            ),
        ],
    )
    .unwrap();
    let request = SimulationRequest::new(
        "deterministic-close",
        path,
        vec![
            Position::new(call, 1).unwrap(),
            Position::new(put, -1).unwrap(),
        ],
        vec![
            ScheduledScenarioAction {
                point_index: 1,
                action: ScenarioAction::Close {
                    contract_id: ContractId("SPY-C100".to_owned()),
                    quantity: 1,
                },
            },
            ScheduledScenarioAction {
                point_index: 1,
                action: ScenarioAction::Close {
                    contract_id: ContractId("SPY-P100".to_owned()),
                    quantity: 1,
                },
            },
        ],
        DeterministicFillModel::CrossAtTouch,
        TransactionCostModel {
            model_id: "fixed-50c.v1".to_owned(),
            per_contract_side_micros: 500_000,
            minimum_ticket_micros: 0,
        },
        vec![
            "quotes are point-in-time fixtures".to_owned(),
            "underlying liquidation costs are zero".to_owned(),
        ],
    )
    .unwrap();

    let first = ScenarioSimulator.run(&request).unwrap();
    let second = ScenarioSimulator.run(&request).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.manifest.seed, 42);
    assert_eq!(first.opening_fills.len(), 2);
    assert_eq!(first.closing_fills.len(), 2);
    assert_eq!(first.pathwise_pnl[0].pnl_micros, -21_000_000);
    assert_eq!(first.terminal_pnl_micros, 658_000_000);
    assert_eq!(
        first
            .pathwise_pnl
            .last()
            .unwrap()
            .leg_attribution
            .iter()
            .map(|leg| leg.equity_micros)
            .sum::<i128>(),
        first.terminal_pnl_micros
    );
    assert!(first
        .final_state
        .positions
        .values()
        .all(|position| position.lifecycle == ContractLifecycle::Closed));
    assert!(first.to_json_line().unwrap().ends_with('}'));
}

#[test]
fn physical_exercise_is_attributed_to_cash_and_underlying() {
    let call = option("SPY-C100", OptionRight::Call, 100);
    let path = ScenarioPath::new(
        "exercise",
        7,
        vec![
            point("entry", 10, 105, [option_price("SPY-C100", 50, 49, 51)]),
            point(
                "pre-dividend",
                15,
                120,
                [option_price("SPY-C100", 200, 199, 201)],
            ),
        ],
    )
    .unwrap();
    let request = SimulationRequest::new(
        "exercise-attribution",
        path,
        vec![Position::new(call, 1).unwrap()],
        vec![ScheduledScenarioAction {
            point_index: 1,
            action: ScenarioAction::Exercise {
                contract_id: ContractId("SPY-C100".to_owned()),
                quantity: 1,
            },
        }],
        DeterministicFillModel::DeclaredMark,
        TransactionCostModel::zero(),
        vec!["declared marks are synthetic fills".to_owned()],
    )
    .unwrap();

    let result = ScenarioSimulator.run(&request).unwrap();
    let terminal = result.pathwise_pnl.last().unwrap();
    assert_eq!(terminal.cash_micros, -10_500_000_000);
    assert_eq!(terminal.underlying_value_micros, 12_000_000_000);
    assert_eq!(terminal.option_mark_value_micros, 0);
    assert_eq!(terminal.pnl_micros, 1_500_000_000);
    assert_eq!(
        result
            .final_state
            .underlying_units
            .get(&UnderlyingId("SPY".to_owned())),
        Some(&100)
    );
}

#[test]
fn executable_fill_model_fails_closed_without_the_required_quote_side() {
    let call = option("SPY-C100", OptionRight::Call, 100);
    let contract_id = ContractId("SPY-C100".to_owned());
    let path = ScenarioPath::new(
        "missing-ask",
        1,
        vec![point(
            "entry",
            10,
            100,
            [(
                contract_id.clone(),
                ScenarioOptionPrice {
                    contract_id,
                    mark: PriceMicros(2_000_000),
                    mark_basis: MarkBasis::LastTrade,
                    bid: Some(PriceMicros(1_900_000)),
                    ask: None,
                },
            )],
        )],
    )
    .unwrap();
    let request = SimulationRequest::new(
        "missing-touch",
        path,
        vec![Position::new(call, 1).unwrap()],
        vec![],
        DeterministicFillModel::CrossAtTouch,
        TransactionCostModel::zero(),
        vec![],
    )
    .unwrap();

    assert!(ScenarioSimulator.run(&request).is_err());
}
