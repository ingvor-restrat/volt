use std::collections::{BTreeMap, BTreeSet};

use volt::{
    CandidateAction, CandidateActionEffect, CandidateGrammar, CandidateMarketContext,
    CanonicalPortfolio, CompiledCandidateAction, ContractId, ExerciseSchedule, ExpiryBucket,
    OptionContract, OptionRight, Position, PriceMicros, Settlement, StrikeOrDeltaBucket,
    TimestampNs, TransitionAction,
};

#[test]
fn add_leg_resolves_declared_buckets_and_canonicalizes() {
    let contracts = listed_contracts();
    let grammar = grammar();
    let empty = CanonicalPortfolio::from_positions(Vec::new()).unwrap();
    let context = context(TimestampNs(10));
    let action = CandidateAction::AddLeg {
        right: OptionRight::Call,
        expiry_bucket: "front".to_owned(),
        strike_or_delta_bucket: "strike-100".to_owned(),
        quantity: 1,
    };

    let compiled = grammar
        .compile_action(&action, &empty, &contracts, &context)
        .unwrap();
    assert_eq!(compiled.len(), 1);
    assert!(matches!(
        &compiled[0],
        CompiledCandidateAction::AddLeg { contract, quantity }
            if contract.contract_id.0 == "SPY-C-100" && *quantity == 1
    ));
    let CandidateActionEffect::Portfolio(portfolio) = compiled[0].apply(&empty).unwrap() else {
        panic!("AddLeg must produce a portfolio");
    };
    assert_eq!(portfolio.positions.len(), 1);
    assert_eq!(portfolio.positions[0].quantity, 1);
    assert_eq!(action.semantic_hash().unwrap().len(), 64);
}

#[test]
fn add_leg_preserves_every_matching_listed_branch_in_id_order() {
    let mut contracts = listed_contracts();
    contracts.push(
        OptionContract::new(
            "SPY-C-105",
            "SPY",
            "USD",
            OptionRight::Call,
            PriceMicros(105_000_000),
            100,
            ExerciseSchedule::American {
                first_exercise: TimestampNs(0),
                expiry: TimestampNs(100),
            },
            Settlement::Physical,
        )
        .unwrap(),
    );
    contracts.reverse();
    let grammar = CandidateGrammar::new(
        vec![ExpiryBucket {
            bucket_id: "front".to_owned(),
            minimum_time_to_expiry_ns: 1,
            maximum_time_to_expiry_ns: 100,
        }],
        vec![StrikeOrDeltaBucket::StrikeRange {
            bucket_id: "wide".to_owned(),
            minimum_strike_micros: 95_000_000,
            maximum_strike_micros: 110_000_000,
        }],
        BTreeSet::from([-1, 1]),
        BTreeSet::from([-1, 1]),
    )
    .unwrap();
    let action = CandidateAction::AddLeg {
        right: OptionRight::Call,
        expiry_bucket: "front".to_owned(),
        strike_or_delta_bucket: "wide".to_owned(),
        quantity: 1,
    };
    let compiled = grammar
        .compile_action(
            &action,
            &CanonicalPortfolio::from_positions(Vec::new()).unwrap(),
            &contracts,
            &context(TimestampNs(10)),
        )
        .unwrap();
    let ids = compiled
        .iter()
        .map(|action| match action {
            CompiledCandidateAction::AddLeg { contract, .. } => contract.contract_id.0.as_str(),
            _ => panic!("expected AddLeg realization"),
        })
        .collect::<Vec<_>>();

    assert_eq!(ids, ["SPY-C-100", "SPY-C-105"]);
}

#[test]
fn remove_resize_wait_and_exercise_have_distinct_authority() {
    let contracts = listed_contracts();
    let grammar = grammar();
    let long =
        CanonicalPortfolio::from_positions(vec![Position::new(contracts[0].clone(), 2).unwrap()])
            .unwrap();
    let context = context(TimestampNs(50));

    let resize = CandidateAction::ResizeLeg {
        contract: ContractId("SPY-C-100".to_owned()),
        delta_quantity: -1,
    };
    let compiled = grammar
        .compile_action(&resize, &long, &contracts, &context)
        .unwrap();
    let CandidateActionEffect::Portfolio(resized) = compiled[0].apply(&long).unwrap() else {
        panic!("ResizeLeg must produce a portfolio");
    };
    assert_eq!(resized.positions[0].quantity, 1);

    let wait = grammar
        .compile_action(&CandidateAction::Wait, &resized, &contracts, &context)
        .unwrap();
    assert_eq!(
        wait[0].apply(&resized).unwrap(),
        CandidateActionEffect::Portfolio(resized.clone())
    );

    let exercise = CandidateAction::Exercise {
        contract: ContractId("SPY-C-100".to_owned()),
        quantity: 1,
    };
    let compiled = grammar
        .compile_action(&exercise, &resized, &contracts, &context)
        .unwrap();
    assert_eq!(
        compiled[0].apply(&resized).unwrap(),
        CandidateActionEffect::Runtime(TransitionAction::Exercise {
            contract_id: ContractId("SPY-C-100".to_owned()),
            quantity: 1,
        })
    );

    let remove = CandidateAction::RemoveLeg {
        contract: ContractId("SPY-C-100".to_owned()),
    };
    let compiled = grammar
        .compile_action(&remove, &resized, &contracts, &context)
        .unwrap();
    let CandidateActionEffect::Portfolio(removed) = compiled[0].apply(&resized).unwrap() else {
        panic!("RemoveLeg must produce a portfolio");
    };
    assert!(removed.positions.is_empty());
}

#[test]
fn grammar_rejects_ambiguous_removal_and_invalid_exercise_authority() {
    let contracts = listed_contracts();
    let grammar = grammar();
    let short =
        CanonicalPortfolio::from_positions(vec![Position::new(contracts[0].clone(), -1).unwrap()])
            .unwrap();
    let exercise = CandidateAction::Exercise {
        contract: ContractId("SPY-C-100".to_owned()),
        quantity: 1,
    };
    assert!(grammar
        .compile_action(&exercise, &short, &contracts, &context(TimestampNs(50)))
        .unwrap_err()
        .to_string()
        .contains("requires 1 long"));

    let long =
        CanonicalPortfolio::from_positions(vec![Position::new(contracts[0].clone(), 1).unwrap()])
            .unwrap();
    assert!(grammar
        .compile_action(&exercise, &long, &contracts, &context(TimestampNs(101)))
        .unwrap_err()
        .to_string()
        .contains("outside the contract opportunity set"));

    let resize_to_zero = CandidateAction::ResizeLeg {
        contract: ContractId("SPY-C-100".to_owned()),
        delta_quantity: -1,
    };
    assert!(grammar
        .compile_action(
            &resize_to_zero,
            &long,
            &contracts,
            &context(TimestampNs(50))
        )
        .unwrap_err()
        .to_string()
        .contains("use RemoveLeg"));

    let duplicated = [contracts.clone(), contracts].concat();
    assert!(grammar
        .compile_action(
            &CandidateAction::Wait,
            &long,
            &duplicated,
            &context(TimestampNs(50))
        )
        .unwrap_err()
        .to_string()
        .contains("repeats contract"));
}

fn grammar() -> CandidateGrammar {
    CandidateGrammar::new(
        vec![ExpiryBucket {
            bucket_id: "front".to_owned(),
            minimum_time_to_expiry_ns: 1,
            maximum_time_to_expiry_ns: 100,
        }],
        vec![
            StrikeOrDeltaBucket::StrikeRange {
                bucket_id: "strike-100".to_owned(),
                minimum_strike_micros: 100_000_000,
                maximum_strike_micros: 100_000_000,
            },
            StrikeOrDeltaBucket::AbsoluteDeltaBps {
                bucket_id: "delta-40-60".to_owned(),
                minimum_absolute_delta_bps: 4_000,
                maximum_absolute_delta_bps: 6_000,
            },
        ],
        BTreeSet::from([-1, 1]),
        BTreeSet::from([-1, 1]),
    )
    .unwrap()
}

fn context(time: TimestampNs) -> CandidateMarketContext {
    CandidateMarketContext {
        valuation_time: time,
        reference_spot: PriceMicros(100_000_000),
        absolute_delta_bps: BTreeMap::from([
            (ContractId("SPY-C-100".to_owned()), 5_000),
            (ContractId("SPY-P-100".to_owned()), 5_000),
        ]),
    }
}

fn listed_contracts() -> Vec<OptionContract> {
    [OptionRight::Call, OptionRight::Put]
        .into_iter()
        .map(|right| {
            OptionContract::new(
                match right {
                    OptionRight::Call => "SPY-C-100",
                    OptionRight::Put => "SPY-P-100",
                },
                "SPY",
                "USD",
                right,
                PriceMicros(100_000_000),
                100,
                ExerciseSchedule::American {
                    first_exercise: TimestampNs(0),
                    expiry: TimestampNs(100),
                },
                Settlement::Physical,
            )
            .unwrap()
        })
        .collect()
}
