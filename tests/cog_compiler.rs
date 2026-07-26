use std::collections::{BTreeMap, BTreeSet};

use volt::{
    CandidateAction, CandidateGrammar, CandidateMarketContext, CogCompiler, CogError,
    CogPositionSource, CogRuleSource, CogSource, CompiledCandidateAction, CompiledStepResult,
    ContractId, Currency, DiagnosticSeverity, ExerciseSchedule, ExpiryBucket, GraphContext,
    GuardExpr, MarketObservation, OptionContract, OptionRight, PositionSide, PriceMicros,
    ScenarioObjectiveSource, Settlement, StateActionGraphSample, StrikeOrDeltaBucket, TimestampNs,
    UnderlyingId, COG_SOURCE_SCHEMA_VERSION,
};

const EXPIRY: TimestampNs = TimestampNs(100);
const VALUATION: TimestampNs = TimestampNs(50);

#[test]
fn structured_json_compiles_to_a_deterministic_valid_artifact() {
    let source = source();
    let json = serde_json::to_string_pretty(&source).unwrap();
    let first = CogCompiler.compile_json(&json);
    let second = CogCompiler.compile(&source);

    assert!(!first.has_errors());
    assert_eq!(first, second);
    let artifact = first.into_result().unwrap();
    artifact.validate().unwrap();
    assert_eq!(artifact.rules[0].rule_id, "add-downside-put");
    assert_eq!(artifact.initial_portfolio.positions.len(), 1);
    assert_eq!(artifact.source_hash.len(), 64);
    assert_eq!(artifact.artifact_hash.len(), 64);

    let round_trip = serde_json::to_string(&artifact).unwrap();
    let decoded: volt::CompiledCog = serde_json::from_str(&round_trip).unwrap();
    assert_eq!(decoded, artifact);
    decoded.validate().unwrap();
}

#[test]
fn compiler_collects_source_addressable_errors_and_emits_no_artifact() {
    let mut source = source();
    source.schema_version = "wrong".to_owned();
    source.program_id.clear();
    source.initial_positions.push(CogPositionSource {
        contract_id: ContractId("MISSING".to_owned()),
        quantity: 0,
    });
    source.rules.push(source.rules[0].clone());
    source.objective.as_mut().unwrap().probability_millionths = vec![1, 2, 3, 4, 5];

    let report = CogCompiler.compile(&source);
    assert!(report.has_errors());
    assert!(report.artifact.is_none());
    let codes = report
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.code.as_str())
        .collect::<BTreeSet<_>>();
    assert!(codes.contains("E0100_SCHEMA"));
    assert!(codes.contains("E0101_PROGRAM_ID"));
    assert!(codes.contains("E0400_ZERO_POSITION"));
    assert!(codes.contains("E0401_UNKNOWN_INITIAL_CONTRACT"));
    assert!(codes.contains("E0502_DUPLICATE_RULE"));
    assert!(codes.contains("E0803_OBJECTIVE_PROBABILITY"));
    assert!(report
        .diagnostics
        .iter()
        .all(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error));

    let malformed = CogCompiler.compile_json("{");
    assert_eq!(malformed.diagnostics[0].code, "E0001_JSON_PARSE");
    assert_eq!(malformed.diagnostics[0].path, "$");
}

#[test]
fn compiled_guards_produce_an_explicit_hard_legal_mask() {
    let artifact = CogCompiler.compile(&source()).into_result().unwrap();
    let state = artifact.initial_state().unwrap();
    let action_set = artifact.action_set(&state, &candidate_context(95)).unwrap();

    assert_eq!(
        action_set
            .candidates
            .iter()
            .map(|candidate| candidate.rule_id.as_str())
            .collect::<Vec<_>>(),
        [
            "add-downside-put",
            "add-upside-call",
            "exercise-call",
            "wait"
        ]
    );
    assert_eq!(action_set.legal_action_mask, [true, false, true, true]);
    assert!(matches!(
        action_set.candidates[0].compiled_action,
        Some(CompiledCandidateAction::AddLeg { ref contract, quantity: 1 })
            if contract.contract_id.0 == "SPY-P-90"
    ));
    assert_eq!(
        action_set.candidates[1].rejection_reason.as_deref(),
        Some("guard evaluated false")
    );
    assert_eq!(action_set.action_set_hash.len(), 64);
}

#[test]
fn one_rule_expands_to_stably_ordered_exact_contract_branches() {
    let mut source = source();
    source.candidate_grammar.strike_or_delta_buckets[0] = StrikeOrDeltaBucket::StrikeRange {
        bucket_id: "strike-90".to_owned(),
        minimum_strike_micros: 90_000_000,
        maximum_strike_micros: 110_000_000,
    };
    let artifact = CogCompiler.compile(&source).into_result().unwrap();
    let action_set =
        artifact.action_set(&artifact.initial_state().unwrap(), &candidate_context(95));
    let action_set = action_set.unwrap();

    let downside_branches = action_set
        .candidates
        .iter()
        .filter(|candidate| candidate.rule_id == "add-downside-put")
        .collect::<Vec<_>>();
    assert_eq!(downside_branches.len(), 3);
    assert_eq!(
        downside_branches
            .iter()
            .map(|candidate| candidate.branch_index)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(
        downside_branches
            .iter()
            .map(
                |candidate| match candidate.compiled_action.as_ref().unwrap() {
                    CompiledCandidateAction::AddLeg { contract, .. } => {
                        contract.contract_id.0.as_str()
                    }
                    _ => panic!("expanded downside action must add one exact leg"),
                }
            )
            .collect::<Vec<_>>(),
        ["SPY-P-100", "SPY-P-110", "SPY-P-90"]
    );
    assert_eq!(
        action_set.legal_action_mask,
        [true, true, true, false, true, true]
    );
}

#[test]
fn design_actions_and_lifecycle_actions_use_existing_exact_semantics() {
    let artifact = CogCompiler.compile(&source()).into_result().unwrap();
    let state = artifact.initial_state().unwrap();
    let action_set = artifact.action_set(&state, &candidate_context(95)).unwrap();
    let observation = observation("decision", 95);

    let design = artifact
        .execute_action(&state, &action_set, 0, &observation)
        .unwrap();
    let CompiledStepResult::Design { portfolio, .. } = design else {
        panic!("AddLeg must be a static design transition");
    };
    assert_eq!(portfolio.positions.len(), 2);
    assert!(portfolio
        .positions
        .iter()
        .any(|position| position.contract.contract_id.0 == "SPY-P-90"));

    let exercise = artifact
        .execute_action(&state, &action_set, 2, &observation)
        .unwrap();
    let CompiledStepResult::Runtime { transition, .. } = exercise else {
        panic!("Exercise must execute through CogRuntime");
    };
    assert_eq!(transition.state.sequence, 1);
    assert_eq!(transition.journal_entry.rule_id, "holder_exercise.v1");

    let wait = artifact
        .execute_action(&state, &action_set, 3, &observation)
        .unwrap();
    let CompiledStepResult::Runtime { transition, .. } = wait else {
        panic!("Wait must execute as a journaled hold");
    };
    assert_eq!(transition.state.sequence, 1);
    assert_eq!(transition.journal_entry.rule_id, "hold.v1");

    assert!(matches!(
        artifact.execute_action(&state, &action_set, 1, &observation),
        Err(CogError::InvalidAction(_))
    ));
}

#[test]
fn state_action_graph_retains_rejected_slots_and_exact_hashes() {
    let artifact = CogCompiler.compile(&source()).into_result().unwrap();
    let state = artifact.initial_state().unwrap();
    let action_set = artifact.action_set(&state, &candidate_context(95)).unwrap();
    let graph_context = GraphContext {
        underlying: UnderlyingId("SPY".to_owned()),
        currency: Currency("USD".to_owned()),
        valuation_time: VALUATION,
        expiry: EXPIRY,
        reference_spot: PriceMicros(95_000_000),
        marks: BTreeMap::from([(ContractId("SPY-C-100".to_owned()), PriceMicros(4_000_000))]),
    };

    let first = StateActionGraphSample::from_state_and_actions(&state, &graph_context, &action_set)
        .unwrap();
    let second =
        StateActionGraphSample::from_state_and_actions(&state, &graph_context, &action_set)
            .unwrap();

    assert_eq!(first, second);
    assert_eq!(first.compiled_cog_hash, artifact.artifact_hash);
    assert_eq!(first.legal_action_mask, [true, false, true, true]);
    assert_eq!(first.action_nodes.len(), 4);
    assert!(!first.action_nodes[1].legal);
    assert_eq!(first.action_nodes[1].rule_id, "add-upside-call");
    assert!(first.action_edges.len() >= first.action_nodes.len());
    assert_eq!(first.sample_id.len(), 64);
}

#[test]
fn tampered_artifacts_and_action_masks_fail_closed() {
    let artifact = CogCompiler.compile(&source()).into_result().unwrap();
    let mut tampered_artifact = artifact.clone();
    tampered_artifact.rules[0].priority += 1;
    assert!(tampered_artifact.validate().is_err());

    let state = artifact.initial_state().unwrap();
    let mut action_set = artifact.action_set(&state, &candidate_context(95)).unwrap();
    action_set.legal_action_mask[1] = true;
    assert!(action_set.validate().is_err());
    assert!(artifact
        .execute_action(&state, &action_set, 1, &observation("tampered", 95))
        .is_err());
}

fn source() -> CogSource {
    let contracts = listed_contracts();
    CogSource {
        schema_version: COG_SOURCE_SCHEMA_VERSION.to_owned(),
        program_id: "protective-overlay".to_owned(),
        description: "Add downside protection, preserve an exercise right, and wait.".to_owned(),
        listed_universe: contracts,
        candidate_grammar: CandidateGrammar::new(
            vec![ExpiryBucket {
                bucket_id: "front".to_owned(),
                minimum_time_to_expiry_ns: 1,
                maximum_time_to_expiry_ns: 100,
            }],
            vec![
                StrikeOrDeltaBucket::StrikeRange {
                    bucket_id: "strike-90".to_owned(),
                    minimum_strike_micros: 90_000_000,
                    maximum_strike_micros: 90_000_000,
                },
                StrikeOrDeltaBucket::StrikeRange {
                    bucket_id: "strike-110".to_owned(),
                    minimum_strike_micros: 110_000_000,
                    maximum_strike_micros: 110_000_000,
                },
            ],
            BTreeSet::from([-1, 1]),
            BTreeSet::from([-1, 1]),
        )
        .unwrap(),
        initial_positions: vec![CogPositionSource {
            contract_id: ContractId("SPY-C-100".to_owned()),
            quantity: 1,
        }],
        rules: vec![
            CogRuleSource {
                rule_id: "add-downside-put".to_owned(),
                priority: 100,
                guard: GuardExpr::All {
                    guards: vec![
                        GuardExpr::SpotAtMost {
                            spot_micros: 100_000_000,
                        },
                        GuardExpr::GrossQuantityAtMost { quantity: 3 },
                    ],
                },
                action: CandidateAction::AddLeg {
                    right: OptionRight::Put,
                    expiry_bucket: "front".to_owned(),
                    strike_or_delta_bucket: "strike-90".to_owned(),
                    quantity: 1,
                },
            },
            CogRuleSource {
                rule_id: "add-upside-call".to_owned(),
                priority: 80,
                guard: GuardExpr::SpotAtLeast {
                    spot_micros: 110_000_000,
                },
                action: CandidateAction::AddLeg {
                    right: OptionRight::Call,
                    expiry_bucket: "front".to_owned(),
                    strike_or_delta_bucket: "strike-110".to_owned(),
                    quantity: -1,
                },
            },
            CogRuleSource {
                rule_id: "exercise-call".to_owned(),
                priority: 70,
                guard: GuardExpr::All {
                    guards: vec![
                        GuardExpr::HasLivePosition {
                            contract_id: ContractId("SPY-C-100".to_owned()),
                            side: PositionSide::Long,
                        },
                        GuardExpr::ExerciseOpen {
                            contract_id: ContractId("SPY-C-100".to_owned()),
                        },
                    ],
                },
                action: CandidateAction::Exercise {
                    contract: ContractId("SPY-C-100".to_owned()),
                    quantity: 1,
                },
            },
            CogRuleSource {
                rule_id: "wait".to_owned(),
                priority: 0,
                guard: GuardExpr::Always,
                action: CandidateAction::Wait,
            },
        ],
        objective: Some(ScenarioObjectiveSource {
            objective_name: "synthetic_risk_adjusted_terminal_pnl_dollars".to_owned(),
            scenario_spots_micros: vec![
                80_000_000,
                90_000_000,
                100_000_000,
                110_000_000,
                120_000_000,
            ],
            probability_millionths: vec![100_000, 200_000, 400_000, 200_000, 100_000],
            downside_weight_bps: 3_500,
            gross_contract_penalty_micros: 250_000,
        }),
    }
}

fn listed_contracts() -> Vec<OptionContract> {
    [OptionRight::Call, OptionRight::Put]
        .into_iter()
        .flat_map(|right| {
            [90_i64, 100, 110].map(move |strike| {
                let code = match right {
                    OptionRight::Call => "C",
                    OptionRight::Put => "P",
                };
                OptionContract::new(
                    format!("SPY-{code}-{strike}"),
                    "SPY",
                    "USD",
                    right,
                    PriceMicros(strike * 1_000_000),
                    100,
                    ExerciseSchedule::American {
                        first_exercise: TimestampNs(0),
                        expiry: EXPIRY,
                    },
                    Settlement::Physical,
                )
                .unwrap()
            })
        })
        .collect()
}

fn candidate_context(spot_dollars: i64) -> CandidateMarketContext {
    CandidateMarketContext {
        valuation_time: VALUATION,
        reference_spot: PriceMicros(spot_dollars * 1_000_000),
        absolute_delta_bps: BTreeMap::new(),
    }
}

fn observation(event_id: &str, spot_dollars: i64) -> MarketObservation {
    MarketObservation {
        event_id: event_id.to_owned(),
        time: VALUATION,
        spot: PriceMicros(spot_dollars * 1_000_000),
    }
}
