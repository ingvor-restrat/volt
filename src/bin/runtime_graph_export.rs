//! Export multi-expiry runtime state/action graphs from structured COG source.
//!
//! `cargo run --bin runtime_graph_export -- --input request.json --output result.json`

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use volt::{
    CandidateMarketContext, CogCompiler, CogSource, CompiledActionSet, PortfolioState,
    RuntimeGraphContext, RuntimeStateActionGraphSample,
};

const REQUEST_SCHEMA_VERSION: &str = "volt.runtime_graph_export_request.v1";
const RESULT_SCHEMA_VERSION: &str = "volt.runtime_graph_export_result.v1";

#[derive(Debug, Deserialize)]
struct RequestedState {
    source_state_id: String,
    state: PortfolioState,
    candidate_context: CandidateMarketContext,
    graph_context: RuntimeGraphContext,
}

#[derive(Debug, Deserialize)]
struct RuntimeGraphExportRequest {
    schema_version: String,
    source: CogSource,
    states: Vec<RequestedState>,
}

#[derive(Debug, Serialize)]
struct ExportedState {
    source_state_id: String,
    action_set: CompiledActionSet,
    sample: RuntimeStateActionGraphSample,
}

#[derive(Debug, Serialize)]
struct RuntimeGraphExportResult {
    schema_version: String,
    compiled_cog_hash: String,
    records: Vec<ExportedState>,
}

fn export_graphs(
    request: RuntimeGraphExportRequest,
) -> Result<RuntimeGraphExportResult, Box<dyn std::error::Error>> {
    if request.schema_version != REQUEST_SCHEMA_VERSION {
        return Err(format!(
            "unsupported runtime graph-export request schema {}",
            request.schema_version
        )
        .into());
    }
    if request.states.is_empty() {
        return Err("runtime graph export requires at least one state".into());
    }
    let compiled = CogCompiler.compile(&request.source).into_result()?;
    let mut records = Vec::with_capacity(request.states.len());
    for requested in request.states {
        if requested.source_state_id.trim().is_empty() {
            return Err("runtime graph source_state_id must not be empty".into());
        }
        if requested.graph_context.valuation_time != requested.candidate_context.valuation_time {
            return Err(format!(
                "runtime graph {} uses inconsistent valuation times",
                requested.source_state_id
            )
            .into());
        }
        let action_set = compiled.action_set(&requested.state, &requested.candidate_context)?;
        let sample = RuntimeStateActionGraphSample::from_state_and_actions(
            &requested.state,
            &requested.graph_context,
            &action_set,
        )?;
        records.push(ExportedState {
            source_state_id: requested.source_state_id,
            action_set,
            sample,
        });
    }
    Ok(RuntimeGraphExportResult {
        schema_version: RESULT_SCHEMA_VERSION.to_owned(),
        compiled_cog_hash: compiled.artifact_hash,
        records,
    })
}

struct Options {
    input: PathBuf,
    output: PathBuf,
}

fn options() -> Result<Options, Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let mut input = None;
    let mut output = None;
    while let Some(flag) = args.next() {
        if flag == "--input" {
            input = Some(PathBuf::from(args.next().ok_or("--input requires a path")?));
        } else if flag == "--output" {
            output = Some(PathBuf::from(
                args.next().ok_or("--output requires a path")?,
            ));
        } else {
            return Err(
                "usage: runtime_graph_export --input REQUEST.json --output RESULT.json".into(),
            );
        }
    }
    Ok(Options {
        input: input.ok_or("--input is required")?,
        output: output.ok_or("--output is required")?,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = options()?;
    let request: RuntimeGraphExportRequest =
        serde_json::from_reader(BufReader::new(File::open(options.input)?))?;
    let result = export_graphs(request)?;
    serde_json::to_writer_pretty(BufWriter::new(File::create(options.output)?), &result)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use volt::{
        CandidateAction, CandidateGrammar, CogPositionSource, CogRuleSource, Currency,
        ExerciseSchedule, ExpiryBucket, GraphAccountLimits, GuardExpr, OptionContract, OptionRight,
        PriceMicros, Settlement, StrikeOrDeltaBucket, TimestampNs, UnderlyingId,
        COG_SOURCE_SCHEMA_VERSION,
    };

    fn source() -> CogSource {
        let front = OptionContract::new(
            "SPY-C-100-FRONT",
            "SPY",
            "USD",
            OptionRight::Call,
            PriceMicros(100_000_000),
            100,
            ExerciseSchedule::American {
                first_exercise: TimestampNs(0),
                expiry: TimestampNs(100),
            },
            Settlement::Physical,
        )
        .unwrap();
        let back = OptionContract::new(
            "SPY-P-100-BACK",
            "SPY",
            "USD",
            OptionRight::Put,
            PriceMicros(100_000_000),
            100,
            ExerciseSchedule::American {
                first_exercise: TimestampNs(0),
                expiry: TimestampNs(200),
            },
            Settlement::Physical,
        )
        .unwrap();
        CogSource {
            schema_version: COG_SOURCE_SCHEMA_VERSION.to_owned(),
            program_id: "runtime-graph-fixture".to_owned(),
            description: "Exercise or wait across two expiries.".to_owned(),
            listed_universe: vec![front, back],
            candidate_grammar: CandidateGrammar::new(
                vec![ExpiryBucket {
                    bucket_id: "all".to_owned(),
                    minimum_time_to_expiry_ns: 1,
                    maximum_time_to_expiry_ns: 1_000,
                }],
                vec![StrikeOrDeltaBucket::StrikeRange {
                    bucket_id: "atm".to_owned(),
                    minimum_strike_micros: 100_000_000,
                    maximum_strike_micros: 100_000_000,
                }],
                BTreeSet::from([-1, 1]),
                BTreeSet::from([-1, 1]),
            )
            .unwrap(),
            initial_positions: vec![
                CogPositionSource {
                    contract_id: volt::ContractId("SPY-C-100-FRONT".to_owned()),
                    quantity: 1,
                },
                CogPositionSource {
                    contract_id: volt::ContractId("SPY-P-100-BACK".to_owned()),
                    quantity: -1,
                },
            ],
            rules: vec![CogRuleSource {
                rule_id: "wait".to_owned(),
                priority: 0,
                guard: GuardExpr::Always,
                action: CandidateAction::Wait,
            }],
            objective: None,
        }
    }

    #[test]
    fn export_compiles_source_and_emits_one_runtime_graph() {
        let source = source();
        let compiled = CogCompiler.compile(&source).into_result().unwrap();
        let mut state = compiled.initial_state().unwrap();
        state
            .cash_micros
            .insert(Currency("USD".to_owned()), 1_000_000);
        let request = RuntimeGraphExportRequest {
            schema_version: REQUEST_SCHEMA_VERSION.to_owned(),
            source,
            states: vec![RequestedState {
                source_state_id: "state-1".to_owned(),
                state,
                candidate_context: CandidateMarketContext {
                    valuation_time: TimestampNs(50),
                    reference_spot: PriceMicros(100_000_000),
                    absolute_delta_bps: BTreeMap::new(),
                },
                graph_context: RuntimeGraphContext {
                    valuation_time: TimestampNs(50),
                    reference_spots: BTreeMap::from([(
                        UnderlyingId("SPY".to_owned()),
                        PriceMicros(100_000_000),
                    )]),
                    marks: BTreeMap::from([
                        (
                            volt::ContractId("SPY-C-100-FRONT".to_owned()),
                            PriceMicros(5_000_000),
                        ),
                        (
                            volt::ContractId("SPY-P-100-BACK".to_owned()),
                            PriceMicros(7_000_000),
                        ),
                    ]),
                    account_limits: GraphAccountLimits {
                        maximum_live_positions: 3,
                        maximum_gross_option_quantity: 3,
                        maximum_absolute_underlying_units: 200,
                        minimum_cash_micros: BTreeMap::from([(
                            Currency("USD".to_owned()),
                            -20_000_000_000,
                        )]),
                        transaction_cost_per_contract_micros: 650_000,
                    },
                },
            }],
        };

        let result = export_graphs(request).unwrap();

        assert_eq!(result.records.len(), 1);
        assert_eq!(result.records[0].source_state_id, "state-1");
        assert_eq!(result.records[0].sample.legal_action_mask, [true]);
    }
}
