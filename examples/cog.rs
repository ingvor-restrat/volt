//! `cog` compiles one structured financial-logic program and emits its first
//! state/action graph plus a journaled wait transition.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::Serialize;
use volt::{
    CandidateAction, CandidateMarketContext, CogCompiler, CompiledActionSet, CompiledCog,
    CompiledStepResult, CompilerDiagnostic, ContractId, GraphContext, JournalEntry,
    MarketObservation, PriceMicros, StateActionGraphSample, TimestampNs,
};

const DEFAULT_SOURCE: &str = include_str!("cogs/protective-put.json");

#[derive(Serialize)]
struct CogDemoOutput<'a> {
    compiled_cog: &'a CompiledCog,
    diagnostics: &'a [CompilerDiagnostic],
    action_set: &'a CompiledActionSet,
    state_action_graph: &'a StateActionGraphSample,
    executed_journal_entry: &'a JournalEntry,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = arguments()?;
    let source_json = match arguments.source {
        Some(path) => fs::read_to_string(path)?,
        None => DEFAULT_SOURCE.to_owned(),
    };
    let report = CogCompiler.compile_json(&source_json);
    let diagnostics = report.diagnostics.clone();
    let compiled_cog = report.into_result()?;
    let state = compiled_cog.initial_state()?;
    let context = CandidateMarketContext {
        valuation_time: TimestampNs(arguments.time_ns),
        reference_spot: PriceMicros(arguments.spot_micros),
        absolute_delta_bps: BTreeMap::new(),
    };
    let action_set = compiled_cog.action_set(&state, &context)?;
    let graph_context = GraphContext {
        underlying: compiled_cog.listed_universe[0].underlying.clone(),
        currency: compiled_cog.listed_universe[0].currency.clone(),
        valuation_time: context.valuation_time,
        expiry: compiled_cog.listed_universe[0].expiry(),
        reference_spot: context.reference_spot,
        marks: compiled_cog
            .initial_portfolio
            .positions
            .iter()
            .map(|position| {
                (
                    position.contract.contract_id.clone(),
                    synthetic_mark(&position.contract.contract_id),
                )
            })
            .collect(),
    };
    let state_action_graph =
        StateActionGraphSample::from_state_and_actions(&state, &graph_context, &action_set)?;
    let wait_index = action_set
        .candidates
        .iter()
        .position(|candidate| candidate.legal && candidate.source_action == CandidateAction::Wait)
        .ok_or("compiled program has no legal wait action")?;
    let observation = MarketObservation {
        event_id: "cog-demo-decision".to_owned(),
        time: context.valuation_time,
        spot: context.reference_spot,
    };
    let CompiledStepResult::Runtime {
        transition: wait_transition,
        ..
    } = compiled_cog.execute_action(&state, &action_set, wait_index, &observation)?
    else {
        return Err("wait action did not execute through CogRuntime".into());
    };
    let output = CogDemoOutput {
        compiled_cog: &compiled_cog,
        diagnostics: &diagnostics,
        action_set: &action_set,
        state_action_graph: &state_action_graph,
        executed_journal_entry: &wait_transition.journal_entry,
    };
    let json = serde_json::to_string_pretty(&output)?;
    match arguments.output {
        Some(path) => fs::write(path, format!("{json}\n"))?,
        None => println!("{json}"),
    }
    Ok(())
}

fn synthetic_mark(contract_id: &ContractId) -> PriceMicros {
    let dollars = match contract_id.0.as_str() {
        "SPY-C-100" => 4,
        _ => 2,
    };
    PriceMicros(dollars * 1_000_000)
}

struct Arguments {
    source: Option<PathBuf>,
    output: Option<PathBuf>,
    time_ns: i64,
    spot_micros: i64,
}

fn arguments() -> Result<Arguments, Box<dyn std::error::Error>> {
    let mut source = None;
    let mut output = None;
    let mut time_ns = 50_i64;
    let mut spot_micros = 95_000_000_i64;
    let mut args = std::env::args_os().skip(1);
    while let Some(argument) = args.next() {
        match argument.to_str() {
            Some("--source") => {
                source = Some(args.next().ok_or("--source requires a path")?.into())
            }
            Some("--output") => {
                output = Some(args.next().ok_or("--output requires a path")?.into())
            }
            Some("--time-ns") => {
                time_ns = args
                    .next()
                    .ok_or("--time-ns requires an integer")?
                    .to_str()
                    .ok_or("--time-ns must be UTF-8")?
                    .parse()?
            }
            Some("--spot-micros") => {
                spot_micros = args
                    .next()
                    .ok_or("--spot-micros requires an integer")?
                    .to_str()
                    .ok_or("--spot-micros must be UTF-8")?
                    .parse()?
            }
            _ => {
                return Err(
                    "usage: cog [--source PATH] [--output PATH] [--time-ns N] [--spot-micros N]"
                        .into(),
                )
            }
        }
    }
    Ok(Arguments {
        source,
        output,
        time_ns,
        spot_micros,
    })
}
