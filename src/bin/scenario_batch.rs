//! Evaluate a batch of complete scenario requests through Volt's runtime.
//!
//! `cargo run --bin scenario_batch -- --input request.json --output result.json`

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use volt::{ScenarioSimulator, SimulationRequest, SimulationResult};

const REQUEST_SCHEMA_VERSION: &str = "volt.scenario_batch_request.v1";
const RESULT_SCHEMA_VERSION: &str = "volt.scenario_batch_result.v1";

#[derive(Debug, Deserialize)]
struct ScenarioBatchRequest {
    schema_version: String,
    requests: Vec<SimulationRequest>,
}

#[derive(Debug, Serialize)]
struct ScenarioBatchResult {
    schema_version: &'static str,
    records: Vec<SimulationResult>,
}

fn parse_args() -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
    let mut input = None;
    let mut output = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--input" => input = arguments.next().map(PathBuf::from),
            "--output" => output = arguments.next().map(PathBuf::from),
            _ => return Err(format!("unknown argument {argument}").into()),
        }
    }
    match (input, output) {
        (Some(input), Some(output)) => Ok((input, output)),
        _ => Err("usage: scenario_batch --input REQUEST.json --output RESULT.json".into()),
    }
}

fn evaluate(
    request: ScenarioBatchRequest,
) -> Result<ScenarioBatchResult, Box<dyn std::error::Error>> {
    if request.schema_version != REQUEST_SCHEMA_VERSION {
        return Err(format!("unsupported request schema {}", request.schema_version).into());
    }
    if request.requests.is_empty() {
        return Err("scenario batch must contain at least one request".into());
    }
    let simulator = ScenarioSimulator;
    let records = request
        .requests
        .iter()
        .map(|item| simulator.run(item))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ScenarioBatchResult {
        schema_version: RESULT_SCHEMA_VERSION,
        records,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (input_path, output_path) = parse_args()?;
    let request: ScenarioBatchRequest =
        serde_json::from_reader(BufReader::new(File::open(input_path)?))?;
    let result = evaluate(request)?;
    serde_json::to_writer_pretty(BufWriter::new(File::create(output_path)?), &result)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt::{
        DeterministicFillModel, ExerciseSchedule, OptionContract, OptionRight, Position,
        PriceMicros, ScenarioOptionPrice, ScenarioPath, ScenarioPoint, Settlement, TimestampNs,
        TransactionCostModel,
    };

    fn request() -> SimulationRequest {
        let contract = OptionContract::new(
            "SPY-C100",
            "SPY",
            "USD",
            OptionRight::Call,
            PriceMicros(100_000_000),
            100,
            ExerciseSchedule::European {
                expiry: TimestampNs(20),
            },
            Settlement::Cash,
        )
        .unwrap();
        let price = ScenarioOptionPrice {
            contract_id: contract.contract_id.clone(),
            mark: PriceMicros(5_000_000),
            mark_basis: volt::MarkBasis::GeneratedModel,
            bid: None,
            ask: None,
        };
        SimulationRequest::new(
            "batch-fixture",
            ScenarioPath::new(
                "batch-path",
                7,
                vec![ScenarioPoint {
                    event_id: "entry".to_owned(),
                    time: TimestampNs(10),
                    spot: PriceMicros(100_000_000),
                    option_prices: [(contract.contract_id.clone(), price)]
                        .into_iter()
                        .collect(),
                }],
            )
            .unwrap(),
            vec![Position::new(contract, 1).unwrap()],
            vec![],
            DeterministicFillModel::DeclaredMark,
            TransactionCostModel::zero(),
            vec!["test fixture".to_owned()],
        )
        .unwrap()
    }

    #[test]
    fn batch_runs_each_request_through_scenario_simulator() {
        let result = evaluate(ScenarioBatchRequest {
            schema_version: REQUEST_SCHEMA_VERSION.to_owned(),
            requests: vec![request()],
        })
        .unwrap();

        assert_eq!(result.schema_version, RESULT_SCHEMA_VERSION);
        assert_eq!(result.records.len(), 1);
        assert_eq!(result.records[0].request_id, "batch-fixture");
    }
}
