//! `graph` emits deterministic portfolio-graph samples as JSON Lines.
//!
//! Marks, scenario probabilities, and targets are synthetic infrastructure
//! fixtures. They are not return estimates or trading results.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use volt::{
    CandidateAction, CandidateActionEffect, CandidateGrammar, CandidateMarketContext,
    CanonicalPortfolio, CogCompiler, CogPositionSource, CogRuleSource, CogSource,
    CompiledPortfolioGraphSample, ContractId, Currency, ExerciseSchedule, ExpiryBucket,
    GraphContext, GraphTarget, GuardExpr, OptionContract, OptionRight, PortfolioGraphSample,
    PriceMicros, ScenarioObjectiveSource, Settlement, StrikeOrDeltaBucket, TimestampNs,
    UnderlyingId, COG_SOURCE_SCHEMA_VERSION,
};

const MICROS_PER_DOLLAR: i64 = 1_000_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = output_options()?;
    let expiry = TimestampNs(1_800_000_000_000_000_000);
    let valuation_time = TimestampNs(1_797_408_000_000_000_000);
    let contracts = fixture_contracts(expiry)?;
    let (grammar, candidate_context) =
        fixture_candidate_compiler(&contracts, valuation_time, expiry)?;
    let marks = fixture_marks(&contracts);
    let context = GraphContext {
        underlying: UnderlyingId("SPY".to_owned()),
        currency: Currency("USD".to_owned()),
        valuation_time,
        expiry,
        reference_spot: PriceMicros(100 * MICROS_PER_DOLLAR),
        marks,
    };
    let spots = [80_i64, 90, 100, 110, 120].map(|spot| spot * MICROS_PER_DOLLAR);
    let fixture_objective = ScenarioObjectiveSource {
        objective_name: "synthetic_risk_adjusted_terminal_pnl_dollars".to_owned(),
        scenario_spots_micros: spots.to_vec(),
        probability_millionths: vec![100_000, 200_000, 400_000, 200_000, 100_000],
        downside_weight_bps: 3_500,
        gross_contract_penalty_micros: 250_000,
    };

    let mut writer: Box<dyn Write> = match options.output {
        Some(path) => Box::new(BufWriter::new(File::create(path)?)),
        None => Box::new(BufWriter::new(io::stdout().lock())),
    };

    let mut rows = 0_usize;
    for q0 in -1_i64..=1 {
        for q1 in -1_i64..=1 {
            for q2 in -1_i64..=1 {
                for q3 in -1_i64..=1 {
                    for q4 in -1_i64..=1 {
                        for q5 in -1_i64..=1 {
                            let quantities = [q0, q1, q2, q3, q4, q5];
                            let compiled_cog = if options.compiled {
                                Some(
                                    CogCompiler
                                        .compile(&fixture_cog_source(
                                            &contracts,
                                            &grammar,
                                            &quantities,
                                            &fixture_objective,
                                        ))
                                        .into_result()?,
                                )
                            } else {
                                None
                            };
                            let portfolio = match &compiled_cog {
                                Some(compiled) => compiled.initial_portfolio.clone(),
                                None => compile_portfolio(
                                    &grammar,
                                    &candidate_context,
                                    &contracts,
                                    &quantities,
                                )?,
                            };
                            let objective = compiled_cog
                                .as_ref()
                                .and_then(|compiled| compiled.objective.as_ref())
                                .unwrap_or(&fixture_objective);
                            let target = synthetic_target(&portfolio, &context, objective)?;
                            let sample = PortfolioGraphSample::from_portfolio(
                                &portfolio,
                                &context,
                                Some(target),
                            )?;
                            match compiled_cog {
                                Some(compiled) => {
                                    let sample = CompiledPortfolioGraphSample::from_compiled_cog(
                                        &compiled, sample,
                                    )?;
                                    serde_json::to_writer(&mut writer, &sample)?;
                                }
                                None => serde_json::to_writer(&mut writer, &sample)?,
                            }
                            writer.write_all(b"\n")?;
                            rows += 1;
                        }
                    }
                }
            }
        }
    }
    writer.flush()?;
    eprintln!("wrote {rows} graph samples");
    Ok(())
}

fn fixture_cog_source(
    contracts: &[OptionContract],
    grammar: &CandidateGrammar,
    quantities: &[i64; 6],
    objective: &ScenarioObjectiveSource,
) -> CogSource {
    CogSource {
        schema_version: COG_SOURCE_SCHEMA_VERSION.to_owned(),
        program_id: "supervised-portfolio-fixture".to_owned(),
        description: "Compiler-bound synthetic supervised portfolio.".to_owned(),
        listed_universe: contracts.to_vec(),
        candidate_grammar: grammar.clone(),
        initial_positions: contracts
            .iter()
            .zip(quantities)
            .filter(|(_, quantity)| **quantity != 0)
            .map(|(contract, quantity)| CogPositionSource {
                contract_id: contract.contract_id.clone(),
                quantity: *quantity,
            })
            .collect(),
        rules: vec![
            CogRuleSource {
                rule_id: "add-downside-protection".to_owned(),
                priority: 100,
                guard: GuardExpr::All {
                    guards: vec![
                        GuardExpr::SpotAtMost {
                            spot_micros: 100 * MICROS_PER_DOLLAR,
                        },
                        GuardExpr::GrossQuantityAtMost { quantity: 6 },
                    ],
                },
                action: CandidateAction::AddLeg {
                    right: OptionRight::Put,
                    expiry_bucket: "fixture-expiry".to_owned(),
                    strike_or_delta_bucket: "strike-90".to_owned(),
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
        objective: Some(objective.clone()),
    }
}

fn fixture_candidate_compiler(
    contracts: &[OptionContract],
    valuation_time: TimestampNs,
    expiry: TimestampNs,
) -> Result<(CandidateGrammar, CandidateMarketContext), Box<dyn std::error::Error>> {
    let time_to_expiry = expiry
        .0
        .checked_sub(valuation_time.0)
        .ok_or("fixture expiry precedes valuation time")?;
    let strike_buckets = [90_i64, 100, 110]
        .into_iter()
        .map(|strike| StrikeOrDeltaBucket::StrikeRange {
            bucket_id: format!("strike-{strike}"),
            minimum_strike_micros: strike * MICROS_PER_DOLLAR,
            maximum_strike_micros: strike * MICROS_PER_DOLLAR,
        })
        .collect();
    let allowed_quantities = BTreeSet::from([-1_i64, 1]);
    let grammar = CandidateGrammar::new(
        vec![ExpiryBucket {
            bucket_id: "fixture-expiry".to_owned(),
            minimum_time_to_expiry_ns: time_to_expiry,
            maximum_time_to_expiry_ns: time_to_expiry,
        }],
        strike_buckets,
        allowed_quantities.clone(),
        allowed_quantities,
    )?;
    let context = CandidateMarketContext {
        valuation_time,
        reference_spot: PriceMicros(100 * MICROS_PER_DOLLAR),
        absolute_delta_bps: BTreeMap::new(),
    };
    // Fail at fixture construction rather than halfway through enumeration.
    for contract in contracts {
        if contract.expiry() != expiry {
            return Err("fixture contracts must share one expiry".into());
        }
    }
    Ok((grammar, context))
}

fn compile_portfolio(
    grammar: &CandidateGrammar,
    context: &CandidateMarketContext,
    contracts: &[OptionContract],
    quantities: &[i64; 6],
) -> Result<CanonicalPortfolio, Box<dyn std::error::Error>> {
    let mut portfolio = CanonicalPortfolio::from_positions(Vec::new())?;
    for (contract, quantity) in contracts.iter().zip(quantities) {
        if *quantity == 0 {
            continue;
        }
        let strike = contract.strike.0 / MICROS_PER_DOLLAR;
        let action = CandidateAction::AddLeg {
            right: contract.right,
            expiry_bucket: "fixture-expiry".to_owned(),
            strike_or_delta_bucket: format!("strike-{strike}"),
            quantity: *quantity,
        };
        let compiled = grammar.compile_action(&action, &portfolio, contracts, context)?;
        if compiled.len() != 1 {
            return Err(format!(
                "fixture action for {} compiled to {} branches",
                contract.contract_id.0,
                compiled.len()
            )
            .into());
        }
        portfolio = match compiled[0].apply(&portfolio)? {
            CandidateActionEffect::Portfolio(next) => next,
            CandidateActionEffect::Runtime(_) => {
                return Err("fixture AddLeg unexpectedly compiled to a runtime action".into())
            }
        };
    }
    Ok(portfolio)
}

struct OutputOptions {
    output: Option<PathBuf>,
    compiled: bool,
}

fn output_options() -> Result<OutputOptions, Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let mut output = None;
    let mut compiled = false;
    while let Some(flag) = args.next() {
        if flag == "--output" {
            if output.is_some() {
                return Err("--output may be supplied only once".into());
            }
            output = Some(PathBuf::from(
                args.next().ok_or("--output requires a path")?,
            ));
        } else if flag == "--compiled" {
            compiled = true;
        } else {
            return Err("usage: graph [--compiled] [--output PATH]".into());
        }
    }
    Ok(OutputOptions { output, compiled })
}

fn fixture_contracts(
    expiry: TimestampNs,
) -> Result<Vec<OptionContract>, Box<dyn std::error::Error>> {
    let mut contracts = Vec::new();
    for right in [OptionRight::Call, OptionRight::Put] {
        for strike in [90_i64, 100, 110] {
            let right_code = match right {
                OptionRight::Call => "C",
                OptionRight::Put => "P",
            };
            contracts.push(OptionContract::new(
                format!("SPY-{right_code}-{strike}"),
                "SPY",
                "USD",
                right,
                PriceMicros(strike * MICROS_PER_DOLLAR),
                100,
                ExerciseSchedule::American {
                    first_exercise: TimestampNs(0),
                    expiry,
                },
                Settlement::Physical,
            )?);
        }
    }
    Ok(contracts)
}

fn fixture_marks(contracts: &[OptionContract]) -> BTreeMap<ContractId, PriceMicros> {
    let call_marks = [12.0_f64, 6.0, 2.5];
    let put_marks = [2.0_f64, 5.5, 11.0];
    contracts
        .iter()
        .enumerate()
        .map(|(index, contract)| {
            let mark = if index < 3 {
                call_marks[index]
            } else {
                put_marks[index - 3]
            };
            (
                contract.contract_id.clone(),
                PriceMicros((mark * MICROS_PER_DOLLAR as f64).round() as i64),
            )
        })
        .collect()
}

fn synthetic_target(
    portfolio: &CanonicalPortfolio,
    context: &GraphContext,
    objective: &ScenarioObjectiveSource,
) -> Result<GraphTarget, Box<dyn std::error::Error>> {
    let premium_micros = portfolio.positions.iter().try_fold(
        0_i128,
        |total, position| -> Result<i128, Box<dyn std::error::Error>> {
            let mark = context
                .marks
                .get(&position.contract.contract_id)
                .ok_or("fixture mark missing")?;
            let value = i128::from(position.quantity)
                * i128::from(position.contract.multiplier)
                * i128::from(mark.0);
            Ok(total + value)
        },
    )?;
    let scenario_pnl_micros = objective
        .scenario_spots_micros
        .iter()
        .map(|spot| {
            portfolio
                .direct_terminal_payoff(PriceMicros(*spot))
                .map(|payoff| payoff - premium_micros)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let expected_pnl = scenario_pnl_micros
        .iter()
        .zip(&objective.probability_millionths)
        .map(|(pnl, probability)| *pnl as f64 * f64::from(*probability) / 1_000_000.0)
        .sum::<f64>();
    let downside_rms = scenario_pnl_micros
        .iter()
        .zip(&objective.probability_millionths)
        .map(|(pnl, probability)| {
            let loss = (*pnl as f64).min(0.0);
            f64::from(*probability) / 1_000_000.0 * loss * loss
        })
        .sum::<f64>()
        .sqrt();
    let gross_quantity = portfolio
        .positions
        .iter()
        .map(|position| position.quantity.unsigned_abs() as f64)
        .sum::<f64>();
    let complexity_penalty = gross_quantity * objective.gross_contract_penalty_micros as f64;
    let objective_value = (expected_pnl
        - f64::from(objective.downside_weight_bps) / 10_000.0 * downside_rms
        - complexity_penalty)
        / MICROS_PER_DOLLAR as f64;

    Ok(GraphTarget {
        objective_name: objective.objective_name.clone(),
        objective: objective_value,
        scenario_spots_micros: objective.scenario_spots_micros.clone(),
        scenario_pnl_micros,
        note: "Infrastructure fixture only: synthetic marks and scenario weights; not a return estimate or trading result.".to_owned(),
    })
}
