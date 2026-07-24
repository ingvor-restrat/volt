//! `zap` removes representational duplication from a bounded option universe.
//!
//! It reports two reductions: ordered action sequences to exact canonical
//! portfolios, and exact portfolios to terminal-payoff classes.

use std::collections::BTreeSet;

use serde_json::json;
use volt::{
    CanonicalPortfolio, ExerciseSchedule, OptionContract, OptionRight, Position, PriceMicros,
    Settlement, TimestampNs,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let expiry = TimestampNs(1_800_000_000_000_000_000);
    let mut atomic_actions = Vec::new();
    let mut contracts = Vec::new();
    for right in [OptionRight::Call, OptionRight::Put] {
        for strike in [90_i64, 100, 110] {
            let right_code = match right {
                OptionRight::Call => "C",
                OptionRight::Put => "P",
            };
            let contract = OptionContract::new(
                format!("SPY-{right_code}-{strike}"),
                "SPY",
                "USD",
                right,
                PriceMicros(strike * 1_000_000),
                100,
                ExerciseSchedule::American {
                    first_exercise: TimestampNs(0),
                    expiry,
                },
                Settlement::Physical,
            )?;
            contracts.push(contract.clone());
            for quantity in [-1_i64, 1] {
                atomic_actions.push(Position::new(contract.clone(), quantity)?);
            }
        }
    }

    let mut raw_action_sequences = 0_usize;
    let mut exact_portfolios = BTreeSet::new();
    let mut terminal_payoffs = BTreeSet::new();
    for first in &atomic_actions {
        for second in &atomic_actions {
            raw_action_sequences += 1;
            let portfolio =
                CanonicalPortfolio::from_positions(vec![first.clone(), second.clone()])?;
            exact_portfolios.insert(portfolio.semantic_hash()?);
            terminal_payoffs.insert(payoff_class(&portfolio)?);
        }
    }

    let exact_reduction = 1.0 - exact_portfolios.len() as f64 / raw_action_sequences as f64;
    let payoff_reduction = 1.0 - terminal_payoffs.len() as f64 / raw_action_sequences as f64;

    let mut bounded_exact_portfolios = BTreeSet::new();
    let mut bounded_terminal_payoffs = BTreeSet::new();
    for q0 in -2_i64..=2 {
        for q1 in -2_i64..=2 {
            for q2 in -2_i64..=2 {
                for q3 in -2_i64..=2 {
                    for q4 in -2_i64..=2 {
                        for q5 in -2_i64..=2 {
                            let quantities = [q0, q1, q2, q3, q4, q5];
                            let positions = contracts
                                .iter()
                                .zip(quantities)
                                .filter(|(_, quantity)| *quantity != 0)
                                .map(|(contract, quantity)| {
                                    Position::new(contract.clone(), quantity).unwrap()
                                })
                                .collect::<Vec<_>>();
                            let portfolio = CanonicalPortfolio::from_positions(positions)?;
                            bounded_exact_portfolios.insert(portfolio.semantic_hash()?);
                            bounded_terminal_payoffs.insert(payoff_class(&portfolio)?);
                        }
                    }
                }
            }
        }
    }
    let bounded_realizations = 5_usize.pow(contracts.len() as u32);
    let bounded_projection_reduction =
        1.0 - bounded_terminal_payoffs.len() as f64 / bounded_exact_portfolios.len() as f64;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": "volt.search_space_reduction.v1",
            "underlying": "SPY",
            "ordered_two_action_experiment": {
                "atomic_actions": atomic_actions.len(),
                "sequence_length": 2,
                "raw_ordered_action_sequences": raw_action_sequences,
                "unique_exact_portfolios": exact_portfolios.len(),
                "unique_terminal_payoff_classes": terminal_payoffs.len(),
                "exact_portfolio_reduction_fraction": exact_reduction,
                "terminal_payoff_reduction_fraction": payoff_reduction
            },
            "bounded_quantity_vector_experiment": {
                "contracts": contracts.len(),
                "quantity_values_per_contract": [-2, -1, 0, 1, 2],
                "raw_execution_realizations": bounded_realizations,
                "unique_exact_portfolios": bounded_exact_portfolios.len(),
                "unique_terminal_payoff_classes": bounded_terminal_payoffs.len(),
                "terminal_projection_reduction_fraction": bounded_projection_reduction
            },
            "note": "Terminal payoff classes are projections, not dynamic American-option equivalence."
        }))?
    );
    Ok(())
}

fn payoff_class(portfolio: &CanonicalPortfolio) -> Result<String, Box<dyn std::error::Error>> {
    if portfolio.positions.is_empty() {
        return Ok("zero".to_owned());
    }
    let projection = portfolio.terminal_payoff_projection()?;
    if projection.cash_micros == 0
        && projection.underlying_units == 0
        && projection.call_knots.is_empty()
    {
        Ok("zero".to_owned())
    } else {
        Ok(projection.semantic_hash()?)
    }
}
