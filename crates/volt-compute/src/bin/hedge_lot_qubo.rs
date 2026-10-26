//! Small financial signed-lot allocation example and JSON interface.
use clap::Parser;
use std::{error::Error, fs::File, io::Read, path::PathBuf};
use volt_compute::{
    OptionRow, OptionSide,
    aad::analytic_greeks,
    hedge_lot_qubo::*,
    second_order::analytic_gamma,
    volatility_surface::{EuropeanInputs, european_price_and_vega},
};
#[derive(Parser)]
#[command(about = "Bounded Greek-risk/fee QUBO with exact financial lot feasibility")]
struct Args {
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    example: bool,
}
fn illustration() -> Result<HedgeLotProblem, Box<dyn Error>> {
    let row = |side| OptionRow {
        row_id: "illustration".into(),
        contract_id: "ATM3m".into(),
        side,
        spot: 100.,
        strike: 100.,
        maturity_years: 0.25,
        volatility: 0.2,
        rate: 0.,
        dividend_yield: 0.,
        quantity: 1,
        multiplier: 1,
    };
    let call = row(OptionSide::Call);
    let put = row(OptionSide::Put);
    let cg = analytic_greeks(&call)?;
    let pg = analytic_greeks(&put)?;
    let cv = |r: &OptionRow| {
        european_price_and_vega(
            &EuropeanInputs {
                spot: r.spot,
                strike: r.strike,
                maturity_years: r.maturity_years,
                rate: r.rate,
                dividend_yield: r.dividend_yield,
                side: r.side,
            },
            r.volatility,
        )
        .map(|x| x.0)
    };
    let call_price = cv(&call)?;
    let put_price = cv(&put)?;
    let context=RiskContext{underlying_id:"SYNTH".into(),currency:"USD".into(),generation:1,as_of_ns:0,max_quote_age_ns:0,greek_convention:"fixed_volatility".into(),provenance:"Synthetic BS illustration: S=K100,T=.25,sigma=.20,r=q0; initial book two short ATM straddles,100-unit contracts. Quotes and Greeks calculated by native CPU routines.".into()};
    let stock = QuotedInstrument {
        instrument_id: "stock".into(),
        quote_id: "stock-quote".into(),
        underlying_id: context.underlying_id.clone(),
        currency: context.currency.clone(),
        generation: 1,
        observed_at_ns: 0,
        greek_convention: context.greek_convention.clone(),
        eligible: true,
        mark: 100.,
        per_unit_risk: RiskVector {
            delta: 1.,
            gamma: 0.,
            vega_per_vol_point: 0.,
        },
        multiplier: 1,
        current_contracts: 0,
        min_contracts: -200,
        max_contracts: 200,
        proportional_fee: 0.0005,
        fixed_fee_per_selected_lot: 0.,
    };
    let option = QuotedInstrument {
        instrument_id: "ATM-call".into(),
        quote_id: "call-quote".into(),
        underlying_id: context.underlying_id.clone(),
        currency: context.currency.clone(),
        generation: 1,
        observed_at_ns: 0,
        greek_convention: context.greek_convention.clone(),
        eligible: true,
        mark: call_price,
        per_unit_risk: RiskVector {
            delta: cg.delta,
            gamma: analytic_gamma(&call)?,
            vega_per_vol_point: cg.vega / 100.,
        },
        multiplier: 100,
        current_contracts: -2,
        min_contracts: -2,
        max_contracts: 2,
        proportional_fee: 0.02,
        fixed_fee_per_selected_lot: 0.,
    };
    let mut lots = Vec::new();
    for q in [-200, -100, 100, 200] {
        lots.push(SignedLot {
            lot_id: format!("stock:{q}"),
            instrument: 0,
            contracts: q,
        });
    }
    for q in [-4, -2, 2, 4] {
        lots.push(SignedLot {
            lot_id: format!("call:{q}"),
            instrument: 1,
            contracts: q,
        });
    }
    Ok(HedgeLotProblem {
        context,
        instruments: vec![stock, option],
        lots,
        current_risk: RiskVector {
            delta: -200. * (cg.delta + pg.delta),
            gamma: -200. * (analytic_gamma(&call)? + analytic_gamma(&put)?),
            vega_per_vol_point: -2. * (cg.vega + pg.vega),
        },
        target_risk: RiskVector {
            delta: 0.,
            gamma: 0.,
            vega_per_vol_point: 0.,
        },
        risk_scales: [100., 10., 50.],
        risk_weights: [1.; 3],
        fee_weight: 0.001,
        cash_before: 200. * (call_price + put_price),
        cash_borrowing_limit: 0.,
    })
}
fn main() -> Result<(), Box<dyn Error>> {
    let a = Args::parse();
    if a.example {
        if a.input.is_some() {
            return Err("--example takes no input".into());
        }
        println!("{}", serde_json::to_string_pretty(&illustration()?)?);
        return Ok(());
    }
    let path = a
        .input
        .ok_or("--input required; use --example for an illustrative request")?;
    let mut bytes = Vec::new();
    File::open(path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 {
        return Err("lot request exceeds 1 MiB".into());
    }
    let p: HedgeLotProblem = serde_json::from_slice(&bytes)?;
    println!("{}", serde_json::to_string_pretty(&solve(&p)?)?);
    Ok(())
}
