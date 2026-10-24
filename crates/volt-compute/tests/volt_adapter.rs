use volt::{
    CanonicalPortfolio, Currency, ExerciseSchedule, OptionContract, OptionRight, Position,
    PriceMicros, Settlement, TimestampNs, UnderlyingId,
};
use volt_compute::{
    OptionSide, reference_price,
    volt_adapter::{
        ContractPricingInput, MARKET_INPUTS_SCHEMA, OPTION_BATCH_SCHEMA, PortfolioMarketInputs,
        UnderlyingMarketInput, VALUATION_REQUEST_SCHEMA, VoltValuationRequest, YearBasis,
        project_portfolio,
    },
};

const AS_OF: TimestampNs = TimestampNs(1_800_000_000_000_000_000);
const YEAR_NS: i64 = 365 * 86_400 * 1_000_000_000;

fn contract(id: &str, right: OptionRight, settlement: Settlement) -> OptionContract {
    OptionContract::new(
        id,
        "XYZ",
        "USD",
        right,
        PriceMicros(100_000_000),
        100,
        ExerciseSchedule::European {
            expiry: TimestampNs(AS_OF.0 + YEAR_NS),
        },
        settlement,
    )
    .unwrap()
}

fn request() -> VoltValuationRequest {
    let call = contract("Z-C", OptionRight::Call, Settlement::Physical);
    let put = contract("A-P", OptionRight::Put, Settlement::Cash);
    let portfolio = CanonicalPortfolio::from_positions([
        Position::new(call.clone(), 2).unwrap(),
        Position::new(put, -2).unwrap(),
        Position::new(call, 1).unwrap(),
    ])
    .unwrap();
    let input = |id: &str| ContractPricingInput {
        contract_id: volt::ContractId(id.into()),
        observed_at: TimestampNs(AS_OF.0 - 100),
        volatility: 0.2,
        rate: 0.05,
        dividend_yield: 0.0,
    };
    VoltValuationRequest {
        schema_version: VALUATION_REQUEST_SCHEMA.into(),
        portfolio,
        market: PortfolioMarketInputs {
            schema_version: MARKET_INPUTS_SCHEMA.into(),
            snapshot_id: "fixture-snapshot".into(),
            as_of: AS_OF,
            max_observation_age_ns: 1_000_000_000,
            year_basis: YearBasis::Actual365Fixed,
            underlyings: vec![UnderlyingMarketInput {
                underlying: UnderlyingId("XYZ".into()),
                observed_at: TimestampNs(AS_OF.0 - 100),
                spot: PriceMicros(100_000_000),
            }],
            contracts: vec![input("Z-C"), input("A-P")],
        },
    }
}

#[test]
fn projection_preserves_canonical_contracts_units_and_known_values() {
    let request = request();
    let batch = request.project().unwrap();
    assert_eq!(batch.schema_version, OPTION_BATCH_SCHEMA);
    assert_eq!(
        batch.portfolio_hash,
        request.portfolio.semantic_hash().unwrap()
    );
    assert_eq!(batch.currency, Currency("USD".into()));
    assert_eq!(batch.as_of, AS_OF);
    assert_eq!(batch.year_basis, YearBasis::Actual365Fixed);
    assert_eq!(batch.rows.len(), 2);
    let put = &batch.rows[0];
    let call = &batch.rows[1];
    assert_eq!(put.contract_id, "A-P");
    assert_eq!(call.contract_id, "Z-C");
    assert!(matches!(put.side, OptionSide::Put));
    assert_eq!(call.quantity, 3);
    assert_eq!(put.quantity, -2);
    assert_eq!(call.multiplier, 100);
    assert_eq!(call.spot, 100.0);
    assert_eq!(call.strike, 100.0);
    assert_eq!(call.maturity_years, 1.0);
    assert_eq!(
        batch.source_contracts[1].contract.settlement,
        Settlement::Physical
    );
    assert_eq!(
        batch.source_contracts[0].contract.settlement,
        Settlement::Cash
    );
    for (source, row) in batch.source_contracts.iter().zip(&batch.rows) {
        assert_eq!(source.row_id, row.row_id);
        assert_eq!(source.contract.contract_id.0, row.contract_id);
        assert_eq!(source.quantity, row.quantity);
    }
    let portfolio_value = batch
        .rows
        .iter()
        .map(|row| reference_price(row).unwrap() * row.quantity as f64 * row.multiplier as f64)
        .sum::<f64>();
    let expected = 300.0 * 10.450583572185565 - 200.0 * 5.573526022256971;
    assert!((portfolio_value - expected).abs() < 1e-9);
}

#[test]
fn market_order_is_irrelevant_but_point_in_time_policy_is_part_of_provenance() {
    let mut request = request();
    let first = request.project().unwrap();
    request.market.contracts.reverse();
    let reordered = request.project().unwrap();
    assert_eq!(first.market_inputs_hash, reordered.market_inputs_hash);
    assert_eq!(first.rows[0].row_id, reordered.rows[0].row_id);
    request.market.max_observation_age_ns += 1;
    let changed = request.project().unwrap();
    assert_ne!(first.market_inputs_hash, changed.market_inputs_hash);
    assert_eq!(first.portfolio_hash, changed.portfolio_hash);
}

#[test]
fn preserves_settlement_identity_even_when_numeric_marks_match() {
    let physical = request();
    let mut cash = physical.clone();
    cash.portfolio.positions[1].contract.settlement = Settlement::Cash;
    let physical_batch = physical.project().unwrap();
    let cash_batch = cash.project().unwrap();
    assert_ne!(physical_batch.portfolio_hash, cash_batch.portfolio_hash);
    assert_eq!(
        reference_price(&physical_batch.rows[1]).unwrap(),
        reference_price(&cash_batch.rows[1]).unwrap()
    );
}

#[test]
fn rejects_future_stale_missing_duplicate_and_unrelated_market_inputs() {
    let valid = request();
    let mut cases = Vec::new();
    let mut bad = valid.clone();
    bad.market.contracts[0].observed_at = TimestampNs(AS_OF.0 + 1);
    cases.push(bad);
    let mut bad = valid.clone();
    bad.market.underlyings[0].observed_at = TimestampNs(AS_OF.0 - 1_000_000_001);
    cases.push(bad);
    let mut bad = valid.clone();
    bad.market.contracts.pop();
    cases.push(bad);
    let mut bad = valid.clone();
    bad.market.underlyings.clear();
    cases.push(bad);
    let mut bad = valid.clone();
    bad.market.contracts.push(bad.market.contracts[0].clone());
    cases.push(bad);
    let mut bad = valid.clone();
    bad.market
        .underlyings
        .push(bad.market.underlyings[0].clone());
    cases.push(bad);
    let mut bad = valid.clone();
    bad.market.contracts[0].contract_id = volt::ContractId("unknown".into());
    cases.push(bad);
    let mut bad = valid;
    bad.market.underlyings[0].underlying = UnderlyingId("unknown".into());
    cases.push(bad);
    for (index, bad) in cases.iter().enumerate() {
        assert!(bad.project().is_err(), "invalid market case {index}");
    }
}

#[test]
fn rejects_noncanonical_and_unsupported_portfolios() {
    let valid = request();
    let mut cases = Vec::new();
    let mut bad = valid.clone();
    bad.portfolio.positions.reverse();
    cases.push(bad);
    let mut bad = valid.clone();
    let mut conflict = bad.portfolio.positions[0].clone();
    conflict.contract.strike = PriceMicros(101_000_000);
    bad.portfolio.positions.push(conflict);
    cases.push(bad);
    let mut bad = valid.clone();
    bad.portfolio.positions[0].quantity = 0;
    cases.push(bad);
    let mut bad = valid.clone();
    bad.portfolio.positions[0].contract.currency = Currency("EUR".into());
    cases.push(bad);
    let mut bad = valid.clone();
    bad.portfolio.positions[0].contract.exercise = ExerciseSchedule::American {
        first_exercise: AS_OF,
        expiry: TimestampNs(AS_OF.0 + YEAR_NS),
    };
    cases.push(bad);
    let mut bad = valid.clone();
    bad.portfolio.positions[0].contract.exercise = ExerciseSchedule::European { expiry: AS_OF };
    cases.push(bad);
    let mut bad = valid;
    bad.portfolio.positions.clear();
    cases.push(bad);
    for (index, bad) in cases.iter().enumerate() {
        assert!(bad.project().is_err(), "invalid portfolio case {index}");
    }
}

#[test]
fn rejects_incoherent_parameters_and_unknown_schemas() {
    let valid = request();
    let mut cases = Vec::new();
    let mut bad = valid.clone();
    bad.market.contracts[0].volatility = 0.0;
    cases.push(bad);
    let mut bad = valid.clone();
    bad.market.contracts[0].rate = f64::NAN;
    cases.push(bad);
    let mut bad = valid.clone();
    bad.market.contracts[0].rate = 0.06;
    cases.push(bad);
    let mut bad = valid.clone();
    bad.market.contracts[0].dividend_yield = 0.01;
    cases.push(bad);
    let mut bad = valid.clone();
    bad.market.underlyings[0].spot = PriceMicros(0);
    cases.push(bad);
    let mut bad = valid.clone();
    bad.market.max_observation_age_ns = -1;
    cases.push(bad);
    let mut bad = valid.clone();
    bad.market.schema_version = "unknown".into();
    cases.push(bad);
    let mut bad = valid.clone();
    bad.portfolio.schema_version = "unknown".into();
    cases.push(bad);
    let mut bad = valid;
    bad.schema_version = "unknown".into();
    cases.push(bad);
    for (index, bad) in cases.iter().enumerate() {
        assert!(
            bad.project().is_err(),
            "invalid parameter/schema case {index}"
        );
    }
}

#[test]
fn request_json_roundtrips_and_rejects_unknown_fields_and_year_basis() {
    let request = request();
    let serialized = serde_json::to_vec(&request).unwrap();
    let decoded: VoltValuationRequest = serde_json::from_slice(&serialized).unwrap();
    assert_eq!(
        decoded.project().unwrap().portfolio_hash,
        request.project().unwrap().portfolio_hash
    );
    let mut value = serde_json::to_value(&request).unwrap();
    value["market"]["year_basis"] = "unsupported_basis".into();
    assert!(serde_json::from_value::<VoltValuationRequest>(value).is_err());
    let mut value = serde_json::to_value(request).unwrap();
    value["unknown"] = 1.into();
    assert!(serde_json::from_value::<VoltValuationRequest>(value).is_err());
}

#[test]
fn distinct_maturities_can_use_a_term_structure() {
    let mut request = request();
    request.portfolio.positions[1].contract.exercise = ExerciseSchedule::European {
        expiry: TimestampNs(AS_OF.0 + 2 * YEAR_NS),
    };
    request.market.contracts[0].rate = 0.06;
    request.market.contracts[0].dividend_yield = 0.01;
    let batch = project_portfolio(&request.portfolio, &request.market).unwrap();
    assert_eq!(batch.rows[1].maturity_years, 2.0);
    assert_eq!(batch.rows[1].rate, 0.06);
}

#[test]
fn projected_portfolio_values_in_order_across_single_row_chunks() {
    let request = request();
    let batch = request.project().unwrap();
    let row_ids = batch
        .rows
        .iter()
        .map(|row| row.row_id.clone())
        .collect::<Vec<_>>();
    let target = std::env::var("VOLT_TEST_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .parse()
        .unwrap();
    let precision = if target == volt_compute::ExecutionTarget::Metal {
        volt_compute::Precision::F32
    } else {
        volt_compute::Precision::F64
    };
    let budget = volt_compute::budget::MemoryBudget {
        max_batch_rows: 1,
        ..Default::default()
    };
    let result =
        volt_compute::chunked::value_rows(&batch.rows, target, precision, &budget).unwrap();
    assert_eq!(result.plan.chunk_rows, 1);
    assert_eq!(result.stats.completed_chunks, 2);
    assert_eq!(result.prices.len(), batch.rows.len());
    let mut expected_total = 0.0;
    let mut gross = 0.0;
    for (row, actual) in batch.rows.iter().zip(&result.prices) {
        let expected = reference_price(row).unwrap();
        let tolerance = if precision == volt_compute::Precision::F64 {
            1e-10 + 1e-10 * expected.abs()
        } else {
            0.0002 + 0.0001 * expected.abs()
        };
        assert!((actual - expected).abs() <= tolerance);
        let amount = expected * row.quantity as f64 * row.multiplier as f64;
        expected_total += amount;
        gross += amount.abs();
    }
    assert!((result.portfolio_value - expected_total).abs() <= 1e-4 * gross);
    assert_eq!(
        batch.portfolio_hash,
        request.portfolio.semantic_hash().unwrap()
    );
    assert_eq!(
        batch
            .rows
            .iter()
            .map(|row| row.row_id.clone())
            .collect::<Vec<_>>(),
        row_ids
    );
}

#[test]
fn checks_forged_canonical_input_and_extreme_timestamp_bounds() {
    let valid = request();
    let mut duplicate = valid.clone();
    duplicate
        .portfolio
        .positions
        .push(duplicate.portfolio.positions[0].clone());
    assert!(duplicate.project().is_err());
    let mut bad_contract_schema = valid.clone();
    bad_contract_schema.portfolio.positions[0]
        .contract
        .schema_version = "unknown".into();
    assert!(bad_contract_schema.project().is_err());
    let mut negative_time = valid.clone();
    negative_time.market.as_of = TimestampNs(i64::MIN);
    assert!(negative_time.project().is_err());
    let mut longest_interval = valid;
    longest_interval.market.as_of = TimestampNs(0);
    longest_interval.market.max_observation_age_ns = 0;
    for input in &mut longest_interval.market.underlyings {
        input.observed_at = TimestampNs(0);
    }
    for input in &mut longest_interval.market.contracts {
        input.observed_at = TimestampNs(0);
    }
    for position in &mut longest_interval.portfolio.positions {
        position.contract.exercise = ExerciseSchedule::European {
            expiry: TimestampNs(i64::MAX),
        };
    }
    let batch = longest_interval.project().unwrap();
    assert!(batch.rows[0].maturity_years.is_finite());
    assert!(batch.rows[0].maturity_years > 292.0);
}
