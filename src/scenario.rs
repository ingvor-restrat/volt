//! Reproducible path simulation built on the auditable runtime.
//!
//! A simulation request declares every path point, fill rule, transaction-cost
//! rule, action, and assumption needed to regenerate its result. Prices and P&L
//! use integer micro-units so repeated runs do not drift through floating-point
//! arithmetic.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::contract::{ContractId, Currency, Position, PriceMicros, TimestampNs};
use crate::hash::canonical_hash;
use crate::market::{MarkBasis, OrderSide};
use crate::runtime::{
    CogRuntime, ContractLifecycle, Emission, JournalEntry, MarketObservation, PortfolioState,
    TransitionAction,
};
use crate::{CogError, CogResult};

/// Schema tag for [`ScenarioPath`].
pub const SCENARIO_PATH_SCHEMA_VERSION: &str = "volt.scenario_path.v1";
/// Schema tag for [`SimulationRequest`].
pub const SIMULATION_REQUEST_SCHEMA_VERSION: &str = "volt.simulation_request.v1";
/// Schema tag for [`SimulationResult`].
pub const SIMULATION_RESULT_SCHEMA_VERSION: &str = "volt.simulation_result.v1";

/// Rule used to turn declared scenario prices into deterministic fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeterministicFillModel {
    /// Use the declared point-in-time mark as a synthetic/proxy fill.
    DeclaredMark,
    /// Buy at the ask and sell at the bid. Both required sides must be present.
    CrossAtTouch,
}

impl DeterministicFillModel {
    fn fill_price(self, price: &ScenarioOptionPrice, side: OrderSide) -> CogResult<PriceMicros> {
        match self {
            Self::DeclaredMark => Ok(price.mark),
            Self::CrossAtTouch => match side {
                OrderSide::Buy => price.ask.ok_or_else(|| {
                    CogError::DataSource(format!(
                        "ask is unavailable for executable buy of {}",
                        price.contract_id.0
                    ))
                }),
                OrderSide::Sell => price.bid.ok_or_else(|| {
                    CogError::DataSource(format!(
                        "bid is unavailable for executable sell of {}",
                        price.contract_id.0
                    ))
                }),
            },
        }
    }
}

/// Deterministic transaction-cost rule applied to each fill.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransactionCostModel {
    /// Stable identifier for the rule and parameterization.
    pub model_id: String,
    /// Variable cost per contract, in millionths of a currency unit.
    pub per_contract_side_micros: i128,
    /// Minimum cost per fill, in millionths of a currency unit.
    pub minimum_ticket_micros: i128,
}

impl TransactionCostModel {
    /// Returns a named model that charges no transaction costs.
    pub fn zero() -> Self {
        Self {
            model_id: "zero.v1".to_owned(),
            per_contract_side_micros: 0,
            minimum_ticket_micros: 0,
        }
    }

    /// Calculates the total cost for a positive contract quantity.
    pub fn cost_micros(&self, quantity: i64) -> CogResult<i128> {
        self.validate()?;
        if quantity <= 0 {
            return Err(CogError::InvalidAction(
                "transaction-cost quantity must be positive".to_owned(),
            ));
        }
        let variable = self
            .per_contract_side_micros
            .checked_mul(i128::from(quantity))
            .ok_or_else(|| CogError::NumericOverflow("transaction cost".to_owned()))?;
        Ok(variable.max(self.minimum_ticket_micros))
    }

    fn validate(&self) -> CogResult<()> {
        if self.model_id.trim().is_empty() {
            return Err(CogError::InvalidDefinition(
                "transaction-cost model_id must not be empty".to_owned(),
            ));
        }
        if self.per_contract_side_micros < 0 || self.minimum_ticket_micros < 0 {
            return Err(CogError::InvalidDefinition(
                "transaction costs must be non-negative".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Declared option price and executable quote at one scenario point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioOptionPrice {
    /// Exact contract to which these prices belong.
    pub contract_id: ContractId,
    /// Non-negative analytical or observed mark.
    pub mark: PriceMicros,
    /// Convention used to select the mark.
    pub mark_basis: MarkBasis,
    /// Optional positive executable bid.
    pub bid: Option<PriceMicros>,
    /// Optional positive executable ask.
    pub ask: Option<PriceMicros>,
}

impl ScenarioOptionPrice {
    fn validate(&self) -> CogResult<()> {
        if self.contract_id.0.trim().is_empty() {
            return Err(CogError::InvalidDefinition(
                "scenario option contract_id must not be empty".to_owned(),
            ));
        }
        if self.mark.0 < 0
            || self.bid.is_some_and(|price| price.0 <= 0)
            || self.ask.is_some_and(|price| price.0 <= 0)
        {
            return Err(CogError::InvalidDefinition(
                "scenario option prices must be non-negative marks and positive quote sides"
                    .to_owned(),
            ));
        }
        if let (Some(bid), Some(ask)) = (self.bid, self.ask) {
            if bid > ask {
                return Err(CogError::InvalidDefinition(format!(
                    "scenario quote is crossed for {}",
                    self.contract_id.0
                )));
            }
        }
        Ok(())
    }
}

/// Complete market observation at one point on a scenario path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioPoint {
    /// Stable event identifier, unique within the path.
    pub event_id: String,
    /// Point time in nanoseconds since the Unix epoch.
    pub time: TimestampNs,
    /// Underlying spot in millionths of one currency unit.
    pub spot: PriceMicros,
    /// Option prices keyed by exact contract identifier.
    pub option_prices: BTreeMap<ContractId, ScenarioOptionPrice>,
}

impl ScenarioPoint {
    fn validate(&self) -> CogResult<()> {
        if self.event_id.trim().is_empty() {
            return Err(CogError::InvalidDefinition(
                "scenario event_id must not be empty".to_owned(),
            ));
        }
        if self.time.0 < 0 || self.spot.0 < 0 {
            return Err(CogError::InvalidDefinition(
                "scenario time and spot must be non-negative".to_owned(),
            ));
        }
        for (contract_id, price) in &self.option_prices {
            price.validate()?;
            if contract_id != &price.contract_id {
                return Err(CogError::InvalidDefinition(format!(
                    "scenario option-price key does not match {}",
                    price.contract_id.0
                )));
            }
        }
        Ok(())
    }
}

/// Ordered, reproducible path of market observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioPath {
    /// Serialization and hashing schema.
    pub schema_version: String,
    /// Caller-supplied scenario identifier.
    pub scenario_id: String,
    /// Seed recorded for reproducibility, even for externally supplied points.
    pub seed: u64,
    /// Non-empty points in strictly increasing time order.
    pub points: Vec<ScenarioPoint>,
}

impl ScenarioPath {
    /// Constructs and validates a scenario path.
    pub fn new(
        scenario_id: impl Into<String>,
        seed: u64,
        points: Vec<ScenarioPoint>,
    ) -> CogResult<Self> {
        let path = Self {
            schema_version: SCENARIO_PATH_SCHEMA_VERSION.to_owned(),
            scenario_id: scenario_id.into(),
            seed,
            points,
        };
        path.validate()?;
        Ok(path)
    }

    /// Returns the versioned canonical content hash of the path.
    pub fn semantic_hash(&self) -> CogResult<String> {
        self.validate()?;
        canonical_hash(SCENARIO_PATH_SCHEMA_VERSION, self)
    }

    /// Checks schema, identity, event uniqueness, prices, and time ordering.
    pub fn validate(&self) -> CogResult<()> {
        if self.schema_version != SCENARIO_PATH_SCHEMA_VERSION {
            return Err(CogError::InvalidDefinition(format!(
                "unsupported scenario path schema {}",
                self.schema_version
            )));
        }
        if self.scenario_id.trim().is_empty() || self.points.is_empty() {
            return Err(CogError::InvalidDefinition(
                "scenario id and at least one point are required".to_owned(),
            ));
        }
        let mut event_ids = BTreeSet::new();
        for (index, point) in self.points.iter().enumerate() {
            point.validate()?;
            if !event_ids.insert(point.event_id.as_str()) {
                return Err(CogError::InvalidDefinition(format!(
                    "duplicate scenario event_id {}",
                    point.event_id
                )));
            }
            if index > 0 && self.points[index - 1].time >= point.time {
                return Err(CogError::InvalidDefinition(
                    "scenario points must have strictly increasing times".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

/// Portfolio action scheduled at a scenario point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ScenarioAction {
    /// Advance through the point without an economic action.
    Hold,
    /// Close part or all of a live option position.
    Close {
        /// Exact contract to close.
        contract_id: ContractId,
        /// Positive number of contracts to consume.
        quantity: i64,
    },
    /// Exercise part or all of a long option.
    Exercise {
        /// Exact contract to exercise.
        contract_id: ContractId,
        /// Positive number of contracts to consume.
        quantity: i64,
    },
    /// Assign part or all of a short option.
    Assign {
        /// Exact contract to assign.
        contract_id: ContractId,
        /// Positive number of contracts to consume.
        quantity: i64,
    },
    /// Expire the complete remaining position without settlement.
    Expire {
        /// Exact contract to expire.
        contract_id: ContractId,
    },
}

/// Action paired with its zero-based scenario-point index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduledScenarioAction {
    /// Index into [`ScenarioPath::points`].
    pub point_index: usize,
    /// Action evaluated at the indexed point.
    pub action: ScenarioAction,
}

/// Complete, hashable input to [`ScenarioSimulator`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimulationRequest {
    /// Serialization and hashing schema.
    pub schema_version: String,
    /// Caller-supplied request identifier.
    pub request_id: String,
    /// Market path used by the run.
    pub path: ScenarioPath,
    /// One to three exact positions on one underlying and in one currency.
    pub positions: Vec<Position>,
    /// Actions evaluated at indexed path points.
    pub actions: Vec<ScheduledScenarioAction>,
    /// Rule used to determine opening and closing prices.
    pub fill_model: DeterministicFillModel,
    /// Rule used to charge each opening and closing fill.
    pub transaction_cost_model: TransactionCostModel,
    /// Non-empty free-form assumptions recorded in the manifest.
    pub assumptions: Vec<String>,
}

impl SimulationRequest {
    /// Constructs and validates a complete simulation request.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: impl Into<String>,
        path: ScenarioPath,
        positions: Vec<Position>,
        actions: Vec<ScheduledScenarioAction>,
        fill_model: DeterministicFillModel,
        transaction_cost_model: TransactionCostModel,
        assumptions: Vec<String>,
    ) -> CogResult<Self> {
        let request = Self {
            schema_version: SIMULATION_REQUEST_SCHEMA_VERSION.to_owned(),
            request_id: request_id.into(),
            path,
            positions,
            actions,
            fill_model,
            transaction_cost_model,
            assumptions,
        };
        request.validate()?;
        Ok(request)
    }

    /// Returns the versioned canonical content hash of the request.
    pub fn semantic_hash(&self) -> CogResult<String> {
        self.validate()?;
        canonical_hash(SIMULATION_REQUEST_SCHEMA_VERSION, self)
    }

    /// Checks schemas, path, model parameters, positions, and scheduled actions.
    pub fn validate(&self) -> CogResult<()> {
        if self.schema_version != SIMULATION_REQUEST_SCHEMA_VERSION {
            return Err(CogError::InvalidDefinition(format!(
                "unsupported simulation request schema {}",
                self.schema_version
            )));
        }
        if self.request_id.trim().is_empty() {
            return Err(CogError::InvalidDefinition(
                "simulation request_id must not be empty".to_owned(),
            ));
        }
        self.path.validate()?;
        self.transaction_cost_model.validate()?;
        if self.positions.is_empty() || self.positions.len() > 3 {
            return Err(CogError::InvalidDefinition(
                "M3 requires between one and three input positions".to_owned(),
            ));
        }
        let mut underlyings = BTreeSet::new();
        let mut currencies = BTreeSet::new();
        let mut contract_ids = BTreeSet::new();
        for position in &self.positions {
            position.contract.validate()?;
            if position.quantity == 0 {
                return Err(CogError::InvalidDefinition(
                    "simulation positions must be non-zero".to_owned(),
                ));
            }
            if !contract_ids.insert(position.contract.contract_id.clone()) {
                return Err(CogError::InvalidDefinition(format!(
                    "duplicate simulation contract {}",
                    position.contract.contract_id.0
                )));
            }
            underlyings.insert(position.contract.underlying.clone());
            currencies.insert(position.contract.currency.clone());
        }
        if underlyings.len() != 1 || currencies.len() != 1 {
            return Err(CogError::InvalidDefinition(
                "M3 requires exactly one underlying and one currency".to_owned(),
            ));
        }
        if self
            .assumptions
            .iter()
            .any(|assumption| assumption.trim().is_empty())
        {
            return Err(CogError::InvalidDefinition(
                "simulation assumptions must not contain empty values".to_owned(),
            ));
        }
        for scheduled in &self.actions {
            if scheduled.point_index >= self.path.points.len() {
                return Err(CogError::InvalidDefinition(format!(
                    "action point index {} is outside the scenario path",
                    scheduled.point_index
                )));
            }
            let contract_id = match &scheduled.action {
                ScenarioAction::Hold => continue,
                ScenarioAction::Close { contract_id, .. }
                | ScenarioAction::Exercise { contract_id, .. }
                | ScenarioAction::Assign { contract_id, .. }
                | ScenarioAction::Expire { contract_id } => contract_id,
            };
            if !contract_ids.contains(contract_id) {
                return Err(CogError::InvalidDefinition(format!(
                    "action refers to unknown contract {}",
                    contract_id.0
                )));
            }
        }
        Ok(())
    }
}

/// Executed opening or closing option trade.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioFill {
    /// Zero-based path-point index at which the fill occurs.
    pub point_index: usize,
    /// Identifier of the path point at which the fill occurs.
    pub event_id: String,
    /// Exact contract traded.
    pub contract_id: ContractId,
    /// Positive opens/increases a long; negative opens/increases a short.
    pub signed_quantity: i64,
    /// Buy or sell from the portfolio's perspective.
    pub side: OrderSide,
    /// Per-unit fill price in millionths of one currency unit.
    pub price: PriceMicros,
    /// Signed cash movement before costs, in micro-units.
    pub gross_cash_micros: i128,
    /// Non-negative transaction cost in micro-units.
    pub transaction_cost_micros: i128,
    /// Signed cash movement after costs, in micro-units.
    pub net_cash_micros: i128,
    /// `true` for initial portfolio acquisition, `false` for a scheduled close.
    pub opening_fill: bool,
}

/// Mark-to-market attribution for one option leg at one path point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegPathAttribution {
    /// Exact contract represented by this row.
    pub contract_id: ContractId,
    /// Signed contracts still open.
    pub open_quantity: i64,
    /// Current runtime lifecycle.
    pub lifecycle: ContractLifecycle,
    /// Cumulative attributed cash movements, in micro-units.
    pub cash_flow_micros: i128,
    /// Cumulative attributed physical underlying units.
    pub underlying_units: i64,
    /// Marked value of attributed underlying units, in micro-units.
    pub underlying_value_micros: i128,
    /// Marked value of the open option position, in micro-units.
    pub option_mark_value_micros: i128,
    /// Sum of cash flow and marked underlying and option values.
    pub equity_micros: i128,
}

/// Whole-portfolio P&L and leg attribution at one path point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathPnlSnapshot {
    /// Zero-based path-point index.
    pub point_index: usize,
    /// Identifier of the corresponding path point.
    pub event_id: String,
    /// Point time in nanoseconds since the Unix epoch.
    pub time: TimestampNs,
    /// Underlying spot in millionths of one currency unit.
    pub spot: PriceMicros,
    /// Portfolio cash ledger in micro-units.
    pub cash_micros: i128,
    /// Marked value of all physical underlying balances.
    pub underlying_value_micros: i128,
    /// Marked value of all live option balances.
    pub option_mark_value_micros: i128,
    /// Account equity relative to zero immediately before opening the portfolio.
    pub pnl_micros: i128,
    /// Per-contract reconciliation of the portfolio total.
    pub leg_attribution: Vec<LegPathAttribution>,
    /// Canonical hash of the runtime state at this point.
    pub state_hash: String,
}

/// Reproducibility inputs copied into a simulation result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimulationManifest {
    /// Canonical hash of the complete request.
    pub request_hash: String,
    /// Canonical hash of the scenario path.
    pub scenario_path_hash: String,
    /// Scenario seed recorded by the path.
    pub seed: u64,
    /// Fill rule used by the run.
    pub fill_model: DeterministicFillModel,
    /// Transaction-cost rule used by the run.
    pub transaction_cost_model: TransactionCostModel,
    /// Caller-declared assumptions used to interpret the run.
    pub assumptions: Vec<String>,
}

/// Complete deterministic output of a scenario run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimulationResult {
    /// Serialization and hashing schema.
    pub schema_version: String,
    /// Identifier copied from the request.
    pub request_id: String,
    /// Inputs required to identify and reproduce the run.
    pub manifest: SimulationManifest,
    /// Initial acquisition fills.
    pub opening_fills: Vec<ScenarioFill>,
    /// Fills generated by scheduled close actions.
    pub closing_fills: Vec<ScenarioFill>,
    /// Runtime audit entries generated along the path.
    pub journal: Vec<JournalEntry>,
    /// One marked P&L snapshot per scenario point.
    pub pathwise_pnl: Vec<PathPnlSnapshot>,
    /// Invariant-checked state after the final point.
    pub final_state: PortfolioState,
    /// P&L from the final path snapshot, in micro-units.
    pub terminal_pnl_micros: i128,
    /// Canonical hash of the complete result body.
    pub result_hash: String,
}

impl SimulationResult {
    /// Serializes the result as one compact JSON record.
    pub fn to_json_line(&self) -> CogResult<String> {
        serde_json::to_string(self).map_err(|error| CogError::Serialization(error.to_string()))
    }
}

#[derive(Debug, Clone, Default)]
struct LegLedger {
    cash_micros: i128,
    underlying_units: i64,
}

/// Stateless deterministic scenario runner.
#[derive(Debug, Default, Clone, Copy)]
pub struct ScenarioSimulator;

impl ScenarioSimulator {
    /// Validates and evaluates a simulation request.
    pub fn run(&self, request: &SimulationRequest) -> CogResult<SimulationResult> {
        request.validate()?;
        let request_hash = request.semantic_hash()?;
        let scenario_path_hash = request.path.semantic_hash()?;
        let first_point = &request.path.points[0];

        let mut state = PortfolioState::from_positions(request.positions.clone())?;
        let mut ledgers = request
            .positions
            .iter()
            .map(|position| (position.contract.contract_id.clone(), LegLedger::default()))
            .collect::<BTreeMap<_, _>>();
        let mut opening_fills = Vec::with_capacity(request.positions.len());

        for position in &request.positions {
            let price = price_for(first_point, &position.contract.contract_id)?;
            let side = if position.quantity > 0 {
                OrderSide::Buy
            } else {
                OrderSide::Sell
            };
            let fill_price = request.fill_model.fill_price(price, side)?;
            let quantity = checked_abs(position.quantity, "opening quantity")?;
            let transaction_cost_micros = request.transaction_cost_model.cost_micros(quantity)?;
            let gross_cash_micros =
                trade_cash_micros(position.quantity, position.contract.multiplier, fill_price)?;
            let net_cash_micros = gross_cash_micros
                .checked_sub(transaction_cost_micros)
                .ok_or_else(|| CogError::NumericOverflow("opening net cash".to_owned()))?;
            add_cash(&mut state, &position.contract.currency, net_cash_micros)?;
            let ledger = ledgers
                .get_mut(&position.contract.contract_id)
                .expect("ledger created for each validated position");
            ledger.cash_micros = net_cash_micros;
            opening_fills.push(ScenarioFill {
                point_index: 0,
                event_id: first_point.event_id.clone(),
                contract_id: position.contract.contract_id.clone(),
                signed_quantity: position.quantity,
                side,
                price: fill_price,
                gross_cash_micros,
                transaction_cost_micros,
                net_cash_micros,
                opening_fill: true,
            });
        }
        state.check_invariants()?;

        let mut actions_by_point: BTreeMap<usize, Vec<&ScenarioAction>> = BTreeMap::new();
        for scheduled in &request.actions {
            actions_by_point
                .entry(scheduled.point_index)
                .or_default()
                .push(&scheduled.action);
        }

        let runtime = CogRuntime;
        let mut closing_fills = Vec::new();
        let mut journal = Vec::new();
        let mut pathwise_pnl = Vec::with_capacity(request.path.points.len());

        for (point_index, point) in request.path.points.iter().enumerate() {
            let scheduled = actions_by_point.get(&point_index);
            if let Some(actions) = scheduled {
                for (action_index, action) in actions.iter().enumerate() {
                    let observation = runtime_observation(point, action_index);
                    let transition_action = match action {
                        ScenarioAction::Hold => TransitionAction::Hold,
                        ScenarioAction::Close {
                            contract_id,
                            quantity,
                        } => {
                            let position = state.positions.get(contract_id).ok_or_else(|| {
                                CogError::InvalidAction(format!(
                                    "unknown contract {}",
                                    contract_id.0
                                ))
                            })?;
                            let side = if position.open_quantity > 0 {
                                OrderSide::Sell
                            } else {
                                OrderSide::Buy
                            };
                            let price = price_for(point, contract_id)?;
                            let fill_price = request.fill_model.fill_price(price, side)?;
                            let transaction_cost_micros =
                                request.transaction_cost_model.cost_micros(*quantity)?;
                            let signed_quantity = quantity
                                .checked_mul(-position.open_quantity.signum())
                                .ok_or_else(|| {
                                    CogError::NumericOverflow("closing signed quantity".to_owned())
                                })?;
                            let gross_cash_micros = trade_cash_micros(
                                signed_quantity,
                                position.contract.multiplier,
                                fill_price,
                            )?;
                            let net_cash_micros = gross_cash_micros
                                .checked_sub(transaction_cost_micros)
                                .ok_or_else(|| {
                                    CogError::NumericOverflow("closing net cash".to_owned())
                                })?;
                            closing_fills.push(ScenarioFill {
                                point_index,
                                event_id: point.event_id.clone(),
                                contract_id: contract_id.clone(),
                                signed_quantity,
                                side,
                                price: fill_price,
                                gross_cash_micros,
                                transaction_cost_micros,
                                net_cash_micros,
                                opening_fill: false,
                            });
                            TransitionAction::Close {
                                contract_id: contract_id.clone(),
                                quantity: *quantity,
                                fill_price,
                                transaction_cost_micros,
                            }
                        }
                        ScenarioAction::Exercise {
                            contract_id,
                            quantity,
                        } => TransitionAction::Exercise {
                            contract_id: contract_id.clone(),
                            quantity: *quantity,
                        },
                        ScenarioAction::Assign {
                            contract_id,
                            quantity,
                        } => TransitionAction::Assign {
                            contract_id: contract_id.clone(),
                            quantity: *quantity,
                        },
                        ScenarioAction::Expire { contract_id } => TransitionAction::Expire {
                            contract_id: contract_id.clone(),
                        },
                    };
                    let result = runtime.transition(&state, &transition_action, &observation)?;
                    if let Some(contract_id) = action_contract_id(action) {
                        attribute_emissions(
                            ledgers
                                .get_mut(contract_id)
                                .expect("validated action contract has a ledger"),
                            &result.journal_entry.emissions,
                        )?;
                    }
                    state = result.state;
                    journal.push(result.journal_entry);
                }
            } else {
                let result = runtime.transition(
                    &state,
                    &TransitionAction::Hold,
                    &runtime_observation(point, 0),
                )?;
                state = result.state;
                journal.push(result.journal_entry);
            }

            pathwise_pnl.push(snapshot(
                point_index,
                point,
                &state,
                &ledgers,
                &request.positions,
            )?);
        }

        let terminal_pnl_micros = pathwise_pnl
            .last()
            .expect("validated scenario contains a point")
            .pnl_micros;
        let manifest = SimulationManifest {
            request_hash,
            scenario_path_hash,
            seed: request.path.seed,
            fill_model: request.fill_model,
            transaction_cost_model: request.transaction_cost_model.clone(),
            assumptions: request.assumptions.clone(),
        };
        let body = SimulationResultBody {
            schema_version: SIMULATION_RESULT_SCHEMA_VERSION,
            request_id: &request.request_id,
            manifest: &manifest,
            opening_fills: &opening_fills,
            closing_fills: &closing_fills,
            journal: &journal,
            pathwise_pnl: &pathwise_pnl,
            final_state: &state,
            terminal_pnl_micros,
        };
        let result_hash = canonical_hash(SIMULATION_RESULT_SCHEMA_VERSION, &body)?;
        Ok(SimulationResult {
            schema_version: SIMULATION_RESULT_SCHEMA_VERSION.to_owned(),
            request_id: request.request_id.clone(),
            manifest,
            opening_fills,
            closing_fills,
            journal,
            pathwise_pnl,
            final_state: state,
            terminal_pnl_micros,
            result_hash,
        })
    }
}

#[derive(Serialize)]
struct SimulationResultBody<'a> {
    schema_version: &'a str,
    request_id: &'a str,
    manifest: &'a SimulationManifest,
    opening_fills: &'a [ScenarioFill],
    closing_fills: &'a [ScenarioFill],
    journal: &'a [JournalEntry],
    pathwise_pnl: &'a [PathPnlSnapshot],
    final_state: &'a PortfolioState,
    terminal_pnl_micros: i128,
}

fn runtime_observation(point: &ScenarioPoint, action_index: usize) -> MarketObservation {
    MarketObservation {
        event_id: format!("{}:{action_index}", point.event_id),
        time: point.time,
        spot: point.spot,
    }
}

fn action_contract_id(action: &ScenarioAction) -> Option<&ContractId> {
    match action {
        ScenarioAction::Hold => None,
        ScenarioAction::Close { contract_id, .. }
        | ScenarioAction::Exercise { contract_id, .. }
        | ScenarioAction::Assign { contract_id, .. }
        | ScenarioAction::Expire { contract_id } => Some(contract_id),
    }
}

fn price_for<'a>(
    point: &'a ScenarioPoint,
    contract_id: &ContractId,
) -> CogResult<&'a ScenarioOptionPrice> {
    point.option_prices.get(contract_id).ok_or_else(|| {
        CogError::DataSource(format!(
            "scenario point {} has no price for live contract {}",
            point.event_id, contract_id.0
        ))
    })
}

fn checked_abs(quantity: i64, name: &str) -> CogResult<i64> {
    quantity
        .checked_abs()
        .ok_or_else(|| CogError::NumericOverflow(name.to_owned()))
}

/// Signed quantity is from the portfolio's perspective: positive is a buy,
/// negative is a sell. Cash therefore has the opposite sign.
fn trade_cash_micros(signed_quantity: i64, multiplier: u32, price: PriceMicros) -> CogResult<i128> {
    i128::from(signed_quantity)
        .checked_mul(i128::from(multiplier))
        .and_then(|amount| amount.checked_mul(i128::from(price.0)))
        .and_then(i128::checked_neg)
        .ok_or_else(|| CogError::NumericOverflow("trade cash".to_owned()))
}

fn add_cash(state: &mut PortfolioState, currency: &Currency, amount_micros: i128) -> CogResult<()> {
    if amount_micros == 0 {
        return Ok(());
    }
    let current = state.cash_micros.get(currency).copied().unwrap_or(0);
    let next = current
        .checked_add(amount_micros)
        .ok_or_else(|| CogError::NumericOverflow("scenario cash ledger".to_owned()))?;
    if next == 0 {
        state.cash_micros.remove(currency);
    } else {
        state.cash_micros.insert(currency.clone(), next);
    }
    Ok(())
}

fn attribute_emissions(ledger: &mut LegLedger, emissions: &[Emission]) -> CogResult<()> {
    for emission in emissions {
        match emission {
            Emission::Cash { amount_micros, .. } => {
                ledger.cash_micros = ledger
                    .cash_micros
                    .checked_add(*amount_micros)
                    .ok_or_else(|| CogError::NumericOverflow("leg cash attribution".to_owned()))?;
            }
            Emission::Underlying { units, .. } => {
                ledger.underlying_units =
                    ledger.underlying_units.checked_add(*units).ok_or_else(|| {
                        CogError::NumericOverflow("leg underlying attribution".to_owned())
                    })?;
            }
            Emission::RightConsumed { .. } => {}
        }
    }
    Ok(())
}

fn snapshot(
    point_index: usize,
    point: &ScenarioPoint,
    state: &PortfolioState,
    ledgers: &BTreeMap<ContractId, LegLedger>,
    initial_positions: &[Position],
) -> CogResult<PathPnlSnapshot> {
    let mut leg_attribution = Vec::with_capacity(initial_positions.len());
    let mut attributed_cash = 0_i128;
    let mut attributed_underlying_value = 0_i128;
    let mut attributed_option_value = 0_i128;

    for initial in initial_positions {
        let contract_id = &initial.contract.contract_id;
        let position = state
            .positions
            .get(contract_id)
            .expect("initial contract remains represented in state");
        let ledger = ledgers
            .get(contract_id)
            .expect("initial contract has attribution ledger");
        let underlying_value_micros = i128::from(ledger.underlying_units)
            .checked_mul(i128::from(point.spot.0))
            .ok_or_else(|| CogError::NumericOverflow("leg underlying value".to_owned()))?;
        let option_mark_value_micros = if position.lifecycle == ContractLifecycle::Live {
            let price = price_for(point, contract_id)?;
            i128::from(position.open_quantity)
                .checked_mul(i128::from(position.contract.multiplier))
                .and_then(|amount| amount.checked_mul(i128::from(price.mark.0)))
                .ok_or_else(|| CogError::NumericOverflow("leg option value".to_owned()))?
        } else {
            0
        };
        let equity_micros = ledger
            .cash_micros
            .checked_add(underlying_value_micros)
            .and_then(|amount| amount.checked_add(option_mark_value_micros))
            .ok_or_else(|| CogError::NumericOverflow("leg equity".to_owned()))?;
        attributed_cash = attributed_cash
            .checked_add(ledger.cash_micros)
            .ok_or_else(|| CogError::NumericOverflow("attributed cash".to_owned()))?;
        attributed_underlying_value = attributed_underlying_value
            .checked_add(underlying_value_micros)
            .ok_or_else(|| CogError::NumericOverflow("attributed underlying value".to_owned()))?;
        attributed_option_value = attributed_option_value
            .checked_add(option_mark_value_micros)
            .ok_or_else(|| CogError::NumericOverflow("attributed option value".to_owned()))?;
        leg_attribution.push(LegPathAttribution {
            contract_id: contract_id.clone(),
            open_quantity: position.open_quantity,
            lifecycle: position.lifecycle,
            cash_flow_micros: ledger.cash_micros,
            underlying_units: ledger.underlying_units,
            underlying_value_micros,
            option_mark_value_micros,
            equity_micros,
        });
    }

    let cash_micros = state
        .cash_micros
        .values()
        .try_fold(0_i128, |total, amount| {
            total
                .checked_add(*amount)
                .ok_or_else(|| CogError::NumericOverflow("portfolio cash".to_owned()))
        })?;
    let underlying_value_micros =
        state
            .underlying_units
            .values()
            .try_fold(0_i128, |total, units| {
                let value = i128::from(*units)
                    .checked_mul(i128::from(point.spot.0))
                    .ok_or_else(|| {
                        CogError::NumericOverflow("portfolio underlying value".to_owned())
                    })?;
                total
                    .checked_add(value)
                    .ok_or_else(|| CogError::NumericOverflow("portfolio equity".to_owned()))
            })?;
    let option_mark_value_micros = attributed_option_value;
    if cash_micros != attributed_cash || underlying_value_micros != attributed_underlying_value {
        return Err(CogError::InvariantViolation(
            "leg attribution does not reconcile to portfolio ledgers".to_owned(),
        ));
    }
    let pnl_micros = cash_micros
        .checked_add(underlying_value_micros)
        .and_then(|amount| amount.checked_add(option_mark_value_micros))
        .ok_or_else(|| CogError::NumericOverflow("pathwise PnL".to_owned()))?;
    let attributed_equity = leg_attribution.iter().try_fold(0_i128, |total, leg| {
        total
            .checked_add(leg.equity_micros)
            .ok_or_else(|| CogError::NumericOverflow("attributed equity".to_owned()))
    })?;
    if pnl_micros != attributed_equity {
        return Err(CogError::InvariantViolation(
            "leg equity does not reconcile to portfolio PnL".to_owned(),
        ));
    }

    Ok(PathPnlSnapshot {
        point_index,
        event_id: point.event_id.clone(),
        time: point.time,
        spot: point.spot,
        cash_micros,
        underlying_value_micros,
        option_mark_value_micros,
        pnl_micros,
        leg_attribution,
        state_hash: state.state_hash()?,
    })
}
