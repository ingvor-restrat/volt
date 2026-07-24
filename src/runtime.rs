//! Auditable state transitions for option portfolios.
//!
//! Every accepted action produces a new invariant-checked state and a
//! content-addressed journal entry. Replaying the same ordered inputs therefore
//! yields the same state and journal hashes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::canonical::CanonicalPortfolio;
use crate::contract::{
    ContractId, Currency, OptionContract, OptionRight, Position, PriceMicros, Settlement,
    TimestampNs, UnderlyingId,
};
use crate::hash::canonical_hash;
use crate::{CogError, CogResult};

/// Schema tag used when hashing and serializing [`PortfolioState`].
pub const RUNTIME_STATE_SCHEMA_VERSION: &str = "volt.runtime_state.v1";
/// Schema tag used when hashing and serializing [`JournalEntry`].
pub const JOURNAL_SCHEMA_VERSION: &str = "volt.journal_entry.v1";

/// Current lifecycle of an option position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractLifecycle {
    /// The contract still has non-zero open quantity.
    Live,
    /// A long option was fully exercised.
    Exercised,
    /// A short option was fully assigned.
    Assigned,
    /// The remaining right expired without settlement.
    Expired,
    /// The position was fully closed by a market trade.
    Closed,
}

/// Runtime state retained for one exact option contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PositionState {
    /// Exact contract definition.
    pub contract: OptionContract,
    /// Signed contracts remaining: positive is long, negative is short.
    pub open_quantity: i64,
    /// Lifecycle consistent with `open_quantity`.
    pub lifecycle: ContractLifecycle,
}

/// Complete deterministic portfolio state at one journal sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortfolioState {
    /// Serialization and hashing schema.
    pub schema_version: String,
    /// Number of transitions applied since construction.
    pub sequence: u64,
    /// Time of the latest observation, if any.
    pub as_of: Option<TimestampNs>,
    /// Option state keyed by exact contract identifier.
    pub positions: BTreeMap<ContractId, PositionState>,
    /// Accumulated cash by currency, in millionths of a currency unit.
    pub cash_micros: BTreeMap<Currency, i128>,
    /// Accumulated physical underlying units by instrument.
    pub underlying_units: BTreeMap<UnderlyingId, i64>,
}

impl PortfolioState {
    /// Builds an initial live state from positions after exact canonicalization.
    pub fn from_positions(positions: impl IntoIterator<Item = Position>) -> CogResult<Self> {
        let canonical = CanonicalPortfolio::from_positions(positions)?;
        let positions = canonical
            .positions
            .into_iter()
            .map(|position| {
                (
                    position.contract.contract_id.clone(),
                    PositionState {
                        contract: position.contract,
                        open_quantity: position.quantity,
                        lifecycle: ContractLifecycle::Live,
                    },
                )
            })
            .collect();
        let state = Self {
            schema_version: RUNTIME_STATE_SCHEMA_VERSION.to_owned(),
            sequence: 0,
            as_of: None,
            positions,
            cash_micros: BTreeMap::new(),
            underlying_units: BTreeMap::new(),
        };
        state.check_invariants()?;
        Ok(state)
    }

    /// Returns the versioned canonical content hash of this state.
    pub fn state_hash(&self) -> CogResult<String> {
        canonical_hash(RUNTIME_STATE_SCHEMA_VERSION, self)
    }

    /// Checks schema, contract, lifecycle, and normalized-ledger invariants.
    pub fn check_invariants(&self) -> CogResult<()> {
        if self.schema_version != RUNTIME_STATE_SCHEMA_VERSION {
            return Err(CogError::InvariantViolation(format!(
                "unsupported runtime state schema {}",
                self.schema_version
            )));
        }
        for (id, position) in &self.positions {
            if id != &position.contract.contract_id {
                return Err(CogError::InvariantViolation(format!(
                    "position map key does not match contract {}",
                    position.contract.contract_id.0
                )));
            }
            position.contract.validate()?;
            match position.lifecycle {
                ContractLifecycle::Live if position.open_quantity == 0 => {
                    return Err(CogError::InvariantViolation(format!(
                        "live contract {} has zero quantity",
                        id.0
                    )));
                }
                ContractLifecycle::Live => {}
                _ if position.open_quantity != 0 => {
                    return Err(CogError::InvariantViolation(format!(
                        "terminal contract {} has remaining quantity",
                        id.0
                    )));
                }
                _ => {}
            }
        }
        if self.cash_micros.values().any(|amount| *amount == 0)
            || self.underlying_units.values().any(|units| *units == 0)
        {
            return Err(CogError::InvariantViolation(
                "zero ledger entries must be removed".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Market observation paired with one runtime transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarketObservation {
    /// Stable identifier supplied by the event source.
    pub event_id: String,
    /// Observation time in nanoseconds since the Unix epoch.
    pub time: TimestampNs,
    /// Underlying spot in millionths of one currency unit.
    pub spot: PriceMicros,
}

impl MarketObservation {
    fn validate(&self) -> CogResult<()> {
        if self.event_id.trim().is_empty() {
            return Err(CogError::InvalidDefinition(
                "observation event_id must not be empty".to_owned(),
            ));
        }
        if self.time.0 < 0 {
            return Err(CogError::InvalidDefinition(
                "observation time must be non-negative".to_owned(),
            ));
        }
        if self.spot.0 < 0 {
            return Err(CogError::InvalidDefinition(
                "observation spot must be non-negative".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Action interpreted by [`CogRuntime`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum TransitionAction {
    /// Advance observation time without changing economic balances.
    Hold,
    /// Close part or all of a live long or short option position.
    Close {
        /// Exact contract to close.
        contract_id: ContractId,
        /// Positive number of contracts to consume.
        quantity: i64,
        /// Per-unit option fill in millionths of one currency unit.
        fill_price: PriceMicros,
        /// Non-negative total cost deducted from close proceeds.
        transaction_cost_micros: i128,
    },
    /// Exercise part or all of a live long option.
    Exercise {
        /// Exact contract to exercise.
        contract_id: ContractId,
        /// Positive number of contracts to consume.
        quantity: i64,
    },
    /// Assign part or all of a live short option.
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

/// One action and the observation under which it is evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionInput {
    /// Requested state transition.
    pub action: TransitionAction,
    /// Market time and spot supplied to the transition.
    pub observation: MarketObservation,
}

/// Economic effect emitted by an accepted transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "emission", rename_all = "snake_case")]
pub enum Emission {
    /// Cash-ledger movement.
    Cash {
        /// Currency whose ledger changes.
        currency: Currency,
        /// Signed cash movement in millionths of a currency unit.
        amount_micros: i128,
        /// Stable human-readable cause.
        reason: String,
    },
    /// Physical-underlying movement.
    Underlying {
        /// Instrument whose ledger changes.
        underlying: UnderlyingId,
        /// Signed number of units delivered or received.
        units: i64,
        /// Stable human-readable cause.
        reason: String,
    },
    /// Record that option rights were consumed without directly changing a ledger.
    RightConsumed {
        /// Contract whose rights were consumed.
        contract_id: ContractId,
        /// Positive number of contracts consumed.
        quantity: i64,
        /// Stable human-readable cause.
        reason: String,
    },
}

/// Content-addressed audit record for one accepted transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalEntry {
    /// Serialization and hashing schema.
    pub schema_version: String,
    /// Resulting state sequence.
    pub sequence: u64,
    /// Canonical hash of the state before the transition.
    pub prior_state_hash: String,
    /// Canonical hash of the state after the transition.
    pub next_state_hash: String,
    /// Action evaluated by the runtime.
    pub action: TransitionAction,
    /// Observation under which the action was evaluated.
    pub observation: MarketObservation,
    /// Ordered economic effects produced by the rule.
    pub emissions: Vec<Emission>,
    /// Versioned rule responsible for the transition.
    pub rule_id: String,
    /// Canonical hash of the complete journal-entry body.
    pub entry_hash: String,
}

/// New state and audit entry returned by one transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionResult {
    /// Invariant-checked state after the action.
    pub state: PortfolioState,
    /// Audit record linking prior and next state hashes.
    pub journal_entry: JournalEntry,
}

/// Final state and ordered journal returned by replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayResult {
    /// State after all supplied inputs.
    pub state: PortfolioState,
    /// One journal entry per supplied input.
    pub journal: Vec<JournalEntry>,
}

/// Stateless interpreter for deterministic portfolio transitions.
#[derive(Debug, Default, Clone, Copy)]
pub struct CogRuntime;

impl CogRuntime {
    /// Applies one action and returns the next state plus its journal entry.
    ///
    /// Observations may share a timestamp but may not move backward in time.
    pub fn transition(
        &self,
        state: &PortfolioState,
        action: &TransitionAction,
        observation: &MarketObservation,
    ) -> CogResult<TransitionResult> {
        state.check_invariants()?;
        observation.validate()?;
        if let Some(as_of) = state.as_of {
            if observation.time < as_of {
                return Err(CogError::InvalidAction(
                    "observation time precedes current state".to_owned(),
                ));
            }
        }

        let prior_state_hash = state.state_hash()?;
        let mut next = state.clone();
        let (emissions, rule_id) = match action {
            TransitionAction::Hold => (Vec::new(), "hold.v1".to_owned()),
            TransitionAction::Close {
                contract_id,
                quantity,
                fill_price,
                transaction_cost_micros,
            } => self.close_option(
                &mut next,
                contract_id,
                *quantity,
                *fill_price,
                *transaction_cost_micros,
            )?,
            TransitionAction::Exercise {
                contract_id,
                quantity,
            } => self.consume_option(
                &mut next,
                contract_id,
                *quantity,
                observation,
                ConsumptionKind::Exercise,
            )?,
            TransitionAction::Assign {
                contract_id,
                quantity,
            } => self.consume_option(
                &mut next,
                contract_id,
                *quantity,
                observation,
                ConsumptionKind::Assignment,
            )?,
            TransitionAction::Expire { contract_id } => {
                self.expire_option(&mut next, contract_id, observation)?
            }
        };

        for emission in &emissions {
            apply_emission(&mut next, emission)?;
        }
        next.sequence = next
            .sequence
            .checked_add(1)
            .ok_or_else(|| CogError::NumericOverflow("state sequence".to_owned()))?;
        next.as_of = Some(observation.time);
        next.check_invariants()?;
        let next_state_hash = next.state_hash()?;

        let body = JournalEntryBody {
            schema_version: JOURNAL_SCHEMA_VERSION,
            sequence: next.sequence,
            prior_state_hash: &prior_state_hash,
            next_state_hash: &next_state_hash,
            action,
            observation,
            emissions: &emissions,
            rule_id: &rule_id,
        };
        let entry_hash = canonical_hash(JOURNAL_SCHEMA_VERSION, &body)?;
        let journal_entry = JournalEntry {
            schema_version: JOURNAL_SCHEMA_VERSION.to_owned(),
            sequence: next.sequence,
            prior_state_hash,
            next_state_hash,
            action: action.clone(),
            observation: observation.clone(),
            emissions,
            rule_id,
            entry_hash,
        };
        Ok(TransitionResult {
            state: next,
            journal_entry,
        })
    }

    /// Replays ordered inputs from an initial state.
    pub fn replay(
        &self,
        initial: &PortfolioState,
        inputs: &[TransitionInput],
    ) -> CogResult<ReplayResult> {
        let mut state = initial.clone();
        let mut journal = Vec::with_capacity(inputs.len());
        for input in inputs {
            let result = self.transition(&state, &input.action, &input.observation)?;
            state = result.state;
            journal.push(result.journal_entry);
        }
        Ok(ReplayResult { state, journal })
    }

    fn close_option(
        &self,
        state: &mut PortfolioState,
        contract_id: &ContractId,
        quantity: i64,
        fill_price: PriceMicros,
        transaction_cost_micros: i128,
    ) -> CogResult<(Vec<Emission>, String)> {
        if quantity <= 0 {
            return Err(CogError::InvalidAction(
                "close quantity must be positive".to_owned(),
            ));
        }
        if fill_price.0 < 0 {
            return Err(CogError::InvalidAction(
                "close fill price must be non-negative".to_owned(),
            ));
        }
        if transaction_cost_micros < 0 {
            return Err(CogError::InvalidAction(
                "close transaction cost must be non-negative".to_owned(),
            ));
        }
        let position = state.positions.get_mut(contract_id).ok_or_else(|| {
            CogError::InvalidAction(format!("unknown contract {}", contract_id.0))
        })?;
        if position.lifecycle != ContractLifecycle::Live {
            return Err(CogError::InvalidAction(format!(
                "contract {} is not live",
                contract_id.0
            )));
        }
        let available = position.open_quantity.unsigned_abs();
        let quantity_unsigned = u64::try_from(quantity)
            .map_err(|_| CogError::NumericOverflow("close quantity".to_owned()))?;
        if quantity_unsigned > available {
            return Err(CogError::InvalidAction(
                "close quantity exceeds open position".to_owned(),
            ));
        }

        let economic_side = position.open_quantity.signum();
        let gross_cash_micros = i128::from(fill_price.0)
            .checked_mul(i128::from(position.contract.multiplier))
            .and_then(|amount| amount.checked_mul(i128::from(quantity)))
            .and_then(|amount| amount.checked_mul(i128::from(economic_side)))
            .ok_or_else(|| CogError::NumericOverflow("close gross cash".to_owned()))?;
        let net_cash_micros = gross_cash_micros
            .checked_sub(transaction_cost_micros)
            .ok_or_else(|| CogError::NumericOverflow("close net cash".to_owned()))?;

        position.open_quantity = position
            .open_quantity
            .checked_sub(
                quantity
                    .checked_mul(economic_side)
                    .ok_or_else(|| CogError::NumericOverflow("close position".to_owned()))?,
            )
            .ok_or_else(|| CogError::NumericOverflow("close position".to_owned()))?;
        if position.open_quantity == 0 {
            position.lifecycle = ContractLifecycle::Closed;
        }

        let mut emissions = nonzero_cash_emission(
            position.contract.currency.clone(),
            net_cash_micros,
            "position_close",
        );
        emissions.push(Emission::RightConsumed {
            contract_id: contract_id.clone(),
            quantity,
            reason: "position_close".to_owned(),
        });
        Ok((emissions, "position_close.v1".to_owned()))
    }

    fn consume_option(
        &self,
        state: &mut PortfolioState,
        contract_id: &ContractId,
        quantity: i64,
        observation: &MarketObservation,
        kind: ConsumptionKind,
    ) -> CogResult<(Vec<Emission>, String)> {
        if quantity <= 0 {
            return Err(CogError::InvalidAction(
                "consumed quantity must be positive".to_owned(),
            ));
        }
        let position = state.positions.get_mut(contract_id).ok_or_else(|| {
            CogError::InvalidAction(format!("unknown contract {}", contract_id.0))
        })?;
        if position.lifecycle != ContractLifecycle::Live {
            return Err(CogError::InvalidAction(format!(
                "contract {} is not live",
                contract_id.0
            )));
        }
        if !position
            .contract
            .exercise
            .permits_exercise(observation.time)
        {
            return Err(CogError::InvalidAction(format!(
                "exercise opportunity is closed for contract {}",
                contract_id.0
            )));
        }

        let economic_side = match kind {
            ConsumptionKind::Exercise => {
                if position.open_quantity <= 0 {
                    return Err(CogError::InvalidAction(
                        "holder exercise requires a long position".to_owned(),
                    ));
                }
                if quantity > position.open_quantity {
                    return Err(CogError::InvalidAction(
                        "exercise quantity exceeds long position".to_owned(),
                    ));
                }
                position.open_quantity -= quantity;
                1_i64
            }
            ConsumptionKind::Assignment => {
                if position.open_quantity >= 0 {
                    return Err(CogError::InvalidAction(
                        "assignment requires a short position".to_owned(),
                    ));
                }
                if quantity > -position.open_quantity {
                    return Err(CogError::InvalidAction(
                        "assignment quantity exceeds short position".to_owned(),
                    ));
                }
                position.open_quantity += quantity;
                -1_i64
            }
        };

        let contract = position.contract.clone();
        if position.open_quantity == 0 {
            position.lifecycle = match kind {
                ConsumptionKind::Exercise => ContractLifecycle::Exercised,
                ConsumptionKind::Assignment => ContractLifecycle::Assigned,
            };
        }
        let reason = match kind {
            ConsumptionKind::Exercise => "holder_exercise",
            ConsumptionKind::Assignment => "short_assignment",
        };
        let mut emissions =
            settlement_emissions(&contract, quantity, economic_side, observation.spot, reason)?;
        emissions.push(Emission::RightConsumed {
            contract_id: contract_id.clone(),
            quantity,
            reason: reason.to_owned(),
        });
        Ok((emissions, format!("{reason}.v1")))
    }

    fn expire_option(
        &self,
        state: &mut PortfolioState,
        contract_id: &ContractId,
        observation: &MarketObservation,
    ) -> CogResult<(Vec<Emission>, String)> {
        let position = state.positions.get_mut(contract_id).ok_or_else(|| {
            CogError::InvalidAction(format!("unknown contract {}", contract_id.0))
        })?;
        if position.lifecycle != ContractLifecycle::Live {
            return Err(CogError::InvalidAction(format!(
                "contract {} is not live",
                contract_id.0
            )));
        }
        if observation.time < position.contract.expiry() {
            return Err(CogError::InvalidAction(format!(
                "contract {} cannot expire before its expiry",
                contract_id.0
            )));
        }
        let quantity = position.open_quantity.unsigned_abs();
        let quantity = i64::try_from(quantity)
            .map_err(|_| CogError::NumericOverflow("expiration quantity".to_owned()))?;
        position.open_quantity = 0;
        position.lifecycle = ContractLifecycle::Expired;
        Ok((
            vec![Emission::RightConsumed {
                contract_id: contract_id.clone(),
                quantity,
                reason: "expiration_without_settlement".to_owned(),
            }],
            "expire_without_settlement.v1".to_owned(),
        ))
    }
}

#[derive(Debug, Clone, Copy)]
enum ConsumptionKind {
    Exercise,
    Assignment,
}

#[derive(Serialize)]
struct JournalEntryBody<'a> {
    schema_version: &'a str,
    sequence: u64,
    prior_state_hash: &'a str,
    next_state_hash: &'a str,
    action: &'a TransitionAction,
    observation: &'a MarketObservation,
    emissions: &'a [Emission],
    rule_id: &'a str,
}

fn settlement_emissions(
    contract: &OptionContract,
    quantity: i64,
    economic_side: i64,
    spot: PriceMicros,
    reason: &str,
) -> CogResult<Vec<Emission>> {
    let contracts_signed = quantity
        .checked_mul(economic_side)
        .ok_or_else(|| CogError::NumericOverflow("signed settlement quantity".to_owned()))?;
    let units = contracts_signed
        .checked_mul(i64::from(contract.multiplier))
        .ok_or_else(|| CogError::NumericOverflow("settlement units".to_owned()))?;

    match contract.settlement {
        Settlement::Cash => {
            let amount_micros = i128::from(contract.intrinsic_per_unit(spot))
                .checked_mul(i128::from(units))
                .ok_or_else(|| CogError::NumericOverflow("cash settlement".to_owned()))?;
            Ok(nonzero_cash_emission(
                contract.currency.clone(),
                amount_micros,
                reason,
            ))
        }
        Settlement::Physical => {
            let underlying_direction = match contract.right {
                OptionRight::Call => 1_i64,
                OptionRight::Put => -1_i64,
            };
            let underlying_units = units.checked_mul(underlying_direction).ok_or_else(|| {
                CogError::NumericOverflow("physical underlying settlement".to_owned())
            })?;
            let cash_micros = i128::from(underlying_units)
                .checked_mul(i128::from(contract.strike.0))
                .and_then(|amount| amount.checked_neg())
                .ok_or_else(|| CogError::NumericOverflow("physical cash settlement".to_owned()))?;
            let mut emissions =
                nonzero_cash_emission(contract.currency.clone(), cash_micros, reason);
            if underlying_units != 0 {
                emissions.push(Emission::Underlying {
                    underlying: contract.underlying.clone(),
                    units: underlying_units,
                    reason: reason.to_owned(),
                });
            }
            Ok(emissions)
        }
    }
}

fn nonzero_cash_emission(currency: Currency, amount_micros: i128, reason: &str) -> Vec<Emission> {
    if amount_micros == 0 {
        Vec::new()
    } else {
        vec![Emission::Cash {
            currency,
            amount_micros,
            reason: reason.to_owned(),
        }]
    }
}

fn apply_emission(state: &mut PortfolioState, emission: &Emission) -> CogResult<()> {
    match emission {
        Emission::Cash {
            currency,
            amount_micros,
            ..
        } => {
            if *amount_micros == 0 {
                return Ok(());
            }
            let current = state.cash_micros.get(currency).copied().unwrap_or(0);
            let next = current
                .checked_add(*amount_micros)
                .ok_or_else(|| CogError::NumericOverflow("cash ledger".to_owned()))?;
            if next == 0 {
                state.cash_micros.remove(currency);
            } else {
                state.cash_micros.insert(currency.clone(), next);
            }
        }
        Emission::Underlying {
            underlying, units, ..
        } => {
            if *units == 0 {
                return Ok(());
            }
            let current = state.underlying_units.get(underlying).copied().unwrap_or(0);
            let next = current
                .checked_add(*units)
                .ok_or_else(|| CogError::NumericOverflow("underlying ledger".to_owned()))?;
            if next == 0 {
                state.underlying_units.remove(underlying);
            } else {
                state.underlying_units.insert(underlying.clone(), next);
            }
        }
        Emission::RightConsumed { .. } => {}
    }
    Ok(())
}
