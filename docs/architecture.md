# How Volt Fits Together

Volt keeps several meanings of an options portfolio synchronized without
pretending they are interchangeable. A position has a listed-contract
identity, a terminal payoff, a lifecycle, an execution realization, and a
possible graph representation. One model is authoritative; the other views
are derived from it.

```text
listed contracts + signed quantities
                 |
                 v
       CanonicalPortfolio
          |      |       \
          |      |        \ graph
          |      |         v
          |      |   PortfolioGraphSample
          |      |
          |      +----------> TerminalPayoffProjection
          |
          +----------> PortfolioState
                           |
              action + observation
                           |
                           v
                      CogRuntime
                       |      |
                       v      v
                  next state  journal entry
```

## Contract identity

`OptionContract` is the listed instrument definition. It records:

- a stable `ContractId`;
- underlying and currency;
- call or put right;
- strike in integer micro-units;
- contract multiplier;
- explicit European, Bermudan, or American exercise opportunities; and
- cash or physical settlement.

Volt does not infer these details from a display symbol. Two contracts with
different identities or lifecycle terms remain different even if they happen
to share a payoff at one spot.

## Exact canonicalization

`CanonicalPortfolio::from_positions` aggregates repeated occurrences of the
same exact contract, rejects conflicting definitions for one identifier,
removes zero net positions, and sorts the result. Its semantic hash is
therefore independent of input ordering.

This is exact equivalence. It is safe because no contract meaning is discarded.

## Terminal-payoff projection

`TerminalPayoffProjection` rewrites a compatible portfolio as:

```text
cash + underlying_units × spot + Σ weight × max(spot - strike, 0)
```

This second canonical form is weaker by design. It forgets early exercise,
assignment, liquidity, settlement realization, and execution history. It is
valuable for payoff comparison and search-space reduction, but two portfolios
with the same projection are not necessarily dynamically equivalent.

## Lifecycle runtime

`CogRuntime` applies one `TransitionAction` to one immutable `PortfolioState`
at a declared `MarketObservation`. It:

1. checks the input state and observation;
2. validates action authority and quantities;
3. computes typed cash, underlying, and consumed-right emissions;
4. applies those emissions to a cloned state;
5. checks the resulting invariants; and
6. links the old and new state hashes in a deterministic journal entry.

`replay` runs the same transition function over saved inputs. Replay is not a
second implementation.

## Scenario layer

`ScenarioSimulator` executes a schedule of hold, close, exercise, assignment,
and expiry actions against a versioned path. Fill semantics are explicit:

- `DeclaredMark` is a synthetic or proxy realization;
- `CrossAtTouch` buys at the ask and sells at the bid, failing closed when the
  required quote side is absent.

Transaction costs are separate from fills. Results reconcile cash, underlying
holdings, option marks, realized actions, and leg attribution.

## Search grammar

`CandidateGrammar` compiles a small typed action vocabulary against a
point-in-time listed universe:

```text
AddLeg | RemoveLeg | ResizeLeg | Wait | Exercise
```

An `AddLeg` bucket may compile to several exact listed contracts. Volt returns
every branch in stable identity order; it never silently chooses one.

This is a candidate-action compiler, not yet a general financial-logic
compiler. The short program name `cog` is reserved for that larger compiler
when its language and output contract exist.

## Graph view

`PortfolioGraphSample` exports the same portfolio as typed nodes and edges for
optimization or learning. The graph carries both exact-portfolio and
terminal-payoff hashes so downstream systems can use either equivalence
without confusing them.

Learning systems may score or propose portfolios. They do not redefine
contract, payoff, lifecycle, or execution semantics.

The optional [supervised graph-learning experiment](supervised-gnn.md) runs
this boundary end to end: candidate-action compilation, exact
canonicalization, graph export, relation-aware message passing, held-out
evaluation, and embedding export. The experiment does not implement the
reserved general `cog` compiler or an RL policy.

## Determinism and versioning

Serialized public structures carry schema versions. Semantic hashes use
canonical JSON with a versioned domain separator. Money is stored in integer
micro-units, and time in integer Unix-epoch nanoseconds. This avoids hidden
floating-point rounding in identity, journals, and cash ledgers.

The boundary is intentional: Volt describes and executes options logic, but it
does not contain brokerage connectivity, account policy, or order authority.
