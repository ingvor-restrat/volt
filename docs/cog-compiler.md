# COG Compiler

COG means **Contract Object Graph**. The compiler's first implemented language
is a versioned structured format for options-design business logic.

The source language describes:

- the exact listed-contract universe;
- an action grammar over expiry, strike, moneyness, or delta buckets;
- an initial exact portfolio;
- prioritized guarded business rules; and
- an optional solver-independent scenario objective declaration.

The source is compiled before execution or learning:

```text
volt.cog_source.v1 JSON
        |
        v
parse + aggregate diagnostics
        |
        v
validated typed source
        |
        v
normalized Financial Logic IR
        |
        v
volt.compiled_cog.v1
        |
        +--> initial PortfolioState
        +--> point-in-time action set + hard legal mask
        +--> canonical design or journaled runtime transition
        +--> portfolio and state/action graph exports
```

This is a real compiler, not a formatter renamed `cog`. It has a source
contract, diagnostics, canonical artifact, execution contract, and stable
content hashes. It is intentionally an options-design compiler first; the
broader specification's arbitrary typed state and emission expression language
remains a later semantic expansion.

## Source program

The reference program is
[`examples/cogs/protective-put.json`](../examples/cogs/protective-put.json).
Its rules state, in structured form:

1. below $100 spot and within a gross-quantity limit, admit a long 90 put;
2. above $110, admit a short 110 call;
3. while the existing 100 call exercise right is open, admit exercise; and
4. always retain a wait branch.

The compiler supports these typed guard expressions:

- `always`, `all`, `any`, and `not`;
- inclusive spot bounds;
- live-leg-count bounds;
- gross-quantity bounds;
- exact live-position and side checks; and
- exact exercise-opportunity checks.

Actions use the existing typed candidate vocabulary:

- `add_leg`;
- `remove_leg`;
- `resize_leg`;
- `wait`; and
- `exercise`.

Bucket references and exact-contract references are checked during compilation.
Point-in-time bucket resolution and state-dependent legality are checked when
the compiled artifact produces an action set.

## Diagnostics and immutable artifact

`CogCompiler::compile_json` parses source and returns every collected
`CompilerDiagnostic`. Diagnostics have:

- `error` or `warning` severity;
- a stable code such as `E0502_DUPLICATE_RULE`;
- a JSON-style source path; and
- a human-readable explanation.

An artifact is emitted only when there are no errors. The compiler:

- validates and sorts the listed universe;
- canonicalizes the initial portfolio;
- validates all guards and actions;
- orders rules by descending priority and stable identifier;
- hashes each compiled rule;
- retains the canonical source hash; and
- hashes the complete immutable artifact body.

Both source and artifact hashes remain attached to compiler-bound graph
samples and learned embeddings.

## Execution boundary

The compiler separates two kinds of transitions that were previously easy to
conflate:

### Design transitions

`AddLeg`, `RemoveLeg`, and `ResizeLeg` transform a
`CanonicalPortfolio`. They describe exact candidate structures, not fills or
broker orders.

### Lifecycle transitions

`Wait` executes as a journaled `Hold`, while `Exercise` delegates to the
existing `CogRuntime`. Runtime transitions retain invariant checks, emissions,
state hashes, and journal replay.

This distinction is deliberate. A future execution adapter may attach fills
and transaction costs to an approved design transition; the compiler does not
invent those observations.

## State/action graph

`StateActionGraphSample` combines:

- the exact live portfolio graph;
- runtime sequence, cash, and delivered-underlying state;
- every compiler-owned action slot;
- typed action-to-state edges; and
- a boolean `legal_action_mask` aligned one-to-one with the action nodes.

Rejected rules remain in the graph with a stable rejection reason. A GNN or RL
policy may score the slots, but it cannot make a rejected slot legal.

## Supervised integration

The supervised generator now supports compiler-bound rows:

```bash
cargo run --example graph -- \
  --compiled \
  --output artifacts/supervised_gnn/graphs.jsonl
```

Each `volt.compiled_portfolio_graph.v1` row contains:

- `source_hash`;
- `compiled_cog_hash`;
- the exact nested `PortfolioGraphSample`;
- exact portfolio and terminal-payoff hashes; and
- the declared supervised target.

The Python trainer accepts the compiled wrapper and writes compiler provenance
into every embedding:

```bash
PYTHON=.venv/bin/python scripts/run_supervised_demo.sh
```

All 729 reference samples are now compiler-bound. The features, split policy,
and target remain unchanged, so the verified held-out metrics remain directly
comparable with the earlier graph-only experiment.

## Run the complete compiler example

```bash
cargo run --example cog -- \
  --source examples/cogs/protective-put.json \
  --output /tmp/volt-cog.json

jq '{
  artifact: .compiled_cog.artifact_hash,
  legal_mask: .action_set.legal_action_mask,
  actions: [.state_action_graph.action_nodes[] |
    {rule_id, action_kind, legal, rejection_reason}],
  journal_rule: .executed_journal_entry.rule_id
}' /tmp/volt-cog.json
```

At the default $95 observation, the mask is:

```text
add-downside-put  true
add-upside-call   false
exercise-call     true
wait              true
```

## Phases 1–6

| Phase | Implemented result |
|---|---|
| 1. Minimal source | `CogSource` and `volt.cog_source.v1` structured JSON |
| 2. Compiler | diagnostics, canonical rules, `CompiledCog`, source/artifact hashes |
| 3. Execution | exact design transitions plus journaled wait/exercise runtime transitions |
| 4. Graph export | compiler-bound portfolio graphs with complete provenance |
| 5. Supervised GNN | trainer consumes compiled graphs and exports provenance with embeddings |
| 6. State/action graph | action nodes, typed edges, rejected slots, and hard legal mask |

RL is not implemented. The state/action graph and legal-mask boundary are the
inputs an RL policy would use later.
