# Financial lot allocation with a QUBO objective

A book with two short at-the-money straddles carries substantial gamma and vega exposure. Stock can change its delta, while an option trade can change all three sensitivities. `hedge_lot_qubo` compares a small set of signed contract lots, accounting for quoted fees and declared cash/inventory limits.

```sh
cargo run -p volt-compute --bin hedge_lot_qubo -- --example > /tmp/hedge-lots.json
cargo run -p volt-compute --bin hedge_lot_qubo -- --input /tmp/hedge-lots.json
cargo test -p volt-compute --test hedge_lot_qubo
```

The example calculates its option price and Greeks with Volt's native European routines. It uses stock 100, strike 100, three months remaining, 20% volatility and zero rates/dividends. Each option contract represents 100 underlying units. The initial book has two short calls and two short puts, with their premium in cash.

## What each binary choice means

A `SignedLot` contains an instrument index and a signed integer number of contracts. One bit selects that entire quoted lot. The example offers stock changes −200, −100, +100, +200 shares and call changes −4, −2, +2, +4 contracts. Holding the current book is the all-zero mask. At most one lot per instrument may be chosen in a decision.

`QuotedInstrument` supplies the per-underlying-unit mark and delta/gamma/vega, multiplier, current contracts, inventory limits and costs. Quantities retain integer contract units. Risk and cash contributions multiply contracts by the instrument multiplier exactly within the declared quantity caps.

All instruments must share an underlying, currency, Greek convention and risk generation. The validator checks quote eligibility, knowledge time and age. Quote-based allocation uses genuine price sensitivities; learning-feature conventions stay labelled in their own pipelines.

Greek units are:

| Coordinate | Unit |
|---|---|
|Delta|Underlying units|
|Gamma|Underlying units per one underlying price unit|
|Vega|Currency per volatility percentage point|

For example, a one-point volatility move is 20% to 21%. The portfolio risk vector is signed and includes the positions already held. `target_risk` defines the desired vector; each selected lot changes the current vector.

## The financial objective and matrix

Let `r` be current risk minus target risk and `g_i` the risk contribution of lot `i`. Positive scales `s_k` give each coordinate its economic normalization. With weights `w_k`, fee weight `a` and binary decisions `z_i`, the objective is

```text
F(z) = sum_k w_k * ((r_k + sum_i g_ik*z_i) / s_k)^2
       + a * sum_i fee_i*z_i.
```

Each lot's actual fee is

```text
abs(contracts_i * multiplier_i * mark_i) * proportional_fee_i
+ fixed_fee_per_selected_lot_i.
```

The risk part is dimensionless after normalization; the fee weight converts currency into those objective units. A user chooses these weights and scales according to the desired local exposures. This is a local Greek-risk/fee proxy; portfolio loss objectives and scenario repricing supply separate assessments.

The exported `QuboObjective` uses an **upper-triangular** convention:

```text
F(z) = constant + sum_i Q_ii*z_i + sum_(i<j) Q_ij*z_i*z_j
constant = sum_k w_k*(r_k/s_k)^2
Q_ii = sum_k w_k*(2*r_k*g_ik + g_ik²)/s_k² + a*fee_i
Q_ij = 2*sum_k w_k*g_ik*g_jk/s_k²,  i<j.
```

Lower-triangle entries are zero. A symmetric matrix representation would split each pair coefficient between its two off-diagonal entries. `QuboObjective::value(mask)` evaluates the declared convention. `evaluate(problem, mask)` independently recomputes risk and fees directly from the financial lots.

The matrix encodes the objective. Cash, inventory and one-lot-per-instrument constraints are enforced as hard filters by the exact enumerator. The exported flag `feasibility_constraints_encoded=false` makes that separation explicit. Encoding general cash constraints in an unconstrained QUBO would require a further construction, such as bounded slack variables and justified penalties.

## Feasible proposals and ordered cash flows

`solve` enumerates all masks for at most 16 binary variables: at most 65,536 possibilities. It evaluates financial feasibility and selects the feasible mask with the smallest direct objective, taking the lowest mask on exact ties. QUBO values are checked against direct evaluations with a reported floating-point expansion discrepancy.

The current inventory and cash must satisfy their declared limits. Each chosen instrument is traded once, and its resulting integer inventory must remain within its bounds. Cash-increasing trades execute first. With one trade per instrument and nonnegative fees, that ordering preserves the cash borrowing allowance at each step whenever both initial and final cash satisfy it. Prefix cash balances are still checked explicitly. Short positions are permitted through the supplied inventory lower bounds.

The returned `ExecutableLot` records the signed purchase notional, fee, resulting cash and inventory, plus the source quote identity. Feasibility is relative to those declared numeric limits and quotes. Broker collateral and live order routing belong to the application executing a proposal. No orders are sent by this module.

For the native example, the selected changes are:

| Trade | Quantity | Purchase notional | Fee |
|---|---:|---:|---:|
|Sell stock|200 shares|−20,000.00|10.00|
|Buy call|4 contracts ×100 units|1,595.10|31.90|

The resulting stock inventory is −200, and net call inventory changes from −2 to +2. Cash finishes at 19,958.10, within a zero cash-borrowing allowance. Delta is approximately `-1.1e-14`; gamma and vega are zero at the quoted state. With scales 100/10/50, unit risk weights and fee weight 0.001, the objective is 0.0419021, entirely the weighted 41.9021 of fees up to floating arithmetic.

This result also follows put–call parity: two long calls and two short puts offset 200 short shares at zero carry. The chosen hedge leaves the straddle book locally neutral at the stated quotes. The example enumerates 256 masks; 8 satisfy the financial filters. Its maximum QUBO/direct-objective discrepancy is approximately `1.4e-14`.

## API and checks

The main public functions are:

```rust,ignore
let objective = hedge_lot_qubo::build(&problem)?;
let direct = hedge_lot_qubo::evaluate(&problem, proposed_mask)?;
let best = hedge_lot_qubo::solve(&problem)?;
assert!(best.selected.feasible);
```

`problem_sha256` binds the exported objective to its original quote/risk context. Applications should rebuild from a new coherent generation when quotes or positions change. Integer contract quantities and bounded enumeration are exact discrete decisions; marks, Greeks, fees and objectives use checked `f64` arithmetic. Fixed-point ledger booking is a subsequent application step.

Tests cover every mask of small problems, arbitrary targets/normalizations, signed quantities and multipliers, fixed/proportional fees, cash/inventory filtering, execution order, stale/mixed-generation quote rejection, overflow and dimension caps. A full 16-bit case actually enumerates 65,536 masks. The CLI example exercises native option pricing/Greeks and the resulting executable proposal.

This financial allocation problem is separate from selecting rollouts or policy replicas. The matrix is available for experimentation with binary solvers, while the bounded CPU enumerator provides a transparent reference optimum for the declared lot set.
