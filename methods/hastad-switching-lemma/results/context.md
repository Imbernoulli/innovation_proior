## Research question

Constant-depth Boolean circuits with unbounded fan-in (`AC0`) are a natural model of fast, highly parallel computation. The central question is how to prove size lower bounds for such circuits, and in particular whether `PARITY_n` can be computed by polynomial-size constant-depth circuits. The setting is the depth-`d` circuit of `AND`, `OR`, and input-negation gates: after pushing negations to the leaves and arranging the gates into alternating layers, the bottom two layers form a collection of small-width DNFs or CNFs. The question is how to control the residual structure of these bottom blocks well enough to argue about the whole circuit.

The technical handle is the random restriction. A restriction partially assigns variables: each variable is fixed to `0` or `1`, or left live as `*`. A random restriction freezes most variables and leaves a small random subset live, and applying it to a formula or circuit gives a residual function on the live variables. The problem is to quantify how a random restriction acts on a width-`w` DNF or CNF, and to use that to reason about a depth-`d` circuit computing parity.

## Background

A restriction `rho` partially assigns the input variables; the live variables are those left as `*`. A random restriction with live probability `p` keeps each variable live with probability `p` and otherwise fixes it to a random Boolean value. The residual `F|rho` is the function on the live variables obtained by substituting the fixed assignments.

Parity is the chosen target object because it is stable under restrictions. Fixing some inputs of `PARITY_n` leaves either parity or negated parity on the surviving variables. So if `m` variables survive, the residual function still has full decision-tree depth `m`, and in depth 2 a DNF or CNF for `PARITY_m` needs a separate full-width term or clause for exponentially many assignments.

A width-`w` DNF or CNF behaves differently under a random assignment: many terms or clauses are killed outright, and the surviving parts typically mention only a small number of live variables. Decision-tree depth is the unit of residual complexity used here, because a function with decision-tree depth at most `s` has both a width-`s` DNF and a width-`s` CNF representation: each root-to-leaf path supplies a conjunction of at most `s` queried literals.

## Core insight

The proof move is to randomly freeze inputs until the residual structure of the bottom blocks becomes simple, rather than to analyze the circuit on every input. Restrictions act in opposite ways on the two objects of interest: small-width depth-2 formulas tend to collapse to shallow residual functions, while parity remains parity on the live variables. The aim is to convert this contrast into a global statement by combining a per-block residual estimate with a union bound over all bottom blocks of the circuit.

When a restricted DNF has a small decision tree it can be rewritten as a small-width CNF, and a restricted CNF as a small-width DNF. If the bottom block is rewritten into the opposite normal form, its new top gate has the same type as the gate above it, and the two adjacent layers merge into one. Random restrictions thus give a candidate depth-reduction step: simplify the bottom blocks, merge a layer, and repeat, while keeping enough live variables that the final restricted parity function is still hard for depth 2.

## Baselines

Furst, Saxe, and Sipser introduced the random-restriction route for `AC0` lower bounds. Their per-block estimate bounds the probability that a restricted bottom block fails to simplify, and iterating it across the `d - 2` rounds of depth reduction yields super-polynomial lower bounds for parity.

Yao obtained an exponential-strength lower bound for parity using sharper probabilistic estimates on how random restrictions act on small-width formulas. In this analysis the restricted block agrees with a simpler form on most inputs, and the approximation error is tracked through the depth-reduction process.

Both approaches share the same architecture: a quantitative bound on the residual decision-tree complexity of a width-`w` DNF or CNF under a random restriction, applied via a union bound over the bottom blocks and iterated to strip layers off the circuit until a depth-2 circuit for parity remains.

## Evaluation settings

The key parameters are the original circuit depth `d`, size `S`, bottom width `w`, restriction live probability `p`, and target residual decision-tree depth `s`. A per-block estimate is used through a union bound over all bottom blocks across `d - 2` rounds of restriction, and the live probability `p` is chosen so that parity still has many live variables after all rounds.

For parity, the iteration is designed to yield a size-depth tradeoff of the form

`size_depth-d(PARITY_n) >= 2^{Omega(n^{1/(d-1)})}`

up to constant choices in the exponent, equivalently that polynomial-size circuits for parity require depth on the order of `log n / log log n`. The analysis is evaluated by how well it balances two forces: restrictions aggressive enough to expose simple structure in small-width formulas, and gentle enough that parity retains many live variables through the final round.
