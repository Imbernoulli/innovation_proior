## Research question

Hastad Switching Lemma explains why constant-depth Boolean circuits with unbounded fan-in still have a rigid weakness: after a carefully chosen random restriction, each small-width DNF or CNF block almost surely becomes simple enough to be represented by a small decision tree. The central question is how this local simplification can be iterated to prove global circuit lower bounds, especially the lower bound that parity is not in `AC0`.

The target model is a depth-`d` circuit of `AND`, `OR`, and input negation gates. After pushing negations to the leaves and alternating layers, the bottom two layers are collections of DNFs or CNFs. If those bottom blocks can be switched to the opposite normal form without a large blowup, the switched top gates merge with the layer above and the whole circuit loses one layer.

## Background

A restriction partially assigns variables: each variable is either fixed to `0` or `1`, or left live as `*`. A random restriction freezes most variables and leaves a small random subset live. Applying a restriction to a formula or circuit gives the residual function on the live variables.

Parity is the hard comparison object because it is stable under restrictions. Fixing some inputs of `PARITY_n` leaves either parity or negated parity on the surviving variables. Thus, if `m` variables survive, the residual function still has full decision-tree depth `m` and still needs exponentially many terms in depth 2.

By contrast, a width-`w` DNF or CNF is fragile. A random assignment kills many terms or clauses outright, and the few parts that survive usually mention only a small number of live variables. The switching lemma gives the sharp quantitative version: for a width-`w` DNF under a random restriction with live probability `p`, the probability that the residual function has decision-tree depth at least `s` is bounded by roughly `(C p w)^s`.

## Core insight

The unique proof move is not to analyze the circuit on every input, but to randomly freeze inputs until hidden simplicity is exposed. A worst-case DNF can look combinatorially tangled before restriction, but after most variables are fixed, its surviving terms can be queried by a shallow decision tree. Decision-tree depth is the right certificate because a depth-`s` decision tree yields both a width-`s` DNF and a width-`s` CNF.

That dual representation is the word "switching." A restricted DNF that has a small decision tree can be written as a small-width CNF, and a restricted CNF can be written as a small-width DNF. Once the bottom block is switched, its top gate has the same type as the gate above it, so the two layers merge. The local collapse of small-width depth-2 formulas becomes a global depth-reduction step.

This creates the lower-bound wedge: random restrictions simplify small `AC0` circuits, but they do not simplify parity. Repeating the restriction-and-switch step strips off layer after layer of the circuit while leaving enough live variables that the final restricted parity function is still hard for depth 2.

## Baselines

Furst, Saxe, and Sipser introduced the random-restriction route for `AC0` lower bounds. Their estimates were strong enough to show super-polynomial lower bounds, but the failure probability for simplifying a block was too weak to survive cleanly through many union bounds and iterations.

Yao obtained the first exponential-strength lower bound using sharper probabilistic ideas, but the switch was approximate: the restricted block agreed with a simpler form on most inputs, rather than being exactly equal to it. That required tracking approximation error through the depth-reduction process.

Hastad's switching lemma made the method both sharp and exact. The failure probability decays exponentially in the target decision-tree depth, and the residual block exactly has the switched representation when the good event occurs. This is the quantitative improvement that turns "random restrictions help" into near-optimal `AC0` lower bounds.

## Evaluation settings

The key parameters are the original circuit depth `d`, size `S`, bottom width `w`, restriction live probability `p`, and switching target `s`. A useful switching estimate must make the bad event small enough that a union bound over all bottom blocks succeeds, while still leaving enough live variables after `d - 2` rounds.

For parity, the iteration yields the standard size-depth tradeoff: any depth-`d` `AC0` circuit computing `PARITY_n` has size at least `2^{Omega(n^{1/(d-1)})}` up to constant choices in the exponent. Equivalently, polynomial-size circuits for parity require depth on the order of `log n / log log n`.

The proof is evaluated by how well it balances two forces: restrictions must be aggressive enough to expose simple structure in small-width formulas, but gentle enough that parity still has many live variables at the end. Hastad's lemma succeeds because it makes the per-block failure probability depend on the switching depth `s`, not on a crude polynomial-in-`n` estimate.
