# Hastad Switching Lemma

Hastad's Switching Lemma says that small-width DNFs and CNFs almost surely become simple after a random restriction. In a common form: if `F` is a width-`w` DNF and `rho` leaves each variable live with probability `p`, then

`Pr[DTdepth(F|rho) >= s] <= (C p w)^s`

for an absolute constant `C` under the usual parameter assumptions. The dual statement holds for CNFs.

The crucial point is not merely that the restricted formula gets smaller. It becomes simple in a form that can be switched. A function with decision-tree depth at most `s` has both a width-`s` DNF and a width-`s` CNF representation: each root-to-leaf path supplies a conjunction of at most `s` queried literals. Thus a restricted DNF can be replaced by a small-width CNF, and a restricted CNF can be replaced by a small-width DNF.

This gives a depth-reduction mechanism for `AC0`. Normalize a constant-depth circuit into alternating `AND` and `OR` layers. The bottom two layers are DNFs or CNFs. Apply a random restriction. With high probability, every bottom block switches to the opposite normal form. Its new top gate then has the same type as the gate above it, so the two adjacent layers merge. One layer has been peeled away.

Iterating this step proves lower bounds because random restrictions treat small circuits and parity very differently. Small-width circuit pieces collapse into shallow decision trees. Parity does not collapse: after any restriction, it remains parity or negated parity on the live variables. After enough iterations, a supposed small depth-`d` circuit for parity becomes a depth-2 circuit for parity on many surviving variables.

Depth-2 parity is exponentially expensive. A DNF or CNF for `PARITY_m` needs a separate full-width term or clause for exponentially many assignments. This final contradiction yields the standard tradeoff:

`size_depth-d(PARITY_n) >= 2^{Omega(n^{1/(d-1)})}`.

So the unique insight is the proof strategy of randomly freezing variables to reveal simple residual structure. The switching lemma makes that strategy quantitative and exact: the failure probability decays exponentially in the target switching depth, allowing a union bound over all bottom blocks and enabling repeated layer removal. That is why a local statement about restricted DNFs/CNFs becomes a strong global lower bound for constant-depth circuits.
