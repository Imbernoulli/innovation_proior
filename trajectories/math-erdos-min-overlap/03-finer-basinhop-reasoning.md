The coarse optimization landed at `0.381240` with two dozen cells and left a clean falsifiable claim: the
cap is resolution, not a missed basin. The whole descent from the flat floor to White's `0.379005` is only
`~0.121`, and the coarse step spent `98%` of it at once; what remains is a thin slice of ten-thousandths,
and I argued two dozen cells cannot represent the fine detuning to shave the envelope thinner — too few
interior knobs for too many competing shifts. If that reading is right, adding pieces should lower the
bound while more coarse restarts should not. So the next move is to lift the optimized profile to more
pieces and refine there. But I have to be careful *how*: the coarse profile carries the gross structure
that bought `98%` of the descent, and re-randomizing at higher `n` throws it away — a fresh random
`120`-cell start lands back in the near-binary chaos where random balanced profiles averaged `0.628`. The
lift must preserve structure.

The clean way to add resolution for free is to upscale: replace each cell by `r` identical finer cells of
the same height. This is the same step function `h` on `[0,2]` on a finer grid — same integral overlap,
same bound — so the upscaled point starts at exactly the coarse `C` with `r` times the degrees of freedom.
Mechanically, repeating each cell `r` times sends the correlation's peak to lag `r·k*` with value exactly
`r` times the coarse peak, and the rescale `2/(rn) = (1/r)(2/n)` cancels the factor precisely, so `C` is
invariant to machine precision — identical to ten digits at `×2, ×5, ×10`. The upscaled peak stays a
unique max at the aligned lag, no intermediate fine shift exceeds it, so upscaling is a true no-op that
hands me a richer neighborhood the optimizer can now carve.

There is a wrinkle, the flip side of the free lift: the upscaled point is a degenerate plateau. The `r`
sub-cells of a former cell are equal, and perturbing two in opposite directions preserves the sum and, to
first order, the overlap — the two enter every `c_k` almost symmetrically. So the objective is flat in a
large subspace right at the upscaled point, and a gradient step launched there finds vanishing slope and
stalls. The flat subspace has about `n_coarse·(r − 1)` directions — for a `×2` upscale of `60` cells, `60`
of them — and those are *precisely* the new degrees of freedom the lift added: it parks me at a point
where all the fresh knobs have zero gradient. So I kick: perturb the upscaled vector slightly and
re-project to feasibility. The kick barely moves `C` — well inside the range where the worst shift stays
put and the score changes only quadratically — but it breaks the block symmetry and gives the optimizer a
non-degenerate gradient. Without it the extra freedom sits unused.

What optimizer refines at this larger `n`? The coarse pass's plain multi-start is now the wrong tool, for
two reinforcing reasons. Its arithmetic turns against me — a dozen blind starts suffice only if a
non-trivial fraction reach the good basin, and that fraction shrinks as more cells make more and narrower
basins. And fresh random starts discard the very structure upscaling just handed me. Both say: perturb the
best-so-far and re-solve rather than start over. That is basin-hopping — solve the annealed ladder to a
local optimum, perturb the best by Gaussian noise, re-project, re-solve, accept only improvements in the
true overlap, shrinking the perturbation over the hops (scale `∝ 0.9^h + 0.1`) so early hops explore
across basins and late hops refine within the best one. This is the perturbation-search-plus-basin-hopping
recipe the agentic-search record uses on this problem, and it fits a non-convex minimax where local
descent is cheap but the basins are many and the good ones close in structure. I accept a hop only if it
improves the true overlap rather than using Metropolis acceptance: the exploration is already supplied by
the perturb-and-re-solve, each hop lands in a genuinely different optimum, and keeping the best-so-far as
anchor with a shrinking kick lets a real jump happen early before the late hops become pure refinement.

The `β` ladder has to be sharper than at the coarse level. At `120` cells there are `239` shifts and the
near-binary profile is spikier, so more shifts crowd the maximum and the surrogate has more competitors to
smear over. At `β = 300` the gap `(2/n)·log(2n−1)/β` is about `3×10⁻⁴`, as large as the prize; annealing
the hop ladder to `β = 3600` brings it to about `2.5×10⁻⁵`, safely below the shave I expect. The fresh
multi-start at `60` and the `120`-cell basin-hop use different ladders on purpose, tracking how cold each
start is: the multi-start cold-starts from random profiles and needs the full soft-to-sharp walk `(60, …,
2400)` to let chaos reorganize before the surrogate kinks up; the basin-hop always starts each solve from
a near-optimal anchor, so it skips the soft floor and runs `(300, 800, 1800, 3600)`.

So the path is `24 → 60` by a fresh multi-start at `60`, then a free `×2` upscale to `120`, then `20`
basin-hops. Re-searching at `60` rather than upscaling `24 → 120` directly is deliberate: `60` is not a
multiple of `24`, and at sixty variables I can still afford to re-solve from scratch and let the optimizer
find the best `60`-cell base — with `60` interior knobs available from the first solve rather than
inherited from a `24`-cell template — which then upscales cleanly by two. Sixty is about the crossover:
the last resolution where blind restarts are still affordable, past which I must switch to
structure-preserving refinement. After each kick I re-project with the same clip-and-redistribute operator;
the kick scale `0.03` is small enough that projection barely distorts its direction while guaranteeing `Σ v
= n/2`, large enough to break the plateau and clear the current basin.

This step is squarely in the grind the flat analysis predicted. The coarse profile already captured the
gross structure, so the gain from `24 → 120` cells is fine shaving, not another jump. I expect to clear
`0.381240` only modestly — landing near `0.38108`, closing most of the distance to the Haugland/AlphaEvolve
landmarks near `0.38092` but not reaching them, still a couple ten-thousandths above the record
`0.38086945`. That modest drop from lifting — with none obtainable from piling more restarts onto `n = 24`
— is exactly the signature confirming the resolution reading over a basin miss: eighty-odd interior knobs
for a `239`-shift envelope separate near-tied shifts more finely than a dozen did, a lower bound bought by
finer detuning, not a changed basin. The coarse drop was `~0.118`; this one, near `1.6×10⁻⁴`, is smaller by
a factor of hundreds — a search closing on a hard limit, and a warning not to expect a large lift to buy a
proportional gain at the endpoint.

One structural prediction I can check on the returned vector: as the envelope flattens under fixed total
mass, the number of shifts essentially tied at the worst overlap should grow — one at the flat floor, a
handful for a coarse profile, an ever-larger active set toward the frontier. A widening active set both
confirms the flattening and warns about the endpoint, since a very large active set is exactly where local
descent has nowhere to go — lowering any tied shift raises another. The cost side warns too: each
basin-hop runs a full SLSQP ladder, and SLSQP finite-differences the surrogate at `O(n)` evaluations of an
`O(n²)` correlation plus a super-linear QP, so per-solve cost grows like `n³` or worse. At `120` cells that
is still seconds, but a naive `×5` to `600` would inflate each solve by two orders of magnitude. So the
endpoint must do two things at once — lift again to the several-hundred-cell scale the records use, and
switch to an optimizer that scales — and finish with a pass that descends the true worst overlap directly.
The one number I watch across that lift is the bound itself: a further small drop says the frontier is
still ahead; a value that holds says I have reached the floor of the basin my recipe selects.
