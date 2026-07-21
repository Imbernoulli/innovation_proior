The Jacobsthal matrix gave multiplier `49`, and the feedback sharpened it: not one of the `841`
single-entry flips increases the determinant. So `Q + I` is a *strict* local maximum under the most
basic move I have — worse than a plateau, since the gradient of `|det|` points nowhere up and any
greedy hill-climb terminates on its first step, still at `49`. The mechanism is the same rigidity that
made the determinant computable: a flip at `(i, j)` perturbs `~14` of that row's overlaps at once,
moving the balanced `14`/`14` pattern of `+1`'s and `−3`'s off the design, always downhill. So the
escape cannot be one lucky flip; it has to be a *sequence* of individually-neutral-or-worse moves that
only collectively reaches better ground. To leave this basin I have to be willing to walk downhill.

Before reaching for annealing I check the cheaper escapes are genuinely dead. Whole-row negation or a
row/column permutation leaves `|det|` invariant (they multiply it by `±1`) — relabelings, not moves.
Two-entry flips blow the neighborhood up to `~3.5×10^5` candidates per step while, at a symmetric
extremum, still most likely offering no uphill direction: greedy with a bigger neighborhood is still
greedy, still trapped. And restarting greedy from random `±1` seeds is worse — the expected squared
determinant of a random `±1` matrix is `29! ≈ 8.8×10^{30}`, so a typical `|det| ≈ √(29!) ≈ 3.0×10^{15}`,
multiplier `≈ 8×10^{−4}`, three-plus orders of magnitude below the `49` I already hold. Random seeds
live in a swamp; that is why I keep the Jacobsthal seed and change only the *acceptance rule*: propose
a random flip, take it if it improves `|det|`, take it anyway with a probability depending on how much
worse it is and on a temperature I cool over time. Simulated annealing.

Two design decisions come from this objective's geometry. First, what should the acceptance rule
compare? The raw determinant is a twenty-one-digit integer near `2×10^{20}`, and its changes under a
flip are enormous in absolute terms while modest in relative terms — one Metropolis temperature cannot
govern additive gaps of `10^{18}` early and `10^{16}` late. But a flip multiplies the determinant by a
ratio, and what I care about is whether that ratio exceeds `1`. So I anneal on `log|det|`: a `49 → 48`
step is `log(48/49) ≈ −0.021`, a doubling `+0.69` — `O(0.01)` to `O(1)` regardless of absolute size,
which one temperature of order `0.01`–`0.1` can cover.

That fixes the schedule concretely. I want the start warm enough that the smallest escape move off the
strict maximum is easy: at `T₀ = 0.06` a `49 → 48`-type step accepts at `exp(−0.021/0.06) ≈ 0.71`, so
most of the smallest downhill moves are taken and the basin is porous, while a catastrophic `Δlog = −1`
accepts at `exp(−1/0.06) ≈ 8×10^{−8}`, essentially never. I cool geometrically to a floor `T₁ =
2×10^{−4}`, where that same move accepts at `≈ 0` and the tail is effectively greedy, committing to the
best basin found. Over `40k` steps the per-step decay is `≈ 0.99986`, a warm-to-cold glide of about
eight halvings, not a quench. The two limits are the two regimes I already know are useless — `T → 0`
is greedy (dead at `49`), `T → ∞` an undirected walk drifting back to the `m ≈ 8×10^{−4}` swamp — and
annealing is the interpolation between them.

Second, how do I evaluate a candidate? The exact Bareiss determinant is `O(n³)` big-integer arithmetic,
and I don't need exactness *during* the search — only a faithful ranking. So inside the loop I use
floating-point `log|det|` via `slogdet`: one LU factorization, fast at order `29`. The float error in
`log|det|` is roughly `29·ε·κ ≈ 6×10^{−12}` even at a pessimistic `κ ≈ 10³`, nine orders below the
`0.02` step scale, so no genuine accept/reject decision flips on float noise. And a flip that makes the
matrix singular gives `slogdet = −∞`, so `Δlog = −∞` and Metropolis auto-rejects it — a clean
consequence of annealing on the log. Each candidate pays a full factorization whether accepted or not;
not clever, but unarguably correct in ranking, and at `n = 29` cheap enough for tens of thousands of
flips per second.

A few proposal choices round it out. I draw `(i, j)` uniformly rather than sweeping in fixed order, so
which entry is tried is decoupled from where in the cooling schedule I am. I keep the schedule
monotone-cooling with no reheating or restarts, so this rung isolates the single claim that accepting
downhill moves clears the strict maximum. I track the *best matrix ever seen*, not the one I end on,
since the search deliberately wanders and often ends worse than something it passed through. I fix the
RNG seed to `11` for reproducibility and seed the chain at `Q + I`, spending the whole budget improving
a structured `m = 49` configuration instead of clawing up from the swamp. The budget is `40k`
proposals: climbing from `49` to the low hundreds is a net `log` gain of order `1`, accumulated out of
`O(0.01)`–`O(0.1)` steps while paying back the downhill escapes.

So I expect the annealing to clear the Jacobsthal wall easily — the `0.71` acceptance of the smallest
escape says the first step off is nearly free — and climb well past `49`, into the low hundreds of
multiplier, a large fraction of the way toward the machine-search frontier near `197`. But I should be
honest about the ceiling. A single-entry flip is a local move; annealing with local moves explores one
basin structure and plateaus once the temperature is low and the remaining gains need *coordinated*
multi-entry changes a single flip cannot make. And a mundane ceiling stacks on it: every candidate
costs a full `slogdet`, so `40k` flips from one seed is about what a reasonable wall-clock buys. So
this rung proves the principle — breaking symmetry by accepting downhill moves buys a large jump — then
stalls short of the frontier, limited by per-flip cost and single-seed reach. If only cost and reach
hold me back, the next rung makes each flip dramatically cheaper.
