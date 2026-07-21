The depth-1 numbers came back exactly along the complexity boundary, and they tell me what to do next
with no ambiguity. `memory_unit` is 1.000 — the constant-function semigroup is a single
copy-from-last-write head, solved exactly, as predicted. `grid_world` is 0.882, high but visibly short of
1.0 — the prefix-sum part is easy but the wall-clamping nonlinearity did *not* fully close in one mixing
step, which is the small leak I worried about. And `random_dfa` is 0.205 — the model is barely above
chance-plus-short-range-structure on the non-solvable environment, exactly the depth-1 collapse Liu et al.
2022 predicts. Let me read those three numbers quantitatively before I move, because the arithmetic is the
brief here. The aggregate is the geometric mean `(1.000 · 0.882 · 0.205)^{1/3}`; the product inside
is `0.1808`, and its cube root is `0.566`, which is what comes back. That single computation
shows me the whole leverage structure: because it is a *geometric* mean, the aggregate is pulled toward the
smallest factor, so raising `random_dfa` from 0.205 has far more effect than any remaining polish on the
other two. If I could somehow lift `random_dfa` to 0.4 while leaving the rest fixed, the mean would jump to
`(0.882·0.4)^{1/3} ≈ 0.70`; the same absolute gain spent on `grid_world` (0.882 → 1.0) only moves the mean
to `(1·0.205)^{1/3}·... ≈ 0.59`. The `random_dfa` factor is where the aggregate lives.

Reading the two failures as mechanisms sharpens the target. `grid_world`'s `0.118` error fraction fits the
back-half concentration I predicted — one mixing step computes the sum but cannot fold in the running-extremum
history — so it reads as a depth leak, not a representational one. And `random_dfa` at 0.205 sits `~12×` above
the `1/60 ≈ 0.017` floor: exactly the short-range structure a single stage can exploit (the first one or two
positions are a direct `δ(q_0, σ_1)` lookup) while the long-prefix compositions dominating positions 10–40 are
left at chance. So the failure is specifically the *composition depth* the non-solvable group demands. The
whole problem is now concentrated in one place — buy effective depth on `random_dfa` (closing the small
`grid_world` leak as a side effect) without throwing away the parameter efficiency that made the shallow model
attractive.

Let me be precise about *why* depth is the missing resource, because that diagnosis determines the fix. The
0.205 is not an optimization failure — the harness trains online with a fresh batch every step, so there is
no overfitting confound, and `memory_unit` hitting 1.0 proves the training loop works. It is a *capacity*
failure of a specific kind: the function `q_t = δ(q_{t-1}, σ_t)` composed `t` times is, for a non-solvable
transition semigroup, not computable by a constant number of attention-mixing stages. Each attention layer
is one round of "every position looks at every other and updates"; one round can route information any
distance in *space* but performs only one stage of *composition*. Simulating a length-`t` automaton run is
inherently a chain of `t` compositions, and Krohn–Rhodes tells me that for a *solvable* automaton this chain
collapses into a constant number of stages (which is why `memory_unit` and most of `grid_world` are fine),
but for a non-solvable one it provably does not collapse below `O(log T)` stages. A single layer is one
stage; I need several. So the lever is the number of *sequential* attention stages the model applies, i.e.
its effective depth.

The obvious way to add depth is to stack more distinct layers — a 6-layer Transformer. But I want to ask
whether I can get the depth *without* paying for it in parameters, because that keeps the comparison honest
(the shallow baseline's appeal was its size) and because there is a structural reason to expect a *shared*
block to be the right object here. Let me do the parameter arithmetic on the two options so the choice is
grounded, not aesthetic. One encoder block is about `0.2M` parameters (attention `≈ 65k`, MLP `≈ 131k`, as I
counted for the shallow probe). Six *distinct* layers is therefore `≈ 1.2M` parameters — six times the model
— whereas six *loops of one shared block* is still `≈ 0.2M`, identical to the depth-1 baseline. Both give six
sequential composition stages; only the shared version keeps the size honest. And the extra parameters of the
distinct stack do not buy me the thing I need: the missing resource is *stages of composition*, which both
options supply equally, not per-stage width, which the depth-1 model already had in surplus (its failure was
never a width problem). So spending 6× the parameters would be paying for capacity I have already shown is
not the bottleneck.

Widening — pushing `d_model` from 128 to, say, 512 — is the cheapest knob, but it buys the wrong axis: a
wider single stage still composes exactly once, reaching two steps of prefix no matter how wide, and the
depth-1 `random_dfa` was never starved for features (it computed the one-step lookups it *could* compute).
Width cannot substitute for composition reach, so it leaves the bottleneck untouched. The missing resource is
*sequential composition stages*, which only depth — looped or stacked — supplies.

Beyond honesty, look at what an automaton actually does: it applies the *same* transition operator at every
step. The computation is not six different transformations; it is one transformation iterated. A network that
ties the same weights across all its sequential stages matches that structure exactly — it learns one
"advance the simulation one round" operator and applies it repeatedly, rather than six independent operators
that happen to compose. That is the recurrence-as-iteration idea: take a single Transformer encoder block and
apply it `n_loops` times to the running representation, feeding each output back as the next input. The depth
is the number of loops; the parameter count is that of a one-layer model. There is also a regularization
argument for tying the weights rather than stacking distinct layers: with six independent layers the model
has six times the freedom and could fit the training stream with a brittle six-stage pipeline that does not
actually implement a clean iterated operator; a shared block forces every stage to be the *same* map, so
whatever it learns must be a function that improves the representation when applied once and is safe to apply
again — which is precisely the algebraic property "apply the transition operator" has. The shared block is
thus not merely cheaper, it is the hypothesis class that contains "iterate one operator" and excludes most
everything else, and on a task that *is* operator iteration that is the bias I want.

I want only the *right* part of that idea. The strong form of looping a Transformer is a programmable-computer
construction — scratchpad/memory/command regions, binary column addresses, attention as addressed reads/writes
executing one instruction of an arbitrary program per pass — but this task exposes none of that: it hands me a
fixed `[B, T]` symbol stream and wants `[B, T, num_states]` logits, with the editable contract just
`build_model` returning an `nn.Module`. So the minimal realization is literally: embed the tokens once, apply
one shared causal-self-attention encoder layer `n_loops` times, then a final norm and the linear state head.
The looping supplies the `n_loops` sequential composition stages a non-solvable automaton needs; the weight
sharing supplies them at one-layer parameter cost.

Let me build that fill against the contract. The embedding is unchanged from the shallow baseline — learned
token + absolute position embeddings, added — because the order-injection argument is identical and the
sequence length is still 40. The novelty is the body: instead of `nn.TransformerEncoder` with
`num_layers=1`, I hold a *single* `nn.TransformerEncoderLayer` (pre-norm, GELU, 4× MLP, `d_model=128`,
4 heads) and call it in a Python loop `for _ in range(n_loops)`, threading the causal mask through every
iteration so each loop is still a prefix-respecting update — the state at `t` may only ever depend on the
prefix `≤ t`, and that must hold at every round of composition, not just the first. The block maps
`[B, 40, 128] → [B, 40, 128]`, so I can feed its output back `n_loops` times without any reshape. Threading
the mask through *every* call matters: drop it on one pass and that pass lets position `t` peek at `t+1`, and
since the loops compose, a single unmasked round contaminates every later one — so the same `[40,40]`
upper-triangular mask goes into each call. After the loops I apply
a final LayerNorm before the head; this matters because pre-norm leaves the residual stream un-normalized at
the top of the stack, and after six residual additions through the *same* block the scale can drift, so a
single closing norm keeps the head's input on a sane scale. Then the linear head to `num_states`. The
`forward_logits` wrapper stays the plain `model(input_ids)` — the looping lives *inside* the module's
forward, so I do not need the scratchpad-style escape hatch the contract offers.

How many loops? I should first understand *why* the depth requirement is `O(log T)` and not `O(T)`, because
that is what tells me six is a principled number rather than a guess. The reason is that composing a chain of
transitions is associative: to know `δ(·,σ_1)∘δ(·,σ_2)∘…∘δ(·,σ_t)` I do not have to apply the operators one
at a time left to right; I can combine them pairwise in a tree. One stage of all-to-all attention can, in
principle, compose every position's partial operator with a partner a fixed stride away, so a parallel-prefix
scan doubles the reach each stage: after `d` stages a position can have absorbed the composition of up to
`2^d` predecessors. To cover a full prefix of length 40 I therefore need `2^d ≥ 40`, i.e. `d ≥ log_2 40 ≈
5.32`, which rounds up to exactly `d = 6`. This is the same computation as the wall-time count but read from
the *expressivity* side, and the two agree: six is the smallest depth at which a looped attention stack can,
purely by counting composition reach, associatively fold the entire length-40 prefix. For a *solvable*
automaton the scan collapses even further (constant depth), which is why the two solvable environments were
already essentially done; six loops is aimed squarely at giving the non-solvable environment the full
prefix-scan reach it structurally requires.

What non-solvability forbids is doing the scan in *fewer* than `log_2 T` stages, not the scan itself. This
also fixes the failure mode of too-few loops: at four loops the reach caps at `2^4 = 16 < 40`, so positions
past index 16 could not have absorbed their full prefix and would mispredict on a non-solvable table. Six is
the first depth that closes that gap, and the depth-vs-difficulty pattern from Liu et al. 2022 corroborates it
— non-solvable groups need depth growing like `O(log T)`, with accuracy climbing steeply before flattening
past the threshold.

Budget is not a constraint on spending those loops: the depth-1 baseline ran the three environments in `62.5`,
`79.9`, and `108.0` seconds against a `~1800`-second cap, so six passes (`6 × 108 ≈ 650` s on the slowest) is
still a third of the budget. I could push to eight or twelve, but the marginal gain per loop shrinks fast past
`log_2 T` and every extra loop is another chance for the shared block to drift, so six sits in the "some
simulation is possible" regime without diminishing returns; the *measured* `random_dfa` gap will decide
whether more depth is even the right axis. I keep AdamW at `lr=3e-4, wd=1e-4` — the settings that worked for
the shallow model; changing them would confound the depth comparison.

One way this could backfire: the same shared block that solves the hard environment must not *break* the easy
one — will applying it six times on `memory_unit` corrupt the answer the copy head already resolves in one
stage? Pre-norm and the residual protect against it. Each loop computes `x ← x + Block(LayerNorm(x))`, so a
loop with nothing left to do can drive its sublayer output toward zero and act as a near-identity, carrying the
settled representation straight through. The block only has to learn "advance one composition, and once the
state is settled, add approximately nothing" — a single consistent map, which is exactly what weight-tying
rewards, so `memory_unit` (finishing in one stage) and `random_dfa` (needing all six) can share one block
without tension.

The falsifiable expectations, against the depth-1 numbers. `memory_unit` stays 1.000 — the extra loops idle
as near-identity, and a drop would mean the shared block is destabilizing the easy case. `grid_world` is the
cleaner test: if six composition stages close the 0.882 leak to ≈1.0, that confirms the leak was a depth
deficit, not a representational one. The real target is `random_dfa`: I expect a clear jump above the 0.205
floor — depth is now in the regime where *some* simulation is possible — but not solved, because `S_5` is
non-solvable and six loops is only just past `O(log T)`, so it lands somewhere in the low tenths, still the
bottleneck that pins the geometric mean.

And the diagnosis for the next step is already legible: if looping lifts `random_dfa` only partway, then even
effective depth 6 — the most a parameter-tied attention stack cheaply offers — is structurally short of what a
non-solvable group demands over a long run. The move then is not "more loops" but a model that does not rely on
a fixed number of parallel composition stages at all: a recurrence performing one *exact* state update per
token, `O(T)` sequential stages, sidestepping the barrier because it never tries to shortcut the composition.
The size of the gap looping cannot close is what motivates it.
