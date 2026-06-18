The depth-1 numbers came back exactly along the complexity boundary, and they tell me what to do next
with no ambiguity. `memory_unit` is 1.000 — the constant-function semigroup is a single
copy-from-last-write head, solved exactly, as predicted. `grid_world` is 0.882, high but visibly short of
1.0 — the prefix-sum part is easy but the wall-clamping nonlinearity did *not* fully close in one mixing
step, which is the small leak I worried about. And `random_dfa` is 0.205 — the model is barely above
chance-plus-short-range-structure on the non-solvable environment, exactly the depth-1 collapse Liu et al.
2022 Thm 4 predicts. The geometric mean is 0.566, and because it is a geometric mean the 0.205 is what
drags everything down: even though two environments are essentially perfect, the third pins the aggregate.
So the entire problem is now concentrated in one place — buy effective depth on `random_dfa` (and close
the small `grid_world` leak as a side effect) without throwing away the parameter efficiency that made the
shallow model attractive.

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
block to be the right object here. Look at what an automaton actually does: it applies the *same* transition
operator at every step. The computation is not six different transformations; it is one transformation
iterated. A network that ties the same weights across all its sequential stages matches that structure
exactly — it learns one "advance the simulation one round" operator and applies it repeatedly, rather than
six independent operators that happen to compose. That is the recurrence-as-iteration idea: take a single
Transformer encoder block and apply it `n_loops` times to the running representation, feeding each output
back as the next input. The depth is the number of loops; the parameter count is that of a one-layer model.
There is also a regularization argument for tying the weights rather than stacking distinct layers: with
six independent layers the model has six times the freedom and could fit the training stream with a brittle
six-stage pipeline that does not actually implement a clean iterated operator; a shared block forces every
stage to be the *same* map, so whatever it learns must be a function that improves the representation when
applied once and is safe to apply again — which is precisely the algebraic property "apply the transition
operator" has. The shared block is thus not merely cheaper, it is the hypothesis class that contains
"iterate one operator" and excludes most everything else, and on a task that *is* operator iteration that is
the bias I want.

I want to be careful that I am importing only the *right* part of that idea and not a more elaborate
machine it could suggest. The strong form of looping a Transformer is a programmable-computer
construction — lay the input out as scratchpad/memory/command regions, give every column a binary address,
and turn attention into addressed reads/writes so the looped block executes one instruction of an arbitrary
program per pass. That is far more than this task needs or exposes. Here there is no instruction set, no
addressing, no scratchpad: the harness hands me a fixed `[B, T]` symbol stream and wants `[B, T,
num_states]` logits, and the editable contract is just `build_model` returning an `nn.Module`. So the
faithful, minimal realization of "depth from looping a shared block" for *this* task is literally: embed
the tokens once, then apply one shared causal-self-attention encoder layer `n_loops` times to the embedded
sequence, then a final norm and the linear state head. No memory layout, no temperature-sharpened
permutation, no FLEQ. The looping supplies the `n_loops` sequential composition stages that a non-solvable
automaton needs; the weight sharing supplies them at one-layer parameter cost; and that is the whole of it.

Let me build that fill against the contract. The embedding is unchanged from the shallow baseline — learned
token + absolute position embeddings, added — because the order-injection argument is identical and the
sequence length is still 40. The novelty is the body: instead of `nn.TransformerEncoder` with
`num_layers=1`, I hold a *single* `nn.TransformerEncoderLayer` (pre-norm, GELU, 4× MLP, `d_model=128`,
4 heads) and call it in a Python loop `for _ in range(n_loops)`, threading the causal mask through every
iteration so each loop is still a prefix-respecting update — the state at `t` may only ever depend on the
prefix `≤ t`, and that must hold at every round of composition, not just the first. After the loops I apply
a final LayerNorm before the head; this matters because pre-norm leaves the residual stream un-normalized at
the top of the stack, and after six residual additions through the *same* block the scale can drift, so a
single closing norm keeps the head's input on a sane scale. Then the linear head to `num_states`. The
`forward_logits` wrapper stays the plain `model(input_ids)` — the looping lives *inside* the module's
forward, so I do not need the scratchpad-style escape hatch the contract offers.

How many loops? The depth-vs-difficulty pattern from Liu et al. 2022 Figure 8 is the guide: non-solvable
groups need depth growing like `O(log T)` and their accuracy climbs steeply with depth, with depth ~6
landing in the high-90s for `A_5` but only the high-20s to low-30s for the harder `S_5`-class column. With
`T=40`, `log T ≈ 5.3`, so six loops is the smallest depth that is comfortably above the `O(log T)` threshold
while staying inside the per-environment wall-time budget — each loop is a full attention+MLP pass, so six
passes is roughly six times the compute of the shallow baseline, which the ~30-minute budget tolerates.
I keep the AdamW recipe at `lr=3e-4, wd=1e-4` — the GPT-2-style settings that worked for the shallow model;
the optimization geometry of a looped shared block is close enough to a shallow stack under pre-norm that I
do not expect to need a different schedule, and changing it would confound the depth comparison.

Now the falsifiable expectations, stated against the depth-1 numbers I am trying to beat. `memory_unit` is
already 1.000 and looping cannot hurt it — a constant-depth-solvable environment stays solved (the extra
loops can learn to act as near-identity after the first), so I expect 1.000 again; if it *drops*, the loop
is destabilizing the easy case and something is wrong with the shared-block optimization. `grid_world` is
the cleaner test of the depth hypothesis: the 0.882 leak was, I argued, one mixing step being too few for
the wall-clamping nonlinearity, so six composition stages should close it to ≈1.0. If `grid_world` reaches
1.0, that confirms the leak was a depth deficit, not a representational one. The real target is
`random_dfa`: I expect a clear jump above the 0.205 floor — the depth is now provably in the regime where
*some* simulation is possible — but I do *not* expect it solved, because `S_5` is non-solvable and six loops
is only just past `O(log T)`; the Figure-8 analogy puts a six-deep model on a non-solvable group somewhere
in the 0.3 range, far from the solvable environments' 1.0. So my honest prediction is: `memory_unit` 1.0,
`grid_world` ≈1.0, `random_dfa` materially up from 0.205 but still the bottleneck — the geometric mean rises
because two environments are now perfect and the third is no longer near-floor, but the third still pins it.

And the diagnosis for the *next* rung is already legible in that prediction. If looping closes the two
solvable environments to 1.0 and lifts `random_dfa` only partway, the lesson is that even effective depth 6
— the most a parameter-tied attention stack can cheaply offer — is structurally short of what a non-solvable
group demands when the run is long. At that point the right move is not "more loops" (the budget and the
`O(log T)` curve both fight diminishing returns) but a model that does *not* rely on a constant or
logarithmic number of parallel composition stages at all: a recurrence that performs one *exact* state
update per token, `O(T)` sequential stages, sidestepping the non-solvability barrier entirely because it
never tries to shortcut the composition. That is the hypothesis the next rung will test, and the size of the
`random_dfa` gap that looping *cannot* close is exactly the quantity that motivates it. The full scaffold
module is in the answer.
