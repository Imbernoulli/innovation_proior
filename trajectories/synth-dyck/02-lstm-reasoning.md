The Transformer told me, in numbers, exactly which problem I have to solve, and it is not the one I might
have feared. In-distribution the attention stack fits well: `id_token_acc` 0.959956 on `dyck-k2-m3`,
0.924174 on `dyck-length-ood`. So the model *can* represent the matching — its one-hop reach finds the
bracket it needs when the positions are familiar. But on the length-OOD splits it sags exactly where I
predicted it would, and I want to read the drops quantitatively rather than eyeball them. The
in-distribution-to-OOD token-accuracy drop is `0.959956 − 0.903061 = 0.0569` on `dyck-k2-m3` (small length
gap, 65–96 vs ≤64), `0.804292 − 0.727853 = 0.0764` on `dyck-k8-m5`, and `0.924174 − 0.734313 = 0.1899` on
`dyck-length-ood`, where training tops out at 64 and the test runs 128–256. That last drop is `0.1899 /
0.0569 ≈ 3.3×` the `dyck-k2-m3` drop. Now hold that against the untrained-position fractions I worked out
before I ran it: roughly 66% of a mid-range length-OOD string sits in untrained position codes versus
roughly 20% for a `dyck-k2-m3` OOD string, a ratio of `66/20 ≈ 3.3`. The measured drop ratio and the
predicted poisoned-fraction ratio agree to the first digit. That is not proof, but it is exactly the
quantitative fingerprint I said the absolute-position story would leave, and it is far more convincing than
the raw scores alone.

The per-string numbers make the diagnosis brutal: `ood_string_acc` is 0.001, 0.0, 0.0 — essentially *no*
OOD string is fully correct on any environment. The model gets most positions right and a handful wrong,
and on a language where one wrong position fails the whole string, that is a failure of generalization, not
of capacity. The signature is fitting-fine, extrapolating-poorly, which is precisely the absolute-position
story: the mechanism that lets attention locate the matching bracket is anchored to absolute coordinates,
and at OOD lengths those coordinates are untrained noise. And `dyck-k8-m5` is worst even in-distribution
(`id_token_acc` 0.804292, `ood_token_acc` 0.727853, `id_string_acc` only 0.024) — 8 bracket types and depth
5 is the hardest content, and the attention stack does not have a clean place to *store* the stack
discipline; it has to re-derive it by position-indexed lookup every time. I file that away: even after I
fix the length problem, the content-hardness of `dyck-k8-m5` may not be fixed by the same move, because its
in-distribution deficit is not a length phenomenon at all.

Before I leave the Transformer behind I should rule out the lazy move, which is to keep it and simply throw
more capacity at it — more layers, more width — because that is the reflex when a model underperforms. The
numbers say capacity is not the limiter. In-distribution the Transformer already reached `id_token_acc`
0.959956 and 0.924174 on two of the three environments with ~266k parameters against a 500k ceiling; the
model demonstrably *has* the capacity to represent the matching, because it does represent it whenever the
positions are trained. The deficit is entirely in the `id → ood` gap (0.057 to 0.19), and that gap is a
function of *which positions were trained*, not of how many parameters compute over them. Doubling the width
would enlarge the position table right along with everything else and hand me more untrained rows to fail
on, not fewer. So "scale the Transformer" fails on its own evidence: the thing that is broken is the
coordinate system, and no amount of capacity fixes a coordinate that was never optimized. That closes the
door on staying with attention-plus-absolute-position and forces the architectural change.

So the next rung must remove the dependence on absolute position, and I have two honest ways to do it. One
keeps attention and swaps the *kind* of position: a relative or rotary encoding scores query-key pairs by
their offset `i − j` rather than by absolute index, so a "matching opener `d` steps back" head could reuse
its learned offset kernel at any absolute position. That is tempting because it preserves attention's `O(1)`
reach, which is the one thing the Transformer clearly did well — recall it fit in-distribution. But when I
walk it a step, the fix is only partial: an OOD string in `dyck-length-ood` contains matching spans of 150+
brackets, so the offset `i − j` at the deepest matches exceeds any offset the kernel saw when training
capped at length 64. The relative kernel would itself be extrapolating past its trained support, softening
the collapse rather than abolishing it. More decisively, I want to test the *cleanest* form of the
hypothesis — is it position *per se* — and the sharpest test is to remove the position table entirely. That
is what a recurrent network does: it processes the sequence as a *stream*, updating its state by the *same*
transition at every step regardless of the absolute index. There is no position table to run off the end
of. If a recurrent model can carry the stack information forward in its hidden state, then a string of
length 200 is processed by the same per-step rule as a string of length 60 — the rule was trained on every
step it ever saw, and there are no untrained positions. That is the structural reason to expect a recurrent
model to length-generalize where the absolute-position Transformer did not. But "recurrent" alone is not
enough; the plain Elman recurrence has its own wall, and I have to understand it precisely or I will just
trade one failure for another.

Follow a single error signal backward through a recurrence. The error that lands on a unit at time `t` and
has to reach a unit `q` steps in the past arrives as a *product of `q` factors*, each factor of the form
`f'(net) · w` — the chain rule telescoped over the path. Stare at that product. If every factor has
magnitude below 1 — the ordinary case, since the logistic derivative peaks at 0.25 and reasonable weights
keep `|f'·w| < 1` — the product shrinks like (something<1)^q: the error *vanishes* exponentially in the
lag. Put a number on it: if the per-step multiplier is 0.5, then across a lag of `q = 40` — the span of a
modest nested Dyck sub-string — the surviving gradient is `0.5^40 ≈ 9 × 10⁻¹³`, utterly gone. If every
factor exceeds 1, it *explodes* symmetrically. Either way the lag `q` sits in the exponent, and no setting
of the weights makes a generic product of many factors stay `O(1)`. For Dyck this is fatal in exactly the
regime I care about: matching a closer to its opener requires carrying gradient across the entire span of
the nested sub-string, and on OOD lengths that span is longer than anything trained — the very lags where
the vanishing is worst. A plain RNN would memorize short-range matching and fail to learn the long-range
discipline, which is no better than the Transformer's position problem. So the plain Elman cell is out on
an arithmetic ground, not a vague one.

If the product of factors is the disease, the cure must make the product *exactly 1* for any `q`. In the
simplest setting — a single unit with a self-connection of weight `w` — the one-step backward multiplier is
`f'(net) · w`, and I want it to equal 1. Read as a constraint on `f`: `f'(net) = 1/w`, a constant, so `f`
must be *linear*, and the clean choice is the identity with `w = 1`. Then the unit's state simply persists,
step after step, and the backpropagated error riding through it is multiplied by exactly 1 at every step —
a constant error carousel, a channel along which gradient survives an arbitrarily long lag. This is the
seed: a protected linear memory state. For Dyck, this is precisely the object I want, because the depth-`m`
stack is information that must be *held* across the nested span and then *read* when the matching closer
arrives — a memory, not a position lookup.

But a bare linear self-loop cannot be wired to the rest of the net without conflict. A single incoming
weight must do two opposite jobs: at the moment relevant information arrives it must *write* it into the
memory, and at all other moments it must *protect* what is stored from being overwritten by irrelevant
input. One weight cannot be context-sensitive; another *unit* can — a unit whose activation depends on the
whole network this step can output "open" in one context and "closed" in another. And the control must be
*multiplicative*, not additive, because protecting the memory means the irrelevant input contributes
exactly zero, which only a multiply by a value in `[0,1]` can do. So I wrap the carousel in multiplicative
gates that are themselves learned sigmoid units. An **input gate** decides when to write — resolving the
write/protect conflict on the input side. An **output gate** decides when to read — resolving the
read/protect conflict on the output side. And because a Dyck stream is not segmented (the harness feeds one
string at a time, but within a string the model must reset its working memory as sub-structures complete),
I also need a way to *release* stored state at the right moments: a **forget gate** that multiplies the
carried-over state, recovering the exact carousel when it is 1 and wiping the memory when it is 0, learned
from context so the model discovers its own reset points — which for Dyck are exactly the bracket closures.

Before I commit to the full three-gate cell, I owe the cheaper gated option a hearing: the GRU folds the
input and forget roles into one coupled gate and drops the separate cell state, so it is smaller and often
trains faster. Why not it? The GRU's memory is its *hidden* state, gated by an interpolation `h_t = (1 − z)
h_{t−1} + z · \tilde h`; there is no separate, un-squashed linear channel — the carousel and the read-out
live on the same vector. The LSTM's virtue for this task is exactly the *separation*: the cell `c_t` is a
protected linear accumulator, and the hidden `h_t = o_t ⊙ tanh(c_t)` is a *gated view* of it, so the model
can hold a value in `c` untouched across a long span while presenting whatever the output gate chooses at
each step. For a stack that must be *held* across a nested span and *read* only at the closer, that
held-versus-read separation is the property I want, and it is worth the extra gate. So GRU is a reasonable
alternative I set down deliberately, not overlook.

Let me convince myself the cell can actually track a stack by hand-tracing the shallowest non-trivial
case, `( ( ) )` — a depth-2, single-type Dyck string — with one idealized memory unit standing for the
depth counter. Initialize `c_0 = 0`. Step 1 reads `(`: the input gate opens (`i ≈ 1`) and writes `+1`, the
forget gate holds (`f ≈ 1`), so `c_1 = 1·0 + 1·1 = 1` — depth one. Step 2 reads another `(`: same gates,
`c_2 = 1·1 + 1·1 = 2` — depth two. Step 3 reads `)`: the controller recognizes a closer and writes `−1`
through the input path while the forget gate still holds the accumulator, `c_3 = 1·2 + 1·(−1) = 1` — the
stack pops back to depth one. Step 4 reads `)`: `c_4 = 1·1 + 1·(−1) = 0` — empty, exactly where a valid
Dyck string must end. At every step the multiplier on the carried state was `f ≈ 1`, so the depth count
survived undistorted, and the *same* four gate settings would run a depth-2 string embedded 150 tokens deep
in a length-256 sequence identically — no step in the trace referenced an absolute index. That is the
length-independence made concrete: the transition is a function of `(x_t, h_{t-1}, c_{t-1})`, never of `t`.
Of course the real task needs more than a depth counter — it must remember *which of k types* sits at each
level to name the right closer — but the trace shows the mechanism is present and index-free; whether 64
units can hold the *typed* stack for `k = 8` crisply is the open question I flag below.

That is the gated memory cell: a linear state `c_t` carried by the forget gate, written through the input
gate, read through the output gate, with the backward state error obeying `ε_s^t = … + f_{t+1} · ε_s^{t+1}`
— unit gain across the lag when the forget gate is open, and a deliberate drop when the cell has chosen to
forget. The forward pass for a layer of cells is the standard form: input/forget/output gates as sigmoids
of `[x_t, h_{t-1}]`, a tanh candidate, `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`, `h_t = o_t ⊙ tanh(c_t)`. This is
exactly `nn.LSTM`, which implements precisely these equations, so I do not re-derive them in code — I use
the fused, optimized version the framework provides. And critically, none of this references absolute
position. The cell's update at step 200 is the same learned function as at step 60; there is no table to
run off the end of. That is the property the Transformer lacked.

Now the fill, against this scaffold. The model is `nn.Embedding(vocab, hidden)` → `nn.LSTM(hidden, hidden,
num_layers=2, batch_first=True)` → `nn.Linear(hidden, vocab)`. Two layers because depth lets the second
layer compute over the first layer's summary of the stack — for `dyck-k8-m5`, where 8 bracket types must be
disambiguated (and where the Transformer's `id_string_acc` was a dismal 0.024), one layer of width 64 may
not cleanly separate the closer identities, and a second layer gives the network room to compose "what is
on top" with "which of 8 types it is." Width is `config.hidden_dim = 64`, which by the bounded-memory result
is generous. Let me put the information-theoretic floor in bits: an exact recognizer must hold a stack of at
most `m` symbols, each one of `k` types, plus a depth counter. For the hardest config `m = 5, k = 8` that is
`5 · log2(8) = 5 · 3 = 15` bits for the stack contents plus `log2(6) ≈ 2.6` bits for the depth (0…5) — call
it ~18 bits. Sixty-four real-valued units can carry far more than 18 bits of usable state, so 64 is
comfortably above the floor, and the question is purely whether gradient descent *finds* the stack-tracking
state, not whether 64 units can hold it. There is no absolute-position table, because the recurrence
supplies order for free from the sequential processing; this is the deliberate contrast with the
Transformer rung.

The parameter count I can predict from the shapes, and it doubles as a budget check. Each LSTM layer has
four gate matrices; with input and hidden both width 64 that is `4 · (64·64 + 64·64 + 64 + 64) = 4 · 8320 =
33 280` per layer, so two layers are `66 560`. Add the embedding `vocab · 64` (384 for `k = 2`) and the head
`64 · vocab + vocab` (390 for `k = 2`) and I get about `67 334` — roughly a quarter of the Transformer's
~266k, and far inside the 500 000 budget. The recurrence is not only more length-robust, it is
*cheaper*, because it spends nothing on a position table. The forward pass embeds, runs the LSTM over the
time axis returning a hidden at every position — output shape `[B, T, 64]` — and projects to logits
`[B, T, vocab]`, exactly the `DyckModel.forward` contract, scored by the harness's next-valid-set evaluator.

One thing I want to be clear-eyed about: the recurrence removes the *absolute-position* failure mode, but
it does not remove every length-generalization risk. The LSTM's memory is still a *fixed-width* vector. The
bounded-memory result guarantees a depth-`m` stack of `k` symbols *fits* in `O(m log k)` units, and `m` is
bounded (≤5), so the stack the model must hold is bounded too — this is why I expect the LSTM to do well
here, where for *unbounded*-depth Dyck a fixed-width state would eventually overflow. The OOD strings are
longer but the *depth* never exceeds `m`; the cell only has to hold a bounded stack and slide it as brackets
open and close, which a forget-gated linear state can do at any length. So the bet is that the LSTM's
length-independent transition plus its bounded stack exactly matches Dyck-(k,m)'s bounded-memory structure —
a length-200 string never presents a *deeper* configuration than a length-60 string, only a longer sequence
of the same bounded operations.

There is also a scoring reason to aim the gain at `dyck-length-ood` specifically, which is worth spelling
out because it tells me where a fixed budget of improvement buys the most task score. The task score is the
geometric mean of the three `ood_token_acc` values, and the Transformer's was `(0.903061 · 0.727853 ·
0.734313)^{1/3} ≈ 0.785`. The geometric mean's log is the *average* of the logs, so its sensitivity to any
one environment is `∂(log G)/∂(log x_i) = 1/3` — a given *relative* lift on the weakest environment moves
the task score exactly as much as the same relative lift anywhere else, which means the largest *absolute*
headroom sits on the smallest factor. Here the two smallest factors are `dyck-k8-m5` (0.727853) and
`dyck-length-ood` (0.734313), nearly tied, and they cap the whole score: even perfect 1.0 on `dyck-k2-m3`
leaves `G ≤ (1 · 0.727853 · 0.734313)^{1/3} ≈ 0.813`. So the recurrence has to deliver on the two hard
environments to move the headline at all — a near-solved `dyck-k2-m3` cannot rescue the mean. That reframes
my expectation: I am not hoping for a uniform lift, I am betting the length-independent transition clears
`dyck-length-ood` (a pure length problem) while watching nervously whether it also clears `dyck-k8-m5`
(partly a content problem).

The falsifiable expectations against the Transformer's numbers, stated so they can be wrong. First and
headline: `ood_token_acc` should *rise* across the board, and rise most on `dyck-length-ood`, where the
Transformer collapsed to 0.734313 — if the absolute-position table was the culprit, removing it should help
most exactly where the extrapolation was largest and the poisoned-fraction was ~66%. I expect
`ood_token_acc` near 0.97+ on the two easier environments and a clear, large gain on `dyck-length-ood`.
Second, the per-string numbers should move off the floor: the Transformer's `ood_string_acc` of
0.001/0.0/0.0 means it never got a full OOD string right; if the LSTM is genuinely tracking the stack rather
than approximating by position, some OOD strings should come out fully correct, so `ood_string_acc` on the
easy `dyck-k2-m3` should climb from 0.001 toward near one. Third, the one place I am *not* confident:
`dyck-k8-m5`. The Transformer's worst environment was the content-hardest one (8 types, depth 5), and its
deficit there showed up *in-distribution* (`id_string_acc` 0.024), so it is not purely a length artifact
the recurrence will erase. A fixed-width recurrent state has to pack the identity of the top-of-stack symbol
cleanly enough that the read-out picks the right closer among 8. If the LSTM also sags on `dyck-k8-m5` —
high `ood_token_acc` on the other two but visibly lower here, and especially with weak per-string accuracy
even in-distribution — that is the tell that a *fixed-width* memory, however well it generalizes across
length, is struggling to maintain a *crisp* discrete stack, and the next rung should give the model an
*explicit* stack data structure rather than asking a dense vector to emulate one. The full scaffold module —
embedding, 2-layer `nn.LSTM`, linear head, no position table — is in the answer.
