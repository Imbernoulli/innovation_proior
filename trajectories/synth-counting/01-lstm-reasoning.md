I start the ladder at the architecture that has the cleanest claim to *actually counting*, because the
research question is about counters and I want the first rung to be the model whose internal mechanism
is a counter by construction rather than by accident. The scaffold default is a 2-layer GRU reading its
last hidden state; before I reach for attention or anything fancier I want to know how far a recurrent
counter gets on its own, because if a recurrent net can hold an integer count across the sequence then
that is the honest floor every later method has to beat, and if it cannot then I have learned precisely
*where* recurrence breaks and that diagnosis seeds the next rung. So rung one is a recurrent encoder,
and I will reason about why an LSTM specifically, and what I expect it to do on each of the three
environments.

There are really four recurrent options in front of me, and it is worth walking them rather than
defaulting. A vanilla `nn.RNN` is the obvious floor; the scaffold ships a 2-layer GRU; an LSTM is the
gated cell with a separate memory line; and I could in principle hand-wire an integer accumulator and
skip learning entirely. The hand-wired accumulator I reject immediately on the contract: the editable
slot is `build_model` returning a learnable `nn.Module` that the *fixed* AdamW loop trains from
initialization, so a frozen non-parametric counter is not really in the design space, and even if I bolted
one on it would be brittle to the three different loss functions and would not teach me anything about
what a *learned* recurrence can do — which is the whole point of the floor. The vanilla RNN I reject on
the gradient, below. The interesting rejection is the GRU, because it is the default and I have to
justify overriding it. A GRU carries its state as a convex interpolation, `h_t = (1 - z_t) ⊙ h_{t-1} +
z_t ⊙ h̃_t`, where the candidate `h̃_t` is itself squashed through a `tanh`; there is no separate,
*unsquashed* memory line. That distinction is exactly what matters for counting. The quantity I want to
accumulate — a running tally — must live somewhere that (a) is not re-compressed through a saturating
nonlinearity every step and (b) passes gradient back at unit gain. The GRU's `h_t` fails (a): even with
the update gate pinned so `z_t → 0` and the state nominally held, the moment the gate opens to write an
increment the new content re-enters through `tanh`, and the held state is the *output* state, the same
one the squash touches. The LSTM separates the two: a linear cell `c_t` that is written additively and a
*separate* output `h_t = o_t ⊙ tanh(c_t)` that squashes only the read-out. So the LSTM can hold an
undamped linear tally in `c_t` while exposing a bounded view of it — the exact separation a counter
wants — and the GRU cannot without fighting its own coupling. That is a computed reason to override the
scaffold default, not a preference.

The reason a plain recurrent net is the wrong place to *stop* — even though it is the right place to
*start* — is the gradient. Trace an error signal backward through a recurrence `h_t = f(h_{t-1}, x_t)`:
over `q` steps it is multiplied by a product of `q` Jacobian factors, each of the form (derivative of
the squashing nonlinearity) times (a recurrent weight). With a sigmoid or tanh unit each such factor is
below one in magnitude for any reasonable weight, so the product decays geometrically in the lag — the
classic vanishing gradient. Put a number on it: if each factor sits at a generous 0.8, then over a lag
of just 30 steps the product is `0.8^30 ≈ 1.2e-3`, and over 60 steps `0.8^60 ≈ 1.5e-6` — the credit
signal is gone. On the `abc` environment the lag is exactly the thing that matters: to decide whether a
string is `a^n b^n c^n` the model must compare the size of the `a`-block, seen at the *front* of the
sequence, against the `c`-block, seen at the *back*, and for the longer training strings (`3n` up to
about 63) those blocks are dozens of steps apart. A vanilla RNN's gradient connecting them has shrunk to
nothing, so it never learns to thread a count from the `a`-block through to the `c`-block; it learns
local texture instead. That is the disease, and it is exactly what the gated memory cell was built to
cure.

So I use an LSTM. The cure is structural: a linear cell state `c_t` carried by a self-loop at unit
gain, so the backpropagated error riding that state is multiplied by *one* per step rather than by a
sub-unit factor, and three sigmoid gates — input (when to write), forget (when to reset), output (when
to read) — that decide, from context, what the cell stores and exposes. The forward dynamics
`c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`, `h_t = o_t ⊙ tanh(c_t)` are, for my purpose, a programmable
accumulator: the network can learn an input gate that opens on `a`, a candidate that contributes `+1`,
a forget gate pinned near one so the running tally is held undamped, and later a contribution that
checks the tally against the `b` and `c` blocks. The backward recursion makes the same point in
gradient form — the state error at time `t` inherits the next step's state error scaled by the forget
gate, `ε_s^t = … + f_{t+1} ⊙ ε_s^{t+1}`, which is unit-gain when the forget gate is open — so the
credit for "increment here" reaches back across the whole block instead of dying in the lag. This is
why a finite-precision LSTM can, in principle, *implement* an integer counter and even extrapolate it
to inputs longer than those seen in training: the counting recipe is a fixed-point of the cell
dynamics, not a length-specific table.

That fixed-point story is the optimistic reading, and I should stress-test it on paper before I trust
it, because the very read-out that makes the cell usable is also where I suspect it will break. The tally
lives in `c_t`, but the head never sees `c_t` directly — it sees `h_t = o_t ⊙ tanh(c_t)`, a *squashed*
view. Suppose the network learns the clean recipe: increment `c` by some fixed step `δ` on each `a`. To
keep the read-out able to *distinguish* successive counts, the difference `tanh(δ·(k+1)) − tanh(δ·k)`
must stay resolvable, and `tanh` flattens hard: `tanh(2) = 0.964`, `tanh(3) = 0.995`, `tanh(4) =
0.9993`. So once `δ·k` passes roughly 3 the read-out is pinned near 1 and consecutive counts are
indistinguishable through the squash. In-range the largest count the model must resolve is about 21 (the
`a`-block of the longest training `abc` string) or up to ~63 (all-`a` `exact` strings), so to keep even
the in-range counts inside the responsive band the network is pushed toward a *small* per-step step, say
`δ ≈ 0.1`, which puts count 21 at `tanh(2.1) = 0.970` — already compressing, but workable if the head
reads the pre-saturation slope. Now extrapolate: at an OOD length of 256 the counts run up toward 85
(abc runs) or beyond, and `δ·85 = 8.5` gives `tanh(8.5) ≈ 0.99999967`. Every large OOD count collapses
onto the same saturated value, `≈ 1`, so the read-out cannot tell a count of 40 from a count of 85 — the
information is arithmetically gone at the squash, *independent* of whether the cell integrated perfectly.
And this is the optimistic branch where `δ` is exact. If the learned increment is `0.99·δ` instead of
`δ`, the error compounds linearly in the number of steps, so a drift that is 1% over 21 steps becomes
4% over 85 — the count read-out is wrong by several units on long strings even before saturation. Both
mechanisms — squash saturation and compounding step error — point the same way, and both grow with
length. So my strongest on-paper prediction is not "the LSTM fails" in general but a *specific split*: it
should be excellent in-range and should degrade sharply, plausibly to the floor, once the counts leave
the band the training lengths shaped.

It is worth walking the ideal counter through a tiny string once, both to convince myself the fixed
point exists and to see exactly where the read-out is fragile. Take `a a b b` (drop CLS for the trace)
and the two-cell recipe: cell 0 tallies `a`, cell 1 tallies `b`. Initialize `c = (0, 0)`. Step 1, token
`a`: input gate for cell 0 opens, candidate `+δ`, forget gates near one, so `c = (δ, 0)`. Step 2, `a`:
`c = (2δ, 0)`. Step 3, token `b`: the `a`-gate closes (its input gate reads the current symbol and stays
shut on `b`), cell 0 holds at `2δ` under its open forget gate, cell 1's input gate opens, `c = (2δ, δ)`.
Step 4, `b`: `c = (2δ, 2δ)`. The final cell state literally holds the two counts, and a membership head
can read "cell 0 equals cell 1" for the two-block version of the decision. The trace shows the recipe is
a genuine fixed point of the gate settings — no length-specific constant appears — which is the
optimistic half. But it also shows the fragility concretely: the head does not see `(2δ, 2δ)`, it sees
`(tanh(2δ), tanh(2δ))`, and the *equality* test survives saturation (both saturate together) while the
*magnitude* test on `exact` does not — `tanh(2δ)` and `tanh(3δ)` are nearly equal once `δ·k > 3`. So the
same trace predicts the membership task (a comparison) should extrapolate *better* than the exact-count
task (a magnitude read), because saturation cancels in a comparison but destroys a magnitude. I file
that as a second, finer falsifiable expectation: if anything survives OOD it is more likely the
equality-structured `abc` decision than the absolute `exact` count. As a degenerate check the other way,
send in the empty content case (only CLS): `lengths = 1`, the gather index clamps to 0, and the head
reads the CLS-position hidden — a defined, count-zero summary rather than an out-of-bounds crash, which
is the behavior the `clamp(min=0)` is there to guarantee.

Before committing the width and depth I check the budget, because the harness asserts at most 5,000,000
parameters and I want to know whether capacity is the binding constraint (it is not, and that is itself
informative). Embedding is `vocab_size × hidden_dim = 5 × 128 = 640`. A single `nn.LSTM` layer with
input and hidden both 128 holds four gate matrices of shape `[4·128, 128]` for input-to-hidden and again
for hidden-to-hidden, i.e. `2 × 4 × 128 × 128 = 131,072` weights, plus `2 × 4 × 128 = 1,024` biases —
about `132k` per layer, so two layers is `≈ 264k`, and a final `LayerNorm(128)` adds `256`. The encoder
lands near `0.27M` parameters, about 5% of the 5M ceiling. So I *could* go to width 256 (roughly
quadruples the recurrent block to ~1M) or stack four layers, and I would still be inside budget. I do not,
for a reason the saturation calculation already gave me: the failure I expect is a *representational*
one — a squashed continuous tally that does not survive length doubling — and neither more width nor
more depth changes that; a wider `tanh` still saturates, a deeper stack still carries the count through
time. Two layers is the *minimum* that buys the one thing depth does help here: division of labor. A
single layer would have to both *maintain* the running tallies and *compute* the `a==b==c` membership
decision from them in the same state; stacking lets the lower layer carry the counts and the upper layer
combine them into the comparison, a cleaner factoring that costs almost nothing against the ceiling.
Capacity is not the lever this task probes, so I spend the minimum on it and keep the comparison against
later rungs honest at matched width and depth.

Now the design choices, each tied to the contract the harness hands me and checked for shape. The
encoder receives `tokens: LongTensor[B, T]` with a CLS at position 0 and must return `[B, hidden_dim]`.
The first layer is an embedding `nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)` — five symbols
(`PAD, a, b, c, CLS`) into a 128-dim space, with the padding row pinned to zero so padded positions
contribute nothing to the recurrence — turning `[B, T]` into `[B, T, 128]`. Then `nn.LSTM(input_size=
hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)` maps `[B, T, 128]` to an output
`[B, T, 128]` (the top-layer hidden at every step). Width 128 matches `hidden_dim` so the returned
summary lands in exactly the shape the fixed head expects, with no extra projection. The pooling then
has to collapse the time axis to a single `[B, 128]` vector, and the index arithmetic is where a shape
bug would hide: I want, per row `b`, the hidden at that row's *true* final position. The harness gives
`lengths: LongTensor[B]`, so `last_idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1,
out.size(-1))` builds a `[B, 1, 128]` gather index — the `.view(-1, 1, 1)` makes it broadcastable along
time and feature, the `.expand(-1, 1, 128)` copies the single time index across all 128 feature channels
so `gather(1, ·)` selects the same timestep for every channel, and `.squeeze(1)` drops the singleton time
axis back to `[B, 128]`. I trace one row to be sure: if `lengths[b] = 10`, the index is 9, and
`out[b, 9, :]` is exactly the top-layer hidden after the recurrence has consumed positions 0…9 — the CLS
plus nine content tokens — which is the whole valid content of that row. Correct.

The pooling rule is the one place I have to respect a scaffold subtlety rather than copy the attention
playbook. The CLS token sits at position 0, *before* any content token. An LSTM is causal and
left-to-right: its hidden state at position 0 has seen only the CLS and nothing else, so reading the CLS
position would summarise an empty sequence. The information accumulates as the recurrence sweeps left to
right, so the state that has seen the *whole* string is the one at the *last valid position*. That is
why I gather at `lengths - 1` rather than at index 0. The `clamp(min=0)` guards the degenerate empty
case; the gather picks per-row the genuinely final hidden, never a padded position. This is the recurrent
reading of the contract — *last valid position*, not CLS — and getting it wrong would silently feed the
head a hidden that never saw the `c`-block, which on `abc` would cap accuracy at chance no matter how
good the cell is. Finally a `nn.LayerNorm(hidden_dim)` on the pooled vector, so the scale handed to the
fixed linear head is stable across the wildly different sequence lengths the three environments produce
(a length-8 `exact` string and a length-256 OOD string would otherwise hand the head summaries on very
different scales, and the single shared head cannot absorb a scale that grows with length).

There is a second reason recurrence is the right floor here, beyond the counter fixed-point, and it comes
from the *structure of the negatives* the `abc` generator draws. The membership decision is not just
"are the three block-counts equal"; the negatives include wrong block counts, *swapped order*, and
interleaved middles. A pure bag-of-counts feature — how many `a`, `b`, `c` — cannot separate `a^n c^n
b^n` from `a^n b^n c^n`: the counts are identical, only the order differs. So the encoder must be
order-sensitive, and a left-to-right recurrence is order-sensitive by construction: the state after
reading `a a b b` is a different trajectory from the state after `a a`-then-something-else, so a swap of
the `b` and `c` blocks lands the cell in a different place. Trace it loosely — the cell that opened its
input gate on `a`, then expected `b`, then `c`, will find its "expected next block" gate mismatched the
moment the order is wrong, and that mismatch is exactly the kind of thing a recurrence can latch. This is
a point in favor of *starting* recurrent rather than jumping straight to a permutation-agnostic model
that would have to be told about order separately.

I should be explicit about what this LSTM is *not*, because the note about same-named baselines applies.
This is not the full qlib-style sequence-to-one regressor with its own Adam, masked MSE,
gradient-value-clipping, and validation early-stopping — all of that optimization machinery lives in the
*fixed* harness here (AdamW at 3e-4, smooth-L1 / BCE, grad-norm-clip 1.0, 6,000 steps, single seed, no
early stop). The harness also fixes the head, the loss, and the data. So the only thing this rung
contributes is the *encoder*: embedding, 2-layer LSTM, last-position read, LayerNorm. Everything the
cell-derivation says about counters still holds — the cell state can implement the tally — but I am
filling a much smaller slot than a full sequence-to-one training loop, and I should reason only about the
encoder's capacity, not about an optimizer I do not control.

What do I expect, environment by environment, so that the numbers can confirm or refute it? On `abc`
in-distribution I expect the LSTM to do *well* — the blocks in the training range are within reach of
the cell's gradient, and a learned increment/compare counter fits the data; this is the regime where
finite-precision LSTMs are known to nail counter languages, so a high in-distribution membership
accuracy is the prediction, and `abc` scores on exactly that. On `exact` in-distribution I likewise
expect a strong rounded-count accuracy: counting `a`'s in a binary string is the *easiest* counter, a
single accumulator with no comparison, so the in-distribution number should be near the top. The place
I am genuinely unsure — and it is the place the task is built to expose — is *out of distribution*, and
the tanh calculation above already told me the mechanism. The LSTM's counter is exact only to the extent
its learned per-step increment is exact and its read-out `tanh(c_t)` has not saturated. At training
lengths up to 64 the cell state stays in a range the model has shaped; at OOD lengths of 128–256 the
accumulated state runs into the flat part of `tanh` (`tanh(8.5) ≈ 1` for a count of 85 at step `δ ≈
0.1`), where successive counts are indistinguishable, and a per-step increment that was 0.99 instead of
1.00 compounds into a several-unit error over twice as many steps. So my falsifiable expectation is a
*split*: high in-distribution accuracy on both `abc` and `exact`, but a large extrapolation gap — the
`exact` OOD accuracy (the `exact` score), which demands the *rounded* count be exactly right at length
256, could collapse all the way to the floor, and the `length-ood` retention could be poor, because the
in-distribution counter the LSTM learns does not transfer cleanly to twice the length. If that split
shows up, it tells me the recurrent counter is *memorized in-range* rather than *implemented as an exact
integer recurrence*, and the next rung has to change the representation of length itself rather than just
the way the count is carried.

So rung one is the honest recurrent floor: an LSTM whose cell state can in principle be an integer
counter, read at the last valid position, normalized into the fixed head. I expect it to win
in-distribution and to reveal, in its OOD numbers, exactly how a continuous-state counter degrades when
the sequence outgrows the lengths that shaped it — and that degradation is the problem the rest of the
ladder exists to attack.

One more reason to start here rather than skip ahead: it keeps the comparison honest. The LSTM and the
later encoders share the same fixed harness — same AdamW, same 6,000 steps, same fixed head, same data,
same single seed — and the same width and depth budget. So whatever the OOD numbers show is attributable
to the *encoder*, not to a training-loop difference I quietly introduced, and not to a capacity
difference either, since I deliberately left ~95% of the parameter ceiling unspent. If the recurrent
counter wins in-distribution while a later encoder wins on retention, that is an architecture result,
cleanly isolated, telling me which property — exact in-range counting versus graceful length behavior —
each family buys. Starting at a model that genuinely *counts*, rather than the scaffold's generic GRU
default, is what makes that read-out interpretable. The full scaffold module is in the answer.
