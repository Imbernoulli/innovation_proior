The LSTM confirmed the diagnosis I closed on, and it confirmed it in the most useful way — by being almost
right, and failing in exactly one place. Removing the absolute-position table did everything I bet it would
for length generalization. On `dyck-k2-m3` the LSTM is essentially perfect: `ood_token_acc` 0.999938,
`ood_string_acc` 0.997 — a fixed-width recurrent state, processed by a length-independent transition,
generalizes to longer strings cleanly when the language is shallow (depth 3, 2 types). On `dyck-length-ood`
it jumps from the Transformer's 0.734313 to 0.973799 token accuracy, and `ood_string_acc` climbs off the
floor from 0.0 to 0.026 — the length-independent rule extrapolates from train ≤64 to test 128–256, exactly
where absolute positions had collapsed. The headline geometric mean rose from 0.785 to 0.962. The recurrence
was the right move, and its win is largest precisely where the extrapolation gap was largest — which is the
falsifiable prediction I made about `dyck-length-ood` coming back confirmed.

But the third environment is the tell, and it is the tell I predicted. On `dyck-k8-m5` — 8 bracket types,
depth 5 — the LSTM gets `ood_token_acc` 0.924061, respectable, but `ood_string_acc` 0.001 and `id_string_acc`
only 0.126. I want to read those numbers against each other rather than in isolation, because the
combination is the whole diagnosis. First, token-versus-string: `ood_token_acc` 0.924 with `ood_string_acc`
0.001 means the model is right at ~92% of positions but almost never right at *all* positions of a string.
Run the string-survival arithmetic backward — if per-position error were independent at rate `ε = 0.076`,
an 80-token string would survive at `(1 − 0.076)^80 ≈ 0.0018`, right at the measured 0.001 — so the errors
behave like a low-rate-but-persistent scatter across positions, and on Dyck the positions that matter most
are the closers, where the model must name *which* of 8 closing brackets matches the symbol on top of the
stack. Second, and more damning, the *in-distribution* string accuracy is also broken: `id_string_acc`
0.126, meaning even on training-length strings the LSTM fully solves only one string in eight. That single
number rules out a length explanation. Length generalization cannot be the culprit for a string the model
saw the length of during training; the failure is present at the lengths it was fit on. Compare the sister
environment: on `dyck-length-ood` the LSTM's `id_string_acc` is 0.666, five times higher — so the LSTM can
hold a depth-4, 4-type stack in-distribution far better than a depth-5, 8-type one. The gap between 0.666
and 0.126 is a *content* gap, not a length gap.

So what is actually wrong on `dyck-k8-m5`? A fixed-width vector of 64 units, sliding a depth-5 stack of
8-way symbols, has to pack the identity of the top-of-stack symbol cleanly enough that the read-out picks
the right one of 8. It mostly does — hence 0.924 token — but it smears under load: deep nesting plus many
symbol types means the dense state is emulating a stack by *superposition*, storing several levels' symbol
identities as an additive mixture in one vector, and the superposition blurs just often enough that almost
no full string survives. This is not a length problem anymore; even `id_string_acc` is 0.126. It is a
*precision* problem, and it is a problem of memory *shape*: a dense vector is the wrong *shape* of memory for
an order-sensitive, multi-type stack, and the harder the content, the more it blurs.

Now I have to pick the fix, and there are three candidates on the table; I want to eliminate two on
arithmetic before I build the third. The lazy candidate is a *wider* LSTM — push `hidden` from 64 up toward
the parameter ceiling and hope the extra units resolve the superposition. But the information budget says
capacity is not the binding constraint. The bounded stack for `k = 8, m = 5` needs only ~`5 · log2(8) = 15`
bits of content plus a depth counter — under 20 bits — and 64 real-valued units already carry vastly more
than that. The LSTM is not failing because 64 units cannot *hold* the stack; it is failing because
superposition into a *single dense vector* does not keep the levels *separable* enough for a clean read-out.
Widening the vector adds representational room but does not change its shape — it is still an additive
mixture with no per-level addressing — so it would chip at the blur, not remove it, and it would spend
parameters (the LSTM is already ~67k) to do it. Wider-is-not-crisper: eliminated on the shape argument.

The second candidate is the *hard* stack: a genuine discrete data structure with argmax PUSH/POP/NO-OP
decisions and integer indexing. That is the correct *shape* — a real last-in-first-out store with per-slot
addressing — but it is undifferentiable, and I have exactly 100 gradient steps and no curriculum. A hard
argmax makes the loss piecewise-constant in the controller's parameters: nudge a weight and, until the
argmax flips, the action and the output and the loss are unchanged, so the gradient is zero almost
everywhere and backprop has nothing to push on. The alternative, a policy-gradient estimator over the
discrete actions, carries credit-assignment variance that grows with sequence length — and my sequences run
to 256 tokens — so in 100 steps it would be pure noise. Eliminated on the trainability argument: right
shape, unlearnable in this budget.

That leaves the third candidate, and the elimination of the first two is exactly what points at it: I want
the *shape* of the hard stack with the *trainability* of a smooth function. The machine that recognizes
context-free languages — and Dyck is the toy core of the context-free class — is the pushdown automaton: a
finite controller plus a stack. The stack is the data structure whose entire job is "remember things in
order, match last-in-first-out, and read the top." If I give the recurrent controller a real stack and let
it learn when to PUSH (on an opener) and when to POP (on a closer), then the top-of-stack symbol is held
*explicitly* in its own slot, not superposed into a dense state, and the read-out reads it directly. The
order is maintained by the stack structure itself, not by a vector that has to encode order implicitly. That
is the structural fix for the precision failure — it addresses the *shape*, which widening did not.

The obstacle to a literal stack is differentiability, the same wall that killed the hard candidate, so I
relax the discreteness without killing the stack. The controller emits a *probability* for each action via
a 3-way softmax, and the next stack is the probability-weighted *superposition* of the three pure operations
applied to the current stack. Softmax (not three independent sigmoids) because the actions are mutually
exclusive and I want a convex combination on the simplex, so blending the candidate stacks gives a weighted
*average* rather than an arbitrary rescaling — the three coefficients sum to 1, so a saturated softmax is
literally one pure action and nothing is double-counted. Crucially the superposition here is over
*operations*, not over stored symbols: the LSTM's fatal superposition mixed the *contents* of different
stack levels into one vector, whereas this mixes three *candidate next-stacks* that each keep the levels in
separate rows, and as training drives the softmax to a corner the mixture collapses to a single crisp stack.
Same word, opposite consequence.

Represent the stack as a tensor of shape `[B, depth, stack_dim]`, top at row 0. The three pure actions are
shifts of this tensor. A pure PUSH shifts every row *down* (row `i` ← row `i−1`) and writes a fresh
candidate vector into row 0. A pure POP shifts every row *up* (row `i` ← row `i+1`). A pure NO-OP leaves
the stack unchanged. I implement the down- and up-shifts as fixed band matrices — `push_shift` with the
sub-diagonal identity, `pop_shift` with the super-diagonal identity — so a whole shift is one `matmul`, and
inject the new top with a one-hot `top_slot` selecting row 0. The update is then `stack = push_p · pushed +
pop_p · popped + noop_p · stack`, where `pushed = push_shift @ stack + top_slot · new_top` and `popped =
pop_shift @ stack`. Every operation is a multiply-add by a continuous softmax probability, so the whole
stack is a smooth function of the parameters and gradients flow through it into the action logits and the
controller. When a probability saturates to 1 the update *is* the corresponding pure action, so the soft
stack contains the hard stack as its limit, and minimizing the prediction loss drives the softmax toward
the corners on its own. The relaxation is a path *to* discrete behavior, not a replacement for it.

I should verify the band matrices actually implement a LIFO before I trust them, because a sign error here
is invisible until the string accuracies come back wrong. Take `pushed = push_shift @ stack`: with
`push_shift[i, j] = 1` exactly when `i = j + 1`, the product gives `pushed[i] = stack[i−1]` for `i ≥ 1` and
`pushed[0] = 0` — every row slid down one, row 0 cleared, ready for the new top. Adding `top_slot · new_top`
writes `new_top` into the cleared row 0. Now pop that: with `pop_shift[i, j] = 1` when `j = i + 1`,
`popped[i] = pushed[i+1]`, so `popped[i] = stack[i]` for `i` up to `depth − 2` and `popped[depth−1] = 0`.
Reading it off: a saturated PUSH followed by a saturated POP returns the original stack exactly on every
occupied level, losing only whatever would have fallen off the bounded bottom — which for a legal
Dyck-(k,m) configuration never reaches the bottom `+2` slack rows anyway. So `pop(push(S)) = S` on the
region that matters: the construction is a genuine last-in-first-out store, and the smoothing sits only on
*which* operation happens, not on the algebra of the operations. That is the check I needed; the shifts are
correct.

It is worth tracing one two-type string through the saturated stack to see *why* this fixes the closer
precision that broke the LSTM, because the whole bet is that a dedicated slot beats a blurred mixture. Take
`( [ ] )` over two bracket types, the exact kind of interleaving where the LSTM had to remember which of the
open brackets sits on top. Start with a zeroed stack. Read `(`: saturated PUSH writes the type-`(` symbol
vector into row 0, so `top = v_(`. Read `[`: PUSH shifts `v_(` down to row 1 and writes `v_[` into row 0,
so now `top = v_[` and row 1 still holds `v_(` *unmixed* — this is the difference from the dense state,
which would have had to superpose `v_(` and `v_[` into one vector. To predict the token after `[`, the
read-out sees `top = v_[` and must emit the matching closer `]`; it reads the type directly off row 0, no
disambiguation among 8 needed. Read `]`: POP shifts row 1 back up, so `top = v_(` again, restored bit-for-bit
because the band-matrix POP is exact — and the read-out now sees `v_(` and predicts `)`. Read `)`: POP,
stack empty. At no step did the model have to decode a superposition; the identity that names the closer was
always sitting alone in row 0. That is the mechanism the `dyck-k8-m5` string accuracy is testing: with 8
types the dense LSTM had to separate 8 symbol identities out of an additive mixture and blurred; the
explicit stack never mixes them, so the read-out's job collapses from "unmix and classify" to "read one
slot." If that reasoning is right the `dyck-k8-m5` string accuracy should move the most of any number on the
board.

Now the controller, and here I make the design choices that fit *this* harness, which differ from the
unbounded-stack-with-curriculum constructions in the prior art — I want the version that trains end-to-end
in 100 steps with no curriculum and no test-time surgery. The controller is a single fused linear over the
concatenation of the token embedding, the *top of the stack*, and the *previous hidden state*, squashed by
`tanh`: `h_t = tanh(W [emb_t, top_{t-1}, h_{t-1}])`. I keep the ordinary hidden-to-hidden recurrence
(`h_{t-1}` is fed in) rather than zeroing it: on these abstract counting probes one might kill the
recurrence to *isolate* the stack, but here I am language-modeling for accuracy, not isolating a mechanism,
so I let the controller use both its dense state and the explicit stack. I feed back only the single top
slot (`k = 1`), not a window of the top few, because the Dyck next-token rule depends on exactly *one* thing
— the symbol on top — so one slot is the right amount of context and keeps the fused projection small. The
new top candidate is `tanh(push_raw)`, bounded so that repeated pushes and blends keep stack entries in a
fixed range and cannot grow without limit across a long string. The read-out is a linear over the
concatenation of the hidden state and the (new) top of the stack, `head([h_t, top_t]) → logits` — so the
prediction sees both the controller's summary and the explicit top.

Two deliberate departures from the unbounded constructions, both because the harness is bounded-depth Dyck
trained for 100 steps. First, the stack depth is *bounded*: I set `stack_depth = config.m + 2`. The language
guarantees nesting never exceeds `m`, so a stack of depth `m` plus a small margin holds every legal
configuration; an unbounded growing stack is unnecessary here and a fixed-depth tensor is far cheaper and
trains in one shot. The `+2` gives slack for the empty-stack and one-over-full transients. Second, I do
*not* use an out-of-range sentinel (`−1`) for the empty slot or a test-time argmax discretization. The
unbounded constructions needed the sentinel so the controller could detect an exhausted stack on tasks like
`a^n b^{2n}`, and needed test-time discretization to kill numerical drift over hundreds of soft steps. Here
the stack is zero-initialized, the depth is bounded so drift has fewer slots to accumulate in, and the model
is evaluated with its soft actions exactly as trained — the harness scores `argmax(logits)`, so any residual
softness in the *actions* only matters if it changes the *token* prediction, which the read-out is trained
to get right. I am betting the bounded depth and the explicit top-of-stack slot make the soft stack crisp
enough without the extra machinery.

Parameter budget, worked from the shapes so I can confirm the model stays the smallest on the board. With
`hidden = stack_dim = 64` and `stack_depth = m + 2`: the fused controller linear maps `hidden + stack_dim +
hidden = 192` inputs to 64, costing `192 · 64 + 64 = 12 352`; the action/candidate projection maps 64 to
`3 + stack_dim = 67`, costing `64 · 67 + 67 = 4 355`; the read-out maps `hidden + stack_dim = 128` to
`vocab`, costing `128 · vocab + vocab`; and the embedding is `vocab · 64`. For the smallest env (`k = 2`,
`vocab = 6`) that totals `12 352 + 4 355 + 774 + 384 ≈ 17 865`; for the largest (`k = 8`, `vocab = 18`) it
is about `20 181` — so I predict roughly 18k–20k parameters, the *smallest* model on the board, about a
quarter of the LSTM's ~67k and an order of magnitude under the 500k budget. The explicit stack buys
precision without buying parameters, because the stack *tensor* is activation, not weights — the band
matrices and `top_slot` are non-persistent buffers with zero trainable cost. That is the point: the fix is
structural, not a capacity purchase, which is exactly why widening the LSTM was the wrong lever and this is
the right one.

The falsifiable expectations against the LSTM's numbers, sharp enough to be wrong. The whole point of the
explicit stack is the precision failure on `dyck-k8-m5`, so that is the headline test: the LSTM got
`ood_token_acc` 0.924061 but `ood_string_acc` 0.001 and `id_string_acc` 0.126 there. If the explicit stack
is genuinely holding the top-of-stack symbol in its own slot rather than superposing it, the *string*
accuracies on `dyck-k8-m5` should jump dramatically — from near zero toward near one — because the scattered
closer errors that broke every string should disappear once the top symbol is read from a dedicated slot
instead of decoded from a blurred mixture. That is the sharp, falsifiable claim: if `dyck-k8-m5` string
accuracy stays low, the explicit stack is not actually crisper than the dense state and my read of the
precision failure is wrong — and in particular, if `id_string_acc` there stays near 0.126, the shape fix
did nothing and I would be forced back toward capacity or a harder discretization. On the two environments
the LSTM already nearly solved (`dyck-k2-m3` at 0.997 OOD string, `dyck-length-ood` at 0.973799 OOD token),
I expect the stack to at least match and ideally close the remaining gap to 1.0 — the bounded depth and
length-independent transition give it the same length-generalization the LSTM had, plus the explicit memory,
so I would be surprised to see any *regression* there; a regression would mean the soft actions are drifting
even at shallow depth. There is a clean scoring target implied by all this. The geometric-mean task score is capped by its
smallest factor, and for the LSTM that cap was `dyck-k8-m5` at `ood_token_acc` 0.924061, holding the mean at
0.962. If the explicit stack lifts `dyck-k8-m5` token accuracy to near 1.0 while the other two stay at their
already-near-1.0 values, the geometric mean is pulled up toward 1.0 — the whole ceiling on the task score is
exactly the environment my shape fix targets. So the headline `dyck-k8-m5` token *and* string jump is not
just a mechanism check, it is the single lever that moves the task score, which is why I aimed the
architectural change squarely at it rather than at the two environments the LSTM already nearly solved. If
the stack reaches near-perfect token *and* string accuracy across all three environments, where the LSTM was
perfect on one, strong on one, and precision-broken on the hardest, then the explicit-stack hypothesis is
confirmed and this is the top of the baseline ladder. The full scaffold
module — embedding, fused tanh controller, fixed-band PUSH/POP/NO-OP shifts, bounded `m + 2` stack,
`[h, top]` read-out — is in the answer.
