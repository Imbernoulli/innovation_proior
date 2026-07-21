The LSTM confirmed the diagnosis in the most useful way — by being almost right, and failing in exactly one
place. Removing the absolute-position table did everything I bet it would for length generalization. On
`dyck-k2-m3` it is essentially perfect (`ood_token_acc` 0.999938, `ood_string_acc` 0.997): a fixed-width
recurrent state with a length-independent transition generalizes cleanly to longer strings when the language
is shallow. On `dyck-length-ood` it jumps from the Transformer's 0.734313 to 0.973799 token accuracy, and
`ood_string_acc` climbs off the floor from 0.0 to 0.026 — the length-independent rule extrapolates from
train ≤64 to test 128–256, exactly where absolute positions collapsed. The geometric mean rose from 0.785 to
0.962, and the win is largest precisely where the extrapolation gap was largest.

The third environment is the tell, and it is the one I predicted. On `dyck-k8-m5` — 8 types, depth 5 — the
LSTM gets `ood_token_acc` 0.924061 but `ood_string_acc` 0.001 and `id_string_acc` only 0.126. Read
together: token 0.924 with string 0.001 means the model is right at ~92% of positions but almost never right
at *all* positions. Run the string-survival arithmetic — independent per-position error `ε = 0.076` gives an
80-token string survival of `(1 − 0.076)^80 ≈ 0.0018`, right at the measured 0.001 — so the errors are a
low-rate persistent scatter, and on Dyck the positions that matter are the closers, where the model must
name *which* of 8 closing brackets matches the top of the stack. More damning, the *in-distribution* string
accuracy is also broken at 0.126: length generalization cannot be the culprit for a string the model saw the
length of during training. Against the sister environment `dyck-length-ood` at `id_string_acc` 0.666, five
times higher, the gap is a *content* gap, not a length gap.

So a fixed-width vector of 64 units, sliding a depth-5 stack of 8-way symbols, has to pack the top-of-stack
identity cleanly enough to pick the right one of 8. It mostly does — hence 0.924 token — but it smears under
load: deep nesting plus many types means the dense state emulates the stack by *superposition*, storing
several levels' identities as an additive mixture in one vector, and the mixture blurs just often enough
that almost no full string survives. This is a *precision* problem, and specifically a problem of memory
*shape*: a dense vector is the wrong shape for an order-sensitive, multi-type stack.

Three candidates for the fix, two eliminable on argument. A *wider* LSTM does not help: the bounded stack
for `k = 8, m = 5` needs ~15 bits of content plus a depth counter, under 20 bits, and 64 units already carry
vastly more — the failure is not that 64 units cannot hold the stack, it is that superposition into a single
vector does not keep the levels *separable* for a clean read-out. Widening adds room but not addressing; it
is still an additive mixture, so it chips at the blur without removing it. A *hard* stack with argmax
PUSH/POP/NO-OP is the correct shape but undifferentiable: a hard argmax makes the loss piecewise-constant in
the controller's parameters, zero gradient almost everywhere, and a policy-gradient estimator carries
credit-assignment variance that grows with the (up to 256-token) sequence length — pure noise in 100 steps
with no curriculum. Right shape, unlearnable in this budget.

The elimination points at the third candidate: I want the *shape* of the hard stack with the *trainability*
of a smooth function. The machine that recognizes context-free languages — and Dyck is their toy core — is
the pushdown automaton, a finite controller plus a stack, and the stack's whole job is "remember in order,
match last-in-first-out, read the top." Give the recurrent controller a real stack and let it learn to PUSH
on openers and POP on closers, and the top-of-stack symbol is held *explicitly* in its own slot rather than
superposed, with order maintained by the structure itself. That addresses the shape, which widening did not.

To keep it differentiable I relax the discreteness without killing the stack. The controller emits a
*probability* for each action via a 3-way softmax, and the next stack is the probability-weighted
superposition of the three pure operations applied to the current stack. Softmax, not three independent
sigmoids, because the actions are mutually exclusive and I want a convex combination on the simplex — the
coefficients sum to 1, so a saturated softmax is literally one pure action with nothing double-counted.
Crucially this superposition is over *operations*, not stored symbols: the LSTM's fatal superposition mixed
the *contents* of different levels into one vector, whereas this mixes three *candidate next-stacks* that
each keep the levels in separate rows, and as training drives the softmax to a corner the mixture collapses
to a single crisp stack. Same word, opposite consequence.

Represent the stack as a tensor `[B, depth, stack_dim]`, top at row 0. The three pure actions are shifts:
PUSH slides every row down (row `i` ← row `i−1`) and writes a fresh candidate into row 0, POP slides every
row up (row `i` ← row `i+1`), NO-OP leaves it unchanged. I implement the shifts as fixed band matrices —
`push_shift` the sub-diagonal identity, `pop_shift` the super-diagonal identity — so a shift is one `matmul`,
and inject the new top with a one-hot `top_slot`. The update is `stack = push_p · pushed + pop_p · popped +
noop_p · stack` with `pushed = push_shift @ stack + top_slot · new_top` and `popped = pop_shift @ stack`.
Every operation is a multiply-add by a continuous probability, so the whole stack is smooth in the
parameters and gradients flow into the action logits; when a probability saturates to 1 the update *is* the
pure action, so the soft stack contains the hard stack as its limit and minimizing the loss drives the
softmax toward the corners on its own.

The shifts compose into a genuine LIFO: a saturated PUSH then POP returns the original stack, losing only
what falls off the bounded bottom — which a legal Dyck-(k,m) configuration never reaches. The smoothing sits only on *which* operation happens,
not on the algebra. This is what fixes the closer precision that broke the LSTM: reading `( [ ] )`, after the
two opens row 0 holds `v_[` and row 1 holds `v_(` *unmixed*, so predicting `]` reads the type directly off
row 0 with no disambiguation among 8, and the POP restores `v_(` for the next prediction. The read-out's job
collapses from "unmix and classify" to "read one slot."

Now the controller, tuned for training end-to-end in 100 steps with no curriculum. It is a single fused
linear over the concatenation of the token embedding, the top of the stack, and the previous hidden state,
squashed by `tanh`: `h_t = tanh(W [emb_t, top_{t-1}, h_{t-1}])`. I keep the ordinary hidden-to-hidden
recurrence rather than zeroing it to isolate the stack — I am language-modeling for accuracy, not isolating
a mechanism, so the controller uses both its dense state and the explicit stack. I feed back only the single
top slot, because the Dyck next-token rule depends on exactly one thing, and this keeps the projection small.
The new top candidate is `tanh(push_raw)`, bounded so repeated pushes and blends keep entries in range across
a long string. The read-out is a linear over `[h_t, top_t]`, so the prediction sees both the summary and the
explicit top.

Two departures from the unbounded-stack constructions, both because this is bounded-depth Dyck trained for
100 steps. First, the stack depth is *bounded* at `stack_depth = config.m + 2`: nesting never exceeds `m`,
so a stack of depth `m` plus slack for the empty/one-over-full transients holds every legal configuration,
and a fixed-depth tensor is far cheaper and trains in one shot. Second, no out-of-range `−1` empty sentinel
and no test-time argmax discretization: the unbounded constructions needed the sentinel to detect an
exhausted stack on tasks like `a^n b^{2n}` and needed discretization to kill drift over hundreds of soft
steps, but here the stack is zero-initialized, the bounded depth leaves drift few slots to accumulate in,
and the harness scores `argmax(logits)`, so residual softness in the *actions* only matters if it changes
the *token* prediction the read-out is trained to get right.

Parameter budget from the shapes: with `hidden = stack_dim = 64` the fused controller costs `192 · 64 + 64
= 12 352`, the action/candidate projection `64 · 67 + 67 = 4 355`, plus the read-out and embedding (`vocab`-
dependent), totalling ~18k for `k = 2` and ~20k for `k = 8` — the *smallest* model on the board, about a
quarter of the LSTM's ~67k. The explicit stack buys precision without buying parameters, because the stack
*tensor* is activation, not weights, and the band matrices are non-persistent buffers with zero trainable
cost. That is the point: the fix is structural, which is why widening the LSTM was the wrong lever.

The headline falsifiable test is `dyck-k8-m5`, where the LSTM got `ood_token_acc` 0.924061 but
`ood_string_acc` 0.001 and `id_string_acc` 0.126. If the explicit stack genuinely holds the top symbol in
its own slot, the *string* accuracies there should jump from near zero, because the scattered closer errors
that broke every string disappear once the top is read from a dedicated slot instead of decoded from a
mixture — and since the geometric-mean score is capped by that same smallest factor, this is also the single
lever that moves the task score. If `id_string_acc` there stays near 0.126, the shape fix did nothing and I
am forced back toward capacity or a harder discretization. On the two environments the LSTM already nearly
solved I expect the stack to at least match; a regression there would mean the soft actions are drifting even
at shallow depth. The full module is in the answer.
