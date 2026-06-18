The LSTM confirmed the diagnosis I closed on, and it confirmed it in the most useful way — by being almost
right, and failing in exactly one place. Removing the absolute-position table did everything I bet it would
for length generalization. On `dyck-k2-m3` the LSTM is essentially perfect: `ood_token_acc` 0.99994,
`ood_string_acc` 0.997 — a fixed-width recurrent state, processed by a length-independent transition,
generalizes to longer strings cleanly when the language is shallow (depth 3, 2 types). On `dyck-length-ood`
it jumps from the Transformer's 0.73 to 0.974 token accuracy, and `ood_string_acc` climbs off the floor
from 0.0 to 0.026 — the length-independent rule extrapolates from train ≤64 to test 128–256, exactly where
absolute positions had collapsed. The headline geometric mean rose from 0.785 to 0.962. The recurrence was
the right move, and its win is largest precisely where the extrapolation gap was largest.

But the third environment is the tell, and it is the tell I predicted. On `dyck-k8-m5` — 8 bracket types,
depth 5 — the LSTM gets `ood_token_acc` 0.924, respectable, but `ood_string_acc` 0.001 and `id_string_acc`
only 0.126. Read those two numbers together. Token accuracy near 0.92 with string accuracy near zero means
the model is right at most positions and wrong at a *scattering* of them — and on Dyck, the positions that
matter most are the closers, where the model must name *which* of 8 closing brackets matches the symbol on
top of the stack. A fixed-width vector of 64 units, sliding a depth-5 stack of 8-way symbols, has to pack
the identity of the top-of-stack symbol cleanly enough that the read-out picks the right one of 8. It mostly
does — hence 0.92 token — but it smears under load: deep nesting plus many symbol types means the dense
state is emulating a stack by superposition, and the superposition blurs just often enough that almost no
full string survives. This is not a length problem anymore; even `id_string_acc` is 0.126. It is a
*precision* problem: a dense vector is the wrong *shape* of memory for an order-sensitive, multi-type stack,
and the harder the content, the more it blurs. The fix is not more recurrence; it is to stop asking a dense
vector to *emulate* a stack and instead give the model an *explicit* stack data structure.

What shape do I need? The machine that recognizes context-free languages — and Dyck is the toy core of the
context-free class — is the pushdown automaton: a finite controller plus a stack. The stack is the data
structure whose entire job is "remember things in order, match last-in-first-out, and read the top." If I
give the recurrent controller a real stack and let it learn when to PUSH (on an opener) and when to POP (on
a closer), then the top-of-stack symbol is held *explicitly* in its own slot, not superposed into a dense
state, and the read-out reads it directly. The order is maintained by the stack structure itself, not by a
vector that has to encode order implicitly. That is the structural fix for the precision failure.

The obstacle to a literal stack is differentiability: a hard PUSH/POP/NO-OP choice makes the loss
piecewise-constant in the controller's parameters — nudge a weight and, until the argmax flips, the action
and the output and the loss are unchanged, so the gradient is zero almost everywhere. Backprop has nothing
to push on, and reinforcement-style estimators have brutal credit-assignment variance on long sequences. So
I relax the discreteness without killing the stack. The controller emits a *probability* for each action
via a 3-way softmax, and the next stack is the probability-weighted *superposition* of the three pure
operations applied to the current stack. Softmax (not three independent sigmoids) because the actions are
mutually exclusive and I want a convex combination on the simplex, so blending the candidate stacks gives a
weighted *average* rather than an arbitrary rescaling.

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

Parameter budget: the embedding, one fused controller linear over `hidden + stack_dim + hidden` inputs, a
small `3 + stack_dim` action/candidate projection, and a read-out over `hidden + stack_dim` — with
`hidden = stack_dim = 64` this is on the order of 18 000–20 000 parameters, the *smallest* model on the
board, a quarter of the LSTM's and an order of magnitude under the budget. The explicit stack buys
precision without buying parameters, because the stack *tensor* is activation, not weights.

The falsifiable expectations against the LSTM's numbers. The whole point of the explicit stack is the
precision failure on `dyck-k8-m5`, so that is the headline test: the LSTM got `ood_token_acc` 0.924 but
`ood_string_acc` 0.001 and `id_string_acc` 0.126 there. If the explicit stack is genuinely holding the
top-of-stack symbol in its own slot rather than superposing it, the *string* accuracies on `dyck-k8-m5`
should jump dramatically — from near zero toward near one — because the scattered closer errors that broke
every string should disappear. That is the sharp, falsifiable claim: if `dyck-k8-m5` string accuracy stays
low, the explicit stack is not actually crisper than the dense state and my read of the precision failure is
wrong. On the two environments the LSTM already nearly solved (`dyck-k2-m3` at 0.997 OOD string,
`dyck-length-ood` at 0.974 OOD token), I expect the stack to at least match and ideally close the remaining
gap to 1.0 — the bounded depth and length-independent transition give it the same length-generalization the
LSTM had, plus the explicit memory. If the stack reaches near-perfect token *and* string accuracy across
all three environments, where the LSTM was perfect on one, strong on one, and precision-broken on the
hardest, then the explicit-stack hypothesis is confirmed and this is the top of the baseline ladder. The
full scaffold module — embedding, fused tanh controller, fixed-band PUSH/POP/NO-OP shifts, bounded
`m + 2` stack, `[h, top]` read-out — is in the answer.
