The Transformer told me, in numbers, exactly which problem I have to solve, and it is not the one I might
have feared. In-distribution the attention stack fits well: 0.96 token accuracy on `dyck-k2-m3`, 0.92 on
`dyck-length-ood`. So the model *can* represent the matching — its one-hop reach finds the bracket it needs
when the positions are familiar. But on the length-OOD splits it sags exactly where I predicted it would:
`ood_token_acc` falls to 0.90 on the easy `dyck-k2-m3` (only a small length gap, 65–96 vs ≤64), and then
collapses to 0.73 on `dyck-length-ood`, where training tops out at 64 and the test runs 128–256 — the
largest extrapolation on the board, and the worst score. The per-string numbers make the diagnosis brutal:
`ood_string_acc` is 0.001, 0.0, 0.0 — essentially *no* OOD string is fully correct. The model gets most
positions right and a handful wrong, and on a language where one wrong position fails the whole string,
that is a failure of generalization, not of capacity. The signature is fitting-fine, extrapolating-poorly,
which is precisely the absolute-position story: the mechanism that lets attention locate the matching
bracket is anchored to absolute coordinates, and at OOD lengths those coordinates are untrained noise. And
`dyck-k8-m5` is worst even in-distribution (0.80 id, 0.73 ood) — 8 bracket types and depth 5 is the hardest
content, and the attention stack does not have a clean place to *store* the stack discipline; it has to
re-derive it by position-indexed lookup every time.

So the next rung must remove the dependence on absolute position. The cleanest way to do that is to stop
processing the sequence as a set of indexed positions and process it as a *stream*: a recurrent model
updates its state by the *same* transition at every step, regardless of the absolute index. There is no
position table to run off the end of. If a recurrent model can carry the stack information forward in its
hidden state, then a string of length 200 is processed by the same per-step rule as a string of length 60
— the rule was trained on every step it ever saw, and there are no untrained positions. That is the
structural reason to expect a recurrent model to length-generalize where the absolute-position Transformer
did not. But "recurrent" alone is not enough; the plain Elman recurrence has its own wall, and I have to
understand it precisely or I will just trade one failure for another.

Follow a single error signal backward through a recurrence. The error that lands on a unit at time `t` and
has to reach a unit `q` steps in the past arrives as a *product of `q` factors*, each factor of the form
`f'(net) · w` — the chain rule telescoped over the path. Stare at that product. If every factor has
magnitude below 1 — the ordinary case, since the logistic derivative peaks at 0.25 and reasonable weights
keep `|f'·w| < 1` — the product shrinks like (something<1)^q: the error *vanishes* exponentially in the
lag. If every factor exceeds 1, it *explodes*. Either way the lag `q` sits in the exponent, and no setting
of the weights makes a generic product of many factors stay `O(1)`. For Dyck this is fatal in exactly the
regime I care about: matching a closer to its opener requires carrying gradient across the entire span of
the nested sub-string, and on OOD lengths that span is longer than anything trained — the very lags where
the vanishing is worst. A plain RNN would memorize short-range matching and fail to learn the long-range
discipline, which is no better than the Transformer's position problem.

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
disambiguated, one layer of width 64 may not cleanly separate the closer identities, and a second layer
gives the network room to compose "what is on top" with "which of 8 types it is." Width is
`config.hidden_dim = 64`, which by the bounded-memory result is generous: an exact recognizer needs only
`O(m log k)` units — for the hardest config `m=5, k=8` that is on the order of `5 · 3 = 15` units — so 64 is
comfortably above the information-theoretic floor, and the question is purely whether gradient descent
*finds* the stack-tracking state, not whether 64 units can hold it. There is no absolute-position table,
because the recurrence supplies order for free from the sequential processing; this is the deliberate
contrast with the Transformer rung. The parameter count is tiny — two LSTM layers of width 64 plus a small
embedding and head come to roughly 67 000 parameters, a quarter of the Transformer's and far inside the
500 000 budget. The forward pass embeds, runs the LSTM over the time axis returning a hidden at every
position, and projects to logits `[B, T, vocab]` — exactly the `DyckModel.forward` contract, scored by the
harness's next-valid-set evaluator.

One thing I want to be clear-eyed about: the recurrence removes the *absolute-position* failure mode, but
it does not remove every length-generalization risk. The LSTM's memory is still a *fixed-width* vector. The
bounded-memory result guarantees a depth-`m` stack of `k` symbols *fits* in `O(m log k)` units, and `m` is
bounded (≤5), so the stack the model must hold is bounded too — this is why I expect the LSTM to do well
here, where for *unbounded*-depth Dyck a fixed-width state would eventually overflow. The OOD strings are
longer but the *depth* never exceeds `m`; the cell only has to hold a bounded stack and slide it as brackets
open and close, which a forget-gated linear state can do at any length. So the bet is that the LSTM's
length-independent transition plus its bounded stack exactly matches Dyck-(k,m)'s bounded-memory structure.

The falsifiable expectations against the Transformer's numbers. First and headline: `ood_token_acc` should
*rise* across the board, and rise most on `dyck-length-ood`, where the Transformer collapsed to 0.73 — if
the absolute-position table was the culprit, removing it should help most exactly where the extrapolation
was largest. I expect `ood_token_acc` near 0.97+ on the two easier environments and a clear gain on
`dyck-length-ood`. Second, the per-string numbers should move off the floor: the Transformer's
`ood_string_acc` of 0.0/0.001/0.0 means it never got a full OOD string right; if the LSTM is genuinely
tracking the stack rather than approximating by position, some OOD strings should come out fully correct.
Third, the one place I am *not* confident: `dyck-k8-m5`. The Transformer's worst environment was the
content-hardest one (8 types, depth 5), and a fixed-width recurrent state has to pack the identity of the
top-of-stack symbol cleanly enough that the read-out picks the right closer among 8. If the LSTM also sags
on `dyck-k8-m5` — high on the other two but visibly lower here, and especially with weak per-string
accuracy — that is the tell that a *fixed-width* memory, however well it generalizes across length, is
struggling to maintain a *crisp* discrete stack, and the next rung should give the model an *explicit*
stack data structure rather than asking a dense vector to emulate one. The full scaffold module — embedding,
2-layer `nn.LSTM`, linear head, no position table — is in the answer.
