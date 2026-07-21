The Transformer told me, in numbers, exactly which problem I have to solve. In-distribution the attention
stack fits well: `id_token_acc` 0.959956 on `dyck-k2-m3`, 0.924174 on `dyck-length-ood`. So the model *can*
represent the matching — its one-hop reach finds the bracket it needs when the positions are familiar. On
the length-OOD splits it sags exactly where I predicted, and the drops are worth reading against my prior
computation. The id→OOD token-accuracy drop is `0.0569` on `dyck-k2-m3` (small gap, 65–96 vs ≤64), `0.0764`
on `dyck-k8-m5`, and `0.1899` on `dyck-length-ood` (train ≤64, test 128–256) — that last is `0.1899 /
0.0569 ≈ 3.3×` the `dyck-k2-m3` drop. Against the untrained-position fractions I worked out beforehand —
~66% of a mid-range length-OOD string versus ~20% for a `dyck-k2-m3` OOD string, a ratio of ~3.3 — the
measured drop ratio and the predicted poisoned-fraction ratio agree to the first digit. That is the
quantitative fingerprint the absolute-position story predicted, far more convincing than the raw scores.

The per-string numbers make the diagnosis brutal: `ood_string_acc` is 0.001, 0.0, 0.0 — essentially *no*
OOD string is fully correct on any environment. The model gets most positions right and a handful wrong,
and on a language where one wrong position fails the whole string, that is a failure of generalization, not
of capacity — the fitting-fine, extrapolating-poorly signature the absolute-position story predicts.
Separately, `dyck-k8-m5` is worst even in-distribution (`id_token_acc` 0.804292, `id_string_acc` only
0.024): 8 bracket types and depth 5 is the hardest content, and the attention stack has no clean place to
*store* the stack discipline — it re-derives it by position-indexed lookup every time. I file that away:
even after I fix the length problem, `dyck-k8-m5`'s in-distribution deficit is not a length phenomenon and
may not be fixed by the same move.

The reflex when a model underperforms is to throw capacity at it — more layers, more width — but the
numbers rule that out. In-distribution the Transformer already reached 0.959956 and 0.924174 on two
environments with ~266k of a 500k budget; it demonstrably *has* the capacity to represent the matching
whenever the positions are trained. The deficit is entirely in the id→OOD gap, a function of *which*
positions were trained, not how many parameters compute over them. Doubling the width enlarges the position
table right along with everything else, handing me more untrained rows to fail on. So scaling the
Transformer fails on its own evidence, and the architectural change is forced.

The next rung must remove the dependence on absolute position. Keeping attention and swapping to relative/
rotary offsets is only a partial fix — an OOD `dyck-length-ood` string has matching spans of 150+ brackets,
so the offset `i − j` at the deepest matches exceeds any offset a length-64-capped training saw, and the
kernel extrapolates past its support. The sharpest test of the hypothesis — is it position *per se* — is to
remove the position table entirely, which is what a recurrent network does: it processes the sequence as a
*stream*, updating state by the *same* transition at every step regardless of absolute index. There is no
table to run off the end of; a length-200 string is processed by the same per-step rule as a length-60 one,
and that rule was trained on every step it ever saw. That is the structural reason to expect recurrence to
length-generalize where absolute positions did not. But "recurrent" alone is not enough — the plain Elman
recurrence has its own wall, and I have to understand it precisely or I will just trade one failure for
another.

Follow one error signal backward through a recurrence: the error landing on a unit at time `t` that has to
reach `q` steps into the past arrives as a *product of `q` factors*, each `f'(net) · w`. If every factor
has magnitude below 1 — the ordinary case, since the logistic derivative peaks at 0.25 — the product
shrinks like (something<1)^q and the error *vanishes* exponentially in the lag; at a per-step multiplier of
0.5, a lag of `q = 40` (the span of a modest nested sub-string) leaves `0.5^40 ≈ 9 × 10⁻¹³`, gone. If every
factor exceeds 1 it explodes. Either way `q` sits in the exponent. For Dyck this is fatal exactly where I
care: matching a closer to its opener requires carrying gradient across the whole nested span, and on OOD
lengths that span is longer than anything trained. A plain RNN would memorize short-range matching and fail
the long-range discipline — no better than the Transformer's position problem.

If the product of factors is the disease, the cure makes the product *exactly 1* for any `q`. For a single
unit with self-connection `w`, the one-step multiplier is `f'(net) · w`; setting it to 1 forces
`f'(net) = 1/w`, a constant, so `f` must be linear — the clean choice is the identity with `w = 1`. Then
the state simply persists and the backpropagated error is multiplied by 1 at every step: a constant error
carousel along which gradient survives an arbitrarily long lag. For Dyck this is exactly the object I want,
because the depth-`m` stack must be *held* across the nested span and *read* when the matching closer
arrives — a memory, not a position lookup.

A bare linear self-loop cannot be wired to the rest of the net: one incoming weight must both *write* new
information when it arrives and *protect* stored information at all other moments, two opposite jobs. The
resolution is context-sensitive, multiplicative control by other units — multiplicative because protection
means the irrelevant input contributes exactly zero, which only a multiply by a value in `[0,1]` achieves.
So I wrap the carousel in learned sigmoid gates: an **input gate** deciding when to write, an **output
gate** deciding when to read, and — because within a Dyck string the model must release working memory as
sub-structures complete — a **forget gate** that multiplies the carried state, recovering the exact carousel
at 1 and wiping memory at 0, learning its own reset points (for Dyck, the bracket closures). The GRU folds
input and forget into one coupled gate and drops the separate cell state, so its carousel and read-out live
on the same vector; the LSTM's virtue here is exactly the *separation* — `c_t` is a protected linear
accumulator and `h_t = o_t ⊙ tanh(c_t)` a gated *view* of it, so a value can be held in `c` untouched across
a long span while the output gate presents whatever it chooses. For a stack that must be held across a span
and read only at the closer, that separation is worth the extra gate.

The gates are index-free: with `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`, `h_t = o_t ⊙ tanh(c_t)`, a depth-2 string
embedded 150 tokens into a length-256 sequence runs under the same gate settings as one at the start,
because the transition is a function of `(x_t, h_{t-1}, c_{t-1})` and never of `t`. That is the property the
Transformer lacked. These equations are exactly `nn.LSTM`, so I use the fused version rather than re-deriving
it: `nn.Embedding(vocab, hidden)` → `nn.LSTM(hidden, hidden, num_layers=2, batch_first=True)` →
`nn.Linear(hidden, vocab)`. Two layers so the second can compute over the first's summary — for
`dyck-k8-m5`, where 8 types must be disambiguated and the Transformer's `id_string_acc` was 0.024, one layer
of width 64 may not cleanly separate the closer identities. Width `config.hidden_dim = 64` is generous
against the information floor: an exact recognizer holds a stack of ≤ `m` symbols of `k` types plus a depth
counter, `5 · log2(8) = 15` bits plus ~2.6 for depth (0…5) at the hardest config — ~18 bits, far under what
64 real-valued units carry, so the question is purely whether gradient descent *finds* the stack-tracking
state, not whether 64 units can hold it. No position table, ~67k params — about a quarter of the
Transformer's ~266k, because the recurrence spends nothing on positions.

I want to be clear-eyed that the recurrence removes the *absolute-position* failure but not every
length-generalization risk: the LSTM's memory is still a fixed-width vector. What saves it here is that
Dyck-(k,m) has *bounded* depth — the OOD strings are longer but never *deeper* than `m ≤ 5`, so the cell
only has to hold a bounded stack and slide it as brackets open and close, which a forget-gated linear state
does at any length. (For *unbounded*-depth Dyck a fixed-width state would eventually overflow; that is not
this task.)

Where to aim the gain is set by the scoring rule. The task score is the geometric mean of the three
`ood_token_acc` values — the Transformer's was `(0.903061 · 0.727853 · 0.734313)^{1/3} ≈ 0.785` — and since
`∂(log G)/∂(log x_i) = 1/3`, a given *relative* lift moves the score equally from any environment, so the
largest *absolute* headroom sits on the smallest factors. Here those are `dyck-k8-m5` (0.727853) and
`dyck-length-ood` (0.734313), nearly tied; even a perfect `dyck-k2-m3` leaves `G ≤ (1 · 0.727853 ·
0.734313)^{1/3} ≈ 0.813`. So the recurrence has to deliver on the two hard environments to move the
headline — a near-solved easy environment cannot rescue the mean.

Falsifiable expectations against the Transformer's numbers. `ood_token_acc` should *rise* across the board
and rise most on `dyck-length-ood`, where the position table collapsed it to 0.734313 — removing the table
helps most where the extrapolation was largest; I expect near-perfect accuracy on the shallow environments.
The per-string numbers should move off the 0.001/0.0/0.0 floor if the LSTM is genuinely tracking the stack.
Where I am *not* confident is `dyck-k8-m5`: its deficit showed up in-distribution (`id_string_acc` 0.024),
so it is not a length artifact the recurrence will erase — a fixed-width state must pack the top-of-stack
symbol cleanly enough to pick the right closer among 8. If the LSTM sags there too — high `ood_token_acc`
elsewhere but visibly lower here, with weak per-string accuracy even in-distribution — that is the tell that
a *dense* memory, however well it length-generalizes, cannot maintain a *crisp* discrete stack, and the next
rung should give the model an *explicit* one. The fused `nn.LSTM` model is in the answer.
