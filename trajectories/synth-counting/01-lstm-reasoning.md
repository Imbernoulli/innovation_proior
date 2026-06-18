I start the ladder at the architecture that has the cleanest claim to *actually counting*, because the
research question is about counters and I want the first rung to be the model whose internal mechanism
is a counter by construction rather than by accident. The scaffold default is a 2-layer GRU reading its
last hidden state; before I reach for attention or anything fancier I want to know how far a recurrent
counter gets on its own, because if a recurrent net can hold an integer count across the sequence then
that is the honest floor every later method has to beat, and if it cannot then I have learned precisely
*where* recurrence breaks and that diagnosis seeds the next rung. So rung one is a recurrent encoder,
and I will reason about why an LSTM specifically, and what I expect it to do on each of the three
environments.

The reason a plain recurrent net is the wrong place to *stop* — even though it is the right place to
*start* — is the gradient. Trace an error signal backward through a recurrence `h_t = f(h_{t-1}, x_t)`:
over `q` steps it is multiplied by a product of `q` Jacobian factors, each of the form (derivative of
the squashing nonlinearity) times (a recurrent weight). With a sigmoid or tanh unit each such factor is
below one in magnitude for any reasonable weight, so the product decays geometrically in the lag — the
classic vanishing gradient. On the `abc` environment the lag is exactly the thing that matters: to
decide whether a string is `a^n b^n c^n` the model must compare the size of the `a`-block, seen at the
*front* of the sequence, against the `c`-block, seen at the *back*, and for the longer training strings
those blocks are dozens of steps apart. A vanilla RNN's gradient connecting them has shrunk to nothing,
so it never learns to thread a count from the `a`-block through to the `c`-block; it learns local
texture instead. That is the disease, and it is exactly what the gated memory cell was built to cure.

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

Now the design choices, each tied to the contract the harness hands me. The encoder receives
`tokens: LongTensor[B, T]` with a CLS at position 0 and must return `[B, hidden_dim]`. The first layer
is an embedding `nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)` — five symbols
(`PAD, a, b, c, CLS`) into a 128-dim space, with the padding row pinned to zero so padded positions
contribute nothing to the recurrence. Then `nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
num_layers=2, batch_first=True)`: two layers because a single layer must both maintain the count *and*
compute the membership decision from it, while stacking lets the lower layer carry the running tallies
and the upper layer combine them into the `a==b==c` comparison — a small amount of depth buys a cleaner
division of labor without exploding the parameter budget. Width 128 matches `hidden_dim` so the
returned summary lands in exactly the shape the fixed head expects, with no extra projection.

The pooling rule is the one place I have to respect a scaffold subtlety rather than copy the attention
playbook. The CLS token sits at position 0, *before* any content token. An LSTM is causal and
left-to-right: its hidden state at position 0 has seen only the CLS and nothing else, so reading the CLS
position would summarise an empty sequence. The information accumulates as the recurrence sweeps left to
right, so the state that has seen the *whole* string is the one at the *last valid position*. The
harness gives me `lengths` (true length including CLS), so I gather the output at index `lengths - 1`:
`last_idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, out.size(-1))`, then
`out.gather(1, last_idx).squeeze(1)`. The `clamp(min=0)` guards the degenerate empty case; the gather
picks per-row the genuinely final hidden, never a padded position. This is the recurrent reading of the
contract — *last valid position*, not CLS — and getting it wrong would silently feed the head a hidden
that never saw the `c`-block. Finally a `nn.LayerNorm(hidden_dim)` on the pooled vector, so the scale
handed to the fixed linear head is stable across the wildly different sequence lengths the three
environments produce (a length-8 `exact` string and a length-256 OOD string would otherwise hand the
head summaries on very different scales).

I should be explicit about what this LSTM is *not*, because the README's warning about same-named
baselines applies. This is not the full qlib-style sequence-to-one regressor with its own Adam, masked
MSE, gradient-value-clipping, and validation early-stopping — all of that optimization machinery lives
in the *fixed* harness here (AdamW at 3e-4, smooth-L1 / BCE, grad-norm-clip 1.0, 6,000 steps, single
seed, no early stop). The harness also fixes the head, the loss, and the data. So the only thing this
rung contributes is the *encoder*: embedding, 2-layer LSTM, last-position read, LayerNorm. Everything
the cell-derivation says about counters still holds — the cell state can implement the tally — but I am
filling a much smaller slot than the paper's training loop, and I should reason only about the encoder's
capacity, not about an optimizer I do not control.

What do I expect, environment by environment, so that the numbers can confirm or refute it? On `abc`
in-distribution I expect the LSTM to do *well* — the blocks in the training range are within reach of
the cell's gradient, and a learned increment/compare counter fits the data; this is the regime where
finite-precision LSTMs are known to nail counter languages, so a high in-distribution membership
accuracy is the prediction, and `abc` scores on exactly that. On `exact` in-distribution I likewise
expect a strong rounded-count accuracy: counting `a`'s in a binary string is the *easiest* counter, a
single accumulator with no comparison, so the in-distribution number should be near the top. The place
I am genuinely unsure — and it is the place the task is built to expose — is *out of distribution*. The
LSTM's counter is exact only to the extent its learned per-step increment is exact and its read-out
`tanh(c_t)` has not saturated. At training lengths up to 64 the cell state stays in a range the model
has shaped; at OOD lengths of 128–256 the accumulated state can drift past where `tanh` is responsive,
or a per-step increment that was 0.99 instead of 1.00 compounds into a large error over twice as many
steps. So my falsifiable expectation is a *split*: high in-distribution accuracy on both `abc` and
`exact`, but a large extrapolation gap — the `exact` OOD accuracy (the `exact` score) could collapse,
and the `length-ood` retention could be poor, because the in-distribution counter the LSTM learns does
not transfer cleanly to twice the length. If that split shows up, it tells me the recurrent counter is
*memorized in-range* rather than *implemented as an exact integer recurrence*, and the next rung has to
change the representation of length itself rather than just the way the count is carried.

So rung one is the honest recurrent floor: an LSTM whose cell state can in principle be an integer
counter, read at the last valid position, normalized into the fixed head. I expect it to win
in-distribution and to reveal, in its OOD numbers, exactly how a continuous-state counter degrades when
the sequence outgrows the lengths that shaped it — and that degradation is the problem the rest of the
ladder exists to attack.

One more reason to start here rather than skip to attention: it keeps the comparison honest. The LSTM
and the later encoders share the same fixed harness — same AdamW, same 6,000 steps, same fixed head,
same data, same single seed — and the same width and depth budget. So whatever the OOD numbers show is
attributable to the *encoder*, not to a training-loop difference I quietly introduced. If the recurrent
counter wins in-distribution while the attention models win on retention, that is an architecture
result, cleanly isolated, telling me which property — exact in-range counting versus graceful length
behavior — each family buys. Starting at a model that genuinely *counts*, rather than the scaffold's
generic GRU default, is what makes that read-out interpretable. The full scaffold module is in the
answer.
