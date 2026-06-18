The transformer numbers move the metric in exactly the direction I predicted, and the *way* they move
is the whole argument for this rung. `length-ood` retention rose from the LSTM's 0.530 to 0.771 — a real
step up on the metric the task is built to measure — and the extrapolation gap shrank from 0.470 to
0.230, confirming that attention's block comparison degrades more gracefully than the LSTM's compounding
accumulator. But every prediction I made about the *limit* also held. The `exact` OOD accuracy is still
exactly 0.0, so the `exact` score is still 0.0: nothing in the sinusoidal absolute code let the encoder
read an exact count at length 256. The `abc`/`length-ood` OOD accuracy is still 0.524 — dead chance —
so the retention gain is "degrades less," not "generalizes." And in-distribution `abc` actually *fell*,
from the LSTM's 0.994 to 0.754, which is the most diagnostic number on the board: the transformer is
*worse* at the in-range membership task than the recurrent counter was. So the transformer bought
graceful degradation at the cost of in-distribution counting sharpness, and it bought *zero* genuine OOD
generalization. The diagnosis I sketched is confirmed: the positional code is a function of *absolute
index*, the encoder only ever trained on indices up to ~65, and at test length 256 the phase patterns
are out-of-distribution, so the model keys off positional structure it never saw and falls back to
chance. The lever is the positional representation, and rung two proved it by being limited exactly
there.

So the question for rung three is sharp: what positional signal would let the encoder read the count at
*any* length, including lengths four times longer than training? I had it half-right at rung two when I
said sinusoids "have a value" at OOD positions but the encoder never learned to read those absolute
phase patterns. The deeper problem is that *absolute index is the wrong thing to encode in the first
place*. Stare at what the counting tasks actually need. On `abc`, deciding `a^n b^n c^n` is deciding
that the run of `a`'s, the run of `b`'s, and the run of `c`'s all have the same length — and "the run of
`a`'s has length `n`" is a statement about *position within the current block of identical symbols*, not
about absolute position in the sequence. On `exact`, counting `a`'s is counting within runs of `a`. The
latent variable both tasks track is the *within-run count*, and absolute index scrambles it: the third
`a` of a string sits at absolute index 4 (after CLS) if it is the first block, but at index 40 if 36
symbols preceded it. The transformer was trying to reconstruct "same block length" from absolute
positions that move around; no wonder it could not transfer. What I want is a positional code that gives
two symbols the *same* value exactly when they sit at the same offset from the start of their own run.

Let me build that and watch it fall out. Walk along the sequence; every time the symbol changes (or a
run begins), reset a counter to 1; give each symbol of the current run the counter's value and
increment. So `a a a b b b c c c` produces run-offsets `1 2 3 | 1 2 3 | 1 2 3`. Feed those integers
through a learned embedding table and *add* the result to the token embedding, absolute-style, at the
bottom of the stack. Now look at what this does. The `k`-th `a`, the `k`-th `b`, and the `k`-th `c` all
receive the *identical* positional vector — offset `k` — so a head can directly check "do the runs reach
the same maximum offset," which is precisely the `a^n b^n c^n` decision, and on `exact` the offset
*is* the running count of the current run. The latent variable the tasks track is now handed to the
model through the position channel, with no extra tokens. This is the count-indexed embedding idea from
the prior-art lineage, ported from per-*number* digit alignment to per-*run* symbol counting — the
adaptation the task's research question demands, because here the "number" whose internal offset matters
is the run of identical symbols, not a multi-digit numeral.

But I have walked straight into the same wall absolute embeddings hit, and it is the wall that killed
rung two — so I have to fix it or this rung is just the transformer with a fancier index. My counter
resets per run, good, but it still *counts up* within a run, and a length-256 OOD string has runs of `n`
up to ~85, so the embedding table needs rows up to ~85. If I train only on `3n ≤ 64` (so `n ≤ 21`), the
embedding rows for offsets 22, 23, …, 85 are *never updated* and are garbage noise at test — exactly the
out-of-distribution-row failure that made the sinusoidal absolute code collapse. The per-run reset fixed
the *significance* problem beautifully and did nothing for the *coverage* problem. I need both, and the
coverage fix is the one piece that makes this rung genuinely extrapolate rather than just re-index.

The coverage trick is to train the large rows even though the training runs are short, by randomizing
the *starting offset* of the count. During training, before laying down a run's offsets, draw a single
shift uniformly from `{0, …, train_offset}` and add it to every positive offset, so a run `1 2 3`
becomes `1+s, 2+s, 3+s`. The within-run step stays exactly `+1` — so adjacency and "same significance"
are preserved — but because the shift ranges up to `train_offset = 100`, over many batches the embedding
rows from 1 all the way to `~100 + max-run-length` get exercised, including all the rows the OOD lengths
will need. At evaluation I add nothing, so the runs use rows `1, 2, 3, …`, the smallest and
most-trained start. Two details are load-bearing. First, the shift is *shared across the batch* — one
draw per forward pass, the same shift for every run in it — not redrawn per run; if I drew an
independent shift per run, the `a`-run might start at offset 7 and the `c`-run at offset 40 *within the
same example*, and "same significance ⇒ same vector" would break in exactly the example where I need it.
With one shift per batch, every run in the example starts at the same place, so within any example the
`k`-th symbol of every run still lands on the same row, and the column-alignment property holds intact
while coverage is bought across batches. Second, the shift is applied *only to positive offsets* — the
non-run positions (PAD and, here, the CLS token) keep offset 0 and use row 0, reserved for "not a
counted symbol," so they are never perturbed.

Now the run-boundary definition has to respect this task's vocabulary, and this is where the adaptation
to the harness is concrete rather than generic. The "run of identical symbols" is computed over the
content tokens, but two tokens must be *excluded* from the run logic entirely: `pad_id` (padding) and
`cls_id` (the CLS at position 0). Padding is obvious — padded positions carry no symbol. CLS is the
subtle one: the CLS sits at position 0 *before* any content, and if I let it participate in run-counting
it would either start a spurious run or, worse, merge with a following `a` and shift every offset by
one, breaking the alignment between the `a`-run and the `b`/`c`-runs. So I build the offsets by sweeping
the sequence, marking a position *valid* only if it is neither PAD nor CLS, and a position *continues a
run* only if it is valid, the previous position was valid, and the token equals the previous token;
otherwise a valid position *starts* a new run at offset 1, and an invalid position gets offset 0. I clamp
the offsets at a generous `max_count` (4096) so the embedding lookup never indexes out of bounds even on
pathological inputs. This per-run-of-identical-symbols rule is the heart of the adaptation: the
single-round derivation indexes offset from the start of a *number* (digits of one numeral, written
least-significant-first); here the analogous unit is the *maximal run of one symbol*, because the
`a^n b^n c^n` membership and the `a`-count are statements about run length, not numeral length.

The rest of the encoder I keep identical to rung two on purpose, so the comparison isolates the
positional change. Token embedding `nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)`, scaled by
`sqrt(hidden_dim)` to match the additive positional amplitude; *add* the count-positional embedding
(initialized small, `std=0.02`, so it starts as a gentle bias and the optimizer grows the rows it
uses); 2 pre-norm `nn.TransformerEncoderLayer`s with 4 heads, FFN width `4*hidden_dim`, GELU,
`dropout=0.0`; `src_key_padding_mask = tokens.eq(pad_id)`; CLS pooling `h[:, 0]`; final `LayerNorm`.
Same depth, same heads, same pooling, same mask as rung two — the *only* thing that changed is that the
sinusoidal absolute PE became a learned, run-indexed, training-shift-randomized embedding. That is the
controlled experiment.

A word on what this rung does *not* import from the single-round arithmetic derivation, per the
same-named-baseline warning, because the differences are exactly the adaptations the harness forces.
There is no digit-token set and no `isin(input_ids, digit_tokens)` — the "counted" tokens here are
"anything that is not PAD or CLS," computed by exclusion, and the run boundary is symbol-change, not the
contiguous-digit-run of a numeral. There is no reversal of operands (least-significant-first) — the
sequence is read left to right and offset 1 is simply the first symbol of a run. The shift range is
`train_offset = 100`, sized to the OOD run lengths this task actually reaches (~85), not the
hundred-digit numbers of the arithmetic setting. And the whole thing is an *encoder* feeding a CLS-pooled
classifier/regressor through a fixed head, not a decoder-only language model. The *idea* — offset from
the start of the current unit, randomized start for coverage, shared shift per batch, additive at the
bottom — is preserved; the *unit* and the *vocabulary* are this task's.

Falsifiable expectations against the transformer's numbers. The prediction that decides whether this
rung is the strongest is OOD generalization, and I expect it to *finally move off chance*: because the
run-offset is the actual latent the `abc` decision needs and the training-shift-randomization keeps the
needed rows in-distribution at test length, the `abc`/`length-ood` OOD accuracy should rise above 0.524
for the first time on the ladder — that is the single claim this whole construction rests on, and if it
stays at chance the run-indexing did not help. Following from that, `length-ood` retention should beat
the transformer's 0.771; if OOD accuracy climbs while in-distribution holds, retention climbs with it,
and this becomes the strongest rung on the metric the task scores. I am honest about two places it may
*not* dominate. In-distribution `abc` may sit *below* the LSTM's 0.994 and even near the transformer's
0.754, because the randomized training shift makes the in-distribution task slightly noisier to fit
(the model must read offsets it sees at many different shifts), trading a little in-range sharpness for
OOD coverage — a trade the task's geometric-mean-of-scores actively rewards, since retention and OOD
matter. And `exact` OOD: this is the hardest split, because reading an exact integer count at length 256
through a pooled CLS is a tall order even with the right positional signal, so I will not be shocked if
the `exact` score stays at 0.0 like both prior rungs — the run-offset helps the *membership/retention*
tasks most directly, and the regression-of-exact-count OOD may remain the residual failure. The rung is
an improvement if retention and `abc`-OOD rise; the place it might still fall short — exact OOD count
regression — would then name what a future method past this ladder would have to attack. The full
scaffold module is in the answer.
