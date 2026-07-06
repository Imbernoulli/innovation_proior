The transformer numbers move the metric in exactly the direction I predicted, and the *way* they move
is the whole argument for this rung. `length-ood` retention rose from the LSTM's 0.530 to 0.771 — a real
step up on the headline metric the task is built to measure — and the extrapolation gap shrank from 0.470
to 0.230. But I warned myself at rung two not to read the retention headline naively, and now the
warning pays off. Retention is `1 − max(0, in_dist − ood)`, and the OOD accuracy on `abc` is *still*
0.524, identical to the LSTM's to three digits — dead chance, about `1.5` sampling standard deviations off
0.5 given the harness's frozen 1,024-example eval split, no retained signal. So plug the identity in both directions: LSTM retention `= 1 − (0.994 −
0.524) = 0.530`; transformer retention `= 1 − (0.754 − 0.524) = 0.770`. The *entire* retention
improvement is arithmetically explained by the drop in in-distribution `abc`, from 0.994 to 0.754, with
the OOD term held fixed at chance. The transformer did not extrapolate one bit better than the LSTM on
`abc`; it simply got 0.240 *worse* in-distribution, and because retention rewards a small in-dist−ood gap,
getting worse in-range *looks* like getting better at length. That is the trap I flagged made concrete,
and it reframes the target for this rung with total clarity: I do not need retention to rise — that is
cheap and hollow — I need the OOD accuracy itself to lift off 0.524 for the first time on the ladder.

The rest of the transformer's numbers confirm the diagnosis from the other side. The `exact` OOD accuracy
is still exactly 0.0, so the `exact` score is still 0.0, and I predicted that would be over-determined —
absolute positions are OOD *and* the target count magnitude (up to ~256) is out of the range the head was
fit on (~63) — so I do not expect this rung to move it either and I will keep watching it as the residual.
And the in-distribution `abc` falling to 0.754 is not noise but mechanism: the transformer has no
accumulator, so it must *reconstruct* block lengths from attention patterns and compare them within two
layers, strictly harder than the LSTM's direct integer tally, exactly as I reasoned. The `exact`
in-distribution count also slipped, from 0.998 to 0.937 — the same reconstruction cost showing up on the
counting task. So the transformer bought graceful in-range softening and *zero* genuine OOD
generalization, and it proved that the lever is the positional representation by being limited exactly
there: absolute index is OOD at test length, so the model keys off positional structure it never saw and
falls back to chance.

So the question for rung three is sharp: what positional signal would let the encoder read the count at
*any* length, including lengths four times longer than training, so that the OOD accuracy actually moves?
I had it half-right at rung two when I said sinusoids "have a value" at OOD positions but the encoder
never learned to read those absolute phase patterns. The deeper problem is that *absolute index is the
wrong thing to encode in the first place*. Stare at what the counting tasks actually need. On `abc`,
deciding `a^n b^n c^n` is deciding that the run of `a`'s, the run of `b`'s, and the run of `c`'s all have
the same length — and "the run of `a`'s has length `n`" is a statement about *position within the current
block of identical symbols*, not about absolute position in the sequence. On `exact`, counting `a`'s is
counting within runs of `a`. The latent variable both tasks track is the *within-run count*, and absolute
index scrambles it: the third `a` of a string sits at absolute index 4 (after CLS) if it is the first
block, but at index 40 if 36 symbols preceded it. The transformer was trying to reconstruct "same block
length" from absolute positions that move around; no wonder it could not transfer. What I want is a
positional code that gives two symbols the *same* value exactly when they sit at the same offset from the
start of their own run.

Before I build that, I owe the design space a walk, because "encode within-run offset" is one of three
ways I could try to lift OOD off chance and I should reject the other two on computed grounds, not taste.
Option one is a relative-position scheme (ALiBi- or RoPE-style): bias or rotate attention by the *distance*
between two tokens. It never indexes an untrained row, which is why it was tempting at rung two, but it
encodes pairwise separation, and pairwise separation is *not* the latent: two `a`'s at run-offsets 3 and
the two `c`'s at run-offsets 3 can be separated by 4 tokens in one string and 40 in another, so a
distance code gives the very pairs I need to identify as "same significance" different encodings. It
answers "how far apart" when the task asks "how far into your own run," so it cannot make the `k`-th `a`
and the `k`-th `c` share a representation. Option two is to hand the model the count as an explicit
engineered scalar feature — append a channel that is the running within-run tally. But that is essentially
the LSTM's accumulator bolted onto attention, it re-imports the magnitude-read fragility (`tanh`-style
saturation, or an unbounded scalar the head cannot calibrate at OOD magnitude), and it does not give the
*equality* structure that a shared *vector* per offset gives — two runs reaching offset `k` should be
literally the same point in embedding space so a head can match them, not two equal scalars the head must
learn to subtract. Option three, the one I take, is to index each symbol by its within-run offset and
pass that integer through a *learned embedding*, adding it to the token embedding. It gives the equality
structure for free (same offset ⇒ same vector), it is bounded (a table lookup, no saturating magnitude),
and — crucially — it converts the OOD problem from "unseen phase patterns" into "unseen table rows," which
is a problem I can *fix by training the rows*, as below. That is the decisive advantage and the reason
option three wins.

Let me build it and watch it fall out. Walk along the sequence; every time the symbol changes (or a
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
up to ~85, so the embedding table needs rows up to ~85. If I train only on `3n ≤ 63` (so `n ≤ 21`), the
embedding rows for offsets 22, 23, …, 85 are *never updated* and are garbage noise at test — exactly the
out-of-distribution-row failure that made the sinusoidal absolute code collapse, now reincarnated as
untrained lookup rows. The per-run reset fixed the *significance* problem beautifully (same offset ⇒ same
vector) and did nothing for the *coverage* problem (the offsets a test string reaches were never trained).
I need both, and the coverage fix is the one piece that turns this from "the transformer with a nicer
index" into a rung that genuinely extrapolates.

The coverage trick is to train the large rows even though the training runs are short, by randomizing
the *starting offset* of the count. During training, before laying down a run's offsets, draw a single
shift uniformly from `{0, …, train_offset}` and add it to every positive offset, so a run `1 2 3`
becomes `1+s, 2+s, 3+s`. The within-run step stays exactly `+1` — so adjacency and "same significance"
are preserved — but because the shift ranges up to `train_offset = 100`, over many batches the embedding
rows from 1 all the way to `~100 + max-run-length` get exercised, including all the rows the OOD lengths
will need. Now the arithmetic that decides whether this actually works. In-range the largest `abc` run is
`n = 21`, so with a shift up to 100 the training exercises offset rows up to `21 + 100 = 121`; the `exact`
generator, whose runs of identical symbols can reach ~63 in-range, pushes the exercised rows up to `63 +
100 = 163`. At test the `abc` OOD runs reach only ~85. And `85 < 121 ≤ 163`: the rows a test string
needs are a *subset* of the rows training already exercised. That is the verification I needed on paper —
the coverage fix closes the exact gap that sank the sinusoidal code, and it closes it for the
`abc`/`length-ood` decision specifically, which is where I need the OOD accuracy to lift. (The one place
coverage can still fail is a pathological all-one-symbol `exact` OOD string whose single run reaches 256 >
163 — which is one more reason the `exact` OOD score may stay at its residual 0.0 while `abc` moves.) At
evaluation I add nothing, so the runs use rows `1, 2, 3, …`, the smallest and most-trained start.

Two details of the shift are load-bearing, and both are alignment arguments I should make precise. First,
the shift is *shared across the batch* — one draw per forward pass, the same shift for every run in it —
not redrawn per run. Trace why this matters on `a a b b` with a per-example draw: if the `a`-run drew
shift 7 and the `b`-run drew shift 40, the `a`-run offsets become `8, 9` and the `b`-run offsets `41, 42`,
so the first `a` and the first `b` land on *different* rows and "same significance ⇒ same vector" is
broken in exactly the example where the decision needs it. With one shift `s` per batch, the same string
becomes `a`-run `1+s, 2+s` and `b`-run `1+s, 2+s` — the first `a` and first `b` both on row `1+s`, the
second `a` and second `b` both on row `2+s`. Column alignment holds intact within every example while
coverage is bought *across* batches as `s` varies. Second, the shift is applied *only to positive
offsets* — the non-run positions (PAD and, here, the CLS token) keep offset 0 and use row 0, reserved for
"not a counted symbol," so they are never perturbed and never accidentally look like a counted symbol at
some shifted offset.

Now the run-boundary definition has to respect this task's vocabulary, and this is where the adaptation
to the harness is concrete rather than generic. The "run of identical symbols" is computed over the
content tokens, but two tokens must be *excluded* from the run logic entirely: `pad_id` (padding) and
`cls_id` (the CLS at position 0). Padding is obvious — padded positions carry no symbol. CLS is the
subtle one: the CLS sits at position 0 *before* any content, and if I let it participate in run-counting
it would either start a spurious run or, worse, merge with a following `a` and shift every offset by
one, breaking the alignment between the `a`-run and the `b`/`c`-runs. Let me trace the intended machine on
`[CLS, a, a, b, b]` to be sure it does the right thing. Mark valid = "neither PAD nor CLS": the mask is
`[F, T, T, T, T]`. Sweep with a running counter `cur`, starting at 0, tracking the previous token and
whether the previous position was valid. Position 0 (CLS) is invalid, so it gets offset 0. Position 1
(`a`) is valid but the previous was invalid, so it *starts* a run at offset 1. Position 2 (`a`) is valid,
previous valid, same token, so it *continues*: offset 2. Position 3 (`b`) is valid but the token differs
from the previous, so it starts a new run: offset 1. Position 4 (`b`) continues: offset 2. The offset
row is `[0, 1, 2, 1, 2]` — CLS parked at 0, and the two runs aligned so the `k`-th of each shares a row,
exactly the property the whole construction rests on. Apply a batch shift `s = 3` to the positive
offsets: `[0, 4, 5, 4, 5]` — CLS still 0, alignment preserved, both runs slid to the higher, now-trained
rows. The trace confirms the mechanism end to end. I clamp the offsets at a generous `max_count` (4096)
so the embedding lookup never indexes out of bounds even on pathological inputs. This
per-run-of-identical-symbols rule is the heart of the adaptation: the single-round derivation indexes
offset from the start of a *number* (digits of one numeral, written least-significant-first); here the
analogous unit is the *maximal run of one symbol*, because the `a^n b^n c^n` membership and the `a`-count
are statements about run length, not numeral length.

On the budget: the count-positional table is `nn.Embedding(max_count + 1, hidden_dim) = 4097 × 128 ≈
524k` parameters, and stacked on the ~0.40M transformer from rung two the whole encoder is about `0.92M`
— still under a fifth of the 5M ceiling, so I can afford the generous `max_count` even though only rows up
to ~163 are ever touched; the untouched rows are inert and cheap, and the clamp guarantees the lookup is
always in bounds. The rest of the encoder I keep identical to rung two on purpose, so the comparison
isolates the positional change. Token embedding `nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)`,
scaled by `sqrt(hidden_dim)` to match the additive positional amplitude; *add* the count-positional
embedding (initialized small, `std=0.02`, so it starts as a gentle bias on the token embeddings and the
optimizer grows the rows it uses rather than starting from a loud random field); 2 pre-norm
`nn.TransformerEncoderLayer`s with 4 heads, FFN width `4·hidden_dim`, GELU, `dropout=0.0`;
`src_key_padding_mask = tokens.eq(pad_id)`; CLS pooling `h[:, 0]`; final `LayerNorm`. Same depth, same
heads, same pooling, same mask as rung two — the *only* thing that changed is that the sinusoidal absolute
PE became a learned, run-indexed, training-shift-randomized embedding. That is the controlled experiment,
and it is why any movement in the OOD accuracy is attributable to the positional representation and
nothing else.

Two smaller design choices inside "add a learned embedding" deserve their reasons, because I could have
made them differently. Why *add* the count-positional vector to the token embedding rather than
*concatenate* it into a widened channel? Additive injection keeps the model width at `hidden_dim = 128`
so the returned summary still matches the fixed head with no projection, and — more to the point — it puts
the positional signal into the *same* subspace the attention `Q/K` projections already read, so a head
can form a query that is sensitive to run-offset without needing a separate slice of the input; a
concatenation would demand the projections learn to route a disjoint block and would spend parameters
widening every matrix for no representational gain the additive form lacks. Why inject only at the bottom
of the stack rather than re-adding at each layer? Because the residual stream carries the positional
information upward — the first layer writes offset-dependent content into the residual and later layers
read it — so a single bottom injection is the standard, sufficient placement, and re-adding would just
double-count the same signal. And why initialize the table small, `std = 0.02`, when the token embedding
is scaled *up* by `sqrt(128) ≈ 11.3`? The scale asymmetry is deliberate: at initialization I want the
network to see mostly token identity and only a gentle positional bias, `0.02`-scale against an
`11.3`-scale token signal, so the run-offset channel starts as a small perturbation and the optimizer
*grows* precisely the rows that earn gradient — the low offsets that appear constantly, then the higher
shifted rows as the batch shift exercises them. A loud random positional field at init would instead
inject noise the first attention layer has to fight before it can read token identity at all. These are
not free parameters I am guessing; each follows from wanting the run-offset to be a clean, growable bias
in the same subspace the attention already reads.

It is worth naming *why* the equality structure helps attention specifically, since that is the whole
reason I preferred a shared vector over an engineered scalar. A head computes compatibility as a dot
product `q_i · k_j`; if the positional contribution to `q` and `k` is the *same vector* `e_offset` for two
tokens at the same run-offset, then a head whose weights emphasize the positional subspace produces a
*large* score between the `k`-th `a` and the `k`-th `c` and a small one between mismatched offsets — the
match is a single dot product, computed in one attention hop, no reconstruction of magnitudes required.
That is the mechanism by which "same offset ⇒ same vector" turns the `a^n b^n c^n` decision into
something two layers can actually route, and it is exactly what a distance code (different vectors for the
same-significance pair) or a scalar tally (equal numbers the head must learn to subtract, not match) would
not give. So the design does not merely *hand over* the latent; it hands it over in the geometric form —
coincident points in embedding space — that the attention primitive is built to exploit.

A word on what this rung does *not* import from the single-round arithmetic derivation, per the
same-named-baseline note, because the differences are exactly the adaptations the harness forces.
There is no digit-token set and no `isin(input_ids, digit_tokens)` — the "counted" tokens here are
"anything that is not PAD or CLS," computed by exclusion, and the run boundary is symbol-change, not the
contiguous-digit-run of a numeral. There is no reversal of operands (least-significant-first) — the
sequence is read left to right and offset 1 is simply the first symbol of a run. The shift range is
`train_offset = 100`, sized to the OOD run lengths this task actually reaches (~85), not the
hundred-digit numbers of the arithmetic setting. And the whole thing is an *encoder* feeding a CLS-pooled
classifier/regressor through a fixed head, not a decoder-only language model. The *idea* — offset from
the start of the current unit, randomized start for coverage, shared shift per batch, additive at the
bottom — is preserved; the *unit* and the *vocabulary* are this task's.

Falsifiable expectations against the transformer's numbers, and I state them in the order of what they
decide. The prediction that decides whether this rung is the strongest is OOD generalization, and I
expect it to *finally move off chance*: because the run-offset is the actual latent the `abc` decision
needs and the shift-randomization keeps the needed rows (up to ~85) inside the trained set (up to ~121),
the `abc`/`length-ood` OOD accuracy should rise above 0.524 for the first time on the ladder — that is the
single claim this whole construction rests on, and by the earlier arithmetic it is the *only* way to earn
retention honestly rather than by softening the in-range fit. If OOD stays at chance, the run-indexing
did not help and I was wrong about the latent. Following from that, `length-ood` retention should beat the
transformer's 0.771 — but now for the right reason, an OOD term that rose, not an in-dist term that fell.
I am honest about two places it may *not* dominate. In-distribution `abc` may sit *below* the LSTM's 0.994
and near the transformer's 0.754, because the randomized training shift makes the in-distribution task
slightly noisier to fit — the model must read the same offsets at many different starting rows — trading
a little in-range sharpness for OOD coverage. That trade is the right bet here: the aggregate is a
geometric mean across the three environment scores and the `exact` score has sat at 0.0 for both prior
rungs, so the aggregate is currently floored regardless, and the live axis separating methods is exactly
the `abc`/retention pair that OOD generalization lifts — so buying OOD at a small in-dist cost is buying
on the axis that actually moves the ranking. And `exact` OOD: this is the hardest split, because reading
an exact integer count at length 256 through a pooled CLS runs into the target-magnitude wall from rung
two *and*, for the pathological long runs, the one coverage gap I noted, so I will not be shocked if the
`exact` score stays at 0.0 like both prior rungs. The rung is an improvement if `abc`-OOD and retention
rise together; the place it might still fall short — exact OOD count regression — would then name what a
future method past this ladder would have to attack. The full scaffold module is in the answer.
