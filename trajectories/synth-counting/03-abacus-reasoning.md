The transformer moved the metric the way I predicted, and the *way* it moved is the whole argument for
this step. `length-ood` retention rose from 0.530 to 0.771 and the extrapolation gap shrank from 0.470 to
0.230 — but I warned myself not to read the retention headline naively, and the warning pays off. The
`abc` OOD accuracy is *still* 0.524, identical to the LSTM's to three digits, dead chance. Plug the
identity both ways: LSTM `1 − (0.994 − 0.524) = 0.530`; transformer `1 − (0.754 − 0.524) = 0.770`. The
*entire* retention improvement is arithmetically explained by the drop in in-distribution `abc`, from
0.994 to 0.754, with the OOD term held fixed at chance. The transformer did not extrapolate one bit
better than the LSTM on `abc`; it got 0.240 *worse* in-range, and because retention rewards a small
in-dist−ood gap, getting worse in-range *looks* like getting better at length. That reframes the target
with total clarity: I do not need retention to rise — that is cheap and hollow — I need the OOD accuracy
itself to lift off 0.524 for the first time.

The rest confirms it. `exact` OOD is still exactly 0.0, over-determined as predicted (absolute positions
OOD *and* target magnitude beyond the head's ~63 range), so I keep it as the residual. In-distribution
`abc` at 0.754 and `exact` slipping from 0.998 to 0.937 are the reconstruction cost showing up: no
accumulator, so block lengths must be rebuilt from attention patterns and compared within two layers. So
the transformer bought graceful in-range softening and *zero* genuine OOD generalization, and it proved
the lever is the positional representation by being limited exactly there.

So the sharp question: what positional signal lets the encoder read the count at *any* length, including
four times training, so the OOD accuracy actually moves? I had it half-right before — sinusoids "have a
value" at OOD positions but the encoder never learned to read those absolute phase patterns. The deeper
problem is that *absolute index is the wrong thing to encode*. Stare at what the tasks actually need: on
`abc`, deciding `a^n b^n c^n` is deciding the run of `a`'s, the run of `b`'s, and the run of `c`'s all
have the same length — and "the run of `a`'s has length `n`" is a statement about *position within the
current run of identical symbols*, not absolute position. On `exact`, counting `a`'s is counting within
runs of `a`. The latent both tasks track is the within-run count, and absolute index scrambles it: the
third `a` sits at index 4 if it starts the string but index 40 if 36 symbols precede it. I want a
positional code that gives two symbols the same value exactly when they sit at the same offset from the
start of their own run.

Three ways to try to lift OOD, and I reject two on computed grounds. A relative scheme (ALiBi/RoPE) never
indexes an untrained row — tempting — but it encodes pairwise *distance*, and distance is not the latent:
two `a`'s and two `c`'s at run-offset 3 can be 4 tokens apart in one string and 40 in another, so a
distance code gives the very pairs I need to match as "same significance" *different* encodings. It
answers "how far apart," the task asks "how far into your own run." An engineered scalar feature — append
the running within-run tally as a channel — is essentially the LSTM's accumulator bolted onto attention:
it re-imports the magnitude-read fragility and, worse, gives no *equality* structure, because two runs
reaching offset `k` should be literally the same point in embedding space for a head to match, not two
equal scalars the head must learn to subtract. The one I take: index each symbol by its within-run offset
and pass that integer through a *learned embedding*, added to the token embedding. Same offset ⇒ same
vector, so a head computing `q_i·k_j` produces a large score between the `k`-th `a` and the `k`-th `c`
and a small one for mismatched offsets — the `a^n b^n c^n` decision becomes a single dot product in one
attention hop, no magnitude reconstruction. It is bounded (a table lookup), and it converts the OOD
problem from "unseen phase patterns" into "unseen table rows," which I can fix by *training the rows*.
This is the count-indexed embedding idea ported from per-*number* digit alignment to per-*run* symbol
counting — the adaptation this task demands, because here the unit whose internal offset matters is the
run of identical symbols, not a multi-digit numeral.

The construction: walk the sequence, reset a counter to 1 whenever the symbol changes or a run begins,
give each symbol the counter's value and increment. `a a a b b b c c c → 1 2 3 | 1 2 3 | 1 2 3`. But this
walks straight into the wall that killed the sinusoidal step: the counter still counts *up*, and a
length-256 OOD string has runs up to ~85, so the table needs rows up to ~85, while training on `3n ≤ 63`
(so `n ≤ 21`) only ever updates rows up to ~21 — rows 22…85 are untrained noise at test, the same
OOD-row failure reincarnated. The per-run reset fixed *significance* (same offset ⇒ same vector) and did
nothing for *coverage* (the offsets a test string reaches were never trained). I need both, and coverage
is the piece that turns this from "the transformer with a nicer index" into something that genuinely
extrapolates.

The coverage fix trains the large rows on short sequences: during training, draw one shift uniformly from
`{0,…,train_offset=100}` and add it to every positive offset, so `1 2 3` becomes `1+s, 2+s, 3+s`. The
within-run step stays `+1` — adjacency and alignment preserved — but over many batches the rows from 1 up
to `~100 + max-run` get exercised. The arithmetic that decides whether this works: in-range the largest
`abc` run is `n=21`, so shifts up to 100 exercise rows up to 121; the `exact` generator's runs of
identical symbols reach ~63, pushing to 163. At test the `abc` OOD runs reach only ~85, and
`85 < 121 ≤ 163` — the rows a test string needs are a *subset* of what training exercised. That closes the
exact gap the sinusoidal code left open, and closes it for the `abc`/`length-ood` decision where I need
OOD to lift. (The one place it can still fail is a pathological all-one-symbol `exact` string whose single
run reaches 256 > 163 — one more reason `exact` OOD may stay at its residual 0.0.) At evaluation I add
nothing, so runs use the smallest, most-trained rows `1,2,3,…`.

Two details of the shift are load-bearing. The shift is *shared across the batch* — one draw per forward
pass — not redrawn per run: with a per-run draw on `a a b b`, if the `a`-run drew 7 and the `b`-run 40,
the first `a` and first `b` land on different rows and "same significance ⇒ same vector" breaks in exactly
the example where the decision needs it; with one `s` per batch both runs start at `1+s` and column
alignment holds within every example while coverage is bought *across* batches as `s` varies. And the
shift applies *only to positive offsets* — PAD and CLS keep offset 0, row 0 reserved for "not a counted
symbol," so they are never perturbed into looking like a counted symbol at some shifted row.

The run boundary has to respect this vocabulary: the run is computed over content tokens, excluding
`pad_id` and `cls_id`. Padding is obvious; CLS is the subtle one — it sits at position 0 before any
content, and letting it into the run logic would either start a spurious run or merge with a following
`a` and shift every offset by one, breaking alignment between the `a`-run and the `b`/`c`-runs. On
`[CLS, a, a, b, b]` the valid mask is `[F,T,T,T,T]`: CLS gets offset 0, the first `a` starts a run at 1,
the second `a` continues to 2, `b` starts a new run at 1, the second `b` continues to 2 — `[0,1,2,1,2]`,
the two runs aligned so the `k`-th of each shares a row. A batch shift `s=3` on the positive offsets gives
`[0,4,5,4,5]` — CLS still 0, alignment preserved, both runs slid to trained rows. Offsets clamp at a
generous `max_count=4096` so the lookup never goes out of bounds on pathological inputs.

I keep the rest of the encoder identical to the sinusoidal step on purpose, so any OOD movement is
attributable to the positional change alone: token embedding scaled by `√hidden_dim`, *add* the
count-positional embedding, 2 pre-norm layers with 4 heads and FFN 4×, GELU, `dropout=0.0`,
`src_key_padding_mask`, CLS pooling, final `LayerNorm`. The positional table is `nn.Embedding(4097, 128)
≈ 524k`, so the whole encoder ~0.92M, still under a fifth of the ceiling; the untouched rows are inert and
cheap.

Two smaller choices inside "add a learned embedding" earn their reasons. I *add* rather than
*concatenate* so the model stays at width 128 (no projection before the fixed head) and the positional
signal lands in the same subspace the attention `Q/K` already read, so a head can be offset-sensitive
without a separate input slice — concatenation would spend parameters widening every matrix for no gain.
I inject only at the bottom because the residual stream carries the positional information upward;
re-adding at each layer would double-count. And I initialize the table small (`std=0.02`) against the
token embedding's `√128 ≈ 11.3` scale so the network sees mostly token identity at init and only a gentle
positional bias, and the optimizer *grows* precisely the rows that earn gradient — the low offsets
constantly, the higher shifted rows as the batch shift exercises them — rather than fighting a loud random
field first.

The digit-token machinery of the arithmetic origin does not carry over: counted tokens are "not PAD and
not CLS," computed by exclusion, not a digit-token set; the run boundary is symbol-change, not a
contiguous-digit run; there is no operand reversal; this is an encoder feeding a CLS-pooled head, not a
decoder LM. What is preserved is the *idea* — offset from the start of the current unit, randomized start
for coverage, shared shift per batch, additive at the bottom — with the unit and vocabulary reset to this
task's.

The prediction this whole construction rests on: `abc`/`length-ood` OOD accuracy should rise above 0.524
for the first time, because the run-offset is the actual latent the decision needs and the
shift-randomization keeps the needed rows (up to ~85) inside the trained set (up to ~121) — the only
honest way to earn retention rather than by softening the in-range fit. If OOD stays at chance, the
run-indexing did not help and I was wrong about the latent. In-distribution `abc` may dip near the
transformer's 0.754 because the randomized shift makes the in-range fit noisier, but that trade is right:
the aggregate is a geometric mean, `exact` OOD has been 0.0 for both prior steps, so the live axis is the
`abc`/retention pair, and buying OOD there at a small in-range cost buys on the axis that moves the
ranking. `exact` OOD may stay 0.0 — the target-magnitude wall plus the coverage gap on pathological long
runs — naming what a method past this ladder still has to attack.