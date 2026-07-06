The binary run told me exactly what one bit costs, and it told me in numbers. Validation loss landed at
2.7352, WikiText-2 perplexity at 81.3, LAMBADA at 110.2, with ARC-Easy at 47.0 and HellaSwag at 28.7 —
barely above the 25 chance floor on the four-way completion task. Let me read the rest of the board the
same way before I decide what to fix, because the shape of the failure is more informative than any one
number. PIQA sits at 58.9, a few points over its two-way 50 floor; WinoGrande at 51.5, essentially at its
50 floor; and the run finished in 22,601 seconds, indistinguishable from a float run of the same length.
So the model *trained* — the absmean scale and the STE made one-bit GPT-2 a real language model rather
than noise, and the pre-projection LayerNorms kept it stable — but it is a weak one, and it is weak in a
specific pattern. The perplexity-of-text metrics are bad-but-alive (an 80s WikiText-2 for a 355M decoder
at the Chinchilla-optimal budget is roughly where a much smaller float model would sit), while the tasks
that demand finely-discriminated structure — HellaSwag most of all — are pinned near chance. That the
wall-clock is identical to the others is the tell that this is not an optimization failure: the loss curve
was stable, nothing diverged, the run just converged to a poor optimum. It is a *capacity* failure, and I
can name the specific deficiency one bit imposes rather than wave at "not enough bits."

Let me quantify the gap so I know how much I am trying to recover, not just its sign. The val_loss 2.7352
is a cross-entropy in nats on FineWeb, so `exp(2.7352) ≈ 15.4` is the per-token perplexity the binary
model achieves on its own validation distribution; a float GPT-2 Medium at this budget would sit several
points lower. The WikiText-2 81.3 against a float model's low-20s is roughly `3.5–4×` worse perplexity —
that is the size of the hole one bit dug. And the two downstream tasks disagree in a way that is itself a
clue: ARC-Easy 47.0 is comfortably above its multiple-choice floor, so the model has learned *some*
usable factual/associative structure, while HellaSwag 28.7 is only 3.7 points over its 25 chance floor,
so the model has almost no ability to discriminate among four plausible continuations. Read together,
binary GPT-2 can carry coarse signal but cannot make fine distinctions — which is exactly the fingerprint
of weights that have direction but no graded magnitude. I will hold that ARC-vs-HellaSwag split in mind,
because whatever I add next has to move HellaSwag to prove it bought real discrimination and not just a
lower loss.

Two more rows finish that decomposition and set a discipline I need for reading the next board. PIQA at 58.9 is 8.9 points over its two-way 50 floor — real but coarse physical-commonsense signal; WinoGrande at 51.5 is 1.5 over 50, essentially chance. Rank the four above-floor margins — ARC-Easy 22, PIQA 8.9, HellaSwag 3.7, WinoGrande 1.5 — and the shape is a model with coarse world-association and almost no fine discrimination or coreference. The discipline: this is a single-seed board, seed 42 only, so a downstream swing of a few tenths is inside what one seed's initialization and data order can manufacture. I will count a metric as having moved next rung only if it clears roughly a point. That guard matters precisely because I am about to predict that the zero-state helps the loss but not discrimination — if the ternary run's HellaSwag wobbles by half a point in either direction, I have to read it as noise, not as evidence for or against the off-switch, and the same caution applies to ARC-Easy near 47 and PIQA near 59.

Here is the deficiency, stared at directly. A two-valued set `{−1,+1}` forces every weight to
participate at full magnitude `±β`. Watch what `sign` does to a weight that is essentially zero — a
connection the network would rather not have at all: it still gets shoved to `+β` or `−β` at full
strength, picking up whichever side of zero the latent noise happened to land it on. The binary code
has no way to express *absence*. And in a 355M Transformer a large fraction of the weights, after
training, genuinely want to be near zero — that is what gives a dense layer its effective sparsity and
its selectivity. Binary cannot represent that; it converts every weak-but-near-zero connection into a
full-strength one with an essentially random sign, which is pure injected noise into the matmul. So the
2.7352 floor is, in large part, the cost of forbidding the network an off-switch. That is the precise
thing I should buy back next, and it costs almost nothing in bits.

What does adding a single zero level buy, concretely? Two things. First, capacity in the obvious
counting sense: a set of three values carries `log₂3 ≈ 1.585` bits per weight instead of one, so I am
spending barely more than a bit and getting a qualitatively richer alphabet. Second, and this is the
real prize, a weight set to exactly zero contributes *nothing* to the dot product — its term drops out
entirely — so the zero acts as explicit feature filtering: the layer can learn which connections to
switch off, and the random-sign noise that binary injected into every near-zero weight disappears,
because those weights now map to a clean 0 instead of `±β`. That is not just expressiveness on paper;
it is the direct removal of the noise source I just diagnosed. So the next grid is `{−1, 0, +1}` —
ternary, scaled.

I cannot reuse the binary derivation verbatim, because `sign` only ever emits `±1`; I need a rule that
rounds a scaled weight to the *nearest* of `{−1, 0, +1}`. The clean construction: normalize the weight
by a per-tensor scale, then map onto the three integer levels with rounding and clipping. Keep the same
absmean scale `γ = mean(|W|)` — it was the L2-optimal scale for the sign grid and it carries over as
the natural unit, parameter-free, one reduction over the tensor, the same cheap statistic that already
kept `n·γ²` controlled in the variance accounting. Normalize `w_n = W/γ`, then snap `w_n` onto
`{−1, 0, +1}`. Let me check the threshold this induces, because it is the whole point of the zero. A
normalized value below `0.5` in magnitude rounds to `0`; between `0.5` and `1.5` rounds to `±1`; beyond
`±1` clips to `±1`. So a weight is gated off precisely when `|W| < γ/2 = mean(|W|)/2` — when its
magnitude is below half the average magnitude, i.e. exactly the genuinely small, weak connections. The
survivors that become `±1` are the above-half-average ones. So the absmean scale does double duty
again: it is the L2 unit *and* the dial that places the zero-threshold at the sensible spot, gating off
the weak connections that binary was forced to corrupt into full-strength noise.

Let me put a number on that threshold, because "half the average magnitude" is only reassuring if it does
not gate off the whole layer. For a Gaussian weight tensor at the `std=0.02` init, `γ = mean(|W|) =
σ√(2/π) ≈ 0.798σ`, so the zero-threshold sits at `γ/2 ≈ 0.399σ`. The fraction of a Gaussian inside `±0.399σ`
is `2Φ(0.399) − 1 ≈ 0.31`. So at init roughly 31% of the weights map to zero and 69% survive to `±1` — a
minority switched off, not a majority. That directly quiets the fear that the absmean threshold might be
too aggressive: at a Gaussian init it is not, it gates the smallest third and keeps the rest. The caveat I
have to keep honest is that init is not convergence — if training drives the weight distribution toward
sparsity, the *converged* fraction below `γ/2` could climb well past a third, and then the threshold might
start eating useful weak connections. I cannot know the converged fraction from here; what I can say is
that the mechanism is sound at init and the risk lives entirely in how much training sparsifies, which the
loss table will indirectly reveal.

Two more sanity numbers before I trust this grid. First, is the ternary alphabet actually *used*, or does
one level dominate and waste the code? At init the trit distribution is ~31% zero and, by symmetry, ~34.5%
each on `±1`, whose Shannon entropy is `−0.31·log₂0.31 − 2·0.345·log₂0.345 ≈ 0.524 + 1.059 ≈ 1.58` bits —
essentially the maximum `log₂3 ≈ 1.585`. So all three symbols are well-populated; I am not paying for a
third level that never fires. Second, what does 1.58 bits buy in storage terms, to check the "barely more
than a bit" claim is honest? Ternary packs efficiently because `3⁵ = 243 ≤ 256`, so five trits fit in one
byte at exactly `8/5 = 1.6` bits per weight; the 355M projection weights are then ~71 MB against binary's
44 MB and `bfloat16`'s 710 MB — still a 10× compression, and only ~27 MB more than binary for a
qualitatively richer alphabet. So the bit accounting supports the move: a small storage premium for the
off-switch.

I should also decide whether `γ/2` is even the threshold I want, since there is a principled alternative
and I am about to spend a measurement. The ternary-weight-network line derives an approximately optimal
symmetric threshold `Δ* ≈ 0.7·mean(|W|)` for Gaussian-ish weights, which for this init is `≈ 0.559σ` and
gates about 42% to zero — noticeably more aggressive than the round-induced `γ/2`. So I have a genuine
fork: tune the threshold toward the reconstruction-optimal `0.7·mean(|W|)`, or take the parameter-free
`γ/2` that falls out of plain round-then-clip. I take `γ/2`, for two reasons that are about the ladder,
not about which threshold minimizes reconstruction. First, `γ/2` is more *conservative* — it keeps more
weights on — so if the off-switch turns out to matter I have not risked over-pruning the very connections
I am trying to study. Second, it reuses the absmean with no new hyperparameter, keeping this rung a
one-variable change from binary. Reconstruction-optimality of the threshold is a refinement I can revisit
if ternary underperforms; it is not what this rung is testing.

There is a matching subtlety in the *scale* that I should name rather than paper over, because the fill
reuses the binary absmean and that is not quite the ternary-optimal choice. If I solved the ternary least
squares honestly — pick `Δ` and a positive `α` to minimize `‖W − α·ternary_Δ(W)‖²` — the optimal `α` is
the mean magnitude of the *surviving* weights, `α = mean(|W_i| : |W_i| > Δ)`, not the global absmean over
all weights including the zeroed ones. Since the survivors are precisely the larger-magnitude weights, that
survivor-mean is strictly bigger than the global `γ = mean(|W|)`; at this init the survivors are the top
69% by magnitude, so their mean magnitude runs a fair bit above `0.798σ`. Using the global absmean as the
dequant scale therefore reconstructs each surviving `±1` weight slightly *too small*. I keep the global
absmean anyway, for the same ladder reason as the threshold: it is the one scale I have carried since rung
1, it needs no extra reduction over a data-dependent mask, and holding it fixed keeps this rung a
single-variable change. The under-scaling it introduces is small and uniform, and folding a survivor-mask
mean into the quantizer would confound the one comparison I want — does the zero help? — with a change to
how the surviving levels are placed. So I take the global absmean and keep the question about the zero.

One subtlety in *how* the snap is written, because the task's fill orders the two non-differentiable
operations in a way worth being explicit about. The textbook ternary quantizer is round-then-clip —
`RoundClip(w_n, −1, 1) = clip(round(w_n), −1, 1)` — and there the clip almost never fires, because
round already produces an integer and clip only catches the rare `|w_n| ≥ 1.5` that rounds to `±2`. The
fill here does it in the other order: it *clips first* to `[−1, 1]`, then rounds. Clamp-then-round
folds the saturation into a continuous pre-round value, so anything with `|w_n| > 1` is pinned to
exactly `±1` before rounding (round of `±1` is `±1`), while values inside `[−1, 1]` round to the
nearest of `{−1, 0, +1}` exactly as before. The two orders agree on the resulting ternary level for
every weight — the set of values that map to `0`, `+1`, `−1` is identical — so the forward function is
the same ternary map; the only difference is which intermediate the STE attaches its identity gradient
to, and since the STE is `(w_q.round() − w_q).detach() + w_q` wrapped around the *clamped* `w_q`, the
gradient flows as identity through the clamped value, which is exactly the saturating behavior I want
(weights driven past `±1` stop receiving a magnitude-increasing gradient through the round). So the
clamp-first ordering is a faithful ternary quantizer with sensible saturation, not a different method.

Let me trace three concrete weights through it to be sure the map is what I claim. With `γ = 0.0175` (an
absmean in the init ballpark): a weak weight `W = 0.006` gives `w_n = 0.343`, clamp leaves it, round →
`0`; a middling `W = 0.014` gives `w_n = 0.8`, clamp leaves it, round → `1`, dequantized to `γ = 0.0175`;
a large `W = 0.05` gives `w_n = 2.86`, clamp to `1.0`, round → `1`, also `0.0175`. So the third weight,
nearly four times the second, gets the *same* forward value — the resolution loss is still there, ternary
only added the ability to say the first weight is off. That is exactly the point I want to hold onto: the
zero is a structural gain, not a magnitude gain.

It reassures me that ternary is a strict generalization of binary, which I can confirm by pushing the
threshold to its two limits. Send the zero-threshold to 0 (imagine normalizing by a scale so large that
no `|w_n|` ever falls below 0.5): then no weight gates off, every weight snaps to `±1`, and ternary is
exactly the binary sign grid — so binary is the threshold-→-0 corner of this rung, and ternary can only
match or beat it if the optimizer would rather use the zero. Send the threshold the other way, toward the
top of the distribution, and every weight collapses to 0 and the layer dies. The sensible operating point
is an interior one, and `γ/2` — gating the smallest ~31% at init — is a principled interior point, not the
degenerate corner in either direction. That framing also tells me the *worst case*: since binary is a
reachable special case, a correctly-trained ternary model should not do worse than binary except through
optimization noise or through the threshold gating off weights the model genuinely needed. If ternary
lands measurably *below* binary on some metric, that second failure mode — over-gating at convergence — is
the first thing I will suspect, and it is the same risk I flagged from the init fraction.

The STE itself is the same wire I needed for binary, for the same reason: `round` and `clip` have zero
derivative almost everywhere, so I pretend they are the identity on the backward pass via the
detached-difference idiom, and the float latent weight keeps accumulating the optimizer's tiny noisy
steps that a discrete variable could never integrate. Nothing about the latent-weight story changes
from rung 1 — I still keep `self.weight` in float, quantize it on the fly, and discard the latent copy
conceptually at inference.

That the latent stays in float has a consequence for the zero level I should check, because "31% of weights map to 0 at init" could be read as a third of the layer going dark and never learning again — a static prune, which would be a disaster. It is not that, and the reason is in the STE. A weight with `|w_n| < 0.5` maps forward to `0`, but the clamp is inactive there (its value sits inside `[−1, 1]`) and the round is bridged by identity, so its backward gradient is the full `1/γ` and the float latent keeps accumulating the optimizer's steps. A zeroed weight whose latent drifts past `0.5·γ` crosses the threshold and switches on to `±1`; a surviving weight whose latent decays back under `0.5·γ` switches off. So the ternary mask is recomputed from the current latent weights on *every* forward pass — a single weight can toggle between `0` and `±1` many times across a run as its accumulator wanders. That is exactly what makes the zero a *learnable* feature filter rather than a one-shot pruning: the layer discovers which connections to switch off and can change its mind mid-training. A hard mask applied once would zero the forward value *and* the gradient and freeze the weight dead; quantizing on the fly over a live float latent is what avoids that, and it is the concrete reason keeping `self.weight` in float matters even for the weights that currently read as `0`. The only weights that do go gradient-quiet are the ones the clamp saturates, `|w_n| > 1` — the solidly-`±1` band, about 42% of the tensor at this init — and that is the intended STE behavior: a weight already pinned to the top level gains nothing from being pushed further out, so it should feel no pull until its latent relaxes back under the knee.

Now I should be honest about what this task's ternary fill keeps from the absmean-ternary idea and what
it drops, because the textbook version carries machinery the harness omits. The canonical ternary
BitLinear is built on a LLaMA-style backbone and *fuses an RMSNorm into the layer* — a sub-layer
normalization placed before the activation quantizer to force `E[x̃²] ≈ 1` and hold the matmul output
variance near 1 across depth — and it scales activations *per token* so an outlier feature cannot crush
the rest of a row. This fill has **neither**. It adds no normalization inside `BitLinear`, and it is
correct not to: the GPT-2 block already applies a `LayerNorm` immediately before each projection, so
the input reaching the ternary `BitLinear` is already normalized and the variance bookkeeping the fused
RMSNorm would have done is done by the frozen substrate. Let me confirm the variance stays sane with the
zero level in play, since a third of the weights now vanish. Write `E[g²]` for the expected squared grid
value; at init it is just the surviving fraction, `E[g²] = P(|w_n| > 0.5) ≈ 0.69`. Then `Var(y) = n·γ²·
E[g²]·E[x̃²] = n·(2/π)σ²·0.69·E[x̃²] ≈ 0.44·n·σ²·E[x̃²]`. So ternary's forward variance at init is about
0.44× the float layer's, a hair quieter than binary's 0.64× because the zeroed weights drop their terms —
still the same order, still cleanly renormalized by the downstream LayerNorm, no SubLN needed. And the
activation quantizer is the *same* per-tensor 8-bit absmax the binary rung used (`Q_b = 127`, clip
`[−127, 127]`, STE), not the per-token scheme. That last choice is deliberate ladder hygiene: the
activation path is held *identical* across all three rungs so that the only thing changing from binary to
ternary is the weight grid. If ternary beats binary, I will know it was the zero level and not a quieter
change to how activations are scaled.

Let me also pin down why ternary and not some other small set, since I am about to spend a measurement
on it. Two values I have already shown is too rigid — no off switch, near-zero weights become
full-strength noise. A non-symmetric set like `{0, 1}` breaks the thing that made binary stable: with
no negative value the matmul cannot subtract, the representation is biased, the code is no longer
centered on zero, and the optimizer fights a one-sided alphabet whose mean drifts. The symmetry of
`{−1, 0, +1}` keeps the matmul a balanced signed add/subtract and the code centered. Could I go richer —
`{−2,−1,0,1}`, or a genuine four-level grid? Each added value costs bits and kernel complexity, and the
*first* set that gives me negative, zero, and positive is `{−1, 0, +1}`. So I stop at ternary here and
let a richer grid be its own test later; the controlled question this rung answers is narrow and clean —
does adding *only* the zero, at 1.58 bits, beat one bit? Mixing in extra magnitude levels now would
confound the answer.

Assemble the fill: `weight_quant` returns `(clamp-then-round-to-{−1,0,1}, mean(|W|))` with STE;
`activation_quant` is the unchanged int8 per-tensor absmax with STE; `forward` runs
`F.linear(x_q, w_q)` and rescales by the product of the two scales before the bias. The recipe stays
inherited — `CONFIG_OVERRIDES` empty, peak LR `6e-4` — so this remains a clean grid-only comparison
against binary under the standard schedule (the distilled module is in the answer).

So the delta from rung 1 is exactly one thing: where binary forced every weak weight to `±β` and
injected random-sign noise, ternary lets `|W| < mean(|W|)/2` collapse to a clean 0. Here is what I
expect that to do against binary's measured numbers, and where I am unsure. I expect validation loss to
fall *modestly* below 2.7352 — the zero removes a real noise source, but it is only one extra level on
a tiny set, so I am buying back a fraction of the gap to float, not closing it. WikiText-2 should drop
from 81.3 toward the high 70s and LAMBADA from 110 toward the high 100s, both small moves in the same
direction. Downstream I expect ARC-Easy and HellaSwag to stay essentially flat near binary's 47.0 and
28.7 — the zero-state helps the loss but the model is still starved of magnitude resolution (the
three-weights trace above is the reason), so I do not expect the completion tasks, which need that
resolution, to move much. The risk I can already feel and have now half-quantified: the absmean
threshold gates ~31% at init but could gate a majority at convergence if training sparsifies, in which
case ternary could lose useful weak connections and land *at or even slightly worse than* binary on some
metric — which is exactly what the numbers will adjudicate. If ternary's gain over binary turns out to be
as small as I suspect, the diagnosis this rung hands forward is already written: the problem was never the
off-switch, it was *magnitude resolution* — three coarse levels still cannot say "this weight is
small-but-not-zero" — so the capability worth buying next is graded magnitude rather than another
structural state, and the open question I am leaving on the table is whether spending the next fraction of
a bit on resolution rather than on the off-switch is what actually moves the loss.
