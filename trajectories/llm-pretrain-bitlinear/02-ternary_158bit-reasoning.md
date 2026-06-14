The binary run told me exactly what one bit costs, and it told me in numbers. Validation loss landed at
2.7352, WikiText-2 perplexity at 81.3, LAMBADA at 110.2, with ARC-Easy at 47.0 and HellaSwag at 28.7 —
barely above the 25 chance floor on the four-way completion task. So the model did *train*: the absmean
scale and the STE made one-bit GPT-2 a real language model rather than noise, and the pre-projection
LayerNorms kept it stable. But it is a weak one. A perplexity in the 80s on WikiText-2 for a 355M
decoder at the Chinchilla-optimal token budget is roughly where a much smaller float model would sit,
and a HellaSwag a few points above chance says the binarized weights are not carrying enough
information to model the long-range structure these completion tasks probe. This is not an optimization
failure — the loss curve was stable, the run finished in 22,601 seconds like the others — it is a
*capacity* failure. One bit per weight is simply too few, and I can name the specific deficiency it
imposes rather than wave at "not enough bits."

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
counting sense: a set of three values carries `log₂(3) ≈ 1.58` bits per weight instead of one, so I am
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
the weak connections that binary was forced to corrupt into full-strength noise. That is the mechanism
I expect to recover the loss binary lost.

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

The STE itself is the same wire I needed for binary, for the same reason: `round` and `clip` have zero
derivative almost everywhere, so I pretend they are the identity on the backward pass via the
detached-difference idiom, and the float latent weight keeps accumulating the optimizer's tiny noisy
steps that a discrete variable could never integrate. Nothing about the latent-weight story changes
from rung 1 — I still keep `self.weight` in float, quantize it on the fly, and discard the latent copy
conceptually at inference.

Now I should be honest about what this task's ternary fill keeps from the absmean-ternary idea and what
it drops, because the textbook version carries machinery the harness omits. The canonical ternary
BitLinear is built on a LLaMA-style backbone and *fuses an RMSNorm into the layer* — a sub-layer
normalization placed before the activation quantizer to force `E[x̃²] ≈ 1` and hold the matmul output
variance near 1 across depth — and it scales activations *per token* so an outlier feature cannot crush
the rest of a row. This fill has **neither**. It adds no normalization inside `BitLinear`, and it is
correct not to: the GPT-2 block already applies a `LayerNorm` immediately before each projection, so
the input reaching the ternary `BitLinear` is already normalized and the variance bookkeeping the fused
RMSNorm would have done is done by the frozen substrate — `Var(y) = n·γ²·E[x̃²]` with `E[x̃²]` held near
1 by that LayerNorm. And the activation quantizer is the *same* per-tensor 8-bit absmax the binary rung
used (`Q_b = 127`, clip `[−127, 127]`, STE), not the per-token scheme. That last choice is deliberate
ladder hygiene: the activation path is held *identical* across all three rungs so that the only thing
changing from binary to ternary is the weight grid. If ternary beats binary, I will know it was the
zero level and not a quieter change to how activations are scaled.

Let me also pin down why ternary and not some other small set, since I am about to spend a measurement
on it. Two values I have already shown is too rigid — no off switch, near-zero weights become
full-strength noise. A non-symmetric set like `{0, 1}` breaks the thing that made binary stable: with
no negative value the matmul cannot subtract, the representation is biased, and the optimizer fights a
one-sided alphabet. The symmetry of `{−1, 0, +1}` keeps the matmul a balanced signed add/subtract and
the code centered on zero. Could I go richer — `{−2,−1,0,1}`, or the four-level grid that is the next
rung? Each added value costs bits and kernel complexity, and the *first* set that gives me negative,
zero, and positive is `{−1, 0, +1}`. So I stop at ternary here and let the four-level grid be its own
test later; the controlled question this rung answers is narrow and clean — does adding *only* the zero,
at 1.58 bits, beat one bit?

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
28.7 — the zero-state helps the loss but the model is still starved of magnitude resolution, so I do
not expect the completion tasks, which need that resolution, to move much. The risk I can already feel:
the absmean threshold might gate off *too aggressively*, sending more than half the weights to zero and
losing useful weak connections, in which case ternary could land *at or even slightly worse than*
binary on some metric — which is exactly what the numbers will adjudicate. If ternary's gain over
binary turns out to be as small as I suspect, the diagnosis for the next rung is already written: the
problem was never the off-switch, it was *magnitude resolution* — three coarse levels still cannot say
"this weight is small-but-not-zero" — and the fix is to spend the next bit on finer levels instead of
on a zero, i.e. trade the explicit off-state for a genuine two-bit grid.
