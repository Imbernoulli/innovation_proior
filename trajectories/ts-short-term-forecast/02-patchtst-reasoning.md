The linear floor told me where the binding constraints are, in numbers. DLinear came in at SMAPE
13.39 on Monthly, 10.50 on Quarterly, 14.36 on Yearly — and the ordering across regimes is exactly
the one the affine hypothesis predicted: best where there are the most seasonal cycles to read
linearly (Quarterly, two-plus cycles in the window), worst on Yearly, the trend-dominated regime with
the twelve-step look-back and the six-step horizon. So the floor is real and honest under the SMAPE
loss. The *shape* of the three numbers is a measurement, not just a ranking: Yearly at 14.36 sits 0.96
above Monthly (7.2% larger SMAPE) and 3.86 above Quarterly (37% larger); Monthly sits 2.89 above
Quarterly. The Yearly-over-Quarterly gap is enormous, and it lands exactly where the cycle-count
argument said the affine map runs out of periodic structure — twelve points, no repeating shape to
lock onto, a fragile trend extrapolation. And Quarterly being the *best* regime, not Monthly, says the
linear seasonal read is genuinely strong when the window holds a clean short period. So the binding
constraint is Yearly, and any model that wants a mean win has the most room to take it there.

Those numbers are diagnostic of two distinct things the affine model cannot do, and they point at
different fixes. The Yearly number is a *representation* failure — a fixed linear combination of a
twelve-point window has no nonlinear structure to exploit once the trend is extrapolated. More general
across all three regimes: one shared affine map serves M4's tens of thousands of series, each at its
own level and scale, with no per-window normalization, so it cannot decouple "what shape" from "what
level." As the floor's limiting-case check established, the affine class *can* represent a perfect ramp
or period-copy, but only with per-series constants baked into `W`, and one shared `W` cannot hold them
for every series at once. So the question now is precise: can I keep the linear model's
robustness on these short series while adding a *learned, nonlinear* representation and a *per-instance
normalization* — the two things DLinear left on the table — without reintroducing the overfitting that
sank the heavy attention forecasters?

The sharpest version of that question is the one the whole attention line refused to ask. A linear map
beat Informer, Autoformer, FEDformer — every clever surgery on the attention kernel. The usual reading
is "attention is the wrong tool for this data." But I want to chase the opposite reading: attention is
fine, and we were feeding it garbage. Look at what a token *is* in those models — a single time step,
one scalar at time `t`. A single time step has no standalone meaning the way a word does; the
information in a series lives in *shapes over short stretches* — a ramp, a dip, a local oscillation.
Point-wise attention asks "how does the scalar at step 14 relate to the scalar at step 3?" and the
answer is mostly noise. That would explain DLinear winning: it never does point-wise comparison, it
reads the whole window at once and so sees the shape. The fix is not a new attention kernel — it is to
change the token. Cut the series into contiguous sub-series *patches* and let each patch be a token: a
length-`P` patch is a little shape, exactly the object attention should compare. This is the vision
move (cut the image into 16×16 patches) carried over to a series.

Patching buys three things, and only one matters here. (1) Local semantic tokens — each token carries
a shape, so attention has something real to compare. (2) An `S²` cut in attention cost, since the token
count drops from `L` to `~L/S` — but on M4 the window is `seq_len = 36` at most, so cost was never the
constraint. (3) Headroom for a longer look-back — moot, because the harness fixes `seq_len = 2·pred_len`.
So on this task the load-bearing benefit is purely (1): shape-tokens instead of meaningless
scalar-tokens, the only honest way to test whether depth-plus-attention can beat the linear floor.

Now I have to count the tokens, because the whole premise is that attention gets enough of them to
find structure, and on these windows that count is alarmingly small. With `patch_len = 16`,
`stride = 8` (half-overlap, so no local shape is split cleanly between two patches), and end-padding
`stride` copies of the last value so the most recent step is never dropped, the padded window is `L + 8`
long and the patch count is `floor((L + 8 − 16)/8) + 1`. Monthly: `floor(28/8)+1 = 4`. Quarterly:
`floor(8/8)+1 = 2`. Yearly: window 12 padded to 20, `floor(4/8)+1 = 1` — a *single* patch token,
because the window is shorter than one patch and only the replication pad keeps even one alive. So the
head width `head_nf = d_model · N` is `512·4`, `512·2`, `512·1`, matching the `int((L−16)/8+2)` the code
computes. This is the number that should worry me: on Yearly, self-attention over one token is a no-op,
so on the regime that is already the binding constraint, patching gives attention nothing to do and the
model degenerates to "embed one patch, run the FFN, flatten, project." Quarterly's two tokens are
barely better; only Monthly's four give attention a real (if tiny) set. So the representation half of my
fix helps least exactly where DLinear was weakest — a real tension I walk into open-eyed.

The obvious response is to shrink the patch to manufacture more tokens, so let me price it. Drop to
`patch_len = 4`, `stride = 2`: Monthly's padded window (38) gives 18 tokens, Quarterly 8, even Yearly 6
— the one-token degeneracy disappears. But it fixes the count by undoing the entire thesis. A length-4
patch is barely a shape — four consecutive points is almost a single point again — so shrinking the
patch slides straight back toward the point-wise tokenization whose meaninglessness is why attention
failed on forecasting. I would be buying token count at the cost of token semantics, exactly the wrong
trade, and 18 emptier tokens through a 512-wide encoder makes the capacity problem worse. So
`patch_len = 16` is a deliberate "a token must be a real shape, not a point" choice, and I accept the
small counts (and the one-token Yearly degeneracy) as the honest consequence. If the result confirms
attention is starved here, the lesson is "attention is the wrong axis on these short windows," not "use
smaller patches."

The second half of why the linear model wins is the channel axis, and M4 makes the decision for me. The
heavy multivariate Transformers *mix* channels — at each step they fuse the whole channel vector into
one token, forcing every channel under one shared attention pattern. DLinear does the opposite, each
channel through its own map, and wins. Mixing is worse for three reasons that all bite harder on short
data: one shared attention map can't be right for a slow-trend channel and a sharp-cycle channel at
once; learning cross-channel and temporal structure jointly is a much larger hypothesis space, starved
on short series; and mixing fits spurious cross-channel coincidences. On M4 it is not even a tradeoff —
every series is its own univariate channel (`enc_in = c_out = 1`), so channel-mixing has nothing to
mix. Run one shared backbone independently per series, channels folded into the batch axis
(`[B, 1, L] → [B, N, P]`); channel-independence costs nothing but a reshape and is the native case here.

The backbone is deliberately *vanilla* — the thesis is that the token was the bug, so I must not sneak
in a fancy kernel or I would muddy what is responsible. Linear-project each patch to `d_model` with a
no-bias embedding (an instance-normalized patch already has its level removed, so a per-patch additive
offset buys nothing and would add `d_model = 512` parameters per patch position for no gain), add a
learned positional embedding (patches are an unordered set to attention, but order is everything in a
series), then standard multi-head scaled dot-product attention over the `N` tokens, a position-wise FFN
`d_model → d_ff → d_model`, and residuals. One non-default choice: normalization is **BatchNorm**, not
LayerNorm. Time series have outliers — a spike, a glitch — and an outlier step inside a patch skews that
token's *within-token* statistics, dragging LayerNorm (which normalizes over each token's feature axis)
around; BatchNorm normalizes each feature *across* the batch of patch positions, so a single outlier
patch is diluted rather than corrupting its own normalization. The head is the simplest faithful thing:
flatten the `d_model × N` encoder output per series and project with one linear layer to the horizon
(`head_nf = d_model · N`), shared across series — sidestepping the oversized joint head a mixing model
needs and that overfits.

One thing I could carry over from the floor and deliberately will not: DLinear's moving-average
decomposition. It clearly helped the linear map, but the single thesis I am testing here is that the token was
the bug; if I add decomposition on top of patching and the model improves, I cannot say whether the
gain came from shape-tokens or from preconditioning I already validated at the floor. Worse, decomposing
then patching each branch doubles the token count on windows that are already token-starved, and on
Yearly's one-patch regime it would split a twelve-point window into two even thinner streams. So I keep
the backbone vanilla and let instance normalization be the only preprocessing, precisely so a win here
is attributable to the token change plus normalization and nothing else.

The one piece that directly answers DLinear's level-tracking failure is the reversible instance
normalization. Before patching, per instance subtract the look-back mean and divide by `√(var + 1e-5)`
(biased variance — I want the window's actual scale, not an unbiased estimator; both statistics
detached, they are normalization constants not parameters). The one boundary case that could bite is a
*constant* window: variance zero, so I divide by `√(1e-5) ≈ 0.00316`; but the numerator `x − mean` is
zero everywhere, so `0 / 0.00316 = 0`, the encoder sees zeros and predicts zeros, and denormalization
adds the mean back — a flat forecast at the window mean, which is correct for a constant series. For a
large-level window the shift-and-scale makes the encoder always see roughly zero-mean unit-variance
shapes regardless of where the window sits, decoupling shape-learning from level-tracking — the exact
decoupling DLinear could not do, and unlike its declined NLinear trick it uses the window's *statistics*
rather than just its last value, so it is robust to a noisy final observation.

A caution before I commit, because the harness protocol is not this method's own. The fixed Custom
settings pass `d_model = 512`, `d_ff = 512`, `e_layers = 2` — much wider than PatchTST's own M4 script
(`d_model = 128`, `e_layers = 3`). Each encoder layer is roughly `4·d_model²` for the attention
projections plus `2·d_model·d_ff` for the FFN, about `6·512² ≈ 1.57M` parameters, so two layers is
~3.1M, plus a `16→512` patch embedding and the head. That is ~3M parameters fitting, on Yearly, a
single patch token drawn from a twelve-step window — capacity the data almost certainly cannot use,
precisely the overfitting failure the channel-independence argument warned about, now driven by the
harness's width. The instance normalization, channel-independent shared weights, and 10-epoch early
stopping (patience 3) are the regularizers holding it in check, and the token-count arithmetic says
exactly where I am most exposed: Yearly (one token) and Quarterly (two).

The forward composes on all three regimes because only `N` and `pred_len` change, both derived from
`configs`: RevIN leaves shapes unchanged, `PatchEmbedding` folds the single channel into the leading
axis and emits `[B, N, d_model]`, the encoder preserves that, and the flatten head projects
`d_model·N → pred_len` before denormalizing. The Yearly case with `N = 1` flattens to `[B, 1, 512]` and
projects `512 → 6`, well-formed even though attention over one token did nothing.

That one-token worry has a silver lining that changes what I expect on the degenerate regimes. Even
where attention is inert, the encoder is *not* reduced to a linear map, because each token still passes
through the position-wise FFN `d_model → d_ff → d_model` with a nonlinearity — a genuine per-patch MLP.
So on Yearly the model is "instance-normalize, embed the one patch, apply a nonlinear MLP, project to
six steps," strictly more than DLinear's affine map: it has RevIN's level-decoupling *and* a nonlinear
read of the patch, even with zero useful attention. So my expectation of beating the floor everywhere
does not rest on attention working on the short regimes — it rests on RevIN plus the FFN, with attention
a bonus only where Monthly's four tokens give it something to do. If Yearly improves, it is
normalization and nonlinearity, not attention.

So the expectation against DLinear's numbers: beat the floor on every regime — the learned nonlinearity
plus reversible instance normalization recover both the shape-modeling the affine map can't express and
the level-decoupling it can't do — with the largest absolute gain on Yearly, where DLinear was most
strained and the level-decoupling has the most drift to remove. What I am *not* confident about is
whether the gain is large: the token count says attention is near its ceiling on 1–4 short-window
tokens. If the improvement over DLinear is only a few tenths of a SMAPE point rather than a clear
separation, that is the tell that attention over so few tokens is spent — and that the next step should
stop adding generic capacity and instead exploit the *multi-scale* structure of these series directly.
