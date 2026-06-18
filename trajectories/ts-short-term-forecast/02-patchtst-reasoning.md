The linear floor told me where the binding constraints are, in numbers. DLinear came in at SMAPE
13.39 on Monthly, 10.50 on Quarterly, 14.36 on Yearly — and the ordering across regimes is exactly
the one the affine hypothesis predicted: best where there are the most seasonal cycles to read
linearly (Quarterly, two-plus cycles in the window), worst on Yearly, the trend-dominated regime with
the twelve-step look-back and the six-step horizon. So the floor is real and it is honest under the
SMAPE loss. But two things in that result are diagnostic of what the affine model *cannot* do, and
they point straight at the next move. First, the Yearly number (14.36) is far above the other two —
a regime where a fixed linear combination of a short window simply runs out of structure to exploit,
because there is no repeating shape to lock onto and the trend extrapolation from twelve points is
fragile. Second, and more general: a single shared affine map has to serve M4's tens of thousands of
series, each at its own level and scale, with one set of weights and *no per-window normalization* —
so it cannot decouple "what shape" from "what level," and any series sitting far from the training
average is systematically off. SMAPE's percentage form hides some of that at training time, but it
cannot manufacture the nonlinear representation the linear map structurally lacks. So the question for
this rung is: can I keep the linear model's robustness on these short series while adding a *learned,
nonlinear* representation and a *per-instance normalization* — the two things DLinear left on the
table — without reintroducing the overfitting that sank the heavy attention forecasters?

The sharpest version of that question is the one the whole attention line refused to ask. A linear map
beat Informer, Autoformer, FEDformer — every clever surgery on the attention kernel. The usual reading
is "attention is the wrong tool for this data." But I want to chase the opposite reading: attention is
fine, and we were feeding it garbage. Look at what a token *is* in those models — a single time step,
one scalar at time `t`. A single time step has no standalone meaning the way a word does; the
information in a series lives in *shapes over short stretches* — a ramp, a dip, a local oscillation.
Point-wise attention asks "how does the scalar at step 14 relate to the scalar at step 3?" and the
answer is mostly noise, because neither scalar means anything alone. That would explain DLinear
winning: it never does point-wise comparison, it reads the *whole window at once* through learned
weights, so it sees the shape. The fix, then, is not a new attention kernel — it is to change the
token. Cut the series into contiguous sub-series *patches* and let each patch be a token: a length-`P`
patch is a little shape, exactly the object attention should compare. This is the vision move (cut the
image into 16×16 patches) carried over to a series.

Patching buys three things at once, and I should keep them straight because two of them barely matter
here and one matters a lot. (1) Local semantic tokens — each token now carries a shape, so attention
has something real to compare. (2) An `S²` cut in attention cost, since the token count drops from `L`
to `~L/S` and attention is quadratic — but on M4 the window is `seq_len = 36` at most, so cost was
never the constraint; this saving is real but irrelevant on these tiny windows. (3) The headroom for a
longer look-back — also moot here, because the harness fixes `seq_len = 2·pred_len`. So on *this* task
the load-bearing benefit of patching is purely (1): it gives the Transformer shape-tokens instead of
meaningless scalar-tokens, which is the only honest way to test whether depth-plus-attention can beat
the linear floor on short series. With `patch_len = 16`, `stride = 8` (half-overlap, so no local shape
is split cleanly down the middle between two patches), and end-padding `stride` copies of the last
value so the most recent step — the one that matters most for the forecast — is never dropped, the
patch count is `N = floor((seq_len − P)/S) + 2`. For Monthly that is `floor(20/8)+2 = 4`; for
Quarterly `floor(0/8)+2 = 2`; for Yearly the window (12) is *shorter than the patch length (16)*, so
`(L − P)` is negative — I have to handle that boundary, and the harness's `PatchEmbedding` does, by
the same replication-pad-then-unfold that always yields at least the two padded patches. So even
Yearly gets a couple of shape-tokens; it will be a thin attention, but it will not crash.

The second half of why the linear model wins is the channel axis, and here M4 makes the decision for
me. The heavy multivariate Transformers *mix* channels: at each step they fuse the whole channel
vector into one token, so every channel is forced under one shared attention pattern. DLinear does the
opposite — each channel through its own (shared-form) map, completely independently — and it wins.
Three reasons mixing is worse, all of which bite harder on short data: adaptability (one shared
attention map can't be right for a slow-trend channel and a sharp-cycle channel at once, whereas
per-series maps adapt), data efficiency (learning cross-channel *and* temporal structure jointly is a
much bigger hypothesis space, starved on short series), and overfitting (mixing fits spurious
cross-channel coincidences and overfits in a few epochs). On M4 this is not even a tradeoff: every
series is its own univariate channel (`enc_in = c_out = 1`), so channel-mixing has *nothing to mix* —
the native case here is one shared backbone run independently per series, channels folded into the
batch axis (`[B, 1, L] → [B, N, P]`). Channel-independence costs nothing but a reshape, and the
univariate M4 setting is its native case, not a special case bolted on.

The backbone is deliberately *vanilla* — the thesis is that the token was the bug, so I must not sneak
in a fancy kernel or I would muddy what is responsible. Linear-project each patch to `d_model` with a
no-bias embedding (an instance-normalized patch already has its level removed, so a per-patch additive
offset buys nothing), add a learned positional embedding (patches are an unordered set to attention,
but order is everything in a series), then standard multi-head scaled dot-product attention over the
`N` patch tokens with the usual `1/√d_k` scaling, a position-wise FFN `d_model → d_ff → d_model`, and
residuals. One non-default choice I should not make on autopilot: normalization is **BatchNorm**, not
LayerNorm. Time series have outliers — a spike, a glitch — and an outlier step inside a patch skews
that token's *within-token* statistics, dragging LayerNorm around; BatchNorm normalizes each feature
*across* the batch of patch positions, so a single outlier patch is diluted rather than corrupting its
own normalization. This is measured to matter for time-series Transformers, so it stays. The head is
the simplest faithful thing: flatten the `d_model × N` encoder output per series and project with one
linear layer to the horizon (`head_nf = d_model · N`), shared across series — sidestepping the
oversized joint `(L·D) × (M·T)` head a mixing model needs and that overfits.

The one piece that directly answers DLinear's level-tracking failure is the reversible instance
normalization, and I want it explicitly because that was the floor's structural weakness. Before
patching, per instance subtract the look-back mean and divide by `√(var + 1e-5)` (biased variance — I
want the window's actual scale, not an unbiased estimator; both statistics detached, they are
normalization constants not parameters). The encoder then always sees roughly zero-mean unit-variance
shapes regardless of where the window sits, decoupling shape-learning from level-tracking — the exact
decoupling DLinear could not do. At the end the forecast is denormalized: multiply by the std, add the
mean. This is the cheap, consistent gain that should help most on the regimes where per-series level
drift is largest.

A real caution before I commit, because the harness protocol is not PatchTST's own. The fixed Custom
settings pass `d_model = 512`, `d_ff = 512`, `e_layers = 2` — a *much* wider model than PatchTST's
own M4 script (`d_model = 128`, `e_layers = 3`). On windows this short (`N` is 2–4 patch tokens) a
512-wide, 2-layer attention encoder with a flatten head of width `512·N → pred_len` is a large model
fitting a handful of tokens from very short series. The instance normalization and channel-independent
shared weights are exactly the regularizers that should keep this from blowing up, and SMAPE training
plus 10-epoch early stopping (patience 3) caps the overfitting window. But this is the rung where I am
most exposed to the protocol mismatch: a wide attention model on two-token sequences is *capacity the
data may not be able to use*, which is precisely the overfitting failure the channel-independence
argument warned about — only now driven by the harness's width rather than by channel-mixing.

So the falsifiable expectation against DLinear's numbers is specific. I expect this rung to **beat the
linear floor on every regime** — the learned nonlinear representation plus reversible instance
normalization should recover both the shape-modeling DLinear's affine map can't express and the
level-decoupling it can't do — so I expect to clear 13.39 (Monthly), 10.50 (Quarterly), and the
fragile 14.36 (Yearly), with the largest absolute gain on Yearly where DLinear was most strained.
What I am *not* confident about is whether it beats the floor by much, or whether the 512-wide encoder
on 2–4 patch tokens leaves capacity stranded; if the gain over DLinear is only marginal — a few tenths
of a SMAPE point rather than a clear separation — that is the tell that attention over so few short-
window tokens is near its ceiling here, and that the next rung should stop adding generic capacity and
instead exploit the *multi-scale* structure of these series directly, mixing coarse and fine views
rather than attending over a handful of patches.
