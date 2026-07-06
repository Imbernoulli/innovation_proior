The linear floor told me where the binding constraints are, in numbers. DLinear came in at SMAPE
13.39 on Monthly, 10.50 on Quarterly, 14.36 on Yearly — and the ordering across regimes is exactly
the one the affine hypothesis predicted: best where there are the most seasonal cycles to read
linearly (Quarterly, two-plus cycles in the window), worst on Yearly, the trend-dominated regime with
the twelve-step look-back and the six-step horizon. So the floor is real and it is honest under the
SMAPE loss. Let me read the spread quantitatively before I move, because the *shape* of the three
numbers is a measurement, not just a ranking. Yearly at 14.36 sits 0.96 above Monthly (a 7.2% larger
SMAPE) and 3.86 above Quarterly (a 37% larger SMAPE); Monthly sits 2.89 above Quarterly. Two things
fall out of that arithmetic. First, the Yearly-over-Quarterly gap is enormous — more than a third
larger error — and it lands exactly where the cycle-count argument said the affine map runs out of
periodic structure to exploit: twelve points, no repeating shape to lock onto, a fragile trend
extrapolation from a handful of samples. Second, Quarterly being the *best* regime, not Monthly, tells
me the linear seasonal read is genuinely strong when the window holds a clean short period, and it is
Yearly that is dragging the ladder. So the binding constraint is Yearly, and any rung that wants a mean
win has the most room to take it there.

Those numbers are diagnostic of two distinct things the affine model *cannot* do, and they point at
different fixes, so I should separate them. First, the Yearly number is a *representation* failure — a
fixed linear combination of a short window simply has no nonlinear structure to exploit once the trend
is extrapolated, and twelve points give it almost nothing to work with. Second, and more general across
all three regimes: a single shared affine map has to serve M4's tens of thousands of series, each at its
own level and scale, with one set of weights and *no per-window normalization* — so it cannot decouple
"what shape" from "what level," and any series sitting far from the training average is systematically
off. The limiting-case check I did at the floor makes this precise: the affine class *can* represent a
perfect ramp or a perfect period-copy, but only with the right per-series `a,b` baked into `W`, and one
shared `W` cannot hold the right constants for every series at once. SMAPE's percentage form hides some
of that at training time — it does not let a large series dominate the gradient — but it cannot
manufacture the nonlinear representation the linear map structurally lacks, nor can it re-center a
window the model never re-centered. So the question for this rung is precise: can I keep the linear
model's robustness on these short series while adding a *learned, nonlinear* representation and a
*per-instance normalization* — the two things DLinear left on the table — without reintroducing the
overfitting that sank the heavy attention forecasters?

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
the linear floor on short series.

Now I have to actually count the tokens, because the whole premise of patching is that attention gets
enough of them to find structure, and on these windows that count is alarmingly small. With
`patch_len = 16`, `stride = 8` (half-overlap, so no local shape is split cleanly down the middle between
two patches), and end-padding `stride` copies of the last value so the most recent step — the one that
matters most for the forecast — is never dropped, the padded window is `L + 8` long and the patch count
is `floor((L + 8 − 16)/8) + 1`. Monthly: `floor((44−16)/8)+1 = floor(3.5)+1 = 4`. Quarterly:
`floor((24−16)/8)+1 = 1+1 = 2`. Yearly: the window is 12, padded to 20, so `floor((20−16)/8)+1 =
floor(0.5)+1 = 1` — a *single* patch token, because the window (12) is shorter than one patch (16) and
only the replication pad keeps even one full patch alive. So the head width `head_nf = d_model · N`
comes out `512·4`, `512·2`, `512·1` on the three regimes, matching the `int((L−16)/8+2)` the code
computes. This is the number that should worry me: on Yearly, self-attention over *one* token is a
no-op — attention needs at least two tokens to relate anything — so on the regime that is already the
binding constraint, patching gives attention nothing to do and the model degenerates to "embed one
patch, run it through the FFN, flatten, project." Quarterly's two tokens are barely better. Only Monthly,
with four, gives attention a real (if tiny) set to compare. So I should expect the *representation* half
of my fix to help least exactly where DLinear was weakest, which is a real tension I am walking into
open-eyed.

The token count is small enough that I am tempted to just shrink the patch to manufacture more tokens,
so let me walk that alternative and price it before rejecting it. Drop to `patch_len = 4`, `stride = 2`:
Monthly's padded window (38) gives `floor((38−4)/2)+1 = 18` tokens, Quarterly gives
`floor((18−4)/2)+1 = 8`, and even Yearly gives `floor((14−4)/2)+1 = 6` — suddenly attention has real sets
to compare on every regime, and the one-token Yearly degeneracy disappears. That looks like it directly
fixes the problem I just flagged. But it fixes it by *undoing the entire thesis*. A length-4 patch is
barely a shape — four consecutive points is almost a single point token again — so shrinking the patch
slides me straight back toward the point-wise tokenization whose meaninglessness is the whole reason
attention failed on forecasting. I would be buying token count at the cost of token semantics, which is
exactly the wrong trade: the load-bearing benefit of patching here is (1), semantic shape-tokens, and a
4-long patch has almost none. And it makes the capacity problem *worse*, not better — 18 tokens through a
512-wide 2-layer encoder is more attention surface over more but emptier tokens. So `patch_len = 16` is a
deliberate "a token must be a real shape, not a point" choice, and I accept the small token counts (and
the one-token Yearly degeneracy) as the honest consequence of insisting on meaningful tokens. If the
result confirms attention is starved here, the lesson should be "attention is the wrong axis on these
short windows," not "use smaller patches" — which is why I hold the patch size at 16 and let the numbers
speak.

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
offset buys nothing — and here it would add `d_model = 512` parameters per patch position for no gain),
add a learned positional embedding (patches are an unordered set to attention, but order is everything
in a series), then standard multi-head scaled dot-product attention over the `N` patch tokens with the
usual `1/√d_k` scaling, a position-wise FFN `d_model → d_ff → d_model`, and residuals. One non-default
choice I should not make on autopilot: normalization is **BatchNorm**, not LayerNorm. Time series have
outliers — a spike, a glitch — and an outlier step inside a patch skews that token's *within-token*
statistics, dragging LayerNorm (which normalizes over the feature axis of each token) around; BatchNorm
normalizes each feature *across* the batch of patch positions, so a single outlier patch is diluted
rather than corrupting its own normalization. This is measured to matter for time-series Transformers,
so it stays. The head is the simplest faithful thing: flatten the `d_model × N` encoder output per
series and project with one linear layer to the horizon (`head_nf = d_model · N`), shared across series
— sidestepping the oversized joint `(L·D) × (M·T)` head a mixing model needs and that overfits.

There is one more thing I could carry over from the floor and deliberately will not: DLinear's
moving-average decomposition. It would be easy to RevIN, split into trend and seasonal, and patch each
branch separately, and it clearly helped the linear map. But I have to be disciplined about what this
rung is testing. Its single thesis is that *the token was the bug*; if I add decomposition on top of
patching and the model improves, I cannot say whether the gain came from shape-tokens or from the
preconditioning I already validated at the floor. Worse, decomposing then patching each branch doubles
the token count on windows that are already token-starved, and on Yearly's one-patch regime it would
split a twelve-point window into two even thinner streams. So I keep the backbone vanilla and let RevIN
be the *only* preprocessing, precisely so a win here is attributable to the token change plus
normalization and nothing else. Decomposition is a lever I consciously leave on the shelf for a later
rung that wants to combine it with something new, not smear it across this comparison.

The one piece that directly answers DLinear's level-tracking failure is the reversible instance
normalization, and I want it explicitly because that was the floor's structural weakness. Before
patching, per instance subtract the look-back mean and divide by `√(var + 1e-5)` (biased variance — I
want the window's actual scale, not an unbiased estimator; both statistics detached, they are
normalization constants not parameters). Let me verify the two boundary behaviors so I know it is safe.
A *constant* window has zero variance, so I divide by `√(0 + 1e-5) ≈ 0.00316`; but the numerator is
`x − mean = 0` everywhere, so `0 / 0.00316 = 0` — the encoder sees zeros, predicts zeros, and
denormalization multiplies by the tiny std and adds the mean back, returning a flat forecast at the
window mean, which is the correct behavior for a constant series. A *large-level* window (say values
around 10000) is shifted to zero-mean and scaled to unit variance, so the encoder always sees roughly
zero-mean unit-variance shapes regardless of where the window sits, decoupling shape-learning from
level-tracking — the exact decoupling DLinear could not do. At the end the forecast is denormalized:
multiply by the std, add the mean. This is the cheap, consistent gain that should help most on the
regimes where per-series level drift is largest, and unlike DLinear's declined NLinear trick it uses the
window's *statistics* rather than just its last value, so it is robust to a noisy final observation.

A real caution before I commit, because the harness protocol is not PatchTST's own, and I should size the
mismatch. The fixed Custom settings pass `d_model = 512`, `d_ff = 512`, `e_layers = 2` — a *much* wider
model than PatchTST's own M4 script (`d_model = 128`, `e_layers = 3`). Count the encoder: each layer is
roughly `4·d_model²` for the attention projections plus `2·d_model·d_ff` for the FFN, about
`4·512² + 2·512² = 6·262144 ≈ 1.57M` parameters, so two layers is ~3.1M, plus a `16→512` patch
embedding and a `512·N → pred_len` head. That is on the order of three million parameters fitting, on
Yearly, a *single* patch token drawn from a twelve-step window. This is capacity the data almost
certainly cannot use — precisely the overfitting failure the channel-independence argument warned about,
only now driven by the harness's width rather than by channel-mixing. The instance normalization and the
channel-independent shared weights are exactly the regularizers that should keep this from blowing up,
and SMAPE training plus 10-epoch early stopping (patience 3) caps the overfitting window. But this is the
rung where I am most exposed to the protocol mismatch, and the token-count arithmetic tells me exactly
where: Yearly (one token) and Quarterly (two) are where the wide encoder has the least to chew on.

Let me trace the whole forward once to confirm the shapes compose, since a channel-independent pipeline
has several reshapes where a wrong axis would silently corrupt the batch. Input `x_enc` is `[B, L, 1]`.
RevIN subtracts the per-instance mean and divides by std, shapes unchanged. Permute to `[B, 1, L]` and
hand to `PatchEmbedding`, which pads, unfolds into `N` patches of length 16, projects each to `d_model`,
and returns `[B·1, N, d_model]` with the channel count `n_vars = 1` folded into the leading axis — so on
M4 the "batch of series" and the "batch of channels" coincide, and the encoder just sees `B` sequences of
`N` tokens. The encoder maps `[B, N, d_model] → [B, N, d_model]`. I reshape to `[B, 1, N, d_model]` and
permute to `[B, 1, d_model, N]`, then the flatten head concatenates the last two axes to
`[B, 1, d_model·N]` and a single `Linear(d_model·N → pred_len)` gives `[B, 1, pred_len]`. Permute to
`[B, pred_len, 1]`, denormalize by the stored std and mean, and the forward slice returns the last
`pred_len` steps. Every axis lines up on all three regimes because only `N` and `pred_len` change and
both are derived from `configs` — the Yearly case with `N = 1` flattens to `[B, 1, 512]` and projects
`512 → 6`, which is well-formed even though attention over one token did nothing.

The token-count worry has a silver lining worth stating, because it changes what I expect even on the
degenerate regimes. Even where attention is inert — Yearly's single token — the encoder is *not* reduced
to a linear map, because each token still passes through the position-wise FFN `d_model → d_ff → d_model`
with a nonlinearity, which is a genuine per-patch MLP. So on Yearly the model is "instance-normalize the
window, embed the one patch, apply a nonlinear MLP, project to six steps," which is strictly more than
DLinear's affine map: it has the level-decoupling RevIN supplies *and* a nonlinear read of the patch,
even with zero useful attention. That means my expectation of beating the floor everywhere does not rest
on attention working on the short regimes — it rests on RevIN plus the FFN nonlinearity, with attention
as a bonus only where Monthly's four tokens give it something to do. Keeping that attribution straight
matters for reading the result: if Yearly improves, it is normalization and nonlinearity, not attention,
and the next rung should not expect *more attention* to help.

So the falsifiable expectation against DLinear's numbers is specific. I expect this rung to **beat the
linear floor on every regime** — the learned nonlinear representation plus reversible instance
normalization should recover both the shape-modeling DLinear's affine map can't express and the
level-decoupling it can't do — so I expect to clear 13.39 (Monthly), 10.50 (Quarterly), and the
fragile 14.36 (Yearly), with the largest absolute gain on Yearly where DLinear was most strained and
the level-decoupling has the most drift to remove. What I am *not* confident about is whether it beats
the floor by much, or whether the 512-wide encoder on 1–4 patch tokens leaves capacity stranded; the
token count says the attention half of the fix is nearly inert on Yearly and Quarterly, so if there is a
gain there it is carried by RevIN and the FFN, not by attention finding cross-patch structure. If the
gain over DLinear is only marginal — a few tenths of a SMAPE point rather than a clear separation — that
is the tell that attention over so few short-window tokens is near its ceiling here, and that the next
rung should stop adding generic capacity and instead exploit the *multi-scale* structure of these series
directly, mixing coarse and fine views rather than attending over a handful of patches.
