The cross-variate rung came back, and the numbers draw the exact picture I predicted — which means I
now know precisely what the last rung has to do. On **Weather**, iTransformer dropped to MSE 0.001346 /
MAE 0.026998, beating PatchTST's 0.001652 / 0.029337: the fusion gap I located really was a fusion gap,
and cross-variate attention closed it. On **ECL** it went to MSE 0.301479 / MAE 0.405040 from
PatchTST's 0.317887 / 0.394697 — MSE improved, exactly as I expected from 321 channels of cross-client
structure, but **MAE slipped** (0.4050 vs 0.3947). That MAE regression is not noise; it is the
signature I flagged in advance. iTransformer's $O(N^2)$ attention is *indiscriminate*: on a 321-channel
panel most of its compute models interactions into channels I never score, and their noise flows into
the target's token, which inflates the typical-case error (MAE) even while the squared-error-dominated
MSE improves. And on **ETTh1** it landed 0.058923 / 0.186680 — essentially tied with, and slightly
behind, PatchTST's 0.058292 / 0.182538. That is the other prediction confirmed: by crushing each
channel's whole series into one token, iTransformer threw away the intra-series temporal detail
PatchTST's patches preserved, and on ETTh1 — small panel, smooth target, little for fusion to add — that
trade was a net loss. So I am holding two rungs that each win where the other loses: PatchTST keeps the
target's *own* fine temporal structure but is blind to covariates; iTransformer reads the covariates but
blurs the target's temporal structure and fuses indiscriminately. The last rung has to stop choosing.

Let me state the requirement sharply, from the target's point of view, because the MS scoring makes the
asymmetry the whole game. I score *one* channel. From that channel's perspective the others are
exogenous side-information that genuinely drives it — and what I want is exactly three things at once.
One: the target's *own* temporal structure modeled at full resolution, the way PatchTST's patches do,
because the ETTh1 result says that resolution is worth real MSE/MAE and iTransformer's single-token
projection gave it up. Two: the exogenous channels allowed to *inform* the target, the way
iTransformer's cross-variate attention does, because the Weather result says that information is worth
real error. Three — and this is the part neither rung gets right — the fusion must be *asymmetric*: I do
not want the target's representation paying $O(N^2)$ compute to model interactions *into* channels it
never predicts, and I do not want their noise flowing back into it the way iTransformer's symmetric
attention let it on ECL. The target should read *from* the exogenous variables; the exogenous variables
should not write turbulence *into* the target beyond what informs it. PatchTST fails requirement two,
iTransformer fails requirements one and three. So the design is forced: keep patching for the target's
own series, keep variate-tokens for the exogenous channels, and connect them by a *directed* fusion that
runs target ← exogenous and not the reverse.

So I split the representation by role. The **endogenous** stream is the target channel alone, treated
exactly the way PatchTST treats a channel: cut its length-96 look-back into patches of length 16,
embed each as a token, and keep $\text{patch\_num}=\lfloor 96/16\rfloor$ patch tokens that carry the
target's local temporal shapes at full resolution. The **exogenous** stream is every *other* channel,
each treated the way iTransformer treats a variate: collapse its whole look-back into a single variate
token via the inverted embedding, so the exogenous side is $N-1$ tokens, one per covariate, each a
clean summary of that channel's temporal profile. Two tokenizations, each matched to its role: fine for
the thing I model in detail, coarse for the things I only need to read.

Now connect them, and the connection is where the asymmetry lives. Within a layer, first let the
endogenous patch tokens attend among *themselves* — ordinary self-attention over the target's patches,
which is PatchTST's intra-series temporal modeling, untouched. Then comes the directed fusion, and the
cost-control trick that makes it scale: I do not let every endogenous patch token cross-attend to every
exogenous token (that would be back to per-patch $O(\text{patch\_num}\cdot N)$ and would let the
exogenous noise into every patch). Instead I add **one learnable global token** to the endogenous
stream — a single token that, through the self-attention step, aggregates the whole target series into
one summary — and I let *only that global token* cross-attend to the exogenous variate tokens. The
cross-attention is directed by construction: queries come from the endogenous global token, keys and
values from the exogenous tokens, so information flows exogenous → target and never the reverse. The
global token is the bottleneck through which all exogenous influence must pass, which is exactly
requirement three — the exogenous channels inform the target through one controlled channel instead of
flooding every patch, so ECL's 321 noisy clients can no longer write turbulence into the target's fine
structure the way iTransformer's symmetric attention let them. After the cross-attention updates the
global token, it is folded back among the endogenous patch tokens (it sat in the same stream), a
feed-forward block mixes everything, and the layer repeats. At the end, the endogenous tokens — patches
plus the now-exogenous-informed global token — are flattened through a linear head to the 96-step
forecast for the target.

There is a subtlety in *why* a single global token is enough to carry all the exogenous influence, and
it is worth pinning down because it is the crux of the asymmetry argument. One might worry that funnelling
every covariate's contribution through one token is a bottleneck so tight it loses information. But think
about what the target actually needs from the exogenous side: not the covariates' fine temporal detail —
that is *their* business, not the target's — but a summary of "given where the covariates are now, which
way is the target being pushed." That is a low-dimensional ask, and a single $d_{model}$-wide token
updated by attention over the covariate tokens is amply expressive for it. Contrast the alternative,
where every endogenous patch token cross-attends to every covariate token: now each of the target's
local shapes is independently re-weighted by 320 noisy clients, and there is no shared, denoised summary
— each patch absorbs its own slice of covariate turbulence, which is precisely the mechanism by which
iTransformer's symmetric fusion let ECL's noise inflate the target's typical-case error. The global
token forces the covariate influence to be *consolidated* before it touches the target's fine structure,
so the same attention that reads useful covariate signal also averages away the per-channel noise. The
bottleneck is not a limitation; it is the denoiser. And because only the global token queries the
exogenous stream, the cost of fusion is $O(1\cdot N)$ per layer in the number of covariates rather than
iTransformer's $O(N^2)$ — the directed design is cheaper *and* cleaner on exactly the large panels where
symmetric fusion struggled most.

Let me check this against the two failures I am trying to fix, concretely. iTransformer's ETTh1 loss
came from giving up patch-level temporal detail; here the endogenous stream *is* patches, so that
detail is preserved — ETTh1 should recover toward PatchTST's level, because the target's own structure
is modeled at PatchTST's resolution while the (small) ETTh1 covariate signal is available through the
global token if it helps. iTransformer's ECL MAE regression came from symmetric, indiscriminate fusion
letting covariate noise into the target; here the fusion is one directed cross-attention through a
single global-token bottleneck, so the exogenous influence is read, not injected wholesale — ECL MAE
should come back down even as the cross-client information keeps MSE low. And Weather's win, which came
from cross-variate fusion in the first place, should hold or improve, because the fusion is still there,
just routed more precisely. The construction is the literal resolution of the PatchTST-vs-iTransformer
tension the last three feedbacks measured.

Now the edit surface, because the loop is fixed and I only fill `Model`, and TimeXer's MS path has a
few specifics I have to get exactly right. The endogenous embedding (`EnEmbedding`) patches the **last**
channel of `x_enc` — `x_enc[:, :, -1:].permute(0,2,1)` — unfolds it into length-16 patches with stride
16, value-embeds each plus a positional embedding, and concatenates one learnable global token per
endogenous variate (here exactly one, since `features=='MS'` makes `n_vars==1`). The exogenous embedding
is `DataEmbedding_inverted` over `x_enc[:, :, :-1]` *together with* `x_mark_enc`, so the calendar
features ride in as exogenous tokens too — free covariate signal. The encoder layer runs endogenous
self-attention, then global-token→exogenous cross-attention (queries from the last token of the
endogenous stream, which is the global token; keys/values from the exogenous tokens), folds the updated
global token back, and a conv feed-forward mixes. The head flattens the endogenous tokens
($\text{patch\_num}+1$ of them, $\times d_{model}$) to `pred_len`. I keep per-instance normalization
(`use_norm`), and crucially the de-normalization in MS mode uses the **target channel's** mean and std
(`stdev[:, 0, -1:]`, `means[:, 0, -1:]`) since I forecast only the target. Configuration:
`patch_len=16`, `e_layers=1`, `d_model=512`, `d_ff=512`, `n_heads=8`, `factor=3`, dropout 0.1,
`use_norm=True`. The full scaffold module is in the answer.

So the bar this rung must clear is concrete and per-dataset, written against the three real numbers
below it. On **ETTh1** it must beat or match iTransformer's 0.058923 / 0.186680 and ideally recover the
ground iTransformer lost to PatchTST's 0.058292 / 0.182538 — the patch endogenous stream should bring
ETTh1 back toward the high-0.056 range, because that is what restoring patch-level resolution buys. On
**Weather** it must hold iTransformer's 0.001346 / 0.026998 — the fusion that won there is preserved, so
I expect a small further drop, not a regression. On **ECL** it must beat iTransformer on *both* metrics,
and the MAE is the one I am really testing: iTransformer slipped to 0.405040 because its fusion was
indiscriminate, and if the directed global-token bottleneck is doing what I designed it to do, ECL MAE
should fall well below that — that single number is the cleanest test of whether asymmetric fusion beats
symmetric fusion on a large noisy panel. If ETTh1 fails to recover the patch-level ground, my claim that
the endogenous patch stream restores PatchTST's resolution is wrong. If ECL MAE does *not* improve over
0.405040, then the directed bottleneck is not actually controlling the covariate noise and the
asymmetry bought nothing. Those are the two falsifiable failure conditions; clearing both — best ETTh1
in the high 0.057s or below, Weather held at 0.0013, and ECL down to roughly 0.26 / 0.36 — is what it
means for the endo/exo separation to be the right answer to the exogenous-fusion question this whole
ladder was asking.
