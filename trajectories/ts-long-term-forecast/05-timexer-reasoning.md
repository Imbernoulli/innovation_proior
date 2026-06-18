TimeMixer came back and it landed almost exactly where I predicted, which means the prediction's logic
now tells me what is still missing. ETTh1 fell to 0.3767 — best on the ladder, recovering the ground
iTransformer lost and edging past PatchTST, just as the "channel-independent again, plus richer temporal
disentangling" argument said it should. Weather fell to 0.1621 / MAE 0.2089 — clearly best, the ideal
multi-scale dataset paying off. So the bet held: multiscale disentanglement generalizes across datasets
better than cross-variate attention. But ECL is the residue, and it is the residue I expected: 0.1561,
better than PatchTST's 0.1819 but *still behind* iTransformer's 0.1482 — TimeMixer gave back some of the
ECL win precisely because it is channel-independent and never explicitly models the 321 coupled channels.
So after four rungs the scoreboard is honest and split: the best temporal model (TimeMixer) owns ETTh1
and Weather but loses ECL to the only model that modeled channels (iTransformer), which in turn lost
ETTh1. The two levers — fine intra-series temporal structure and cross-variate correlation — have never
been held in the same model that also keeps both their virtues. Every rung that grabbed one let go of the
other. The next rung has to hold both at once, and ECL is the dataset that will prove or disprove it.

So I go back to first principles on the multivariate problem itself, channels included this time. Fix
attention on any one channel I am predicting — the *target* — and the others are side channels that
genuinely drive it. In ECL, one client's consumption is shaped by the regional load that the other
clients also reflect; the side channels are causal levers for that target, not decoration. From the
target's point of view I want the side channels to *inform* its forecast and nothing else: I do not want
the target's representation paying to model interactions *into* channels it does not need, and I do not
want their noise flowing back into it. That per-channel asymmetry is the whole thing, and looking back up
the ladder, none of the four rungs is shaped around it.

Line up the failure as the ladder reveals it. The linear control and PatchTST are channel-independent:
excellent intra-target temporal modeling (PatchTST's patches read off local shape cleanly), zero
cross-channel influence — the side channels are invisible, which is exactly why both trailed on ECL.
iTransformer inverted to fix that: it crushes the target's whole series into one coarse vector by a
single linear projection, so all the fine intra-series detail PatchTST preserved is gone, and it treats
every channel as an equal token, so self-attention is O(C²) — for ECL's 321 channels, quadratic in
exactly the dimension that is largest, and most of that compute models interactions *into* channels I
never predict while letting their noise flow straight into the target. That is why it won ECL on raw
cross-variate signal but lost ETTh1, where there is almost no coupling and the lost temporal detail
dominated. TimeMixer kept the target's temporal detail (channel-independent, multiscale) and that is why
it owns ETTh1/Weather — but for the same reason it has no channel pathway, so it could not close the ECL
gap. The two virtues live in two camps and each camp's virtue is the other's missing piece. I want both
at once, and the ladder has shown me that I cannot get there by picking *one* granularity and applying it
to *all* channels.

That is the assumption to break. I have been taking it for granted that whatever granularity I pick —
patch everything, or variate-token everything — I apply it uniformly. But the target and the side
channels play genuinely different roles. The target is the thing whose internal temporal wiggle I must
nail step by step to get the value right. The side channels are only there to nudge it — I care *that*
side channel z shifts the target, not about the precise micro-structure inside z. Different demands, so
they deserve different granularities. So: patch the *target* finely (PatchTST's good idea, the one that
preserves intra-target temporal detail) and variate-token the *exogenous* channels coarsely
(iTransformer's good idea, the one that gives clean cross-variate correlation), and let the two
representations meet through a controlled, asymmetric channel.

Concretely, the endogenous (target) path: take the target series, cut it into P-length patches with the
patch embedding, and additionally prepend a single learnable *global token* per channel — a summary slot
that will stand in for "the whole target series" when it talks to the exogenous side. Run self-attention
*within* the target's patch tokens (plus its global token): this is the fine intra-target temporal
modeling PatchTST had, fully preserved, with no channel noise leaking in because only target tokens
participate. The exogenous path: embed each side channel's whole look-back into one variate token with
`DataEmbedding_inverted` — iTransformer's coarse, physically-coherent per-channel token. Then the cross:
the target's *global token* (and only the global token) attends to the exogenous variate tokens via a
cross-attention. This is the asymmetry made architecture. Cross-variate information enters through one
narrow gate — the global summary slot — so the patch tokens that carry the target's fine temporal detail
are never overwritten by channel mixing, the side channels' noise can only reach the target through a
single learned bottleneck, and the cost is O(patch_num · C) for the cross-attention rather than
iTransformer's O(C²) over all channels — linear in the side-channel count, spent only on informing the
one target I predict. The encoder layer therefore has *two* attentions: self-attention over the target's
patch-plus-global tokens, then cross-attention from the global token into the exogenous tokens, then the
position-wise feed-forward — stacked e_layers deep. The head is the direct multi-step flatten-and-project
that has worked at every rung: flatten the target's patch features (now informed by the exogenous side
through the global token) and map to the full horizon.

I should pin down why the global token is the right gate rather than, say, letting every patch token
cross-attend to the exogenous side. If all patch tokens attended to the channels, two things break that
the ladder already taught me to fear. First, the cross-attention cost becomes O(patch_num · C) per token
times patch_num tokens — back toward the quadratic blow-up, and on ECL's 321 channels that is exactly the
regime where iTransformer spent most of its compute modeling interactions into channels it never
predicts. Second, and worse, every patch token's fine temporal content would be partially rewritten by a
channel mixture at every layer, which is precisely how iTransformer lost the intra-series detail that
cost it ETTh1. A single global token per channel is the minimal sufficient interface: it is one slot that
summarizes "what does the target need from the rest of the panel," it is updated by cross-attention while
the patch tokens are not, and then it is folded back among the patch tokens by the feed-forward so the
target's prediction can use the imported channel signal without any patch ever having been overwritten.
That is the asymmetry the per-channel-role argument demanded, realized as a one-token bottleneck.

There is a subtlety in stacking this e_layers deep that I want to get right. Within a layer the order is
self-attention (target patches refine their own temporal representation), then cross-attention (the
global token pulls in channel context), then feed-forward (the channel context diffuses back across the
patches). Stacking means the imported channel signal at layer ℓ feeds the target's self-attention at
layer ℓ+1, so over depth the target's temporal modeling and the channel context co-adapt — the target
learns to attend to its own patches *conditioned on* what the side channels are saying, which is richer
than a single late fusion. But because the gate stays one token wide at every layer, the cost stays
linear in C and the patch tokens are never directly mixed, so depth buys co-adaptation without
reintroducing either failure mode. This is why I keep the two-attention layer rather than a single fused
attention over a concatenated token set: concatenating target patches and exogenous tokens into one
sequence and running plain self-attention would be Crossformer's move, and it both pays O((patch_num+C)²)
and lets channel noise hit the patches directly — the very things the asymmetric design exists to avoid.

This is the unification the ladder has been pointing at. Endogenous self-attention = PatchTST's
intra-target temporal modeling, the ETTh1/Weather strength. Exogenous cross-attention through the global
token = iTransformer's cross-variate correlation, the ECL strength — but routed asymmetrically so it
informs without overwriting and costs linear, not quadratic, in channels. And the per-window instance
normalization that every rung needed wraps it all. In the multivariate (`features='M'`) setting, every
channel is in turn the endogenous target while all channels serve as the exogenous pool, so the
endogenous patching is applied to all channels and the cross-attention lets each channel's global token
read the full variate set — the symmetric-multivariate form of the same idea.

The edit-surface gap is, as always, what I must state plainly. The TimeXer training script tunes per
dataset: `d_model=256`, `batch_size=4`, `e_layers=1` on ETTh1 and Weather, `d_model=512`, `e_layers=4`
on ECL. The Custom edit gets the fixed scaffold config `d_model=512`, `e_layers=2`, `batch_size=32`,
`n_heads=8`, and `lr=1e-4`, `train_epochs=10` for all three. So I run TimeXer at a *fixed* capacity and a
much larger batch than its own tiny `batch_size=4` recipe — a real regime mismatch, because the small
batch is part of how the method regularizes on these datasets. `patch_len` I read from the config
(default 16, giving seq_len // patch_len = 6 patches at L=96, non-overlapping), and `use_norm` from the
config (default 1). The fixed `d_model=512` is again over-parameterized on tiny ETTh1, so the method's
*own* tuned ETTh1 score is not what I should expect here; the question is only whether, under the
identical fixed loop, the endogenous/exogenous split beats the best the other four managed under the same
loop.

Falsifiable expectations against the four prior rungs. The single claim this rung must clear: be the
first model to hold both levers, which means winning ECL *without* losing ETTh1/Weather. On ECL — the
residue — I expect TimeXer to beat TimeMixer's 0.1561 *and* match or beat iTransformer's 0.1482, landing
around 0.14, because the asymmetric exogenous cross-attention is cross-variate correlation done right
(linear cost, no noise leakage). That is the decisive test: if ECL drops below 0.1482 while the temporal
detail is preserved, the unification worked. On Weather I expect it competitive with TimeMixer's 0.1621,
plausibly a touch better (low 0.15s) since exogenous channels there carry real signal. On ETTh1 I expect
the soft spot: only 7 channels means the exogenous path has little to add, and the fixed `d_model=512`
over-parameterizes a 7-channel problem, so TimeXer may *not* beat TimeMixer's 0.3767 there and could sit
slightly above it — the endogenous/exogenous machinery earns its keep when there are many informative
side channels, which ETTh1 lacks. So the expected signature is: best-on-ladder ECL and Weather, ETTh1
roughly level with or just behind TimeMixer, and the lowest mean MSE on the ladder — carried by finally
modeling channels without sacrificing temporal detail. If instead ECL fails to reach iTransformer's
0.1482, the unification did not actually hold both levers, and the honest conclusion would be that the
asymmetric gate is too narrow at this fixed capacity.
