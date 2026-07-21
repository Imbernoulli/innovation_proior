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

Let me put the three comparisons in numbers, because the *sizes and signs* are what tell me the two
rungs are genuinely complementary rather than one dominating. On Weather, fusion bought MSE
$0.001652\to0.001346$, a $18.5\%$ drop, and MAE $0.029337\to0.026998$, $8\%$ — a clean, unambiguous win
for reading covariates. On ECL the verdict *split*: MSE $0.317887\to0.301479$, down $5.2\%$, but MAE
$0.394697\to0.405040$, *up* $2.6\%$. That opposite-sign pair on one dataset is not a wash to be averaged
away; it is the precise fingerprint I pre-registered for indiscriminate fusion — squared-error metric
improves because a few big cross-client misses are corrected, typical-error metric worsens because 320
mostly-irrelevant clients each leak a little noise into the target token. And on ETTh1 iTransformer went
$0.058292\to0.058923$ MSE (up $1.1\%$) and $0.182538\to0.186680$ MAE (up $2.3\%$) — a small but
consistent loss of exactly the magnitude I expected from discarding patch-level resolution on a dataset
that barely needs fusion. So the ledger is unambiguous: iTransformer is strictly better on Weather,
strictly worse on ETTh1, and *ambiguous* on ECL in a way that points at its own mechanism. No single
prior rung is the answer, and the ECL split in particular says the fix is not "more fusion" or "less
fusion" but *differently routed* fusion.

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

There is a tempting shortcut to the asymmetry that I should reject explicitly, because it would save me
building a second stream. I could take iTransformer as-is and simply *mask* its attention so that only
the target's variate token is allowed to query the others — one row of the $N\times N$ map kept, the
rest dropped. That gives requirement three for free: information flows only into the target, at $O(N)$
cost. But it does nothing for requirement one, and requirement one is half the problem. A masked
iTransformer still represents the target as a *single collapsed token* — the whole 96-step target series
crushed into one 512-vector by one linear map — which is exactly the representation that lost ETTh1.
Masking the attention cannot give back the intra-series resolution that the collapse destroyed; the
detail is gone before attention even runs. So the masked-iTransformer buys directedness at the price of
keeping iTransformer's temporal blur, and I would be back to failing requirement one. That is the tell
that asymmetry alone is not enough: I genuinely need the target represented as *patches* (for its own
structure) while the covariates stay *variate tokens* (for cheap reading), which means two different
tokenizations coexisting in one model — not one tokenization with a masked attention. The two-stream
split is not decoration; it is the only way to satisfy requirement one and requirement three at once.

So I split the representation by role. The **endogenous** stream is the target channel alone, treated
exactly the way PatchTST treats a channel: cut its length-96 look-back into patches of length 16,
embed each as a token, and keep $\text{patch\_num}=\lfloor 96/16\rfloor$ patch tokens that carry the
target's local temporal shapes at full resolution. The **exogenous** stream is every *other* channel,
each treated the way iTransformer treats a variate: collapse its whole look-back into a single variate
token via the inverted embedding, so the exogenous side is $N-1$ tokens, one per covariate, each a
clean summary of that channel's temporal profile. Two tokenizations, each matched to its role: fine for
the thing I model in detail, coarse for the things I only need to read.

Note one difference from PatchTST's patching that follows from only having to patch a single series here:
I use stride 16, not 8 — non-overlapping patches. With `patch_len=16`, `stride=16`, the target's 96-step
look-back gives $\lfloor96/16\rfloor = 6$ endogenous patch tokens, plus the one global token, so the
endogenous stream is just 7 tokens wide. PatchTST needed the stride-8 overlap and its 12 tokens because
it was the *whole* model applied to every channel; here the endogenous stream is a small, cheap object —
6 patches of the one channel I actually score — and the model's capacity can go into the fusion instead.
That is a real budget consequence of the endo/exo split: I patch exactly one series, so I can afford to
patch it cleanly and keep the token count tiny.

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

Funnelling every covariate through one token might look like a bottleneck too tight to carry the signal,
but the target does not need the covariates' fine temporal detail — that is *their* business — only a
summary of which way they are pushing it, and a single $d_{model}$-wide token updated by attention over
the covariate tokens is amply expressive for that. The bottleneck consolidates the covariate influence
before it touches the target's fine structure, so the same attention that reads useful signal also
averages away the per-channel noise — it is a denoiser, not a limitation. And because only the global
token queries the exogenous stream, the fusion costs $O(1\cdot N)$ per layer rather than iTransformer's
$O(N^2)$: on ECL, one query against 320 keys versus a full $321\times321\approx 103{,}000$-entry map,
and every retained interaction terminates *at the target* rather than at some pair of unscored clients.

The denoising is not just intuition: if each covariate token carries the same useful signal plus
independent noise of variance $\sigma^2$, the global token's softmax-weighted aggregate passes the
signal at full strength while the noise variance drops to $\sigma^2\sum_c w_c^2 \approx \sigma^2/M$ under
roughly uniform weights — suppressed by a factor of $M$. Letting each of the 6 patches run its own
cross-attention instead would give no averaging across patches, so the fine structure would carry six
separately-corrupted injections. The single global token maximizes covariate signal-to-noise before any
of it touches the patches, which is the mechanism behind the ECL MAE fix. Concretely, then, the endo
patch stream preserves the detail iTransformer gave up on ETTh1; the directed bottleneck reads exogenous
influence without injecting it wholesale, so ECL MAE should come back down while cross-client information
keeps MSE low; and Weather's fusion win holds, just routed more precisely — the literal resolution of the
PatchTST-vs-iTransformer tension the last three feedbacks measured.

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

The one shape detail that matters is that the global token is picked out as a single query row — on ECL,
$x[:, -1, :]$ is a $(B,1,512)$ query cross-attending to the $(B,320{+}m,512)$ exogenous keys/values,
producing a $(B,1,512)$ update — so the directed $O(N)$ cost is literal, and the exogenous tokens never
appear on the output path except through that single global-token update. That is the whole asymmetry
made concrete.

One configuration choice deserves its own justification: `e_layers=1`, where PatchTST used 3 and
iTransformer 2. It looks like under-provisioning until I count what one layer here actually contains. A
single TimeXer layer does *three* things — endogenous self-attention over the target's 6 patches, one
directed cross-attention that reads the whole 320-channel exogenous set into the global token, and a conv
FFN — so it already spans both the intra-target temporal modeling and the full fusion in one pass. And
the endogenous stream is tiny (7 tokens over one series), so stacking more self-attention layers over it
buys little; the target's own structure is not deep, it is just fine-grained, and 6 patches with one
attention pass capture it. Adding layers would mostly re-fuse the exogenous set repeatedly, which on a
noisy 320-channel panel is more opportunity for turbulence than for signal. So one layer is not a corner
cut; it is the right depth for a design whose single layer is already doing the work the previous two
rungs split across a deeper stack.

So the bar this rung must clear is concrete and per-dataset, written against the three real numbers
below it. On **ETTh1** it must beat or match iTransformer's 0.058923 / 0.186680 and recover the ground
iTransformer lost to PatchTST's 0.058292 / 0.182538 — the patch endogenous stream should bring ETTh1
back to PatchTST's level, because that is what restoring patch-level resolution buys. On **Weather** it
must hold iTransformer's 0.001346 / 0.026998 — the fusion that won there is preserved, so I expect no
regression; if Weather instead *regressed*, the patch stream had starved the fusion of covariate signal
and the global-token routing was too tight. On **ECL** it must beat iTransformer on *both* metrics, and
the MAE is the one I am really testing: iTransformer slipped to 0.405040 because its fusion was
indiscriminate, and if the directed bottleneck works, ECL MAE should fall well below that — the cleanest
test of whether asymmetric fusion beats symmetric fusion on a large noisy panel. Those are the two
falsifiable failure conditions: if ETTh1 fails to recover the patch-level ground, the claim that the
endogenous patch stream restores PatchTST's resolution is wrong; if ECL MAE does not improve over
0.405040, the directed bottleneck is not controlling the covariate noise and the asymmetry bought
nothing. Clearing both requires this rung to be best-or-near-best on all three datasets at once, which no
prior rung managed — DLinear led nowhere, PatchTST owned ETTh1 but lost Weather and ECL, iTransformer
owned Weather but lost ETTh1 and split ECL — so a rung that tops every dataset would be the first to
dominate rather than trade, the metric-level statement that the endo/exo split subsumes both
channel-independent patching and symmetric cross-variate fusion instead of interpolating between them.
