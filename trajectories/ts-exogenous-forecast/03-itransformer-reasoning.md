The patch Transformer did its job as a control, and the three numbers settle the question I built it to
settle. On **ETTh1** it dropped to MSE 0.058292 / MAE 0.182538, beating DLinear's 0.0644 / 0.1878 — and
since ETTh1 is the dataset where cross-channel fusion has the *least* to add, that gain can only be the
per-channel temporal model finally being strong enough. So my step-2 diagnosis was right: part of
DLinear's loss everywhere was a weak temporal model, not just missing fusion. On **Weather** it went to
MSE 0.001652 / MAE 0.029337 — a huge drop from DLinear's 0.005652 — and on **ECL** to 0.317887 /
0.394697 from 0.3873. Good. But now I have exactly the residual I predicted, and I can finally read it
cleanly, because the temporal-model confound is gone: whatever distance remains between this
channel-independent model and the achievable floor on Weather and ECL is the cost of *not reading the
other channels*. The patch model still folds the channel axis into the batch; the weather covariates
still never touch the target; the other 320 clients still never touch the one I score. That residual is
the fusion gap, isolated. So the next rung is no longer optional book-keeping — it is the first one that
goes after the actual research question, and the numbers tell me where it will pay: on Weather and ECL,
not on ETTh1.

Let me read the improvements as differences and ratios, because the *sizes* are informative and one of
them corrects something I half-believed last rung. ETTh1: MSE $0.064359\to0.058292$, a drop of $0.0061$,
about $9.4\%$; MAE $0.187833\to0.182538$, only $2.8\%$. Weather: MSE $0.005652\to0.001652$, a drop of
$0.0040$ — that is $71\%$ of the error gone — and MAE $0.062667\to0.029337$, $53\%$. ECL: MSE
$0.387333\to0.317887$, a drop of $0.069$, about $18\%$; MAE $0.451222\to0.394697$, $12.5\%$. Now, last
rung I floated the idea that Weather's uniform error texture meant its looseness was mostly a systematic
missing-covariate bias. The $71\%$ collapse from *purely* upgrading the per-channel temporal model — no
fusion added — says I was substantially wrong: most of DLinear's Weather looseness was a weak temporal
model, not a missing input. That is a genuine recalibration, and it makes the residual reasoning sharper
rather than weaker: whatever Weather error survives *after* a strong channel-independent temporal model
is, by construction, the part that better temporal modeling could not touch — and that surviving part is
the cleanest estimate I will ever get of the true fusion gap, uncontaminated by the temporal confound.
The ETTh1 numbers point the same way from the other side: a $9.4\%$ MSE gain but a nearly flat MAE, and
essentially no room left, exactly what I expect on the dataset where fusion has little to offer. So the
mandate is precise: go after the Weather and ECL residuals, expect little on ETTh1, and — this is the
part the sizes warn me about — do not assume the Weather residual is huge, because most of that error is
already gone.

Here is the trap I have to avoid, though, and it is the same trap that made the linear map beat the
point-wise Transformers in the first place. The naive way to "add cross-channel attention" is to go
back to a token-per-timestamp layout — embed the $N$-variate slice $\mathbf{X}_{t,:}$ at each instant
into a $D$-vector, get $T$ temporal tokens, and run self-attention over them. That is the layout I just
spent a whole rung arguing is broken, and it is *worse* for fusion than it looks. Look at what is
inside one such token: whatever every sensor happened to read at the same wall-clock instant — in
Weather, temperature next to humidity next to pressure next to wind, different physical quantities,
different units, different distributions, jammed into one vector. And they are not even time-aligned in
the way that matters: a front moves through and hits the pressure sensor minutes before the wind sensor
responds, so "the same timestamp" lumps the early phase of an event at one channel with a different
phase at another. The token is a fruit salad of time-misaligned, incommensurable numbers with a
receptive field of a single tick. There is almost no temporal content in it; the temporal content lives
*across* tokens. Worse, the LayerNorm in that layout normalizes across the feature dimension of a
token — which here is the variate mixture at a fixed $t$ — so at every instant I am centering and
scaling temperature against humidity against pressure together, injecting interaction noise between
unrelated, possibly lagged processes. And the attention map over $T$ temporal tokens tells me which
*instants* resemble which other instants in this scrambled space — not which *variables* drive which,
which is the multivariate structure I actually want. So I will not get fusion by bolting cross-attention
onto the timestamp layout. I need to change what a token *is*, again — but in the opposite direction
from patching.

Before I abandon the patch backbone entirely, I owe the cheaper option a fair hearing: keep PatchTST
exactly as it is and bolt a channel-mixing step onto it — at each patch position, run a small MLP across
the channel axis so covariates can write into the target's patch tokens. It would reuse everything I
already have. But walk it two steps and it breaks on the same rocks as the timestamp layout. Mixing all
$C$ channels symmetrically at every one of the 12 patch positions re-introduces the units-salad problem
— temperature, humidity, and pressure patches averaged together by a shared MLP — and it scales the
mixing weights with $C$, so on ECL that is a $321\times321$-ish mixer at every patch, applied
indiscriminately whether or not a channel matters to the target. It also fixes the coupling in the
weights rather than *learning per-window which channels are relevant*, which is exactly the adaptivity I
want when the covariate that drives the target changes with regime. The channel-mixing MLP is the
symmetric, static, units-blind version of fusion; I can do better by making the fusion an attention that
learns the coupling from data. So I keep looking for the layout where cross-channel comparison is the
*native* operation, not a bolt-on.

Patching split one channel's time axis into many tokens. For cross-channel fusion I want the dual move:
collapse one channel's *entire* time axis into a single token, so that a token *is a variate*. Take
channel $i$'s whole look-back $\mathbf{X}_{:,i}\in\mathbb{R}^{T}$ — all ninety-six steps — and map it
with one linear projection $\mathbb{R}^{T}\to\mathbb{R}^{D}$ into a single $D$-dimensional variate
token. Do that for every channel and I have $N$ tokens, one per channel, and now I run the *stock*
self-attention over them. Read what the attention score between channel $i$'s token and channel $j$'s
token means: it is a similarity between the full temporal profile of variate $i$ and the full temporal
profile of variate $j$ — a clean **cross-variate correlation**. Finally the side channels can
influence the target, because the target is one token among $N$ and attention lets every other variate
token write into it. This is the inversion: instead of attention across time and a shared MLP across
channels, it is attention across *channels* and an MLP within each variate token modeling its temporal
representation. I keep the entire stock encoder — `DataEmbedding_inverted` for the variate tokens, then
`Encoder`/`EncoderLayer`/`FullAttention`/`AttentionLayer` unchanged — and only the *meaning* of a token
changes. That is the elegance: no new attention kernel, no new layer, just the right tokenization for
the structure I want.

Let me trace the shapes and the cost, because the complexity profile of this layout is the opposite of
PatchTST's and I need to know where it will strain. `DataEmbedding_inverted` takes $x_{enc}$ of shape
$(B,96,N)$, transposes to $(B,N,96)$, and applies one $\mathbb{R}^{96}\to\mathbb{R}^{512}$ linear map
along the last axis, producing $(B,N,512)$ — exactly $N$ variate tokens, each a 512-dim summary of that
channel's whole 96-step look-back. The calendar features `x_mark_enc` ride in as a few extra tokens
appended on the $N$ axis, free covariate information. Self-attention then runs over the token axis, so
the attention map is $N\times N$. This is where the cost lives, and it is *quadratic in the number of
channels*: ETTh1's 7 channels give a trivial $7\times7$; Weather's 21 a $21\times21$; but ECL's 321 give
a $321\times321\approx 103{,}000$-entry map per layer per head. Contrast PatchTST, whose attention was
$12\times12$ but replicated across $B\cdot C$ folded sequences — there the channel count inflated the
*batch*, here it inflates the *attention map itself*. So this layout is cheap on the temporal axis (one
projection, no per-step tokens) and expensive on the channel axis, precisely the inverse trade, and it
warns me that ECL — 321 tokens, most of them channels I never score — is where the quadratic cost and any
channel noise will concentrate.

I should also size the parameters, because $D$ jumped from PatchTST's 128 to 512 and that is not free.
The embedding is $96\cdot512\approx 49$k; each encoder layer is about $4D^2 = 4\cdot512^2 \approx 1.05$M
for attention plus $2\cdot D\cdot d\_ff = 2\cdot512\cdot512\approx 0.52$M for the FFN, so $\approx 1.57$M
per layer, and with $e\_layers=2$ that is $\approx 3.1$M; the projection head is another $512\cdot96
\approx 49$k. Total on the order of $3.2$M — roughly six times PatchTST's $0.55$M. The width is not
gratuitous: each token now has to compress a full 96-step series *and* carry enough capacity for the FFN
to model that series' temporal representation, since in this layout the FFN, not attention, does the
within-channel temporal work. A 128-wide token that sufficed for a 16-step patch would be too thin to
summarize the whole window, so the jump to 512 is the price of the inverted tokenization.

Watch what this fixes about the LayerNorm complaint, because it is not a side benefit, it is central. In
the variate-token layout, LayerNorm normalizes across the feature dimension of a *variate* token — i.e.
across the learned representation of one channel's series — which is exactly the per-variate
normalization that removes that channel's distribution shift without mixing channels. The cross-channel
mixing happens *only* in the attention scores, which is where I want it and nowhere else. And the
embedding $\mathbb{R}^{T}\to\mathbb{R}^{D}$ has the *whole* series in its receptive field, so each token
carries real temporal content rather than a single tick. The two diseases of the timestamp layout — no
temporal content per token, channel mixing smeared through norm — are both gone, and they are gone
because of the tokenization, not because of any extra machinery.

The division of labor is clean and inverted from the usual Transformer: attention does the
*cross-channel* work, the FFN the *within-channel temporal* work, and neither steps on the other — which
is why the stock encoder, unmodified, suddenly becomes a competent multivariate model. The layout, not
the layer, was the lever. A limiting case forces the point: if the target is a pure lagged copy of one
covariate, $x^{(\text{tgt})}_t = x^{(\text{cov})}_{t-k}$, and the rest is noise, then PatchTST — seeing
only the target's own past — is structurally blind to a covariate move the target has not yet responded
to, while iTransformer can place nearly all of the target token's weight on the covariate token (their
profiles align up to a shift) and let the value path and FFN implement the $k$-step shift. So there is a
concrete class of dependence only the cross-variate layout can represent, with no lag parameter and no
specified coupling — and Weather's covariate-driven target is its real-world echo.

The forecast head is the dual of the embedding: each variate token, after the encoder has let the other
channels write into it, is projected $\mathbb{R}^{D}\to\mathbb{R}^{\text{pred\_len}}$ back to a
ninety-six-step forecast, and slicing off the mark tokens leaves one forecast per channel with the
target in the last column where the harness slices. I keep the same per-instance normalization as the
patch rung; it reduces over the time axis, so it stays per-channel and all cross-channel mixing happens
in the attention scores and nowhere else, exactly the property I argued for. The calendar features
`x_mark_enc` fold in as extra tokens — free covariate information. Configuration is the standard one:
`e_layers=2`, `d_model=512`, `d_ff=512`, `n_heads=8`, dropout 0.1. The full scaffold module is in the
answer.

I should be honest about what this rung trades away relative to the patch rung, because it predicts
exactly where it might *not* win. By crushing each channel's whole series into a single token with one
linear map, I throw away the fine intra-series temporal detail that patching preserved — the local
ramps and bumps the patch tokens captured. So on a dataset where the target's *own* fine temporal
structure is the dominant signal and the cross-channel coupling is weak, this rung could actually do
*worse* than the patch rung even though it can now read other channels, because it gave up temporal
resolution to buy cross-variate attention. ETTh1 is exactly that case: small homogeneous panel, smooth
strongly-autocorrelated target, little for fusion to add. The other risk is the quadratic-in-channels
cost and noise: self-attention over $N$ variate tokens is $O(N^2)$, and on ECL with 321 channels most
of that compute models interactions *into* channels I never score, while their noise can flow into the
target's token — fusion that is indiscriminate about *which* channels matter to the target.

So the falsifiable expectations against the patch rung's measured numbers, dataset by dataset. On
**Weather**, where I located the fusion gap, this rung should *beat* PatchTST's 0.001652 / 0.029337 —
the covariates genuinely drive the target and cross-variate attention is the first thing that can read
them; if it does, that improvement is the fusion gap being closed, and it is the headline result of the
whole climb so far. On **ECL**, where 321 clients carry strong cross-client structure, I also expect to
beat PatchTST's 0.317887 on MSE, though the indiscriminate $O(N^2)$ fusion and channel noise make MAE
the riskier of the two — I would not be shocked if MAE moves less or even slips. On **ETTh1**, by
contrast, I expect this rung to roughly *tie or slightly lose* to PatchTST's 0.058292, because I traded
intra-series temporal resolution for a cross-channel capability that ETTh1 barely needs — and if it
does lose on ETTh1 while winning on Weather and ECL, that pattern is not a failure, it is the precise
signature of "fusion helps where covariates matter and costs where they don't," which is exactly the
finding the research question is after. The next rung's job, then, writes itself: keep the per-channel
temporal resolution that PatchTST's patches gave *and* the cross-variate reach that this rung gives,
and stop spending quadratic compute and noise on channels the target never needs.

The ECL split, if it appears, is the diagnostic that licenses the next rung: MSE down but MAE flat-to-up
would be the fingerprint of *indiscriminate* fusion — reading every channel whether or not it helps — and
the fix it points to is a fusion that becomes selective about which channels write into the target and by
how much, rather than symmetric across all $N$.
