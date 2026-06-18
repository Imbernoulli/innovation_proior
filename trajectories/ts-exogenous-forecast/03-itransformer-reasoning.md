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

Watch what this fixes about the LayerNorm complaint, because it is not a side benefit, it is central. In
the variate-token layout, LayerNorm normalizes across the feature dimension of a *variate* token — i.e.
across the learned representation of one channel's series — which is exactly the per-variate
normalization that removes that channel's distribution shift without mixing channels. The cross-channel
mixing happens *only* in the attention scores, which is where I want it and nowhere else. And the
embedding $\mathbb{R}^{T}\to\mathbb{R}^{D}$ has the *whole* series in its receptive field, so each token
carries real temporal content rather than a single tick. The two diseases of the timestamp layout — no
temporal content per token, channel mixing smeared through norm — are both gone, and they are gone
because of the tokenization, not because of any extra machinery.

It is worth being precise about *why* the cross-variate score is the right object, because it is the
whole justification for the rung. In the timestamp layout the attention map answered "which instants
look alike," a quantity that is at best a proxy for periodicity and at worst noise. In the variate-token
layout the score between token $i$ and token $j$ is computed from two vectors that each summarize a full
ninety-six-step series, so the dot product reads off how the *temporal profile* of variate $i$ aligns
with that of variate $j$ — a learned, soft, lag-tolerant analogue of cross-correlation. That is exactly
the statistic a forecaster wants when the target leans on covariates: it says "the wind channel's
recent profile resembles a configuration that historically preceded this kind of move in the target,"
and the softmax-weighted value aggregation then writes the relevant covariate representations into the
target's token. None of this needs me to specify *which* channels matter; the attention learns the
coupling from data. And because the projection that builds each token sees the entire look-back, the
representation it summarizes is already temporally rich — the FFN inside each layer then refines that
per-variate temporal representation further. So the division of labor is clean and inverted from the
usual Transformer: attention does the *cross-channel* work, the FFN does the *within-channel temporal*
work, and neither steps on the other. That separation is the reason the stock encoder, unmodified,
suddenly becomes a competent multivariate model — the layout, not the layer, was the lever all along.

The forecast head is then the dual of the embedding: each variate token, after the encoder has let the
other channels write into it, is projected $\mathbb{R}^{D}\to\mathbb{R}^{\text{pred\_len}}$ back to a
ninety-six-step forecast for that channel. Because `c_out == enc_in` and I produce a token per channel,
the output already has the right shape and the harness slices the last (target) channel. I keep the
same per-instance normalization (subtract the look-back mean, divide by std, add back after) for the
same distribution-shift reason as the patch rung — applied per channel, so it does not smuggle in
coupling. The calendar features `x_mark_enc` can be folded in as extra tokens by the inverted embedding,
which is free covariate information. Configuration is the standard one: `e_layers=2`, `d_model=512`,
`d_ff=512`, `n_heads=8`, dropout 0.1. The full scaffold module is in the answer.

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
