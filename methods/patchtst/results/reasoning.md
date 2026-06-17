Let me start from the thing that actually stings. I have a stack of Transformer forecasters — Informer, Autoformer, FEDformer, Pyraformer — each one a clever surgery on the attention kernel to beat down its quadratic cost: ProbSparse here, auto-correlation and decomposition there, a Fourier block, a pyramid of edges. And a linear model — literally a per-channel linear map from the look-back window to the horizon, with a trend/seasonal split — matches or beats all of them on the standard long-horizon benchmarks. That is not a small embarrassment; it undercuts the premise the whole line was built on. The premise was "the problem with Transformers for forecasting is that O(L^2) attention is too expensive, so let's make attention cheaper." But if a plain linear projection wins, then either self-attention is the wrong tool for this data, or — and this is the possibility I want to chase — attention is fine and we have been feeding the series into it in a way that destroys exactly the structure attention is good at finding. So I refuse to touch the attention kernel. I want to ask instead: what is the *token*, and is the token the bug?

Look at what a token is in every one of these models. It is a single time step. Either one scalar at time t for a univariate series, or the full M-channel vector at time t for a multivariate one. Now sit with that for a second, because I think it is genuinely broken. In NLP a token is a word or a sub-word — it has standalone meaning, attention between two words is comparing two semantic units. What is the meaning of the value of a sensor at 14:03? Nothing, in isolation. A single time step has no more semantic content than a single character — less, really, because a character at least belongs to a small alphabet. The information in a time series lives in *shapes over short stretches*: a rising edge, a dip, a local oscillation, the slope of a ramp. Point-wise attention asks "how does the scalar at 14:03 relate to the scalar at 09:17?" and the answer is almost always noise, because neither scalar means anything by itself. The attention map is being computed over the wrong objects. That would explain the linear model winning: the linear model isn't doing point-wise comparison at all; it is reading the whole window at once through learned weights, so it gets to see the *shape*. Attention's failure here might be a tokenization failure, not an attention failure.

And there's a second symptom pointing the same way. Token count N equals sequence length L when every step is a token, so the attention map is L×L and lengthening the look-back is quadratically punished. Yet I know — it's a reported fact about these datasets — that a longer look-back genuinely helps: pushing L from 96 to 336 drops MSE from around 0.52 to around 0.40 on a representative case. So there's information sitting in the older history that the model would benefit from, but the O(L^2) cost makes everyone default to short windows and throw that information away. Worse, people have tried to *keep* a long window cheaply by down-sampling — take every fourth step, feed a shorter token sequence — and it still forecasts well, sometimes better than the most-recent short window at the same token count. Read that carefully: it says the time axis is *redundant*, that neighboring steps carry overlapping, compressible structure rather than independent information. So the natural move is staring at me. If neighboring steps are redundant and a single step is meaningless, don't tokenize per step. Group a local stretch of steps into one token.

This is exactly what vision did when it hit the same wall. An image has H×W pixels; one pixel is meaningless and per-pixel attention is O((HW)^2), hopeless. The Vision Transformer's answer was to cut the image into 16×16 patches and call each patch a token — suddenly each token carries a local visual concept and the token count collapses by 256×. The analogy to a series is immediate: cut the length-L series into contiguous sub-series patches, and let each patch be a token. A patch of, say, sixteen consecutive time steps is a little shape — it can encode a ramp, a bump, a level — which is the kind of thing attention should be comparing. Let me make this concrete. Take the i-th univariate series x^(i) of length L. Pick a patch length P and a stride S — the hop between the starts of consecutive patches. Slide a window of width P along the series in steps of S, and each placement gives me one patch, a vector in R^P. The number of patches is N = floor((L - P)/S) + 1 by the usual count of how many length-P windows of stride S fit in length L.

Wait — I want to be careful at the boundary, because the last few steps of the series matter most for forecasting and I don't want the windowing to drop them. If (L - P) isn't a multiple of S, the last window doesn't reach x_L, and the most recent value is precisely the one I least want to lose. So before patching I pad the end: append S repeated copies of the last value x_L. Padding by exactly the stride guarantees one more full window slides into existence that includes the true end, and it adds exactly one patch. So the count becomes N = floor((L - P)/S) + 2. Let me sanity-check with the numbers I'll actually use. With L = 336, P = 16, S = 8: floor((336 - 16)/8) + 2 = floor(320/8) + 2 = 40 + 2 = 42 patches. With L = 512: floor(496/8) + 2 = 62 + 2 = 64. Good — clean, and it explains why I'd label two configurations "/42" and "/64" by their patch counts.

Now the payoff, which I should pin down quantitatively rather than wave at. The attention map is N×N, cost O(N^2 D) per layer. Without patching N = L. With patching N ≈ (L - P)/S + 2 ≈ L/S for L ≫ P. So I've cut the token count by a factor of S, and since attention is quadratic in N, the time and memory of the attention map drop by a factor of about S^2. With S = 8 that's roughly a 64× reduction in the attention cost. That is the whole game changing: the quadratic-in-L wall that everyone else attacked by mutilating the attention kernel, I just walked around by changing what a token is. And the saving compounds with the earlier observation — because patching makes a long L affordable, I can now *afford* the long look-back that the data wants, instead of being forced to L = 96. So patching buys me three things at once that I should keep straight: local semantic tokens (the thing attention can actually use), an S^2 cut in attention cost, and the headroom to feed a longer, more informative history. I notice the stride S should probably be a bit smaller than P so consecutive patches overlap and no edge shape gets split cleanly down the middle between two patches; S = 8 with P = 16 means each patch overlaps its neighbor by half, which keeps local continuity. And the patch-length choice: too small and I'm back toward point tokens with little local shape; too large and N shrinks so far that attention has almost nothing to attend over and I lose temporal resolution. Something in the 8–16 range feels right as a balance, and that's borne out as robust across patch-length sweeps.

So one piece of the input is settled: patch each series, project each patch to the model dimension. But there's a second, independent decision I've been deferring, and it's the other half of why the linear model wins. How do I handle the M channels? The standard multivariate Transformer mixes them: at each time step it takes the whole M-vector and projects it jointly into one token, so cross-channel information is fused at the input and every channel ends up sharing one attention pattern. The linear model that's beating everyone does the opposite — it processes each channel with its own (shared-form) linear map, completely independently. So the empirical fact is right there: channel mixing is *not* helping; the channel-independent linear model wins. Let me think about why mixing should be worse, because my instinct says the opposite — surely a model that *can* see cross-channel correlations is strictly more powerful?

It's more powerful in capacity, yes, but that's exactly the trap. Three things go wrong with mixing, and I can reason each one out. First, adaptability. If I mix channels into one token stream, the attention map is computed once over those mixed tokens and *all* channels are forced to live under that single attention pattern. But different channels can have completely different temporal behavior — one is a slow trend, another a sharp daily cycle, another near-noise. A single shared attention pattern can't be right for all of them at once; it's a compromise. If instead each channel goes through the backbone separately, each series produces its *own* attention map, free to attend to whatever lag structure suits it. Correlated series will naturally end up with similar maps; unrelated series with different ones — and that's the model adapting per series rather than averaging. Second, data efficiency. Learning genuine cross-channel correlation *jointly* with the temporal structure is a much bigger hypothesis space — you're learning interactions across both channels and time at once — and that needs a lot of data to pin down. These benchmark datasets are not large; the M4 series in particular are short. The channel-independent model only has to learn structure along the time axis, a far smaller space, so it converges with far less data. The flexibility of mixing is a double-edged sword: more expressive, but data-hungry on exactly the datasets we have. Third, and this is the one I can see most directly, overfitting. With mixing, the model can fit spurious cross-channel coincidences that hold in the training window and don't generalize — and indeed the mixing models are observed to overfit after just a few epochs while the independent ones keep improving. So: process each channel independently.

But "independently" raises a worry: if every channel has its own separate weights, I have M times the parameters and I learn nothing shared across series — and the linear model that wins actually *shares* its weight form across channels. So the right design isn't M separate models, it's one model applied M times: a single Transformer backbone with one set of weights, run independently on each univariate series. Channel-independent in the forward pass, weight-shared in the parameters. This is a *global univariate model* — it sees each series alone, but it learns from all of them through the shared weights. And this has a lovely structural consequence I should note: because the weights don't know or care how many channels there are, the number of series at training time need not match the number at test time. The same backbone trained on a dataset with one channel count can be applied to another. That's free generality, and it matters for transfer and pretraining later. It also tells me the univariate case isn't a special case I bolt on — it's the *native* case. A many-series univariate benchmark like M4, where every series is its own channel with one variable, is just this model with M = 1 per forward pass, the shared weights learning across all the series at once.

Let me check the implementation cost of channel-independence, because "run the backbone M times" sounds expensive. It isn't, and it needs no special operator. Start with the batch x of shape [B, M, L]. Patch it: that produces, per series, N patches of length P, so a 4-D tensor [B, M, N, P]. Now just *fold the channel axis into the batch axis*: reshape to [B·M, N, P]. From the Transformer's point of view this is simply a batch of B·M independent length-N token sequences, each token a length-P vector — a perfectly ordinary input that any standard Transformer consumes. Run the encoder, get [B·M, N, D], reshape back to [B, M, N, D] at the end. So channel-independence costs nothing but a reshape; the M copies of the weights are conceptual, the actual weight tensor is shared and the channels ride in the batch dimension.

Now the backbone itself. I keep insisting it's a *vanilla* Transformer encoder because the whole thesis is that the input representation, not the attention kernel, was the problem — so I must not sneak in a fancy kernel or I'd muddy what's responsible. So: take the patches x^(i)_p ∈ R^{P×N} for series i, map each patch into the model space of dimension D with a trainable linear projection W_p ∈ R^{D×P}, giving x^(i)_d = W_p x^(i)_p ∈ R^{D×N}. No bias on this projection — it's a pure patch embedding, like ViT's, and an instance-normalized patch already has its level removed so a per-patch additive offset buys nothing. The patches are now an unordered *set* as far as attention is concerned — attention is permutation-invariant — but their temporal order is everything in a time series. So I add a positional encoding W_pos to mark which patch came first: x^(i)_d = W_p x^(i)_p + W_pos, one position vector per patch slot. Then standard multi-head self-attention over the N patch tokens: for each head h, Q^(i)_h = (x^(i)_d)^T W^Q_h, K^(i)_h = (x^(i)_d)^T W^K_h, V^(i)_h = (x^(i)_d)^T W^V_h, and the attention output is Softmax(Q K^T / sqrt(d_k)) V, the usual scaled dot-product — divided by sqrt(d_k) so the logits don't grow with head dimension and saturate the softmax into near-one-hot. Multiple heads so different heads can lock onto different lag relationships among the patches. Then the position-wise feed-forward sublayer, D → F → D with a nonlinearity, residual connections around both sublayers. Standard, deliberately.

One choice inside the encoder I should not make on autopilot: where does the normalization go and which kind? The default Transformer uses LayerNorm, normalizing across the feature dimension per token. But for time-series tokens that's risky, because time series have outliers — a spike, a sensor glitch, a regime jump — and an outlier time step landing in a patch will skew that token's per-token statistics, and LayerNorm, computing its mean and variance *within* the token, will be dragged around by it. BatchNorm instead normalizes each feature across the batch (here, across the B·M·N patch positions), so a single outlier patch is diluted by all the others rather than corrupting its own normalization. This is the kind of thing that's easy to get wrong and it's been measured: BatchNorm beats LayerNorm for time-series Transformers. So I'll use BatchNorm in the encoder. Mechanically that's a small wrinkle — BatchNorm1d wants the feature axis in position 1, so I transpose to put D there, BatchNorm over D, transpose back.

Now the head. I have, per series, the encoder output z^(i) ∈ R^{D×N}, and I need to turn it into a forecast of the next T values, x̂^(i) ∈ R^{1×T}. The simplest faithful thing: flatten the D×N representation into a single D·N vector and apply one linear layer to T. Flatten-and-project. I briefly worry this could be a huge matrix — but per series it's only (D·N) × T, which is modest, and crucially it is *per series and shared across series*, not the monstrous (L·D) × (M·T) matrix that a point-wise channel-mixing model needs to map all L step-representations of all M channels to all outputs jointly. That oversized joint head is exactly a thing that overfits when downstream data is scarce; by being channel-independent and patch-based I've sidestepped it. So: a flatten head, D·N → T, per series, weights shared across series. The width D·N is fixed by the patch count: D·N = D·(floor((L−P)/S)+2).

There's one more thing the data forces on me before the architecture is done, and it's the train/test distribution shift. These series are non-stationary — the mean and scale of a window drift over time, and a test window can sit at a level the training windows never visited. If I feed raw values, the model has to simultaneously learn the shape *and* track an offset/scale it has no stable handle on. The clean fix is to normalize each instance: for each univariate series in the window, subtract its mean and divide by its standard deviation, before patching; then, because the forecast comes out in normalized units, multiply it back by that std and add the mean back. Per-instance, reversible. Concretely, over the look-back axis: mean = (1/L) Σ_t x_t, then center x ← x − mean, then stdev = sqrt( (1/L) Σ_t x_t^2 + eps ) with eps ≈ 1e-5 to keep a flat series from dividing by zero (a biased/uncorrected variance — I want the actual scale of the window, not an unbiased estimator), then x ← x / stdev. Detach both statistics — they're normalization constants, not parameters to backprop through. The encoder then always sees roughly zero-mean unit-variance shapes regardless of where the window sits, which decouples shape-learning from level-tracking and directly counters the drift. At the very end, the forecast x̂ is denormalized: x̂ ← x̂ · stdev + mean, broadcast over the T forecast steps. This is a modest accuracy gain on its own — the bulk of the win is patching plus channel-independence — but it's cheap and consistent, so it stays.

Now the loss. For the long-horizon MSE benchmarks the objective is the obvious squared error between forecast and ground truth, averaged over channels: L = E_x (1/M) Σ_i ‖x̂^(i) − x^(i)‖_2^2. But the short, univariate, many-series setting — M4 — is judged by a *percentage* error, SMAPE, the symmetric absolute percentage error (200/T) Σ_t |y_t − ŷ_t| / (|y_t| + |ŷ_t|), and the right thing is to train on the metric I'm scored by rather than on raw MSE, because the series live at wildly different magnitudes and a squared error would let the few large-magnitude series dominate the gradient. So on M4 the training loss is SMAPE itself. Nothing about the architecture changes — the head still projects to T, the model is still univariate-per-forward — only the objective swaps. And the M4 config is small for a reason that follows directly from the channel-independence reasoning: the series are short and there are no cross-channel correlations to learn (M = 1), so a big model would just overfit; a couple of encoder layers, a handful of heads, a small d_model, patch_len 16 and stride 8, Adam at 1e-3 for about ten epochs is the right scale.

Now step back: have I actually got something, or just rearranged furniture? The provocation was a linear model beating every complex Transformer, suggesting attention was useless for this data. My claim is the opposite — attention was being fed garbage tokens. Patching gives attention meaningful local-shape tokens *and* cuts its cost by S^2 *and* affords a longer look-back. Channel-independence with shared weights matches what the winning linear model does on the channel axis — per-series adaptability, data efficiency, less overfitting — while keeping the Transformer's nonlinear depth that the linear model lacks. So I'd expect to both vindicate the vanilla Transformer *and* surpass the linear model, getting the linear model's robustness plus the representation capacity it can't offer.

And that representation capacity is worth chasing one step further, because it's the thing a linear model can never give. With the patch token in hand, masked self-supervised pretraining becomes natural and, more importantly, *non-trivial*. The earlier time-series attempt masked individual time steps — but that's a weak signal, because of the very temporal redundancy I leaned on for patching: a single masked step can be reconstructed by interpolating its immediate neighbors, no global understanding required, so the model never has to capture the whole signal's structure. The fix falls out of the patch design: mask whole *non-overlapping* patches at random (non-overlapping so a visible patch can't leak a masked neighbor's contents) and train a D → P head to reconstruct the missing sub-series shapes. Reconstructing an entire missing patch from the surrounding patches genuinely requires understanding the signal, so the learned representation carries real structure — and because the backbone is channel-independent with shared weights, the pretraining data can even have a different channel count than the downstream data, pretrain broadly, fine-tune narrowly, the global-univariate generality paying off exactly as I flagged. For the supervised forecasting task in front of me I don't need that stage — I train end to end to forecast — but it's the same architecture with the head swapped, and it's why the patch token is the load-bearing choice.

Let me write it as the code I'd ship, filling the empty slots in the harness — the instance normalization, the input representation, the encoder, and the head — with everything I just derived, channels folded into the batch axis.

```python
import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    # BatchNorm1d wants the feature axis at dim=1; we transpose D into place and back.
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    # per-series head: flatten the D x N representation, linear-project to the horizon T
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)           # (D, N) -> (D*N)
        self.linear = nn.Linear(nf, target_window)         # (D*N) -> T
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs, nvars, D, N]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name           # e.g. 'short_term_forecast'
        self.seq_len = configs.seq_len               # L
        self.pred_len = configs.pred_len             # T
        padding = stride                             # pad S copies of the last value (one extra patch)

        # patch + linear-project each patch to D (no bias) + add positional encoding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # vanilla Transformer encoder over the N patch tokens; BatchNorm (not LayerNorm)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),   # multi-head scaled dot-product
                    configs.d_model,
                    configs.d_ff,                            # FFN D -> F -> D
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2),
                                     nn.BatchNorm1d(configs.d_model),
                                     Transpose(1, 2))
        )

        # head input width = D * (number of patches N); N = (L - P)/S + 2
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        # forecasting: flatten the encoded patches and project to the T-step horizon
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # reversible per-instance normalization against train/test distribution shift
        means = x_enc.mean(1, keepdim=True).detach()                       # mean over look-back
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # window scale
        x_enc /= stdev

        # channel-independence: put channels on dim 1, fold them into the batch inside patch_embedding
        x_enc = x_enc.permute(0, 2, 1)               # [bs, nvars, L]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [bs*nvars, N, D]

        # vanilla encoder over patch tokens
        enc_out, attns = self.encoder(enc_out)         # [bs*nvars, N, D]
        # unfold channels back out: [bs, nvars, N, D] -> [bs, nvars, D, N]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # per-series flatten head -> horizon T, then back to [bs, T, nvars]
        dec_out = self.head(enc_out)                   # [bs, nvars, T]
        dec_out = dec_out.permute(0, 2, 1)             # [bs, T, nvars]

        # de-normalize: undo the instance norm on the forecast
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]      # [bs, T, nvars]
        return None
```

with the patch-embedding slot being exactly the tokenizer I argued for — pad S copies of the last value, unfold into length-P patches, fold channels into the batch, linear-project each patch to D with no bias, add a positional embedding, dropout:

```python
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))      # repeat last value S times
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)  # W_p: R^P -> R^D
        self.position_embedding = PositionalEmbedding(d_model)            # W_pos
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                              # x: [bs, nvars, L]
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [bs, nvars, N, P]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # [bs*nvars, N, P]
        x = self.value_embedding(x) + self.position_embedding(x)           # project + position
        return self.dropout(x), n_vars
```

The causal chain, start to finish. A trivial linear model beat every complex Transformer, which undercut the field's premise that the problem was attention's O(L^2) cost and pointed instead at how the series was fed in. The token was a single time step, which carries no standalone meaning and gives attention nothing locally coherent to compare, while forcing N = L and a quadratic window cost; and channels were mixed into one token, forcing all series under one attention pattern. So I made the token a sub-series patch — local shapes attention can actually use, token count cut to ~L/S so attention cost drops ~S^2, and a longer informative window made affordable — with a padded last value so the most recent step is never dropped and N = floor((L−P)/S)+2. I made the channels independent but weight-shared, a global univariate model, for per-series adaptability, data efficiency, and less overfitting, implemented free by folding channels into the batch axis — which makes a one-variable-per-series benchmark like M4 the native case, not a special one. The backbone stayed a deliberately vanilla Transformer encoder — to show the input was the fix, not the kernel — with scaled dot-product multi-head attention over patches, a learnable positional embedding because patches are otherwise an unordered set, BatchNorm rather than LayerNorm because outlier time steps corrupt per-token statistics, and a per-series flatten-and-project head to the horizon T. Reversible per-instance normalization wraps the whole thing to undo train/test distribution shift. The loss is MSE on the long-horizon benchmarks and SMAPE on M4, where training on the percentage metric keeps large-magnitude series from dominating; the small M4 config follows from the same channel-independence reasoning, since short single-variable series want a small model. And the patch token is the load-bearing choice not only for forecasting but because masking whole patches, not steps, is what later lets this same channel-independent encoder learn transferable structure the linear model could never produce.
