Patched temporal attention resolved the fork: PatchTST dropped ETTh1 to MSE $0.3794$ / MAE $0.3986$ — below the linear control's $0.3962$, on the dataset the linear model was strongest on — which confirms that attention over time, when the token is a local patch shape rather than a meaningless single step, does extract more than a linear temporal map. Weather fell hard ($0.1962\to 0.1738$) and the mean MSE went $0.2676\to 0.2451$. But ECL is the diagnosis: $0.2104\to 0.1819$, improved yet now the *relative laggard* of the three, with the worst MAE ($0.2743$) by a wide margin. ECL is the 321-channel dataset, and PatchTST, like the linear control, is channel-independent — it never lets one channel inform another. The temporal lever is now well used; the channel lever is untouched. So the next rung is not a better temporal operator but cross-variate structure — and the question is how to add it without throwing away what just worked.

The reflex is to keep the temporal-token layout and bolt channel mixing on top, but I think the layout itself is the bug. In the standard layout I have a panel $X\in\mathbb{R}^{T\times N}$ of $T$ timesteps and $N$ variates, and the token is the slice $X_{t,:}\in\mathbb{R}^N$ — all $N$ variates at one instant — embedded into a $d_{\text{model}}$ vector, giving $T$ tokens and self-attention over time. But look inside one of those tokens. A word token is a coherent semantic unit; $X_{t,:}$ is whatever every sensor happened to read at the same wall-clock instant — in Weather, temperature next to rainfall next to pressure, different physical quantities and units jammed into one vector, and worse, an event hits one channel and only later the next, so the "same timestamp" lumps together different phases of the same process. The temporal token is a fruit salad of time-misaligned, incommensurable numbers with a single-instant receptive field; the temporal content lives *across* tokens, and the model must reconstruct it from a sequence of scrambled snapshots. This layout also makes LayerNorm harmful: it normalizes across the feature dimension of a token, which here is the variate mixture at fixed $t$, so it centers temperature against rainfall against pressure together — injecting interaction noise between unrelated, possibly lagged processes. And there is a cleaner argument that the time axis is the wrong place for attention at all: attention is permutation-invariant in its tokens, which is exactly why language bolts on positional encodings, but time *has* order, so a permutation-invariant operator on the temporal axis is a structural mismatch we then patch with positional encodings just to undo. PatchTST mitigated this by making each token a local patch, but it still ran attention along time and still refused channels.

So I propose **iTransformer**: invert the token. The axis I should not permute is time; the axis *without* inherent order, where permutation invariance is *correct*, is the set of variates — channel 5 and channel 12 have no canonical ordering. So make the token the whole series of one variate. Take the column $X_{:,n}\in\mathbb{R}^T$ — the entire look-back of channel $n$ — and embed *that* into a $d_{\text{model}}$ vector $h_n$ with one linear map $\mathbb{R}^T\to\mathbb{R}^D$. Now there are $N$ tokens, one per variate, each describing a single physically coherent series over its whole history, with a receptive field of the entire window. This is the extreme of patching: PatchTST grouped a handful of consecutive steps into a token; push that all the way and the patch is the complete series. Crucially, attention between two of these tokens, $h_i\cdot h_j$, is now a clean cross-variate correlation — how channel $i$'s whole history relates to channel $j$'s — exactly the structure ECL needs and which neither the linear control nor PatchTST could touch.

Three consequences fall out, each a fix to a harm I named. First, attention now runs over the unordered variate set, so permutation invariance is *correct* and no temporal positional encoding is needed — the mismatch is gone, not patched. Second, LayerNorm now normalizes across the feature dimension of a *variate* token, the learned representation of one channel's series — a sensible per-series normalization, not a cross-channel blend — so the interaction noise is removed by construction. Third, the cost is $O(N^2)$ in the number of *variates*, independent of $T$; lengthening the look-back only widens the embedding's input, so longer history is cheap, the opposite of the temporal-token regime. The division of labor inverts the standard one and follows from the layout: attention mixes variate tokens (cross-variate correlation), and the position-wise feed-forward then acts *within* each variate token as the per-series temporal feature extractor, learning a nonlinear representation of that channel's history from its $d_{\text{model}}$ embedding, shared across channels. The head keeps the direct multi-step generation of every rung: one linear map $d_{\text{model}}\to \text{pred\_len}$ applied to each variate token, transposed to $[B, \text{pred\_len}, N]$, no decoder. I wrap it in the same reversible per-window instance normalization PatchTST used, since variate tokens still need level/scale drift removed before embedding. The edit-surface gap shapes what to expect on the small dataset: the method's own script tunes $d_{\text{model}}=128$ on ETTh1 (only 7 channels, so 7 tokens), but the fixed scaffold config is $d_{\text{model}}=512$, $e_{\text{layers}}=2$ for all three — badly over-parameterized on ETTh1, where 7 weakly coupled channels give the cross-variate attention almost nothing to find and a lot of capacity to overfit, so I expect the inversion to pay on ECL's 321 richly coupled channels and not on ETTh1.

```python
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """iTransformer: attention across variate tokens; FFN as the per-series temporal extractor."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # reversible per-window instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)        # [B, N(+marks), d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]   # [B, pred_len, N]
        # de-normalize
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
