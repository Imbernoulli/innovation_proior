The patch Transformer did its job as a control. ETTh1 dropped to MSE 0.058292 / MAE 0.182538, beating DLinear's 0.0644 / 0.1878, and since ETTh1 is where cross-channel fusion has the least to add, that gain can only be the per-channel temporal model finally being strong enough — confirming the temporal model was part of DLinear's loss everywhere. Weather went to 0.001652 / 0.029337 and ECL to 0.317887 / 0.394697, both big drops. But the temporal-model confound is now gone, so whatever distance remains to the achievable floor on Weather and ECL is the cost of *not reading the other channels*: PatchTST still folds the channel axis into the batch, the covariates still never touch the target. That residual is the fusion gap, isolated, and the next rung is the first to go after the actual research question.

There is a trap to avoid — the same one that let the linear map beat the point-wise Transformers. The naive way to "add cross-channel attention" is to go back to a token-per-timestamp layout: embed the $N$-variate slice $\mathbf{X}_{t,:}$ at each instant into a $D$-vector, get $T$ temporal tokens, and run self-attention over them. That layout is *worse* for fusion than it looks. One such token jams together whatever every sensor read at the same wall-clock instant — temperature next to humidity next to pressure next to wind, different units and distributions — and they are not even time-aligned in the way that matters, since a front hits the pressure sensor minutes before the wind sensor responds. Worse, the LayerNorm in that layout normalizes across the feature dimension of a token, which here is the variate mixture at fixed $t$, so it centers and scales temperature against humidity against pressure together, injecting interaction noise between unrelated, lagged processes. And the attention map over $T$ temporal tokens tells me which *instants* resemble which — not which *variables* drive which, which is the structure I actually want.

I propose **iTransformer**: cross-variate attention by inverting the token. Patching split one channel's time axis into many tokens; for fusion I want the dual move — collapse one channel's *entire* time axis into a single token, so that a token *is a variate*. Take channel $i$'s whole look-back $\mathbf{X}_{:,i} \in \mathbb{R}^T$ (all ninety-six steps) and map it with one linear projection $\mathbb{R}^T \to \mathbb{R}^D$ into a single $D$-dimensional variate token. Do that for every channel, get $N$ tokens, and run the *stock* self-attention over them. The attention score between channel $i$'s token and channel $j$'s token is then a similarity between the full temporal profile of variate $i$ and the full temporal profile of variate $j$ — a clean **cross-variate correlation**. Finally the side channels can influence the target, because the target is one token among $N$ and attention lets every other variate token write into it. I keep the entire stock encoder unchanged — `DataEmbedding_inverted` for the variate tokens, then `Encoder`/`EncoderLayer`/`FullAttention`/`AttentionLayer` — and only the *meaning* of a token changes. That is the elegance: no new attention kernel, just the right tokenization for the structure I want.

This inversion is also what cures the LayerNorm complaint, and that is central, not incidental. In the variate-token layout, LayerNorm normalizes across the feature dimension of a *variate* token — the learned representation of one channel's series — which is exactly per-variate normalization that removes that channel's distribution shift without mixing channels. The cross-channel mixing now happens *only* in the attention scores, where it is wanted. And because the projection $\mathbb{R}^T \to \mathbb{R}^D$ has the whole series in its receptive field, each token carries real temporal content rather than a single tick.

It is worth being precise about *why* the cross-variate score is the right object, because it is the whole justification. The score between token $i$ and token $j$ is computed from two vectors that each summarize a full ninety-six-step series, so the dot product reads off how the temporal profile of variate $i$ aligns with that of variate $j$ — a learned, soft, lag-tolerant analogue of cross-correlation. That is exactly the statistic a forecaster wants when the target leans on covariates: it says "the wind channel's recent profile resembles a configuration that historically preceded this kind of move in the target," and the softmax-weighted value aggregation writes the relevant covariate representations into the target's token. Nothing specifies *which* channels matter; the attention learns the coupling from data. The division of labor is clean and inverted from the usual Transformer: attention does the cross-channel work, the FFN inside each layer does the within-channel temporal work, and neither steps on the other — which is why the stock encoder, unmodified, suddenly becomes a competent multivariate model. The layout, not the layer, was the lever.

The forecast head is the dual of the embedding: each variate token, after the encoder has let the other channels write into it, is projected $\mathbb{R}^D \to \mathbb{R}^{\text{pred\_len}}$ back to a ninety-six-step forecast for that channel. Since `c_out == enc_in` and I produce a token per channel, the output already has the right shape and the harness slices the target. I keep the same per-instance normalization (subtract the look-back mean, divide by std, add back after) for the same distribution-shift reason, applied per channel so it smuggles in no coupling. The calendar features `x_mark_enc` fold in as extra tokens via the inverted embedding — free covariate information. Configuration is the standard `e_layers=2`, `d_model=512`, `d_ff=512`, `n_heads=8`, dropout 0.1.

I should be honest about the trade this rung makes, because it predicts exactly where it might not win. Crushing each channel's whole series into one token with one linear map throws away the fine intra-series temporal detail patching preserved — the local ramps and bumps. So where the target's *own* fine temporal structure dominates and cross-channel coupling is weak, this rung could do *worse* than PatchTST even though it can now read other channels. ETTh1 is that case. I expect iTransformer to beat PatchTST on Weather (covariates genuinely drive the target — the fusion gap closing) and on ECL MSE (321 channels of cross-client structure), to roughly tie or slightly lose on ETTh1, and on ECL MAE to be the riskier metric — the $O(N^2)$ fusion is indiscriminate, so most of its compute models interactions into channels I never score while their noise flows into the target's token. That pattern — fusion helps where covariates matter and costs where they don't — is exactly the finding the research question is after, and it writes the next rung's job: keep the per-channel temporal resolution PatchTST gave *and* the cross-variate reach this rung gives, and stop spending quadratic compute and noise on channels the target never needs.

```python
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """Cross-variate attention forecaster (iTransformer): one token per channel."""

    def __init__(self, configs):
        super(Model, self).__init__()
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
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # per-instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        # one token per variate (whole series -> D); attention runs across channels
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]   # [B, pred_len, N]

        # de-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
