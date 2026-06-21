Reading the four baselines together tells me what the ladder has and has not solved. DLinear set the affine floor (13.39 / 10.50 / 14.36). PatchTST added a learned nonlinear representation and instance normalization and beat it everywhere but only by tenths (12.97 / 10.22 / 13.68), because attention over two-to-four patch tokens on a $2\cdot\text{pred\_len}$ window is starved. TimeMixer added explicit multi-scale decomposition-mixing and won on every regime (12.80 / 10.21 / 13.38), its gains concentrated where trend and season separate across scales. TimesNet added FFT period-discovery and a 2D intra/interperiod layout and edged ahead overall (12.803 / 10.089 / 13.442) — but the win is *entirely Quarterly*: it took the cleanest-period regime from 10.21 to 10.089, essentially tied TimeMixer on Monthly, and came in *worse* on Yearly (13.442 vs 13.378), exactly where there is no period to discover and trend is the whole story. So the strongest baseline, at mean SMAPE about 12.11, is strong on periodic regimes and no better than a multi-scale MLP on the trend-dominated one — and across the whole ladder the Yearly number has barely moved from DLinear's 14.36 down to roughly 13.38–13.68, a far smaller relative gain than Monthly or Quarterly saw. The unsolved problem is sharp: every rung so far is a single architecture serving trend-only Yearly and strong-seasonal Monthly with *one* fixed inductive structure, and whichever structure it picks — pooling ladder, period FFT, patch attention — is mismatched to at least one regime. What I want is an architecture whose inductive bias is not "one lens" but *sequential refinement*: peel off whatever the previous part could not explain, so it adapts to a pure-trend series and a strongly seasonal one without a regime-specific lens.

I propose **N-BEATS**: a deep stack of fully-connected **blocks** wired by a **double residual** loop, each block constrained to emit a basis expansion. The output strategy is settled by the same direct-multi-step argument that has held since DLinear — emit the whole horizon at once, no roll-out, no error accumulation — and a feed-forward network whose output spans the horizon does this natively. The novelty is what goes between input and output. Instead of one network explaining the whole look-back at once (which spends capacity wherever the loss is largest — the trend-starves-season pathology DLinear's decomposition was a crude fix for), I build a *stack* in which each block reads the *residual* look-back — what earlier blocks have not yet explained — and outputs two things: a partial forecast (its contribution to the horizon) and a backcast (its reconstruction of the part of the look-back it just used). The double-residual loop is
$$r_b = r_{b-1} - \hat{x}_b, \qquad y \mathrel{+}= \hat{y}_b,$$
subtracting this block's backcast from the running look-back residual so the next block faces a cleaner signal, and accumulating this block's forecast. The backcast stream flows backward, peeling the input apart component by component; the forecast stream flows forward as a sum. This is the boosting/ResNet idea applied to the look-back: each block models only a small correction, so a deep stack stays trainable, and — the part that matters for the Yearly/Monthly tension — the decomposition is *emergent and per-series*, not a fixed pooling window or FFT lens. A pure-trend Yearly series and a strong-seasonal Monthly series are decomposed by whatever each block learns to peel off, so one architecture adapts to both.

Inside a block I want the output *constrained*, because the M4 series are short and an unconstrained per-block horizon would overfit training noise — the failure that sank naive MLPs on M4. So a block maps its residual input through a small FC stack (a few `Linear`+`ReLU` layers of width $W$) to a coefficient vector $\theta$, and the output is $V\cdot\theta$ for a fixed basis $V$. The **generic** basis is the identity — two linear maps, $\text{hidden} \to \text{seq\_len}$ for the backcast and $\text{hidden} \to \text{pred\_len}$ for the forecast — which is maximally flexible and is the raw-accuracy configuration I reach for here, because across three heterogeneous regimes I care more about accuracy than about reading off the components. (The **interpretable** alternative constrains the basis to a low-degree polynomial in trend blocks and a Fourier sine/cosine basis in seasonality blocks, so stacking a trend stack then a seasonality stack yields an *explicit* learned trend/season split — the same decomposition STL imposes by hand, here emergent. That variant trades a little accuracy for transparency; the generic stack already gets decomposition-by-refinement for free through the residual loop, so it is the right default for a leaderboard.) Either way the backcast and forecast share the block's hidden representation but use different basis sizes ($\text{seq\_len}$ vs $\text{pred\_len}$). The interpretable configuration additionally shares weights across blocks within a stack — a strong regularizer; the generic configuration I ship keeps the blocks *distinct* (the canonical generic factory), relying instead on the small theta dimension, the persistence seed, and the 10-epoch early-stopping cap for the regularization the short M4 series demand and the naive-MLP entrants lacked.

Two details the loop needs. First, reverse the look-back before feeding it in, so the most recent point sits in a stable leading position regardless of $\text{seq\_len}$ — recent values matter most, and I want them read consistently across the three regimes' different window lengths. Second, seed the running forecast with the *last observed value* ($x[:, -1:]$) rather than zero, so the network only learns the *deviation* from a persistence forecast. A naive-last baseline is already decent on short horizons, and starting the residual sum there makes each block's job a correction on top of persistence — both easier to learn and a sensible prior on the trend-dominated Yearly regime where the strongest baselines stalled.

Grounding in this harness, where I fill only `models/Custom.py` under the fixed protocol: it gives $\text{seq\_len}=2\cdot\text{pred\_len}$ univariate windows, calls `model(batch_x, None, dec_inp, None)` with no time marks (N-BEATS never wanted them — it reads only the raw look-back), optimizes SMAPE, and passes $d_{\text{model}}=512$, $e_{\text{layers}}=2$, $\text{batch}=16$, $\text{lr}=10^{-3}$, 10 epochs, patience 3. N-BEATS does not use $d_{\text{model}}/e_{\text{layers}}$ as an embedding/attention depth, so I repurpose what the harness offers: FC width $W=512$ — reusing the wide channels the previous rungs left stranded, here feeding a fully-connected block that can actually use width — a moderate number of generic blocks, 4 FC layers each. I squeeze the channel axis ($\text{enc\_in}=1$) to feed the $[\text{batch}, \text{seq\_len}]$ vector to the blocks and unsqueeze the $[\text{batch}, \text{pred\_len}]$ forecast back; the input mask is all-ones (M4 windows are dense), so the backcast subtraction is unmasked. There is **no ensembling** — the harness trains one model at one look-back for 10 epochs — which is the honest limitation: N-BEATS' published M4 result leans on ensembling over look-back lengths, inits, and losses to average away short-series variance, and a single un-ensembled fit under a 10-epoch cap will not reach that. The bar I clear is not the paper's ensembled number; it is this ladder's strongest baseline under this exact protocol.

The case for N-BEATS is strongest exactly where the ladder stalled. **Yearly** is the region of greatest hope: it is trend-dominated, the strongest baselines were no better than the multi-scale MLP there (13.442 / 13.378), and persistence-seeded residual refinement plus trend-friendly generic blocks are built for pure-trend short horizons — seeding the forecast at the last value and correcting it is exactly the right prior for six trend steps, so I expect to beat 13.44 here, the clearest opportunity of the three. **Quarterly** is the hardest to beat: TimesNet's FFT lock on the sharp 4-step period drove it to 10.089, and a generic FC stack with no explicit period basis must rediscover that periodicity under a 10-epoch cap — I expect to be competitive (around or slightly above 10.09) but would not be surprised to fall just short. **Monthly** I expect close to the 12.80 cluster: the residual stack can model the 12-step seasonality through its generic basis, but without the period prior TimesNet/TimeMixer encode, so a tie-to-slight-win is the realistic target. The finale's claim to the endpoint rests on a *mean* win driven by Yearly: if the trend-regime gain outweighs a possible Quarterly shortfall and Monthly is a wash, the persistence-seeded refinement architecture takes the ladder's mean SMAPE below TimesNet's about 12.11 — and it does so by being the first rung whose inductive bias is *adaptive refinement* rather than a single fixed lens, the exact thing the per-regime split in the baselines showed was missing. If instead Yearly does not move, that falsifies the bet that the unsolved regime was a refinement problem rather than a capacity problem, and would point back to ensembling — the one piece of N-BEATS' recipe this fixed protocol withholds — as the missing ingredient.

```python
import numpy as np
import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):
    """One block: FC stack -> theta -> identity (generic) basis split."""

    def __init__(self, input_size, theta_size, backcast_size, forecast_size, layers, layer_size):
        super(NBeatsBlock, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_size, layer_size)]
            + [nn.Linear(layer_size, layer_size) for _ in range(layers - 1)]
        )
        self.basis_parameters = nn.Linear(layer_size, theta_size)
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, x):                                # x: [B, input_size]
        h = x
        for layer in self.layers:
            h = torch.relu(layer(h))
        theta = self.basis_parameters(h)                # [B, theta_size]
        backcast = theta[:, :self.backcast_size]        # generic identity basis
        forecast = theta[:, -self.forecast_size:]
        return backcast, forecast


class Model(nn.Module):
    """N-BEATS (generic): double-residual stack of FC blocks, direct multi-step."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        layer_size = getattr(configs, 'd_model', 512)
        layers = getattr(configs, 'nbeats_layers', 4)
        n_blocks = getattr(configs, 'nbeats_blocks', 6)
        theta_size = self.seq_len + self.pred_len       # generic: backcast (L) + forecast (H)

        # generic config: a flat stack of distinct generic blocks (ServiceNow generic factory)
        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=self.seq_len,
                theta_size=theta_size,
                backcast_size=self.seq_len,
                forecast_size=self.pred_len,
                layers=layers,
                layer_size=layer_size,
            )
            for _ in range(n_blocks)
        ])

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # univariate: [B, L, 1] -> [B, L]
        x = x_enc[:, :, 0]
        residuals = x.flip(dims=(1,))                   # most recent first
        forecast = x[:, -1:]                            # seed at last observed value (persistence)
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast            # peel explained part (dense mask -> no scaling)
            forecast = forecast + block_forecast
        return forecast.unsqueeze(-1)                   # [B, H, 1]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]       # [B, H, 1]
        return None
```
