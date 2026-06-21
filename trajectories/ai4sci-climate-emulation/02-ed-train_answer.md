The heteroskedastic rung came in exactly where I feared. At the long budget HSR sits at NMSE $0.639$, R² only $0.361$, and the breakdown is the tell: `ml_nmse` $0.655$ against `sl_nmse` $0.088$. The eight single-level diagnostics are predicted well, but the 360 multi-level tendency profiles — the actual physics — are barely better than predicting their own mean, and the model does not climb much with budget ($0.745 \to 0.673 \to 0.639$). The diagnosis holds: the inverse-variance weighting fired before $\mu$ was any good, so the network explained away its underfitting of the hard, high-variance boundary-layer tendencies as "noise," down-weighted exactly those columns, and froze them out — and the uncertainty machinery bought nothing on the point metric. The lesson is not "uncertainty was a bad idea" but that on a *flat* backbone the probabilistic readout was a distraction from the thing that actually limits skill: the flat MLP treats the 556-vector as an unordered bag and leaves the largest gains on the table. So I want to attack the architecture now, not the loss — and I want a structural prior that gives a compact, committed handle on the map rather than half a million unstructured weights, much of which HSR's flat $0.65$ ml_nmse says are not being spent on the multi-level physics.

The classical way to get a low-dimensional, inspectable summary of a high-dimensional state is to project onto its leading variance directions — PCA — and regress on those. But PCA is linear, and the convection map is sharply nonlinear; a linear projection throws away most of the predictable signal and underfits the tendencies, the same failure as any linear reduced model. I want the *nonlinear* analogue of "squeeze the state through a few numbers," and that is an autoencoder. The sanity check that the framing is right: strip the nonlinearity out, make every activation the identity, and an autoencoder collapses back to a linear subspace projection — PCA again — so the nonlinear version is exactly the generalization I want. But a vanilla autoencoder reconstructs its *input*, while my problem is a map *across* domains, from the 556-dim state $X$ to the *different* 368-dim tendencies $Y$. So I propose the **encoder-decoder (ED)** with a hard low-dimensional bottleneck: encode $X$ down to a small code $z$ through a tapering nonlinear funnel, then decode $z$ up to $Y$, trained on plain MSE between predicted and true $Y$ — an autoencoder whose decoder targets something other than what the encoder ingested.

The design question that matters is how small the bottleneck can be without wrecking skill, and that is where a physics hypothesis becomes a measurement. Reduced-complexity atmospheric models — multi-cloud schemes, the quasi-equilibrium tropical circulation model — are all built on the premise that the number of degrees of freedom that really govern convection is *small*, far smaller than the raw state vector. If that holds for the learned map, a tiny bottleneck should still predict $Y$ well, and the width where skill starts to fall off is itself a measurement of the effective dimensionality. So the bottleneck width is an *instrument*, not just a hyperparameter. I set it to a **5-node latent** — aggressive, small enough that a handful of physical factors (large-scale temperature/humidity structure, radiative forcing, convective intensity) could each occupy a coordinate if the data support that separation, but a genuinely brutal compression. It is the opposite extreme from HSR's wide flat body: if HSR's problem was an over-wide unstructured trunk, a 5-node bottleneck is the maximal structural commitment in the other direction, and the reconstruction loss will adjudicate the dimensionality claim immediately.

The funnel descends smoothly so no single layer has to do a violent squeeze, and it starts *wider* than the input so the first layers can lift $X$ into a richer representation before compressing. The encoder is six layers, $556 \to 768 \to 512 \to 384 \to 256 \to 128 \to 5$, with the final projection into the code kept *linear* so the latent is an unconstrained continuous bottleneck rather than one squashed by a nonlinearity; the decoder runs the funnel symmetrically backwards, $5 \to 128 \to 256 \to 384 \to 512 \to 768 \to 368$, symmetric because there is no reason to shape expansion differently from compression. The widths follow the established ClimSim ED recipe (768/512/384/256/128/64) and the latent is held at the published 5, so the taper is reused, not invented. Each hidden block is $\text{Linear}\to\text{LayerNorm}\to\text{ELU}\to\text{Dropout}(0.1)$ — LayerNorm to keep activations conditioned through the deep stack, dropout for regularization — and the nonlinearity is decided by the output. The targets are normalized tendencies and fluxes, real-valued and frequently *negative*: cooling, drying, downward fluxes are negative numbers. A ReLU output would clip every negative to zero, so the network literally could not represent a cooling tendency; the output projection stays linear, and the hidden units use ELU — identity for positive inputs, $\alpha(e^x-1)$ for negative ones — which pushes mean activations toward zero, dodges the dying-ReLU problem, and offers a mild sensible bias on signed data.

I deliberately leave out two extensions the autoencoder framing tempts. The first is making the decoder reconstruct $X$ alongside $Y$, to force the latent to encode the climate state and not just the convective response — but the leaderboard scores NMSE on $Y$, not the interpretability of $z$, so spending output width on rebuilding $X$ trades the metric for a diagnostic I am not graded on. The second, and the more tempting one given HSR, is to make the latent *probabilistic* — emit a mean and log-variance per code coordinate, sample with the reparameterization trick, add a KL to an $\mathcal N(0,I)$ prior, anneal it — the textbook cure for variance-squashing. But I just watched the variance machinery, run inside this fixed MSE-locked trainer, *hurt* the point metric, and the variational version needs a KL term I cannot add cleanly: the only way in is the same monkey-patch shim, now carrying a KL-annealing schedule the trainer cannot see, re-importing the exact failure mode I am climbing out of. So this rung is the *deterministic* encoder-decoder — encode $X$ to five nodes, decode to $Y$, plain MSE — and it is precisely the KL-off, reconstruction-only corner of the more general object, the corner that fits this scaffold cleanly.

The delta from HSR is therefore sharp: where HSR kept a wide flat 768-unit trunk and spent its effort on a probabilistic readout, I throw away the flat body for a tapering encoder-decoder with a hard 5-node bottleneck and a plain MSE objective, betting that a compact nonlinear code is a better inductive bias for the convection map than an over-wide unstructured one, and that dropping the inverse-variance reweighting removes HSR's self-sabotage. The honest risk is that five nodes is *too* brutal: if the effective dimensionality is well above five the ED will underfit, with the damage concentrated again in `ml_nmse`, because the multi-level tendencies are the high-dimensional part of $Y$ a 5-node code would starve first. That is the specific risk of the instrument, and either way the bottleneck width tells me something measurable — if the ED's NMSE comes in well above HSR's, the 5-node bottleneck is the wrong dimensionality, and the next rung must give the network back representational room while *keeping* structure, which points straight at putting the vertical axis back into the architecture instead of crushing it through a scalar bottleneck.

```python
class _EDBlock(nn.Module):
    """FC + LayerNorm + ELU + Dropout, one rung of the encoder/decoder ladder."""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.net(x)


class Custom(nn.Module):
    """Wide Encoder-Decoder with 5-node latent bottleneck.

    Encoder: 6 FC blocks 556 -> 768 -> 512 -> 384 -> 256 -> 128 -> 5
    Latent:  5 nodes (paper-faithful)
    Decoder: 6 FC blocks 5 -> 128 -> 256 -> 384 -> 512 -> 768 -> 368
    """

    LATENT_DIM = 5
    ENC_DIMS = [768, 512, 384, 256, 128]   # 6 FC layers (the 6th = projection to LATENT)
    DEC_DIMS = [128, 256, 384, 512, 768]   # mirrors encoder

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # ---- Encoder: 6 FC blocks ending at the 5-node latent ----
        enc_layers = []
        prev = input_dim
        for d in self.ENC_DIMS:
            enc_layers.append(_EDBlock(prev, d, dropout=0.1))
            prev = d
        # 6th FC: projection into the bottleneck (no nonlinearity → linear code)
        enc_layers.append(nn.Linear(prev, self.LATENT_DIM))
        self.encoder = nn.Sequential(*enc_layers)

        # ---- Decoder: 6 FC blocks expanding from the 5-node latent ----
        dec_layers = []
        prev = self.LATENT_DIM
        for d in self.DEC_DIMS:
            dec_layers.append(_EDBlock(prev, d, dropout=0.1))
            prev = d
        # 6th FC: projection to output (linear)
        dec_layers.append(nn.Linear(prev, output_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)              # [B, 5]
        y = self.decoder(z)              # [B, output_dim]
        return y
```
