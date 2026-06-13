**Problem (from step 1).** On a flat MLP backbone, HSR's heteroskedastic readout bought nothing on the
point metric (long NMSE 0.639, ml_nmse 0.655) — the inverse-variance weighting froze out the hard
multi-level columns, and the flat body ignores the column's structure. The architecture, not the loss, is the
bottleneck. Attack it with a structural prior and a plain MSE objective.

**Key idea.** A nonlinear encoder-decoder with a hard low-dimensional bottleneck: a nonlinear PCA for the
X→Y map. Encode the 556-dim state through a tapering FC funnel to a **5-node latent**, then decode
symmetrically up to the 368-dim tendencies. The bottleneck width is an *instrument* — it tests the
reduced-complexity premise that few degrees of freedom govern convection. Train on plain MSE; deliberately
drop the variational/KL and X-reconstruction machinery (the deterministic ED is the KL-off corner of that
object, and the fixed MSE trainer makes the variational version risky — exactly HSR's failure mode).

**Why it works (or not).** A nonlinear bottleneck is a stronger inductive bias for the convection map than
HSR's over-wide unstructured trunk, and plain MSE removes HSR's early-weighting pathology. The risk is that
5 nodes is too brutal a compression; if the effective dimensionality is well above five the ED underfits,
and the damage would again concentrate in ml_nmse (the high-dimensional part of Y).

**Scaffold edit / hyperparameters.** Encoder: 6 FC blocks 556→768→512→384→256→128 then linear→5
(widths = ClimSim Table A). Decoder: mirror, 5→128→256→384→512→768 then linear→368. Each block =
Linear→LayerNorm→ELU→Dropout(0.1) (ELU + linear output so signed/negative tendencies are representable).
Latent = 5 (published). AdamW + cosine LR + MSE, all fixed.

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
