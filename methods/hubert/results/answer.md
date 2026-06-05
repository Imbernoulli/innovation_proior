# HuBERT (Hidden-unit BERT)

## Problem

Learn speech representations from raw audio with no transcripts, so recognition
needs little labeled data. The obstacle unique to speech: unlike text (a known
word-piece vocabulary) or images (one instance), audio is a continuous sequence
with no given inventory of units and no known unit boundaries — a masked-prediction
objective has nothing clean to predict.

## Key idea

Manufacture discrete targets by **offline clustering** of frame features, then
train a BERT-style model to **predict the cluster assignments of masked frames**:

- **Discovered units as targets.** k-means over frame features (MFCCs at first)
  yields per-frame cluster ids z_t that correlate with phones. They are noisy but
  carry phonetic structure.
- **Loss on masked frames only (α→1).** Predicting hidden frames' units from
  context forces both acoustic modeling (of visible frames) and long-range
  temporal modeling. Because the model cannot copy a per-frame teacher label, it
  learns whatever is *consistently* predictable — making it robust to label noise.
  Loss only on *unmasked* frames (α=0) merely mimics the clusterer.
- **Continuous input, discrete target.** The Transformer ingests masked
  continuous features (not quantized tokens) to keep maximal information.
- **Better targets two ways.** Ensemble several clusterings of different
  granularity (multi-task, one head each), and *iteratively refine*: re-cluster the
  trained model's own hidden features to seed the next generation.

## Final objective

Mask spans (sample p% of frames as starts, mask l consecutive). Per-frame unit
distribution from cosine similarity scaled by τ:

  p_f^{(k)}(c | X̃, t) = exp(sim(A^{(k)} o_t, e_c)/τ) / Σ_{c'} exp(sim(A^{(k)} o_t, e_{c'})/τ)

Loss = α·L_m + (1−α)·L_u, with L_m, L_u the per-frame cross-entropies summed over
masked / unmasked indices and over clusterings k; use α≈1. τ=0.1. Fine-tune by
removing the unit heads, attaching a CTC softmax (26 letters + space + apostrophe
+ blank), freezing the convolutional encoder.

Config: conv encoder 7 blocks, 512 channels, strides (5,2,2,2,2,2,2), kernels
(10,3,3,3,3,2,2) → 20 ms frames (320× downsample). Transformer Base (12 layers,
d=768, FFN 3072, 8 heads, ~95M), Large (24, 1024, 4096, 16, ~317M), X-Large (48,
1280, 5120, 16, ~1B). Projection dim 256/768/1024.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

class ConvFeatureEncoder(nn.Module):
    def __init__(self, dims=(512,)*7,
                 kernels=(10,3,3,3,3,2,2), strides=(5,2,2,2,2,2,2)):
        super().__init__()
        layers, c_in = [], 1
        for c_out, k, s in zip(dims, kernels, strides):
            layers += [nn.Conv1d(c_in, c_out, k, s),
                       nn.GroupNorm(1, c_out), nn.GELU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
    def forward(self, wav):
        return self.conv(wav.unsqueeze(1)).transpose(1, 2)        # (B, T, 512)

class TransformerEncoder(nn.Module):
    def __init__(self, d=768, layers=12, heads=8, ffn=3072):
        super().__init__()
        self.pos_conv = nn.Conv1d(d, d, 128, padding=64, groups=16)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(d, heads, ffn, 0.1, F.gelu, batch_first=True)
            for _ in range(layers))
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        p = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)[:, :x.size(1)]
        x = self.ln(x + F.gelu(p))
        for l in self.layers:
            x = l(x)
        return x

class UnitPredictor(nn.Module):                      # o_t -> logits over C units
    def __init__(self, d=768, proj=256, n_units=100, tau=0.1):
        super().__init__()
        self.A = nn.Linear(d, proj)
        self.embed = nn.Embedding(n_units, proj)
        self.tau = tau
    def forward(self, o):
        h = F.normalize(self.A(o), dim=-1)
        e = F.normalize(self.embed.weight, dim=-1)
        return (h @ e.t()) / self.tau                # (B, T, C)

def span_mask(x, mask_emb, p=0.08, l=10):
    B, T, _ = x.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    for b in range(B):
        for s in (torch.rand(T, device=x.device) < p).nonzero().flatten():
            mask[b, s:s + l] = True
    x = x.clone(); x[mask] = mask_emb
    return x, mask

def masked_prediction_loss(enc, heads, x, targets, mask_emb, alpha=1.0):
    xm, mask = span_mask(x, mask_emb)
    o = enc(xm)
    Lm = Lu = 0.0
    for head, z in zip(heads, targets):              # one head per clustering
        logits = head(o)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             z.reshape(-1), reduction='none').view(x.size(0), -1)
        Lm = Lm + ce[mask].mean()                    # masked frames -> learn context
        Lu = Lu + ce[~mask].mean()                   # unmasked frames -> mimic teacher
    return alpha * Lm + (1 - alpha) * Lu             # alpha -> 1

def make_targets(feature_extractor, dataset, n_clusters, layer=None):
    feats = torch.cat([feature_extractor(x, layer) for x in dataset])   # MFCC, then hidden
    return MiniBatchKMeans(n_clusters).fit(feats.numpy())               # re-cluster to refine

class CTCHead(nn.Module):                            # fine-tuning: 26 + space + apostrophe + blank
    def __init__(self, d=768, n_vocab=29):
        super().__init__()
        self.proj = nn.Linear(d, n_vocab)
    def forward(self, o):
        return self.proj(o).log_softmax(-1)
```
