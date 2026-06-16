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

- **Discovered units as targets.** 100-way k-means over 39-dimensional MFCC
  frame features (13 coefficients plus first and second derivatives) yields
  per-frame cluster ids z_t with non-trivial acoustic-unit correlation. They are
  noisy but carry phonetic structure.
- **Loss on masked frames only (α=1).** Predicting hidden frames' units from
  context forces both acoustic modeling (of visible frames) and long-range
  temporal modeling. Because the model cannot copy a per-frame teacher label, it
  learns whatever is *consistently* predictable — making it robust to label noise.
  Loss only on *unmasked* frames (α=0) merely mimics the clusterer.
- **Continuous input, discrete target.** The Transformer ingests masked
  continuous features (not quantized tokens) to keep maximal information.
- **Better targets two ways.** Ensemble several clusterings of different
  granularity (multi-task, one head each), and *iteratively refine*: re-cluster the
  trained model's own intermediate hidden features with 500-way k-means to seed the
  next generation. For the second Base generation, cluster the 6th Transformer
  layer from the first generation; for the larger models, cluster the 9th layer
  from the second Base generation.

## Final objective

Mask spans (sample p% of frames as starts, mask l consecutive). Per-frame unit
distribution from cosine similarity scaled by τ:

  p_f^{(k)}(c | X̃, t) = exp(sim(A^{(k)} o_t, e_c)/τ) / Σ_{c'} exp(sim(A^{(k)} o_t, e_{c'})/τ)

Minimize the cross-entropy version L = α·L_m + (1−α)·L_u, with L_m, L_u summed
over masked / unmasked indices and over clusterings k; use α=1 for
masked-frame-only training. τ=0.1. Fine-tune by
removing the unit heads, attaching a CTC softmax (26 letters + space + apostrophe
+ blank), freezing the convolutional encoder.

Config: conv encoder 7 blocks, 512 channels, strides (5,2,2,2,2,2,2), kernels
(10,3,3,3,3,2,2) → 20 ms frames (320× downsample). Transformer Base (12 layers,
d=768, FFN 3072, 8 heads, 95M), Large (24, 1024, 4096, 16, 317M), X-Large (48,
1280, 5120, 16, 964M). Projection dim 256/768/1024.
Use p=8%, l=10 for masking. Fit k-means with MiniBatchKMeans, mini-batches of
10,000 frames, k-means++ initialization, and 20 random starts; for hidden-feature
clustering, fit on a random 10% sample because the feature matrix is large.
Pre-training uses Adam with β=(0.9,0.98), 8% linear warmup, then linear decay.
Peak learning rates are 5e-4/1.5e-3/3e-3 for Base/Large/X-Large.

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
        starts = (torch.rand(T, device=x.device) < p).nonzero(as_tuple=False).flatten()
        for s in starts.tolist():
            mask[b, s:min(s + l, T)] = True
    x = x.clone(); x[mask] = mask_emb
    return x, mask

def masked_prediction_loss(enc, heads, x, targets, mask_emb, alpha=1.0):
    xm, mask = span_mask(x, mask_emb)
    o = enc(xm)
    Lm = o.new_tensor(0.0)
    Lu = o.new_tensor(0.0)
    def selected_mean(values, selected):
        return values[selected].mean() if selected.any() else values.new_tensor(0.0)
    for head, z in zip(heads, targets):              # one head per clustering
        logits = head(o)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             z.reshape(-1), reduction='none').view(x.size(0), -1)
        Lm = Lm + selected_mean(ce, mask)            # masked frames -> learn context
        Lu = Lu + selected_mean(ce, ~mask)           # unmasked frames -> mimic teacher
    return alpha * Lm + (1 - alpha) * Lu             # alpha=1

def fit_kmeans_targets(feature_extractor, dataset, n_clusters, layer=None,
                       sample_ratio=1.0, batch_size=10_000):
    chunks = []
    for x in dataset:
        f = feature_extractor(x) if layer is None else feature_extractor(x, layer)
        chunks.append(f.reshape(-1, f.size(-1)))
    feats = torch.cat(chunks, dim=0)                    # all frame features, (N, d)
    if sample_ratio < 1.0:
        keep = torch.randperm(feats.size(0))[:int(sample_ratio * feats.size(0))]
        feats = feats[keep]
    return MiniBatchKMeans(n_clusters=n_clusters, init="k-means++",
                           n_init=20, batch_size=batch_size).fit(
                               feats.detach().cpu().numpy())

def first_generation_targets(mfcc_extractor, dataset):
    return fit_kmeans_targets(mfcc_extractor, dataset, n_clusters=100)

def refined_targets(hidden_extractor, dataset, layer=6):
    # Use layer=9 when clustering the second Base generation for larger models.
    return fit_kmeans_targets(hidden_extractor, dataset, n_clusters=500,
                              layer=layer, sample_ratio=0.10)

class CTCHead(nn.Module):                            # fine-tuning: 26 + space + apostrophe + blank
    def __init__(self, d=768, n_vocab=29):
        super().__init__()
        self.proj = nn.Linear(d, n_vocab)
    def forward(self, o):
        return self.proj(o).log_softmax(-1)
```
