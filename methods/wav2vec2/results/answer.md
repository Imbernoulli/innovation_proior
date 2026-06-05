# wav2vec 2.0

## Problem

Learn powerful speech representations from raw, untranscribed audio so that a
recognizer can be fitted on top with very little labeled data. Transcripts are
scarce for almost every language; raw audio is abundant. The method pre-trains
self-supervised on audio alone, then fine-tunes for recognition with as little as
ten minutes of labels.

## Key idea

Encode the waveform to continuous latent frames, contextualize them with a
Transformer, and train with a **masked contrastive** objective whose targets are
a **jointly-learned discretization** of the latents:

- **Continuous input, discrete target.** The Transformer receives the rich
  continuous latents (masked in spans); the contrastive *targets* are quantized
  latents, which strip nuisance detail (speaker, channel) and become phonetically
  meaningful. The quantizer is trained end-to-end with the model, not frozen in a
  separate first stage.
- **Contrastive, not reconstructive.** At each masked frame the context vector
  must identify the true quantized target among distractors drawn from other
  masked frames of the same utterance — no generation, no exact-vector regression.
- **Product-quantized, Gumbel-selected codebook.** G groups of V entries express
  up to Vᴳ codes; a hard Gumbel-softmax makes the discrete choice differentiable
  (straight-through), with temperature annealed from high to low.
- **Diversity penalty** keeps the codebook from collapsing by driving the
  batch-averaged code usage toward uniform.

## Final objective

Mask spans of the latent sequence (sample each frame as a span start with prob.
p≈0.065, expand by M=10, ~49% masked). With cosine similarity sim, temperature κ,
and target set Q_t (true q_t plus K=100 same-utterance distractors):

  L_m = −log[ exp(sim(c_t, q_t)/κ) / Σ_{q̃∈Q_t} exp(sim(c_t, q̃)/κ) ]

Quantizer: per-group probabilities use Gumbel noise n=−log(−log(u)), u~U(0,1),

  p_{g,v} = exp((l_{g,v}+n_v)/τ) / Σ_k exp((l_{g,k}+n_k)/τ),

forward = argmax (hard), backward = soft (straight-through). Diversity term over
the noise/temperature-free batch-averaged usage p̄_g:

  L_d = (1/GV) Σ_g −H(p̄_g) = (1/GV) Σ_g Σ_v p̄_{g,v} log p̄_{g,v}

Total: L = L_m + α L_d, with α≈0.1. Fine-tune with CTC, feature encoder frozen.

Config (Base): 7-block conv encoder, channels 512, strides (5,2,2,2,2,2,2),
kernels (10,3,3,3,3,2,2) → ~49 Hz frames; 12 Transformer blocks, d=768, FFN 3072,
8 heads; quantizer G=2, V=320, κ=0.1, τ annealed 2→0.5. Optimize with Adam,
warmup then linear decay. Relative positions via a grouped conv (kernel 128, 16
groups), not sinusoidal embeddings.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ConvFeatureEncoder(nn.Module):                 # waveform -> latent frames z
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

class GumbelProductQuantizer(nn.Module):             # continuous z -> discrete target q
    def __init__(self, in_dim=512, G=2, V=320, out_dim=256):
        super().__init__()
        self.G, self.V = G, V
        self.weight   = nn.Linear(in_dim, G * V)
        self.codebook = nn.Parameter(torch.randn(1, G * V, out_dim // G))
        self.proj     = nn.Linear(out_dim, out_dim)
        self.tau = 2.0
    def forward(self, z):
        B, T, _ = z.shape
        logits = self.weight(z).view(B * T * self.G, self.V)
        oh = F.gumbel_softmax(logits, tau=self.tau, hard=True)
        codes = (oh.unsqueeze(-1) * self.codebook.view(self.G, self.V, -1)
                 .repeat(B * T, 1, 1)).sum(-2)
        q = self.proj(codes.view(B, T, -1))
        probs = F.softmax(logits.view(B * T, self.G, self.V), dim=-1).mean(0)   # (G,V)
        return q, probs

class TransformerContext(nn.Module):
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

def mask_spans(x, mask_emb, p=0.065, M=10):
    B, T, _ = x.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    for b in range(B):
        for s in (torch.rand(T, device=x.device) < p).nonzero().flatten():
            mask[b, s:s + M] = True
    x = x.clone(); x[mask] = mask_emb
    return x, mask

def diversity_loss(probs):                            # probs (G,V), batch-averaged
    return (probs * (probs + 1e-7).log()).sum() / probs.numel()   # = (1/GV) sum -H(p_g)

def contrastive_loss(c, q, mask, K=100, kappa=0.1):
    total, n = 0.0, 0
    for b in range(c.size(0)):
        idx = mask[b].nonzero().flatten()
        cb, qb = c[b, idx], q[b, idx]
        sim = F.cosine_similarity(cb.unsqueeze(1), qb.unsqueeze(0), dim=-1) / kappa
        m = idx.numel()
        for i in range(m):
            negs = torch.randperm(m, device=c.device)
            negs = negs[negs != i][:K]
            cols = torch.cat([torch.tensor([i], device=c.device), negs])
            total += F.cross_entropy(sim[i, cols].unsqueeze(0),
                                     torch.zeros(1, dtype=torch.long, device=c.device))
            n += 1
    return total / max(n, 1)

def pretrain_step(wav, enc, quant, ctx, in_proj, mask_emb, alpha=0.1):
    z = enc(wav)
    q, probs = quant(z)                               # targets from UN-masked z
    x, mask = mask_spans(in_proj(z), mask_emb)        # mask the context input only
    c = ctx(x)
    return contrastive_loss(c, q, mask) + alpha * diversity_loss(probs)

class CTCHead(nn.Module):                             # fine-tuning: 29 chars + boundary + blank
    def __init__(self, d=768, n_vocab=32):
        super().__init__()
        self.proj = nn.Linear(d, n_vocab)
    def forward(self, c):
        return self.proj(c).log_softmax(-1)
```
