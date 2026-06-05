# DeBERTa (Disentangled Attention)

## Problem

Standard Transformer encoders represent each token by *summing* its content and position
embeddings and computing attention on the sum, which entangles content and position and
hides their interaction. Existing relative-position schemes add only the
content-to-position part of that interaction, omitting position-to-content. And absolute
position — needed to disambiguate same-context words — is injected at the input layer,
where it can interfere with learning relative structure. Goal: a better content/position
parameterization that improves pre-training efficiency and downstream accuracy at fixed
scale.

## Key idea

**Disentangled attention.** Represent each token by two vectors — content `H_i` and
relative position `P` — and compute the attention score as a sum of disentangled terms.
Expanding `{H_i, P_{i|j}}·{H_j, P_{j|i}}ᵀ` gives four terms; keep three (drop
position-to-position, which is content-free under relative encoding):

```
Ã_{i,j} = Q^c_i·K^c_jᵀ            (a) content-to-content
        + Q^c_i·K^r_{δ(i,j)}ᵀ     (b) content-to-position
        + K^c_j·Q^r_{δ(j,i)}ᵀ     (c) position-to-content
H_o = softmax( Ã / √(3d) ) · V_c
```

with `Q_c=HW_{q,c}, K_c=HW_{k,c}, V_c=HW_{v,c}`, and relative-position projections
`Q_r=PW_{q,r}, K_r=PW_{k,r}` from a layer-shared table `P ∈ R^{2k×d}`. The two mixed
terms use *different* relative indices: `δ(i,j)` for (b), `δ(j,i)` for (c). Because the
score sums three terms (vs one), the softmax scale is `1/√(3d)`, not `1/√d`.

**Relative-distance bucketing** (`k` = max relative distance):
```
δ(i,j) = 0        if i−j ≤ −k
       = 2k−1     if i−j ≥  k
       = i−j+k    otherwise
```
maps signed offsets into `[0, 2k)`, centered at `k` for offset 0.

**Efficient implementation.** Per-query relative embeddings would cost `O(N²d)`; instead
compute the small `N×2k` products `Q_c K_rᵀ` and `K_c Q_rᵀ` once and *gather* the column
at `δ(i,j)` / `δ(j,i)` — `O(kd)` memory. `k=512` for pre-training.

**Enhanced Mask Decoder (EMD).** All Transformer layers use only content + relative
position; absolute position is folded in *late*, right before the masked-token softmax,
as complementary information (and avoids competing with relative learning in every
layer). The decoder takes the encoder output as static K/V and absolute-position
embeddings to form Q for the masked slots; seeding masked-slot queries with position
(not the token's own content) also removes the self-peek on the 10% unchanged masked
tokens.

**Scale-invariant Fine-Tuning (SiFT).** Virtual adversarial training perturbs word
embeddings, but embedding norms vary across words and grow with model size, destabilizing
the perturbation. Normalize embeddings before applying the perturbation, then require the
output distribution to match the clean one. Applied to the largest models.

## Code

```python
import math, torch, torch.nn as nn

def relative_position(N, k, device):
    idx = torch.arange(N, device=device)
    rel = idx[:, None] - idx[None, :]                       # i - j
    return torch.where(rel <= -k, torch.zeros_like(rel),
           torch.where(rel >=  k, torch.full_like(rel, 2 * k - 1), rel + k))

class DisentangledAttention(nn.Module):
    def __init__(self, d, k=512, pos_terms=2):              # c2p + p2c
        super().__init__()
        self.d, self.k = d, k
        self.Wqc, self.Wkc, self.Wvc = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.Wqr, self.Wkr = (nn.Linear(d, d, bias=False) for _ in range(2))
        self.scale_factor = 1 + pos_terms                   # = 3

    def forward(self, H, P, attn_mask):
        N = H.size(0)
        Qc, Kc, Vc = self.Wqc(H), self.Wkc(H), self.Wvc(H)
        Qr, Kr = self.Wqr(P), self.Wkr(P)                   # P: (2k, d)
        delta = relative_position(N, self.k, H.device)      # (N, N), delta[i,j]=δ(i,j)
        scale = 1.0 / math.sqrt(self.d * self.scale_factor) # 1/sqrt(3d)

        A = Qc @ Kc.t()                                      # (a)
        c2p = Qc @ Kr.t()                                    # (N, 2k)
        A = A + torch.gather(c2p, -1, delta)                # (b) gather col δ(i,j)
        p2c = Kc @ Qr.t()                                    # (N, 2k)
        A = A + torch.gather(p2c, -1, delta).t()            # (c) gather col δ(j,i), transpose
        A = A * scale

        A = A.masked_fill(attn_mask == 0, float('-inf'))
        return torch.softmax(A, dim=-1) @ Vc

def enhanced_mask_decode(encoder_out, abs_pos_emb, mask_positions,
                         decoder_layer, vocab_proj, n_steps=2):
    K = V = encoder_out
    Q = encoder_out.clone()
    Q[mask_positions] = abs_pos_emb[mask_positions]         # absolute position only here
    for _ in range(n_steps):
        Q = decoder_layer(Q, K, V)
    return vocab_proj(Q[mask_positions])

def sift_finetune_step(model, batch, eps, task_loss, kl_div):
    emb   = model.embed(batch.input_ids)
    emb_n = emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)   # normalize, then perturb
    logits_clean = model.from_embeddings(emb_n, batch)
    delta = torch.zeros_like(emb_n).uniform_(-eps, eps).requires_grad_(True)
    logits_adv   = model.from_embeddings(emb_n + delta, batch)
    return task_loss(logits_clean, batch.labels) + kl_div(logits_clean.detach(), logits_adv)
```

Base: 12 layers, d=768, 12 heads, FFN 3072. Large: 24 layers, d=1024, 16 heads, FFN
4096. `k=512`. Pre-trained with MLM (+ span masking) on ~78GB of text, Adam with
decoupled weight decay, 10k warmup, batch 2k, up to 1M steps. Sharing the position and
content projection matrices (`W_{q,r}=W_{q,c}`, `W_{k,r}=W_{k,c}`) keeps the parameter
count equal to a same-size absolute-position encoder.
