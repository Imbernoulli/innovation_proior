DeBERTa is a Transformer encoder whose attention keeps token content and token position as separate factors. Standard encoders add content and absolute-position embeddings at the input, then attend with a single dot product. That makes the attention logit carry mixed content/position effects implicitly. Existing relative-position attention recovers only one mixed term, content-to-position, and leaves out position-to-content.

The attention score starts by expanding a two-vector representation for query position `i` and key position `j`:

```
A_{i,j} = {H_i, P_{i|j}} · {H_j, P_{j|i}}^T
        = H_i H_j^T + H_i P_{j|i}^T + P_{i|j} H_j^T + P_{i|j} P_{j|i}^T.
```

With relative positions, the position-to-position term carries no token content, so the implemented score keeps the other three terms:

```
Q_c = H W_{q,c},  K_c = H W_{k,c},  V_c = H W_{v,c}
Q_r = P W_{q,r},  K_r = P W_{k,r}

A~_{i,j} = Q^c_i K^{cT}_j              # content-to-content
          + Q^c_i K^{rT}_{delta(i,j)}  # content-to-position
          + K^c_j Q^{rT}_{delta(j,i)}  # position-to-content
H_o = softmax(A~ / sqrt(3d)) V_c.
```

The two mixed terms intentionally use opposite relative directions. If `delta(i,j)` indexes the key position as seen from the query, then the position-to-content term must use `delta(j,i)`, because it asks how the key content at `j` scores against the query position at `i`. The relative-distance index is bounded by a maximum distance `k`:

```
delta(i,j) = 0       if i - j <= -k
           = 2k - 1  if i - j >=  k
           = i - j + k otherwise.
```

The score has three dot-product terms, so the stabilizing scale is `1/sqrt(3d)`. A direct per-query relative-position tensor would take `O(N^2 d)` memory; the efficient implementation computes `Q_c K_r^T` and `K_c Q_r^T` against the `2k` relative rows, then gathers the needed relative-distance column for each `(i,j)`.

The enhanced mask decoder keeps absolute position out of the encoder stack and injects it at masked-token decoding time. The encoder learns content plus relative position. The decoder uses the encoder output as static `K,V`; the query for the first decoder layer adds the absolute-position embedding to the encoder hidden state at each masked slot, and each shared decoder layer output becomes the next query input. The language-model head receives only the final masked-slot decoder states. Folding absolute position into the query this way also suppresses the self-peek path: even for the 10% MLM targets kept unchanged in the input, the query carries position rather than the bare content embedding, so the decoder cannot trivially copy the target from itself.

Scale-invariant fine-tuning adapts virtual adversarial training to embeddings whose norms vary widely across words and model sizes. The perturbation is applied after normalizing word embeddings, so a fixed perturbation radius has a consistent meaning.

```python
import math, torch, torch.nn as nn

def relative_position(N, k, device):
    idx = torch.arange(N, device=device)
    rel = idx[:, None] - idx[None, :]
    return torch.where(rel <= -k, torch.zeros_like(rel),
           torch.where(rel >=  k, torch.full_like(rel, 2 * k - 1), rel + k))

class DisentangledAttention(nn.Module):
    def __init__(self, d, k=512, pos_terms=2):
        super().__init__()
        self.d, self.k = d, k
        self.Wqc, self.Wkc, self.Wvc = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.Wqr, self.Wkr = (nn.Linear(d, d, bias=False) for _ in range(2))
        self.scale_factor = 1 + pos_terms

    def forward(self, H, P, attn_mask):
        N = H.size(0)
        Qc, Kc, Vc = self.Wqc(H), self.Wkc(H), self.Wvc(H)
        Qr, Kr = self.Wqr(P), self.Wkr(P)
        delta = relative_position(N, self.k, H.device)
        scale = 1.0 / math.sqrt(self.d * self.scale_factor)

        A = Qc @ Kc.t()
        c2p = Qc @ Kr.t()
        A = A + torch.gather(c2p, -1, delta)
        p2c = Kc @ Qr.t()
        A = A + torch.gather(p2c, -1, delta).t()
        A = A * scale

        A = A.masked_fill(attn_mask == 0, float("-inf"))
        return torch.softmax(A, dim=-1) @ Vc

def enhanced_mask_decode(encoder_out, abs_pos_emb, target_ids,
                         decoder_layer, vocab_proj, n_steps=2):
    K = V = encoder_out                                       # static encoder memory
    mask_positions = (target_ids > 0).nonzero(as_tuple=True)[0]
    Q = encoder_out[mask_positions] + abs_pos_emb[mask_positions]  # add absolute position to content
    for _ in range(n_steps):
        Q = decoder_layer(Q, K, V, query_positions=mask_positions)
    return vocab_proj(Q)

def sift_finetune_step(model, batch, eps, task_loss, kl_div):
    emb = model.embed(batch.input_ids)
    emb_n = emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)
    logits_clean = model.from_embeddings(emb_n, batch)
    delta = torch.zeros_like(emb_n).uniform_(-eps, eps).requires_grad_(True)
    logits_adv = model.from_embeddings(emb_n + delta, batch)
    return task_loss(logits_clean, batch.labels) + kl_div(logits_clean.detach(), logits_adv)
```

Base uses 12 layers, hidden size 768, 12 heads, head size 64, FFN size 3072. Large uses 24 layers, hidden size 1024, 16 heads, FFN size 4096. The setup uses `k=512`, MLM with span masking on roughly 78GB of text, Adam with decoupled weight decay, 10k warmup, batch size 2k, and up to 1M pre-training steps. Sharing `W_{q,r}=W_{q,c}` and `W_{k,r}=W_{k,c}` keeps the parameter count at the RoBERTa level.
