The core problem is how a Transformer encoder should represent word position and let it interact with content during self-attention. The usual recipe adds a content embedding and a position embedding into one vector at the input layer, then computes attention on that single vector. That entangles two signals that do different work: the attention between two tokens depends on what the tokens are, where they are relative to each other, and the interaction between those two factors, but a summed vector forces the model to recover all of that inside one dot product. Relative-position methods improve on absolute embeddings by injecting a distance-keyed bias into the attention logit, but in practice they only add one mixed term, content-to-position, and ignore the complementary position-to-content interaction. Absolute position is also still needed for some predictions, yet feeding it in at the bottom risks re-entangling everything we wanted to keep separate.

The method is DeBERTa, short for Decoding-enhanced BERT with Disentangled Attention. It keeps content and position as separate vectors and expands the attention score into explicit interaction terms, injects absolute position only at the decoding step, and uses a scale-invariant virtual-adversarial fine-tuning regularizer for large models. In attention, each token at position i is represented by a content vector H_i and a relative-position vector P_{i|j} that encodes i as seen from key position j. The score is the inner product of the concatenated representations, which expands into four terms: content-to-content, content-to-position, position-to-content, and position-to-position. The last one carries no content under relative encoding, so it is dropped, leaving three additive terms. Content-to-content is the usual query-key dot product. Content-to-position compares the query content to the key's relative position, using the relative distance delta(i, j). Position-to-content compares the query position to the key content, using the opposite relative distance delta(j, i), because the asymmetry is what makes it a distinct signal. A shared learnable table P in R^{2k x d} indexed by relative distance supplies the position vectors; the maximum distance k is clamped at both ends, so delta maps signed offsets into the fixed range [0, 2k). Because the score is a sum of three comparable dot products, the softmax scale is 1 / sqrt(3d) instead of 1 / sqrt(d) to keep logits from running hot.

Memory is the subtle part. A naive implementation would allocate an N x N x d position tensor, but only 2k distinct relative-position rows ever appear. DeBERTA computes Q_c K_r^T and K_c Q_r^T once as N x 2k matrices and gathers the column delta(i, j) or delta(j, i) for each query-key pair. This reduces position memory from O(N^2 d) to O(k d). The encoder therefore operates purely on content plus relative position, with no absolute position mixed into the hidden states.

Relative position alone cannot resolve every ambiguity. Consider two masked words that both follow the same word at the same relative offset but play different syntactic roles; their absolute positions differ. DeBERTa handles this with an enhanced mask decoder. The encoder output serves as static keys and values. For each masked slot, the decoder's first query is the encoder hidden state plus an absolute-position embedding, so absolute position enters only at prediction time rather than being summed into content at the input. Two shared decoder layers refine the query against the fixed encoder memory, and the language-model head sees only the final masked-slot states. Adding absolute position to the query also damps the self-peek shortcut for the 10 percent of masked tokens that are kept unchanged in the input, because the query is no longer identical to the content vector at that position.

For fine-tuning, virtual adversarial training perturbs word embeddings toward an adversarial direction and penalizes output-distribution change. Because embedding norms vary widely across words and grow with model size, a fixed perturbation magnitude has inconsistent effect. DeBERTa normalizes the embeddings to unit scale before applying the perturbation, making the adversarial radius invariant to embedding scale and model size.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def relative_position(N, k, device):
    idx = torch.arange(N, device=device)
    rel = idx[:, None] - idx[None, :]  # entry (i, j) is i - j
    return torch.where(
        rel <= -k,
        torch.zeros_like(rel),
        torch.where(
            rel >= k,
            torch.full_like(rel, 2 * k - 1),
            rel + k,
        ),
    )


class DisentangledAttention(nn.Module):
    def __init__(self, d, k=512, pos_terms=2):
        super().__init__()
        self.d = d
        self.k = k
        self.Wqc, self.Wkc, self.Wvc = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.Wqr, self.Wkr = (nn.Linear(d, d, bias=False) for _ in range(2))
        self.scale_factor = 1 + pos_terms  # c2c + c2p + p2c

    def forward(self, H, P, attn_mask):
        N = H.size(0)
        Qc, Kc, Vc = self.Wqc(H), self.Wkc(H), self.Wvc(H)
        Qr, Kr = self.Wqr(P), self.Wkr(P)
        delta = relative_position(N, self.k, H.device)
        scale = 1.0 / math.sqrt(self.d * self.scale_factor)

        # (a) content-to-content
        A = Qc @ Kc.t()

        # (b) content-to-position: gather by delta(i, j)
        c2p = Qc @ Kr.t()
        A = A + torch.gather(c2p, dim=-1, index=delta)

        # (c) position-to-content: gather by delta(j, i), then transpose
        p2c = Kc @ Qr.t()
        A = A + torch.gather(p2c, dim=-1, index=delta).t()

        A = A * scale
        A = A.masked_fill(attn_mask == 0, float("-inf"))
        return torch.softmax(A, dim=-1) @ Vc


def enhanced_mask_decode(encoder_out, abs_pos_emb, target_ids,
                         decoder_layer, vocab_proj, n_steps=2):
    K = V = encoder_out
    mask_positions = (target_ids > 0).nonzero(as_tuple=True)[0]
    Q = encoder_out[mask_positions] + abs_pos_emb[mask_positions]
    for _ in range(n_steps):
        Q = decoder_layer(Q, K, V, query_positions=mask_positions)
    return vocab_proj(Q)


def scale_invariant_finetune_step(model, batch, eps, task_loss, kl_div):
    emb = model.embed(batch.input_ids)
    emb_n = emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)
    logits_clean = model.from_embeddings(emb_n, batch)
    delta = torch.zeros_like(emb_n).uniform_(-eps, eps).requires_grad_(True)
    logits_adv = model.from_embeddings(emb_n + delta, batch)
    return task_loss(logits_clean, batch.labels) + kl_div(logits_clean.detach(), logits_adv)
```
