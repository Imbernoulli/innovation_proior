Let me think about the one design choice in a Transformer encoder that I've always felt is sloppy: how position gets into attention. The standard recipe gives each token a single input vector that is the *sum* of its content embedding and its position embedding, and then attention is computed on these summed vectors. Adding them bothers me. Content and position are doing different jobs. The attention between two words depends on what they are, on where they are relative to each other, and on the *interaction* between those — "deep" attends strongly to "learning" when they're adjacent, weakly when they're sentences apart. A single summed vector smears content and position into one thing before attention ever looks at them, so the model has to disentangle, inside the dot product, two signals I carelessly glued together at the input. Let me see if representing them separately and being honest about their interactions buys something.

Represent a token at position `i` by *two* vectors: a content vector `H_i` and a position vector `P_{i|j}` encoding its position relative to the token at `j`. Now write the attention score between `i` and `j` as the inner product of the full (content, position) representations:
`A_{i,j} = {H_i, P_{i|j}} · {H_j, P_{j|i}}ᵀ`.
Expand it. A product of two-part vectors gives four cross terms:
`A_{i,j} = H_i H_jᵀ + H_i P_{j|i}ᵀ + P_{i|j} H_jᵀ + P_{i|j} P_{j|i}ᵀ`.
Read them off: content-to-content (how the two contents match), content-to-position (query content vs key position), position-to-content (query position vs key content), and position-to-position. That decomposition is the whole point — it makes explicit the interactions a summed vector hides.

Now compare to what existing relative-position methods actually do. They add a relative-position bias into the attention logit that amounts to a query *content* vector dotted with a relative-position *key* vector — that's exactly the content-to-position term, and *only* that term. They leave out position-to-content entirely. But position-to-content is not redundant: it asks how much a key's *content* matters given the *query's* position, which is a genuinely different question from how a query's content matters given the key's position. Attention isn't symmetric in that sense, so dropping one of the two mixed terms throws away half of the content–position interaction. So I want at least content-to-content, content-to-position, *and* position-to-content.

What about the fourth term, position-to-position? I'm using *relative* position embeddings, so `P_{i|j}` and `P_{j|i}` are both functions of the relative distance only — the position-to-position term is then a function of relative distance alone, carrying no content information, and it's largely already implied by how the other position terms index distance. It adds little. Drop it. So three terms.

Let me now write attention concretely with relative position. I have content projections `Q_c = HW_{q,c}`, `K_c = HW_{k,c}`, `V_c = HW_{v,c}` as usual. For position I need projected relative-position vectors. Let `P ∈ R^{2k×d}` be a learnable embedding indexed by *relative distance*, shared across all layers (it doesn't depend on the actual tokens, only on offsets, so one table for the whole network). Project it two ways: `Q_r = P W_{q,r}` and `K_r = P W_{k,r}`. Then the disentangled score is
`Ã_{i,j} = Q^c_i · K^c_jᵀ  +  Q^c_i · K^r_{δ(i,j)}ᵀ  +  K^c_j · Q^r_{δ(j,i)}ᵀ`,
the three terms being (a) content-to-content, (b) content-to-position, (c) position-to-content.

I have to get the index on each position term exactly right, because the relative distances are *not* the same in the two mixed terms. For content-to-position (b): the query is the content at `i`, and it's comparing against the *position* of the key `j` as seen from `i` — that's the relative distance from `i` to `j`, i.e. `δ(i,j)`, so I index `K^r` at `δ(i,j)`. For position-to-content (c): now the query is a *position* (the query position `i`) and it compares against the *content* of key `j`. The relative distance here is from the key's perspective — how far `j` is from `i` — which is `δ(j,i)`, *not* `δ(i,j)`. So I index `Q^r` at `δ(j,i)`. Getting this backwards would silently swap the two off-diagonal directions of the relative bias; the asymmetry `δ(j,i) ≠ δ(i,j)` is the thing that makes position-to-content a distinct term and not a mislabeled copy of content-to-position. So: term (b) uses `δ(i,j)` into `K_r`, term (c) uses `δ(j,i)` into `Q_r`.

Now define `δ`. Raw relative offset `i − j` ranges over `[−(N−1), N−1]`, which is unbounded in `N` and signed; I want a bounded, non-negative index into a fixed table of size `2k`. Cap the maximum relative distance at `k` and map the offset into `[0, 2k)`:
`δ(i,j) = 0` if `i − j ≤ −k` (the key is far to the right, clamp to the smallest bucket),
`δ(i,j) = 2k − 1` if `i − j ≥ k` (the key is far to the left, clamp to the largest bucket),
`δ(i,j) = i − j + k` otherwise.
Check the middle case at the boundaries: at `i − j = −(k−1)` it gives `1`; at `i − j = k − 1` it gives `2k − 1`; and offset `0` (a token to itself) maps to `k`, the center of the table. So distances within the window get a distinct bucket, anything beyond `±k` is clamped to the two ends, and the table has exactly `2k` rows. Good — bounded, non-negative, and signed offsets are preserved (left vs right of the query land on opposite sides of the center `k`).

Now the scale. Standard self-attention divides by `√d` to keep the logit variance from growing with dimension so the softmax doesn't saturate. Here each `Ã_{i,j}` is a *sum of three* dot-product terms instead of one, so its variance is roughly tripled — if I keep dividing by `√d` the logits run hot and training destabilizes, more so at large scale. The terms are comparable in scale (each is a `d`-dimensional dot product), so to restore the original logit magnitude I divide by `√(3d)` instead of `√d`. In general, with `m` additive score terms the scale is `√(m·d)`; here `m = 3`. So
`H_o = softmax( Ã / √(3d) ) V_c`.

There's a memory trap in the position terms if I implement them naively. For content-to-position I want, for each query `i` and key `j`, the dot product `Q^c_i · K^r_{δ(i,j)}`. The relative index `δ(i,j)` varies per `(i,j)` pair, so if I literally build a relative-position embedding per query I'd allocate an `N×N×d` tensor — `O(N²d)` memory. But notice the only distinct position vectors that can ever appear are the `2k` rows of `K_r`; for any query there are at most `2k` possibilities. So compute the full small product `Q_c K_rᵀ ∈ R^{N×2k}` once (every query against every relative-distance row), then *gather*: for entry `(i,j)`, pick out column `δ(i,j)`. Symmetrically for position-to-content, compute `K_c Q_rᵀ ∈ R^{N×2k}` and gather column `δ(j,i)` for entry `(i,j)`. This stores only `K_r, Q_r ∈ R^{2k×d}` plus the small `N×2k` products — `O(kd)` for the embeddings instead of `O(N²d)`. The gather is the efficient implementation.

Let me write the disentangled attention.

```python
import math, torch, torch.nn as nn

def relative_position(N, k, device):
    # raw offsets i - j, then bucket via delta(i,j) in [0, 2k)
    idx = torch.arange(N, device=device)
    rel = idx[:, None] - idx[None, :]                       # (N, N), entry = i - j
    delta = torch.where(rel <= -k, torch.zeros_like(rel),
            torch.where(rel >=  k, torch.full_like(rel, 2 * k - 1),
                        rel + k))
    return delta                                            # delta[i, j] = δ(i, j)

class DisentangledAttention(nn.Module):
    def __init__(self, d, k=512, pos_terms=2):              # pos_terms = c2p + p2c = 2
        super().__init__()
        self.d, self.k = d, k
        self.Wqc, self.Wkc, self.Wvc = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.Wqr, self.Wkr = (nn.Linear(d, d, bias=False) for _ in range(2))
        self.scale_factor = 1 + pos_terms                   # 1 (c2c) + 2 (c2p, p2c) = 3

    def forward(self, H, P, attn_mask):
        N = H.size(0)
        Qc, Kc, Vc = self.Wqc(H), self.Wkc(H), self.Wvc(H)
        Qr, Kr = self.Wqr(P), self.Wkr(P)                   # P: (2k, d) relative-position table
        delta = relative_position(N, self.k, H.device)      # (N, N)
        scale = 1.0 / math.sqrt(self.d * self.scale_factor) # 1/sqrt(3d)

        # (a) content-to-content
        A = Qc @ Kc.t()                                      # (N, N)
        # (b) content-to-position: Qc against the 2k relative-position keys, gather col δ(i,j)
        c2p = Qc @ Kr.t()                                    # (N, 2k)
        A = A + torch.gather(c2p, dim=-1, index=delta)       # index [i, j] -> δ(i, j)
        # (c) position-to-content: Kc against the 2k relative-position queries, gather col δ(j,i)
        p2c = Kc @ Qr.t()                                    # (N, 2k)
        p2c = torch.gather(p2c, dim=-1, index=delta)         # p2c[j, i'] picked at δ(j, i')
        A = A + p2c.t()                                      # transpose so entry (i, j) uses δ(j, i)
        A = A * scale

        A = A.masked_fill(attn_mask == 0, float('-inf'))
        return torch.softmax(A, dim=-1) @ Vc
```

The `p2c` transpose deserves a beat: I compute `p2c[j, ·] = K_c[j] · Q_rᵀ` and gather at `δ(j, i)`, giving a value keyed by `(j, i)`; transposing puts it at attention entry `(i, j)`, which is exactly term (c) wanting `δ(j,i)` for the score from `i` to `j`. That's the asymmetric index landing in the right cell.

So far I've handled content and *relative* position thoroughly. But relative position isn't everything for pre-training. Back to "a new **store** opened beside the new **mall**." Mask *store* and *mall*. Each masked slot sees similar surrounding content and identical relative offsets to *new*, so relative-only attention can't tell them apart — yet they differ syntactically, and that difference is tied to their *absolute* positions (one is the subject). So I need absolute position somewhere. The default place is to add it at the input, the way absolute-position models do. But I deliberately built the whole stack on relative position because relative is the more effective signal; if I sum absolute-position embeddings into the input, I re-entangle position with content at the bottom and risk drowning out the relative structure I want every layer to learn. So inject absolute position *late* instead: let all the Transformer layers operate purely on content + relative position, and only fold in absolute positions right before the softmax that decodes the masked tokens. The layers learn rich relative representations undisturbed; absolute position enters as *complementary* information exactly where the prediction is made.

Concretely, treat the top of the network as a small decoder for the masked tokens that takes the encoder's contextual outputs together with absolute-position embeddings. The encoder produces `H` over `N−1` (or `L−1`) layers as `K` and `V`; the decode step uses absolute positions to form the query that attends over those, then predicts the masked token via the softmax. Call this the enhanced mask decoder. Because the relative-only layers are unchanged and I only add a thin decoding step, it costs essentially no extra parameters (I can reuse the last encoder layer's weights), and I can stack a couple of such decode steps if it helps. The reason this beats input-layer absolute position is empirical and mechanistic: early absolute position competes with relative learning in every layer; late absolute position complements it and only at the point of decoding.

```python
def enhanced_mask_decode(encoder_out, abs_pos_emb, mask_positions,
                          decoder_layer, vocab_proj, n_steps=2):
    # K, V come from the encoder output (content + relative position, all layers);
    # Q is seeded with absolute-position embeddings at the masked slots.
    K = V = encoder_out
    Q = encoder_out.clone()
    Q[mask_positions] = abs_pos_emb[mask_positions]          # absolute position enters only here
    for _ in range(n_steps):
        Q = decoder_layer(Q, K, V)                           # attend Q over static K, V
    logits = vocab_proj(Q[mask_positions])                   # decode masked tokens
    return logits
```

There's a small leak to plug in masking. The standard MLM keeps 10% of masked positions *unchanged* (to reduce the `[MASK]`-never-seen-downstream mismatch). But an unchanged token can be predicted by peeking at itself through attention, which is too easy and teaches nothing. Since the decoder query for masked slots is seeded with the absolute-position embedding rather than the token's own content, the unchanged-token slot decodes from position + context without seeing its own content directly — the self-peek is removed for free.

Last piece: fine-tuning regularization. Virtual adversarial training perturbs the input toward an adversarial direction and forces the model to keep the same output distribution — a good generalization regularizer. For text the perturbation lands on word embeddings. But word-embedding norms vary a lot across words, and the variance grows with model size (billions of parameters), so a fixed perturbation magnitude is huge for some embeddings and negligible for others — the adversarial training destabilizes, worse for big models. Layer normalization handles exactly this kind of scale disparity by normalizing vectors before use, so apply the same idea: *normalize* the word embeddings into unit-scale stochastic vectors first, then add the perturbation to the normalized embeddings. The perturbation now has a consistent meaning across words and model sizes. Call it scale-invariant fine-tuning; reserve it for the largest models where the instability actually bites.

```python
def scale_invariant_finetune_step(model, batch, eps):
    emb = model.embed(batch.input_ids)
    emb_n = emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)    # normalize before perturbing
    logits_clean = model.from_embeddings(emb_n, batch)
    delta = torch.zeros_like(emb_n).uniform_(-eps, eps).requires_grad_(True)
    logits_adv = model.from_embeddings(emb_n + delta, batch)  # perturb normalized embeddings
    adv_loss = kl_div(logits_clean.detach(), logits_adv)      # match clean output distribution
    return task_loss(logits_clean, batch.labels) + adv_loss
```

So the chain: summing content and position at the input entangles signals that have distinct roles and hides their interaction; representing each token by separate content and relative-position vectors and expanding the attention score gives four interaction terms, of which content-to-content, content-to-position, and position-to-content matter (position-to-position is content-free under relative encoding and dropped); the two mixed terms use *different* relative indices `δ(i,j)` and `δ(j,i)`, with `δ` a bounded `[0,2k)` bucketing of the signed offset, and since the score now sums three terms the softmax scale becomes `1/√(3d)`; the per-query relative embeddings would cost `O(N²d)` but collapse to a gather over the `2k`-row tables for `O(kd)`; relative position alone can't disambiguate same-context words, so absolute position is folded in late, in an enhanced mask decoder right before the softmax, which also removes the self-peek on unchanged masked tokens; and for fine-tuning, normalizing embeddings before adversarial perturbation makes the perturbation scale-invariant across words and model sizes.
