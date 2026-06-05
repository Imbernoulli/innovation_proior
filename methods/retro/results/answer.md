# RETRO (Retrieval-Enhanced Transformer)

## Problem

Scaling a Transformer LM conflates two benefits — more *computation* and more *memorization* of training data in weights. RETRO decouples them: it augments an autoregressive LM with direct access to a massive (trillions of tokens) text database, getting the memorization benefit from retrieval rather than parameters, without significantly increasing computation. Constraints: the database must scale to trillions of tokens, and incorporating retrieved text must cost time *linear* in the amount retrieved.

## Key ideas

1. **Frozen retriever, trainable fusion.** Embed text with a *frozen* pre-trained BERT (mean-pooled) → keys never drift → neighbours precomputable once → the database scales to trillions of tokens (unlike REALM/RAG, which must re-index during training). A *trainable* encoder + cross-attention then lets the network *reason* over retrieved text (unlike kNN-LM, which only interpolates output probabilities).
2. **Chunk-level retrieval.** Split each n-token sequence (n=2048) into l chunks of size m=64. Retrieve per chunk, not per token — cuts lookups by a factor of m and gives coherent passages to reason over.
3. **Store neighbour + continuation.** Database is key→value: key = BERT(N) (frozen, mean-pooled); value = [N, F] where N is the matched chunk and F is its *continuation* in the source document (length 64 each, so r=128). The continuation supplies "what follows text like this." Retrieve k approximate L2-nearest neighbours via SCaNN (O(log T), ~10ms over 2T tokens). Filter out same-document neighbours to prevent leakage.

## Objective (autoregressive, retrieval from earlier chunks only)

  L(X | θ, D) = Σ_{u=1}^{l} Σ_{i=1}^{m} ℓ_θ( x_{(u-1)m+i} | (x_j)_{j<(u-1)m+i}, (Ret(C_{u'}))_{u'<u} ),  Ret(C₁)=∅.

A token in chunk u uses neighbours of strictly-earlier chunks only — preserving autoregressivity (and samplability), since Ret(C_u) was computed from C_u itself.

## Architecture

Decoder-only Transformer (RMSNorm, relative positions) with retrieval blocks interleaved among standard LM blocks (one Retro-block every 3 layers, from layer 6):

  Retro(H, E) = FFW( CCA( Attn(H), E ) ),   LM(H) = FFW( Attn(H) ).

- **Neighbour encoder**: a small (~2-layer, d′=896) *bidirectional* Transformer over each neighbour, *conditioned via cross-attention on the retrieving chunk's activations H_u* — so the retrieval representation is differentiably modulated by the query chunk. Encodes all l·k neighbours in parallel → E ∈ ℝ^{l×k×r×d′}.
- **Chunked cross-attention (CCA)**: the core operator. Build *attending chunks* shifted by one token, H_u⁺ = (h_{um+i−1})_{i∈[1,m]} — so the last token of C_u is the first position to see Ret(C_u) (it predicts the next token, in chunk u+1, which is allowed). Each H_u⁺ cross-attends over E_u with neighbour and time dimensions merged (k·r attended positions) and relative positional logits (chunks ≈ aligned with neighbours). First m−1 tokens: identity. CrossAttention(h, Y) = softmax(Y K Qᵀ h) Y V (multi-head).
  - **Linear cost**: each token attends to one chunk's k·r retrieved tokens → O(n·k·r), independent of sequence length.
  - **Full prior-retrieval dependence**: though each CCA touches only E_{u-1}, self-attention propagates information so chunk u's tokens depend on *all* Ret(C_{u'})_{u'<u} — without quadratic cross-attention cost.

## Evaluation (leakage-aware)

Retrieval LMs can copy leaked training chunks at eval. So report *filtered* bits-per-byte over evaluation chunks whose maximal contiguous overlap r(C) with the nearest training chunk is below α — separating generalization from copying. Train data deduplicated (drop ≥0.8 13-gram MinHash-Jaccard matches). Benchmarks: Wikitext103, the Pile, C4 (bpb); Natural Questions (EM). MassiveText (>5T tokens), SentencePiece vocab 128k; 600B-token train retrieval DB, ~1.75T-token eval DB.

## QA fine-tuning

The retrieval pathway accepts any source: feed top-20 DPR Wikipedia passages as neighbours, format "question: {q} \n answer: {a}", left-pad so "answer:" ends the first 64-token chunk (aligns the answer with the first retrieving position), fine-tune all weights.

## Code

```python
import jax, jax.numpy as jnp
Q = jnp.zeros((d, d)); K = jnp.zeros((d, d)); V = jnp.zeros((d, d))

def cross_attention(chunk, neighbour):                  # (m,d),(r,d)
    return (chunk @ Q) @ (neighbour @ K).T, neighbour @ V    # logits (m,r), values (r,d)

def multi_neighbour_cross_attention(chunk, neighbours):     # neighbours (k,r,d)
    logits, values = jnp.vectorize(cross_attention,
        signature='(m,d),(r,d)->(m,r),(r,d)')(chunk, neighbours)
    logits += relative_positional_encodings(m, r)[None]
    logits = jnp.moveaxis(logits, 0, -1).reshape((m, r * k))    # attend over k*r at once
    values = jnp.moveaxis(values, 0, 1).reshape((r * k, d))
    return jax.nn.softmax(logits) @ values                      # (m,d), linear in k*r

def chunked_cross_attention(H, neighbours):                 # H (n,d), neighbours (l,k,r,d)
    attending = jnp.pad(H[m-1:], ((0, m-1), (0, 0))).reshape(l, m, d)   # shift by m-1
    out = jnp.vectorize(multi_neighbour_cross_attention,
                        signature='(m,d),(k,r,d)->(m,d)')(attending, neighbours)
    return jnp.pad(out.reshape(n, d), ((m-1, 0), (0, 0)))[:n]    # first m-1 tokens identity
```

```python
import torch.nn as nn
class RetroBlock(nn.Module):                                # FFW(CCA(Attn(H), E))
    def forward(self, H, E):
        H = self.attn(H)                                    # causal; propagates across chunks
        H = H + chunked_cross_attention(H, E)               # fuse retrieved content
        return self.ffw(H)

class RetrievalLM(nn.Module):
    def __init__(self, L=12, fuse_layers=(6, 9, 12)):
        super().__init__()
        self.embed, self.enc, self.readout = nn.Embedding(VOCAB, D), NeighbourEncoder(), nn.Linear(D, VOCAB)
        self.layers = nn.ModuleList([RetroBlock(D) if p in fuse_layers else LMBlock(D)
                                     for p in range(1, L+1)])
        self.fuse_layers = fuse_layers
    def forward(self, X, ret_tokens):
        H, E = self.embed(X), None
        for p, layer in enumerate(self.layers, 1):
            H = layer.attn(H)
            if p == min(self.fuse_layers):
                E = self.enc(ret_tokens, split_chunks(H, m))    # conditioned on retrieving chunk
            H = layer(H, E) if p in self.fuse_layers else layer(H)
        return self.readout(H)
```

RETRO is a semi-parametric LM: a frozen chunk-level BERT retriever over a trillion-token database, fused into a Transformer decoder by chunked cross-attention at cost linear in retrieved data, giving a large fraction of a 10×-bigger model's benefit while leaving knowledge inspectable and updatable.
