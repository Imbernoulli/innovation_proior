The problem is to build a single neural network that can ingest an arbitrary set of heterogeneous geophysical variables on a shared latitude-longitude grid. A single example has the shape V × H × W, where V is the number of physical variables and the variables differ across data sources, across pretraining runs, and across deployment settings. The standard approach treats the V axis as the channel axis of an image model, which welds the first layer to a fixed V and discards variable identity. That prevents pretraining across datasets with different variable sets, accepting a subset of variables at finetune time, or ingesting a variable that was not in the original channel list. The alternative is to move the variables onto a token axis so that each variable has its own tokens, but that multiplies the backbone sequence length by V and makes self-attention cost O((V · h · w)²), which is impractical for the few dozen variables in realistic inputs. It also leaves the backbone with a heterogeneous soup of tokens drawn from very different physical groundings, units, scales, and dynamics.

The simple reductions also fail for subtler reasons. Mean pooling over the V tokens per location accepts any V and is cheap, but it weights every variable identically regardless of location, atmospheric state, or which variable is informative. A learned scalar weight per variable is slightly better, but the weights are still static functions of variable identity, not functions of the token contents, so the combination remains content-independent and merely rescales raw tokens without re-projecting them into a shared representation. What is needed is a per-location reduction that produces a normalized, content-dependent convex combination over the variable set and re-expresses the tokens in a shared space.

The method is cross-attention variable aggregation. At each spatial location independently, a single learnable query vector attends over the V variable tokens at that location using multi-head cross-attention. The variable tokens serve as keys and values; learned projections W^K and W^V map them into the query/key and value spaces, and W^V in particular re-projects each variable token into a unified representation before mixing. For each head with dimension d_k = D / num_heads, the compatibility scores are (W^Q q) · (W^K x_v) / sqrt(d_k), normalized by a softmax over the V keys, and the head output is the weighted sum of W^V x_v. The heads are concatenated and passed through the output projection W^O. Because the softmax runs over the set of keys, the operation is permutation-invariant in the variables and defined for any number of variables V, which is simply read from the input shape at runtime. The 1/sqrt(d_k) scaling keeps the logits unit-variance and prevents the softmax from saturating into a near-one-hot regime where gradients vanish. Using multiple heads allows several distinct "which variables matter" patterns to remain separate rather than being blurred into one compromise weighting, and with d_k = D / num_heads the total cost is comparable to a single full-width head.

The learnable query is the key design choice. There is no natural source for a query because the desired output is not a transformed version of any input token but a summary of the whole set. A single trainable vector, reused at every spatial location and every example, is optimized to ask the right question of the variable set. This is the same trainable-query-summarizes-a-set pattern as the Vision Transformer class token, but applied over the variable axis instead of the spatial axis. Zero-initializing the query is important: with zero attention biases at initialization, every score is zero, so the softmax is uniform and the layer begins with the same equal-weighting behavior as mean pooling applied to the projected value tokens. From that safe prior, training learns content-dependent weighting. In fact, mean pooling and fixed learned scalar weighting are degenerate cases of this same object: mean pooling corresponds to a forced-uniform softmax with identity projections, and fixed scalar weights correspond to logits that are learned constants rather than functions of the token contents. Cross-attention variable aggregation is the general form that fixes both defects.

The cost is linear in V: O(V) per location and O(V · h · w) total for the aggregation. After aggregation the backbone sequence is back to h · w, so backbone self-attention is O((h · w)²) and independent of the number of variables. One layer and one query suffice because a single query already emits exactly one token per location; stacking more would re-attend a length-one sequence into the variable set and buy little. In practice, embed_dim D = 1024 and num_heads = 16 give d_k = 64 per head, matching standard Transformer head widths.

```python
import torch
import torch.nn as nn


class VariableAggregator(nn.Module):
    """Cross-attention variable aggregation.

    A single learnable query attends over the V variable tokens at each
    spatial location, producing one token per location.

    Input:  x : [B, V, L, D]  -- V variable tokens at each of L spatial patches.
    Output:     [B, L, D]     -- one aggregated token per spatial location.
    V is read from the input shape at runtime, so any number of variables is accepted.
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        # One learnable query. Zero-init gives uniform attention at step 0,
        # so fine-tuning starts from mean-pooling behavior and learns
        # content-dependent weights from there.
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        b, v, l, d = x.shape
        # Treat each (example, location) pair as an independent set of V tokens.
        x = x.permute(0, 2, 1, 3)   # [B, L, V, D]
        x = x.reshape(b * l, v, d)  # [B*L, V, D]

        query = self.var_query.expand(b * l, -1, -1)  # [B*L, 1, D]
        out, _ = self.var_agg(query, x, x)           # [B*L, 1, D]
        out = out.squeeze(1)                          # [B*L, D]

        out = out.reshape(b, l, d)  # [B, L, D]
        return out
```
