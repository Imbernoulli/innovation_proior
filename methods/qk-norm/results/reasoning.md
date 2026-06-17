Let me start from something that has been bothering me about training attention on these small translation sets. I keep getting heads that collapse: I look at the attention heatmaps and a head will put almost all of its weight on one token, row after row, a near one-hot. Sometimes that is right — a head that copies, or that tracks a single antecedent, *should* be peaky. But it happens even where I would expect a diffuse, smeared-out distribution to be useful, and once a head is that sharp it has stopped being able to express anything else. So the question is whether something in the mechanism is *forcing* heads toward winner-take-all, rather than letting the data decide.

Walk through what the softmax actually does to the scores. For one query position the attention weights are softmax of a row of compatibility scores, and the scores are dot products `q · k_j` over all keys `j`, divided by `sqrt(d_k)`. The softmax of a vector depends only on the *differences* between its entries — add a constant to every entry and the output is unchanged. Fine. But now combine that with the fact that the dot product is unbounded, and stare at a concrete row. Suppose the raw scores are `[760, 752, 750]`. The softmax of that is `[0.99962, 0.00034, 0.00005]` — essentially one-hot. And the softmax of `[12, 4, 2]` is *exactly the same distribution*, because the adjacent gaps are the same: 8 and 2 either way. So a gap of 8 between the top score and the next is enough to all-but-silence everything else, and whether that gap rides on a baseline of 6 or a baseline of 750 is irrelevant. The trouble is that when the scores live at large magnitudes — hundreds — an 8-point gap can be *small on a relative basis*, just over one percent, yet it produces a near-deterministic choice. The model can stumble into winner-take-all not because one connection is decisively more important but because unbounded dot products make fixed absolute gaps easy to create from small relative differences. That is the mechanism I was looking for: score differences, not common offsets, saturate the softmax; large query/key norms make those differences grow without a bound. In saturation the softmax has tiny gradients and can only express sharp distributions. If I want a head to be able to learn a diffuse pattern, I have to keep the score differences it sees from blowing up.

So why is the `1/sqrt(d_k)` not already handling this? Let me re-derive what that factor is for, because I want to know exactly how far it goes. Model the components of `q` and `k` as independent, mean zero, variance one. Then `q · k = sum_{i=1}^{d_k} q_i k_i`. Each product `q_i k_i` has mean `E[q_i]E[k_i] = 0` and variance `E[q_i^2 k_i^2] = E[q_i^2]E[k_i^2] = 1` (independence), so the sum of `d_k` of them has mean 0 and variance `d_k`. The typical magnitude of the dot product therefore scales like `sqrt(d_k)`, which is exactly why for large head dimension the raw score spread can grow large enough for the softmax to saturate — and dividing by `sqrt(d_k)` pulls the variance back to 1. Good. But read what I just proved: it controls the scale *under the initialization assumptions* — independent, unit-variance, uncorrelated `q` and `k`. The moment training starts moving the projection matrices, those assumptions rot. The query and key vectors grow norms, their components correlate, and `q · k` is free to grow again. `1/sqrt(d_k)` is a one-time variance normalization at `t = 0`; it sets the *expected* scale of a fresh layer and then does nothing. It is not a *bound*. Nothing in the mechanism stops the scores from drifting back up into the saturated regime over training. That is the gap.

What I actually want is a *bound* on the per-entry score, something that holds for all of training, not just at init. Let me think about where else I have seen "an unbounded dot product feeding a softmax causes trouble," because I have a feeling this disease has a known shape. There is the output layer of a language model. The logit for word `i` is `z_i = x_i · h`, the dot product of the word embedding with the prediction vector, and people write it in polar form, `z_i = ||x_i|| · ||h|| · cos(theta_i) + b_i`, where `theta_i` is the angle between the word vector and the prediction. There is a striking empirical finding here: the *norm* term `||x_i||` dominates the *angle* term `cos(theta_i)`. The word embeddings spread their norms over a wide range, but their angles relative to a given prediction direction fall into a narrow band, so it is the norms, not the directions, that decide which word wins the softmax. Words with small norms are systematically improbable no matter the context — there is even a clean geometric statement of it, that any word whose embedding lies in the interior of the convex hull of the vocabulary is probability-bounded — and the upshot is a structural ceiling on what the output softmax can express. The lesson generalizes past the output layer. Whenever you feed a *dot product* — a product of two magnitudes and a cosine — into a softmax, the magnitudes can drown out the directional agreement that you presumably cared about. The attention score `q · k = ||q|| ||k|| cos(angle)` is *the same object*, one layer down. The magnitudes `||q||` and `||k||` can swamp the part I actually want the head to reason with, the angular agreement between the query and the key.

That reframes the fix. If the magnitudes are the saboteur — they are what is unbounded, what `1/sqrt(d_k)` only momentarily tames, and what drowns out the direction — then take them out of the score entirely. Keep only the cosine. Replace `q · k` with `cos(angle between q and k) = (q · k) / (||q|| ||k||)`. That is exactly cosine similarity, the same move cosine normalization makes inside fully-connected layers, where the inner product `w · x` is replaced by `w · x / (||w|| ||x||)` to bound the pre-activation. Mechanically it just means normalizing each query and each key to unit length before I take their dot product:

  q_hat = q / ||q||,   k_hat = k / ||k||,   score = q_hat · k_hat.

Now the score is bounded: `q_hat · k_hat` is a cosine, so it lives in `[−1, 1]` no matter what training does to the projection norms. The saturation-by-magnitude failure mode is gone by construction — an entry can never run off to hundreds. This is the kind of guarantee `1/sqrt(d_k)` could never give, because it is a property of the operation rather than a calibration at init.

But I should not normalize blindly; let me be careful about *which vectors* and *along which dimension*, because there are several plausible places to put an `l2` normalization and they are not equivalent. The existing low-resource recipe already normalizes activations — ScaleNorm is `g · x / ||x||`, an `l2` normalization along the *embedding* dimension, applied to the whole residual-stream vector before the attention projections and head split. If that already normalizes things, why isn't it solving my problem? Because it is normalizing the wrong object for this purpose. The dot product that actually feeds the softmax is computed *per head*, between the head-dimension slices `q_i` and `k_j` of length `d_k = d / M`, *after* the projections and multi-head split. Normalizing the full pre-split embedding vector to unit length does not make the per-head query/key slices unit length, and it is the per-head slice geometry that the softmax sees. So the normalization has to happen *after* the split, *along the head dimension* — that is the dimension over which the dot product is contracted, and bounding the cosine over that dimension is what bounds the score. Normalize `q` and `k` along their last (head) dimension, once they are already shaped `[batch * heads, length, head_dim]`.

And which of `q`, `k`, `v`? Only `q` and `k`. The values `v` never go through the softmax — they are the things being *averaged* by the attention weights, not scored against each other. Bounding the value vectors would be solving a problem that does not exist and would distort the mixed output for no reason. So: normalize the per-head queries and keys along the head dimension, leave the values alone. If I were to test value normalization, the mechanism predicts no help and possible harm, because it deforms the representation being averaged rather than the scores doing the selecting. This is also why it should *complement* the residual-stream normalization, not replace it. ScaleNorm and LayerNorm normalize the activations flowing along the residual path; what I am doing lives strictly *inside* attention, on the per-head scores. They are orthogonal jobs. I keep a normalization on the residual stream — vanilla LayerNorm is fine — and add the query-key normalization inside the attention scoring.

Now the wall. I bounded the score to `[−1, 1]`, which kills saturation-by-magnitude — but it creates the opposite disease. Picture a row of `m` cosine scores. With no extra scale, even the best possible separation, one entry at `1` and every competitor at `−1`, gives the top entry probability

  e^1 / (e^1 + (m - 1)e^-1) = e^2 / (e^2 + m - 1).

For a row around length 75, that is only about 0.091. Typical cosine gaps are smaller than the full width 2, so the real distribution would be flatter still. I have made it impossible for a head to *concentrate*. That is just as bad as the original problem, only mirror-imaged: before, the head could too easily become peaky; now it cannot make a near-copy selection even when that is what the task needs. So bounding the score is necessary but not sufficient; I have to be able to *expand* the bounded range back out when the head wants to be sharp.

The fix is to multiply the scores up before the softmax. Note the direction has flipped relative to scaled dot product: there the dot products were too big and I *divided* by `sqrt(d_k)`; here the cosines are too small and I need to *multiply* by some factor `g > 1` to stretch `[−1, 1]` into a range where softmax can express both flat and sharp distributions. So the attention becomes

  softmax(g · q_hat k_hat^T) · v,

replacing `softmax(q k^T / sqrt(d_k)) · v`. The cosine bound gives me the floor — scores cannot run away — and the multiplier `g` buys back the ceiling — scores can be stretched far enough apart for the softmax to single one out when needed. The two together give the head the *full* range from diffuse to peaky, which is what I wanted from the start.

Now, what should `g` be? My first instinct is a fixed constant, the mirror of `1/sqrt(d_k)`. But the right amount of stretch is not a fixed property of the head dimension — it depends on the row length and on how sharp a particular attention module's heads need to become. Even at a fixed length, how much spread a given module wants is something the model is in a better position to discover than I am. The existing recipe already taught me this lesson — ScaleNorm uses a single *learned* scalar `g` and lets each layer find its own ideal global scale. So make `g` a *learnable* scalar, one per attention module, and let backprop tune how much each module's scores get stretched. That is the natural object: a learned temperature on the bounded cosine scores.

I do still have to *initialize* `g`, and a bad initialization on a learnable parameter can waste a lot of the training budget before it crawls to a sensible value, so I need a starting point. The job of `g` is to make it *possible* for the maximum entry in a softmax row to come out near 1, i.e. to dominate all the others. With scale `g`, the best-case row probability is

  e^g / (e^g + (m - 1)e^-g) = 1 / (1 + (m - 1)e^(-2g)).

So longer sequences need more scale, but a purely derived row-length constant would still be a crude approximation because real cosine matrices are not made of one `1` and many `−1`s. The initialization should therefore be an empirical rule of thumb, not a theorem. I use the number of off-diagonal entries in a typical full `L × L` similarity matrix as a coarse proxy for similarity-matrix size, then compress that count with a logarithm:

  g_0 = log2(L^2 − L),

with `L` taken as the 97.5th-percentile sequence length over the training data (source and target), a robust stand-in for typical matrix size rather than the maximum, which a few outlier-length sentences would inflate. For the benchmark `L` values, 72 through 79, this gives `g_0` from `log2(72^2 - 72) = 12.32` to `log2(79^2 - 79) = 12.59`. It stretches the cosine range of `[−1, 1]` to roughly `[−12.5, 12.5]`, but only as an initialization; the parameter is free to move. I will not pretend `log2(L^2 − L)` is derived from first principles. It is a rule of thumb found by applying softmax to similarity matrices under several heuristics and then tuning the percentile choice. The important part for the mechanism is that `g` is learned, while this formula gives it a plausible high-scale starting value.

I want to sanity-check that the learnable `g` is really load-bearing and not decoration, because if the bounded cosine alone did most of the work I would be overcomplicating things. Imagine fixing `g = 1`, leaving the scores stuck in `[−1, 1]`. By the probability bound above, a length-75 row cannot put even 10% on one entry in the best possible case, and typical rows will be flatter. That is not a usable replacement for attention heads that sometimes need near-copy behavior. This tells me the scale-up is the indispensable complement to the bound: cosine-normalizing without the learned re-scaling is not a half-measure, it removes the head's ability to make sharp selections.

There is a useful head-count sanity check. Because I normalize *per head*, the head dimension `d_k = d / M` only enters through the cosine, and a cosine over a 16-dimensional slice is just as well-defined as over a 256-dimensional one — I am not relying on `d_k` being large for the initial scale to be right, because I am not relying on the `sqrt(d_k)` variance argument anymore. That does not prove head-count invariance; it only says the mechanism should not break just because `d_k` becomes smaller. I would test this by sweeping the number of heads, including small head dimensions. And where this sits relative to the rest of the recipe: I keep PreNorm (residual path stays an identity map, so the residual gradient does not pick up exploding/vanishing terms from the normalization, which is what makes warmup-light low-resource training stable) and FixNorm (unit-length word embeddings, so a word's norm does not drown out its direction at the *embedding* layer — note this is the very same magnitude-vs-direction principle I just used inside attention, applied at the input side). The query-key normalization slots in as the attention-internal member of the same family. I can use vanilla LayerNorm on the residual stream because this attention-internal normalization is not meant to replace residual-stream normalization; it is aimed at the score computation.

So let me write the scoring path the way I would actually ship it, filling the one open slot — how the per-head queries and keys become the scores that feed the softmax. The learnable scale is a one-parameter module so it lives naturally in the module's parameter list and gets its gradient like everything else:

```python
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


class ScaleUp(nn.Module):
    """One learnable scalar g that stretches the bounded cosine scores before softmax."""
    def __init__(self, scale):
        super().__init__()
        self.scale = Parameter(torch.tensor(scale))

    def forward(self, x):
        return x * self.scale


class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = args.att_dropout
        self.use_bias = args.use_bias
        self.mha_sup = args.mha_sup
        self.seq_len_threshold = args.seq_len_threshold

        # packed q/k/v/o projections; smaller-than-Xavier std for stability
        self.weights = Parameter(torch.Tensor(4 * self.embed_dim, self.embed_dim))
        if self.use_bias:
            self.biases = Parameter(torch.Tensor(4 * self.embed_dim))
        nn.init.normal_(self.weights, mean=0.0, std=(2 / (5 * self.embed_dim)) ** 0.5)
        if self.use_bias:
            nn.init.constant_(self.biases, 0.)

        if self.mha_sup:
            self.mha_scale = ScaleUp(np.log2(self.seq_len_threshold ** 2 - self.seq_len_threshold))
        else:
            self.scale = self.head_dim ** -0.5

    def _split_heads(self, x):
        bsz, length, _ = x.size()
        return (x.reshape(bsz, length, self.num_heads, self.head_dim)
                 .transpose(1, 2)
                 .reshape(bsz * self.num_heads, -1, self.head_dim))

    def forward(self, q, k, v, mask, do_proj_qkv=True):
        if do_proj_qkv:
            q, k, v = self.proj_qkv(q, k, v)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        if self.mha_sup:
            # QKNorm branch: cosine scores, then learned scale-up before softmax.
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            att = self.mha_scale(torch.bmm(q, k.transpose(1, 2)))
        else:
            att = torch.bmm(q, k.transpose(1, 2)) * self.scale

        bsz_x_num_heads, src_len, tgt_len = att.size()
        bsz = bsz_x_num_heads // self.num_heads
        att = att.reshape(bsz, self.num_heads, src_len, tgt_len)
        if mask is not None:
            att.masked_fill_(mask, -1e9)
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)

        out = torch.bmm(att.reshape(bsz_x_num_heads, src_len, tgt_len), v)   # values are NOT normalized
        out = (out.reshape(bsz, self.num_heads, src_len, self.head_dim)
                  .transpose(1, 2)
                  .reshape(bsz, src_len, self.embed_dim))
        return self.proj_o(out), att

    # proj_qkv / proj_o: existing packed linear projections
```

Let me lay out the causal chain so I am sure it holds together. Heads on low-resource data were collapsing into winner-take-all even where diffuse attention should help; the cause is that softmax sees only score *differences* and the scaled dot product is unbounded, so query/key norm growth can make a slight, even accidental lead saturate the softmax. The `1/sqrt(d_k)` factor only normalizes the score variance *at initialization* under independence assumptions that decay as training moves the projections; it bounds nothing. The output-softmax of language models shows the general disease — in a dot product `||q|| ||k|| cos`, the magnitudes drown out the angular agreement — so the fix is to discard the magnitudes and keep the cosine, i.e. `l2`-normalize the queries and keys, which must be done *per head along the head dimension* (the dimension the dot product contracts, after the split) and to `q` and `k` only (the values are averaged, not scored). That bounds every score to `[−1, 1]` for all of training, killing saturation-by-magnitude — but it overshoots, because with `g = 1` the best-case top probability in a length-`m` row is only `e^2/(e^2+m-1)`, so long-sequence heads cannot make sharp selections. The missing piece is a *learnable* per-module scalar `g`, initialized by the empirical `log2(L^2 − L)` matrix-size heuristic, which buys back sharpness while keeping the bound. The result, `softmax(g · q_hat k_hat^T) v`, sits inside attention and complements rather than replaces the residual-stream normalization, pairs naturally with PreNorm and FixNorm, and has a clean head-count sanity check because the per-head score remains a cosine even when `d_k` changes.
