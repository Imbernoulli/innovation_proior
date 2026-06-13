Let me start from something that has been bothering me about training attention on these small translation sets. I keep getting heads that collapse: I look at the attention heatmaps and a head will put almost all of its weight on one token, row after row, a near one-hot. Sometimes that is right — a head that copies, or that tracks a single antecedent, *should* be peaky. But it happens even where I would expect a diffuse, smeared-out distribution to be useful, and once a head is that sharp it has stopped being able to express anything else. So the question is whether something in the mechanism is *forcing* heads toward winner-take-all, rather than letting the data decide.

Walk through what the softmax actually does to the scores. For one query position the attention weights are softmax of a row of compatibility scores, and the scores are dot products `q · k_j` over all keys `j`, divided by `sqrt(d_k)`. The softmax of a vector depends only on the *differences* between its entries — add a constant to every entry and the output is unchanged. Fine. But now combine that with the fact that the dot product is unbounded, and stare at a concrete row. Suppose the raw scores are `[760, 752, 750]`. The softmax of that is `[0.99962, 0.00034, 0.00005]` — essentially one-hot. And the softmax of `[12, 4, 2]` is *exactly the same distribution*, because the differences are the same: 8 and 2 either way. So a gap of 8 between the top score and the next is enough to all-but-silence everything else, and whether that gap rides on a baseline of 6 or a baseline of 750 is irrelevant. The trouble is that when the scores live at large magnitudes — hundreds — an 8-point gap is *small on a relative basis*, just over one percent, yet it produces a near-deterministic choice. The model can stumble into winner-take-all not because one connection is decisively more important but because the dot products happen to be big, so even a slight, possibly accidental, lead gets amplified into a near one-hot row. That is the mechanism I was looking for: large unbounded scores push the softmax into saturation, and in saturation it has tiny gradients and can only express sharp distributions. If I want a head to be able to learn a diffuse pattern, I have to keep the scores it sees from blowing up.

So why is the `1/sqrt(d_k)` not already handling this? Let me re-derive what that factor is for, because I want to know exactly how far it goes. Model the components of `q` and `k` as independent, mean zero, variance one. Then `q · k = sum_{i=1}^{d_k} q_i k_i`. Each product `q_i k_i` has mean `E[q_i]E[k_i] = 0` and variance `E[q_i^2 k_i^2] = E[q_i^2]E[k_i^2] = 1` (independence), so the sum of `d_k` of them has mean 0 and variance `d_k`. The typical magnitude of the dot product therefore scales like `sqrt(d_k)`, which is exactly why for large head dimension the raw scores are large and the softmax saturates — and dividing by `sqrt(d_k)` pulls the variance back to 1. Good. But read what I just proved: it controls the scale *under the initialization assumptions* — independent, unit-variance, uncorrelated `q` and `k`. The moment training starts moving the projection matrices, those assumptions rot. The query and key vectors grow norms, their components correlate, and `q · k` is free to grow again. `1/sqrt(d_k)` is a one-time variance normalization at `t = 0`; it sets the *expected* scale of a fresh layer and then does nothing. It is not a *bound*. Nothing in the mechanism stops the scores from drifting back up into the saturated regime over training. That is the gap.

What I actually want is a *bound* on the per-entry score, something that holds for all of training, not just at init. Let me think about where else I have seen "an unbounded dot product feeding a softmax causes trouble," because I have a feeling this disease has a known shape. There is the output layer of a language model. The logit for word `i` is `z_i = x_i · h`, the dot product of the word embedding with the prediction vector, and people write it in polar form, `z_i = ||x_i|| · ||h|| · cos(theta_i) + b_i`, where `theta_i` is the angle between the word vector and the prediction. There is a striking empirical finding here: the *norm* term `||x_i||` dominates the *angle* term `cos(theta_i)`. The word embeddings spread their norms over a wide range, but their angles relative to a given prediction direction fall into a narrow band, so it is the norms, not the directions, that decide which word wins the softmax. Words with small norms are systematically improbable no matter the context — there is even a clean geometric statement of it, that any word whose embedding lies in the interior of the convex hull of the vocabulary is probability-bounded — and the upshot is a structural ceiling on what the output softmax can express. The lesson generalizes past the output layer. Whenever you feed a *dot product* — a product of two magnitudes and a cosine — into a softmax, the magnitudes can drown out the directional agreement that you presumably cared about. The attention score `q · k = ||q|| ||k|| cos(angle)` is *the same object*, one layer down. The magnitudes `||q||` and `||k||` can swamp the part I actually want the head to reason with, the angular agreement between the query and the key.

That reframes the fix. If the magnitudes are the saboteur — they are what is unbounded, what `1/sqrt(d_k)` only momentarily tames, and what drowns out the direction — then take them out of the score entirely. Keep only the cosine. Replace `q · k` with `cos(angle between q and k) = (q · k) / (||q|| ||k||)`. That is exactly cosine similarity, the same move cosine normalization makes inside fully-connected layers, where the inner product `w · x` is replaced by `w · x / (||w|| ||x||)` to bound the pre-activation. Mechanically it just means normalizing each query and each key to unit length before I take their dot product:

  q_hat = q / ||q||,   k_hat = k / ||k||,   score = q_hat · k_hat.

Now the score is bounded: `q_hat · k_hat` is a cosine, so it lives in `[−1, 1]` no matter what training does to the projection norms. The saturation-by-magnitude failure mode is gone by construction — an entry can never run off to hundreds. This is the kind of guarantee `1/sqrt(d_k)` could never give, because it is a property of the operation rather than a calibration at init.

But I should not normalize blindly; let me be careful about *which vectors* and *along which dimension*, because there are several plausible places to put an `l2` normalization and they are not equivalent. The existing low-resource recipe already normalizes activations — ScaleNorm is `g · x / ||x||`, an `l2` normalization along the *embedding* dimension, applied to the whole vector *before* it gets split into heads, and applied to queries, keys *and* values. If that already normalizes things, why isn't it solving my problem? Because it is normalizing the wrong object for this purpose. The dot product that actually feeds the softmax is computed *per head*, between the head-dimension slices `q_i` and `k_j` of length `d_k = d / M`, *after* the multi-head split. Normalizing the full pre-split embedding vector to unit length does not make the per-head slices unit length, and it is the per-head slice geometry that the softmax sees. So the normalization has to happen *after* the split, *along the head dimension* — that is the dimension over which the dot product is contracted, and bounding the cosine over that dimension is what bounds the score. Normalize `q` and `k` along their last (head) dimension, once they are already shaped `[batch * heads, length, head_dim]`.

And which of `q`, `k`, `v`? Only `q` and `k`. The values `v` never go through the softmax — they are the things being *averaged* by the attention weights, not scored against each other. Bounding the value vectors would be solving a problem that does not exist and would distort the mixed output for no reason. So: normalize the per-head queries and keys along the head dimension, leave the values alone. (I can already imagine the ablation: throwing `v` into the normalization should not help and may hurt, because it deforms the averaged representation — and indeed I would bet on it being slightly worse, not better.) This is also why it should *complement* the residual-stream normalization, not replace it. ScaleNorm and LayerNorm normalize the activations flowing along the residual path; what I am doing lives strictly *inside* attention, on the per-head scores. They are orthogonal jobs. I keep a normalization on the residual stream — vanilla LayerNorm is fine — and add the query-key normalization inside the attention scoring.

Now the wall. I bounded the score to `[−1, 1]`, which kills saturation-by-magnitude — but it creates the opposite disease. Picture a row of cosine scores: they are all in `[−1, 1]`, so the largest gap between any two entries is at most 2, and typically much smaller. Push that row through softmax. The softmax of values squeezed into `[−1, 1]` is nearly *uniform* — the differences are too small for it to ever put decisive weight anywhere. I have made it impossible for a head to *concentrate*. That is just as bad as the original problem, only mirror-imaged: before, the head could only be peaky; now it can only be flat. softmax is a function of differences, and I have made all the differences tiny. A head that needs to copy from one specific token, to attend almost entirely to a single antecedent, simply cannot, because the most it can do is `cos = 1` vs `cos = something a little less than 1`, a gap of a fraction, which softmax turns into a mild preference, not a selection. So bounding the score is necessary but not sufficient; I have to be able to *expand* the bounded range back out when the head wants to be sharp.

The fix is to multiply the scores up before the softmax. Note the direction has flipped relative to scaled dot product: there the dot products were too big and I *divided* by `sqrt(d_k)`; here the cosines are too small and I need to *multiply* by some factor `g > 1` to stretch `[−1, 1]` into a range where softmax can express both flat and sharp distributions. So the attention becomes

  softmax(g · q_hat k_hat^T) · v,

replacing `softmax(q k^T / sqrt(d_k)) · v`. The cosine bound gives me the floor — scores cannot run away — and the multiplier `g` buys back the ceiling — scores can be stretched far enough apart for the softmax to single one out when needed. The two together give the head the *full* range from diffuse to peaky, which is what I wanted from the start.

Now, what should `g` be? My first instinct is a fixed constant, the mirror of `1/sqrt(d_k)`. But the right amount of stretch is not a fixed property of the head dimension — it depends on how *many* entries are competing in each softmax row, which is the sequence length, which is a property of the *data*, not the architecture. And more: even at a fixed length, how much spread a given layer wants is something the model is in a better position to discover than I am. The existing recipe already taught me this lesson — ScaleNorm replaced a hand-set `sqrt(d)` scale with a single *learned* scalar `g` and showed that one learnable scalar per normalization is enough, and that it lets each layer find its own ideal global scale. So make `g` a *learnable* scalar, one per attention layer, and let backprop tune how much each head's scores get stretched. That is the natural object: a per-layer learned temperature on the bounded cosine scores. (And I would expect the learned `g` values to come out *larger* in deeper layers, where representations are more discriminative and the head wants sharper selection — a small prediction I can check later by reading the trained values off.)

I do still have to *initialize* `g`, and a bad initialization on a learnable parameter can waste a lot of the training budget before it crawls to a sensible value, so let me reason about the right starting point rather than guess. The job of `g` is to make it *possible* for the maximum entry in a softmax row to come out near 1, i.e. to dominate all the others. How hard that is depends on how many others there are. In a row over a length-`L` sequence there are `L` competing entries, and across the full `L × L` score matrix the number of off-diagonal entries — the entries that a given position has to out-compete — is `L^2 − L`. Longer sequences mean more competitors, which means a single entry has to be stretched further above the rest to win the softmax, which means a larger `g`. So I want `g_0` to grow with the entry count `L^2 − L`. It should not grow *linearly* — that would be enormous and would slam every head into saturation from step one, undoing the whole point — so a compressive function of the count is what I want. A logarithm is the natural compressive choice, and base 2 gives convenient magnitudes, so

  g_0 = log2(L^2 − L),

with `L` taken as the 97.5th-percentile sequence length over the training data (source and target), a robust stand-in for "how long are the rows the softmax has to handle," rather than the maximum, which a few outlier-length sentences would inflate. For a typical `L` around 72-79 this puts `g_0` around 12.3-12.6, which stretches a cosine range of `[−1, 1]` out to roughly `[−12.5, 12.5]` — wide enough that a top entry can dominate, narrow enough that a head can still be diffuse if it wants. I will not pretend this `log2(L^2 − L)` is derived from first principles; it is a rule of thumb, arrived at by applying softmax to similarity matrices scaled by various heuristics and seeing what lets the max soft-max to 1 — exactly the same epistemic status as `sqrt(d_k)` in scaled dot product, which is *also* a rule of thumb. The difference is that here it only sets the *starting point* of a parameter the model is free to move. (If I sweep which percentile to use for `L` — 75th, 90th, 95th, 97.5th, 99th, the max — I would expect a sweet spot rather than monotone behavior, since too small under-scales short-batch rows and too large over-scales; my bet is the high-but-not-extreme percentile wins.)

I want to sanity-check that the learnable `g` is really load-bearing and not decoration, because if the bounded cosine alone did most of the work I would be overcomplicating things. Imagine ablating `g` — equivalently fixing `g = 1`, leaving the scores stuck in `[−1, 1]`. Then by the argument three paragraphs up the softmax is nearly uniform everywhere, the head can never concentrate, and translation quality should *crater*, not merely dip. I would expect this to be by far the largest drop of any ablation, much larger than removing the residual-stream normalization, because without `g` the attention is essentially broken rather than merely worse. That prediction is exactly what tells me the scale-up is the indispensable complement to the bound: cosine-normalizing without the learned re-scaling is not a half-measure, it is a non-functional attention.

There is a nice consequence for the head count. Because I normalize *per head*, the head dimension `d_k = d / M` only enters through the cosine, and a cosine over a 16-dimensional slice is just as well-defined as over a 256-dimensional one — I am not relying on `d_k` being large for the scale to be right, because I am not relying on the `sqrt(d_k)` variance argument at all anymore. So the method should be robust to the head count in a way scaled dot product is not, where a tiny `d_k` makes the `sqrt(d_k)` calibration shaky. I would expect performance to hold roughly flat from a handful of heads up to many small heads. And where this sits relative to the rest of the recipe: I keep PreNorm (residual path stays an identity map, so the residual gradient does not pick up exploding/vanishing terms from the normalization, which is what makes warmup-light low-resource training stable) and FixNorm (unit-length word embeddings, so a word's norm does not drown out its direction at the *embedding* layer — note this is the very same magnitude-vs-direction principle I just used inside attention, applied at the input side). The query-key normalization slots in as the attention-internal member of the same family. And I keep vanilla LayerNorm on the residual stream rather than ScaleNorm here, because my method already supplies the bounded-cosine geometry inside attention and I do not need ScaleNorm's competing global-scale bias on the residual path on top of it.

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

        # packed q/k/v/o projections; smaller-than-Xavier std for stability
        self.weights = Parameter(torch.Tensor(4 * self.embed_dim, self.embed_dim))
        nn.init.normal_(self.weights, mean=0.0, std=(2 / (5 * self.embed_dim)) ** 0.5)

        # g, initialized from the typical row size so the max entry can softmax to ~1:
        # L^2 - L is the off-diagonal count of an L x L score matrix; log2 compresses it.
        L = args.seq_len_threshold                      # 97.5th-percentile sequence length
        self.mha_scale = ScaleUp(np.log2(L ** 2 - L))

    def _split_heads(self, x):
        bsz, length, _ = x.size()
        return (x.reshape(bsz, length, self.num_heads, self.head_dim)
                 .transpose(1, 2)
                 .reshape(bsz * self.num_heads, -1, self.head_dim))

    def forward(self, q, k, v, mask):
        q, k, v = self.proj_qkv(q, k, v)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        # bound each score to [-1, 1] by normalizing per-head q and k along the head dim,
        # so the entry q_hat . k_hat is the cosine similarity (magnitudes can't saturate softmax);
        # then stretch by the learnable g so the head can still be sharp when it wants.
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        att = self.mha_scale(torch.bmm(q, k.transpose(1, 2)))   # g * q_hat k_hat^T

        bh, src_len, tgt_len = att.size()
        att = att.reshape(-1, self.num_heads, src_len, tgt_len)
        if mask is not None:
            att.masked_fill_(mask, -1e9)
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)

        out = torch.bmm(att.reshape(bh, src_len, tgt_len), v)   # values are NOT normalized
        out = (out.reshape(-1, self.num_heads, src_len, self.head_dim)
                  .transpose(1, 2)
                  .reshape(-1, src_len, self.embed_dim))
        return self.proj_o(out), att

    # proj_qkv / proj_o: existing packed linear projections
```

Let me lay out the causal chain so I am sure it holds together. Heads on low-resource data were collapsing into winner-take-all even where diffuse attention should help; the cause is that softmax sees only score *differences* and the scaled dot product is unbounded, so large scores let a slight, even accidental lead saturate the softmax. The `1/sqrt(d_k)` factor only normalizes the score variance *at initialization* under independence assumptions that decay as training moves the projections; it bounds nothing. The output-softmax of language models shows the general disease — in a dot product `||q|| ||k|| cos`, the magnitudes drown out the angular agreement — so the fix is to discard the magnitudes and keep the cosine, i.e. `l2`-normalize the queries and keys, which must be done *per head along the head dimension* (the dimension the dot product contracts, after the split) and to `q` and `k` only (the values are averaged, not scored). That bounds every score to `[−1, 1]` for all of training, killing saturation-by-magnitude — but it overshoots, because cosines squeezed into `[−1, 1]` make the softmax nearly uniform and a head can no longer concentrate. So multiply the bounded scores by a *learnable* per-layer scalar `g`, initialized to `log2(L^2 − L)` from the typical row size (so the maximum entry can still softmax toward 1), which buys back the ability to be sharp while keeping the floor; without `g` the attention is non-functional, which is why the scale-up is the indispensable complement to the bound. The result, `softmax(g · q_hat k_hat^T) v`, sits inside attention and complements rather than replaces the residual-stream normalization, pairs naturally with PreNorm and FixNorm, and is robust to the head count because the per-head cosine never depended on `d_k` being large.
