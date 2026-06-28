Let me start from something that has been bothering me about training attention on these small translation sets. I keep getting heads that collapse: I look at the attention heatmaps and a head will put almost all of its weight on one token, row after row, a near one-hot. Sometimes that is right — a head that copies, or that tracks a single antecedent, *should* be peaky. But it happens even where I would expect a diffuse, smeared-out distribution to be useful, and once a head is that sharp it has stopped being able to express anything else. So the question is whether something in the mechanism is *forcing* heads toward winner-take-all, rather than letting the data decide.

Walk through what the softmax actually does to the scores. For one query position the attention weights are softmax of a row of compatibility scores, and the scores are dot products `q · k_j` over all keys `j`, divided by `sqrt(d_k)`. The softmax of a vector depends only on the *differences* between its entries — add a constant to every entry and the output is unchanged. Fine. But now combine that with the fact that the dot product is unbounded, and stare at a concrete row. Suppose the raw scores are `[760, 752, 750]`. Let me actually take the softmax: subtract the max, `[0, -8, -10]`, exponentiate, `[1, e^-8, e^-10] = [1, 0.000335, 0.0000454]`, normalize by the sum `1.000381` — that gives `[0.99962, 0.00034, 0.00005]`. Essentially one-hot. Now do `[12, 4, 2]`: subtract the max, `[0, -8, -10]` — *the same vector*. So the softmax is `[0.99962, 0.00034, 0.00005]`, identical. The adjacent gaps were 8 and 2 in both rows, and only the gaps survive. So a gap of 8 between the top score and the next is enough to all-but-silence everything else, and whether that gap rides on a baseline of 6 or a baseline of 750 is irrelevant. The thing that catches my eye is the second row's perspective: when the scores live at large magnitudes — hundreds — an 8-point gap is *small on a relative basis*, just over one percent of 750, yet it produces a near-deterministic choice. So a head can be pushed into winner-take-all not because one connection is decisively more important but because unbounded dot products make a fixed absolute gap easy to open up from a small relative difference. What saturates the softmax is the score *differences*, not the common offset; and what lets those differences grow without limit is large query/key norms. In saturation the softmax has tiny gradients and can only express sharp distributions, so a head that drifts there gets stuck. If I want a head to keep the option of a diffuse pattern, I have to keep the score differences it sees from blowing up.

So why is the `1/sqrt(d_k)` not already handling this? Let me re-derive what that factor is for, because I want to know exactly how far it goes. Model the components of `q` and `k` as independent, mean zero, variance one. Then `q · k = sum_{i=1}^{d_k} q_i k_i`. Each product `q_i k_i` has mean `E[q_i]E[k_i] = 0` and variance `E[q_i^2 k_i^2] = E[q_i^2]E[k_i^2] = 1` (independence), so the sum of `d_k` of them has mean 0 and variance `d_k`. Before I trust this let me just simulate it: draw `q, k` as 64-dimensional standard Gaussians, take `q · k` twenty thousand times, and look at the spread. The empirical variance comes out about 64.5 against a theoretical 64, and the standard deviation about 8.0 against `sqrt(64) = 8`. So the typical magnitude of the dot product really does scale like `sqrt(d_k)`, which is exactly why for large head dimension the raw score spread can grow into the saturated regime — and dividing by `sqrt(d_k)` pulls the standard deviation back to 1. Good. But read what I just established: it controls the scale *under the initialization assumptions* — independent, unit-variance, uncorrelated `q` and `k`. The moment training starts moving the projection matrices, those assumptions rot. The query and key vectors grow norms, their components correlate, and `q · k` is free to grow again. `1/sqrt(d_k)` is a one-time variance normalization at `t = 0`; it sets the *expected* scale of a fresh layer and then does nothing. It divides by a constant — it does not cap anything. Nothing in the mechanism stops the scores from drifting back up into the saturated regime over training. That is the gap.

So what I actually want is a *bound* on the per-entry score, something that holds for all of training, not just at init. Let me think about where else I have seen "an unbounded dot product feeding a softmax causes trouble," because I have a feeling this disease has a known shape. There is the output layer of a language model. The logit for word `i` is `z_i = x_i · h`, the dot product of the word embedding with the prediction vector, and people write it in polar form, `z_i = ||x_i|| · ||h|| · cos(theta_i) + b_i`, where `theta_i` is the angle between the word vector and the prediction. There is a striking empirical finding here: the *norm* term `||x_i||` dominates the *angle* term `cos(theta_i)`. The word embeddings spread their norms over a wide range, but their angles relative to a given prediction direction fall into a narrow band, so it is the norms, not the directions, that decide which word wins the softmax. Words with small norms are systematically improbable no matter the context — there is even a clean geometric statement of it, that any word whose embedding lies in the interior of the convex hull of the vocabulary is probability-bounded — and the upshot is a structural ceiling on what the output softmax can express. The lesson seems to generalize past the output layer. Whenever you feed a *dot product* — a product of two magnitudes and a cosine — into a softmax, the magnitudes can drown out the directional agreement that you presumably cared about. And the attention score is literally the same object one layer down: `q · k = ||q|| ||k|| cos(angle)`. The magnitudes `||q||` and `||k||` can swamp the part I actually want the head to reason with, the angular agreement between the query and the key.

That suggests a direction for the fix. If the magnitudes are the saboteur — they are what is unbounded, what `1/sqrt(d_k)` only momentarily tames, and what drowns out the direction — then take them out of the score entirely and keep only the cosine. Replace `q · k` with `(q · k) / (||q|| ||k||)`, which is the cosine of the angle and also the cosine similarity that cosine normalization uses inside fully-connected layers to bound a pre-activation. Mechanically it just means normalizing each query and each key to unit length before I take their dot product:

  q_hat = q / ||q||,   k_hat = k / ||k||,   score = q_hat · k_hat.

Does this give me the bound I wanted? `q_hat · k_hat` is a dot product of two unit vectors, so by Cauchy–Schwarz it lies in `[−1, 1]` regardless of what `||q||` or `||k||` were before normalization. So yes — the score is now capped at `±1` for the entire run, not calibrated to a scale at `t = 0` and then left to drift. An entry can never run off to hundreds the way `[760, 752, 750]` did. This is the property `1/sqrt(d_k)` does not have, because it rescales by a constant rather than constraining the operation.

But I should not normalize blindly; let me be careful about *which vectors* and *along which dimension*, because there are several plausible places to put an `l2` normalization and they are not equivalent. The existing low-resource recipe already normalizes activations — ScaleNorm is `g · x / ||x||`, an `l2` normalization along the *embedding* dimension, applied to the whole residual-stream vector before the attention projections and head split. So if that already normalizes things, why hasn't it solved my problem? My suspicion is that it normalizes the wrong object for this purpose, but let me check it rather than assert it. Take a 512-dim residual vector, normalize the *whole* thing to unit length, then split it into 8 heads of dimension 64 and measure the norm of each head slice. The slice norms come out around `[0.342, 0.350, 0.357, 0.346, 0.356, 0.360, 0.316, 0.395]` — none of them anywhere near 1. And that makes sense: normalizing the full vector forces the *sum* of squared slice norms to 1 (I get 1.000 when I add the squares), so eight roughly equal slices each sit near `1/sqrt(8) ≈ 0.354`, not at unit length. So the per-head queries and keys that the per-head dot product actually contracts are *not* unit vectors after ScaleNorm, and their cosines are not bounded the way I need. The normalization therefore has to happen *after* the split, *along the head dimension* — the dimension over which the dot product is contracted — because that is the slice geometry the softmax sees. Normalize `q` and `k` along their last (head) dimension, once they are already shaped `[batch * heads, length, head_dim]`.

And which of `q`, `k`, `v`? Only `q` and `k`. The values `v` never go through the softmax — they are the things being *averaged* by the attention weights, not scored against each other. Bounding the value vectors would be solving a problem that does not exist and would distort the mixed output for no reason. So: normalize the per-head queries and keys along the head dimension, leave the values alone. If I were to test value normalization, the mechanism predicts no help and possible harm, because it deforms the representation being averaged rather than the scores doing the selecting — I would want to confirm that empirically before claiming it. This is also why I expect it to *complement* the residual-stream normalization, not replace it. ScaleNorm and LayerNorm normalize the activations flowing along the residual path; what I am doing lives strictly *inside* attention, on the per-head scores. They are orthogonal jobs. So I keep a normalization on the residual stream — vanilla LayerNorm is fine — and add the query-key normalization inside the attention scoring.

Now the part I did not see coming until I worked the numbers. Bounding the score to `[−1, 1]` kills saturation-by-magnitude — but let me check what it does to a head that *should* be sharp, because the bound cuts both ways. Picture a row of `m` cosine scores. The most favorable case for concentration is one entry at the maximum `+1` and every competitor at the minimum `−1`. With no extra scale, the top entry's softmax probability is

  e^1 / (e^1 + (m - 1)e^-1) = e^2 / (e^2 + m - 1).

For a row around length `m = 75`, plug it in: `e^2 = 7.389`, denominator `7.389 + 74 = 81.389`, so the top probability is `7.389 / 81.389 ≈ 0.091`. That is the *best case* — perfect `+1`-vs-`−1` separation — and it still cannot put even a tenth of the mass on one token. Real cosine gaps are far smaller than the full width of 2, so the actual distribution would be flatter still. I have made it impossible for a head to concentrate. That is just as bad as the original problem, only mirror-imaged: before, the head could too easily become peaky; now it cannot make a near-copy selection even when that is what the task needs. So bounding the score is necessary but not sufficient; I also have to be able to *expand* the bounded range back out when the head wants to be sharp.

The remedy is to multiply the scores up before the softmax. Note the direction has flipped relative to scaled dot product: there the dot products were too big and I *divided* by `sqrt(d_k)`; here the cosines are too small and I need to *multiply* by some factor `g > 1` to stretch `[−1, 1]` into a range where softmax can express both flat and sharp distributions. So the attention becomes

  softmax(g · q_hat k_hat^T) · v,

replacing `softmax(q k^T / sqrt(d_k)) · v`. The cosine bound gives me the floor — scores cannot run away — and the multiplier `g` buys back the ceiling. Let me check that `g` actually does buy it back, with the same `m = 75` best-case row. The top probability becomes

  e^g / (e^g + (m - 1)e^-g) = 1 / (1 + (m - 1)e^(-2g)).

At `g = 1` this is the 0.091 I just computed. At `g = 6` it is `1/(1 + 74·e^-12) = 1/(1 + 74·0.0000061) = 0.99955`. So a scale of about 6 already lets the best-case row become essentially one-hot; pushing `g` to 12 makes the top probability indistinguishable from 1. So the two pieces together — bound plus scale — give the head the *full* range from diffuse to peaky, which is what I wanted from the start. The bound alone could not do it, and a glance at these numbers shows why.

Now, what should `g` be? My first instinct is a fixed constant, the mirror of `1/sqrt(d_k)`. But the right amount of stretch is not a fixed property of the head dimension — the formula above shows it depends on the row length `m`, and even at a fixed length, how much spread a given module wants is something the model is in a better position to discover than I am. The existing recipe already pointed this way — ScaleNorm uses a single *learned* scalar `g` and lets each layer find its own ideal global scale. So make `g` a *learnable* scalar, one per attention module, and let backprop tune how much each module's scores get stretched. That is the natural object: a learned temperature on the bounded cosine scores.

I do still have to *initialize* `g`, and a bad initialization on a learnable parameter can waste a lot of the training budget before it crawls to a sensible value, so I need a starting point. The job of `g` is to make it *possible* for the maximum entry in a softmax row to dominate the others when the head wants that. Working the bound the other way: to reach a top probability of, say, 0.99 on a length-75 row I need `1/(1 + 74·e^(-2g)) = 0.99`, i.e. `74·e^(-2g) = 1/0.99 − 1 = 0.0101`, so `e^(-2g) = 0.000136` and `g ≈ 4.45`. So somewhere in the single digits is already enough to permit sharp attention on these rows; I want an initialization that lands at least there, with some headroom, and then let training tune it down or up. A purely derived row-length constant would still be a crude approximation, because real cosine matrices are not made of one `+1` and many `−1`s. So the initialization should be an empirical rule of thumb, not a theorem. I take the number of off-diagonal entries in a typical full `L × L` similarity matrix as a coarse proxy for similarity-matrix size, then compress that count with a logarithm:

  g_0 = log2(L^2 − L),

with `L` taken as the 97.5th-percentile sequence length over the training data (source and target), a robust stand-in for typical matrix size rather than the maximum, which a few outlier-length sentences would inflate. For the benchmark `L` values, 72 through 79, this gives `g_0 = log2(72^2 − 72) = log2(5112) = 12.32` up to `log2(79^2 − 79) = log2(6162) = 12.59`. So it stretches the cosine range `[−1, 1]` to roughly `[−12.5, 12.5]` — comfortably past the `≈4.45` that the bound said suffices for sharp rows, which is the headroom I wanted, but only as an initialization; the parameter is free to move. I will not pretend `log2(L^2 − L)` is derived from first principles. It is a rule of thumb found by applying softmax to similarity matrices under several heuristics and then tuning the percentile choice. The load-bearing claim is just that `g` is learned and starts high enough; this formula gives it a plausible high-scale starting value.

I want to make sure the learnable `g` is really load-bearing and not decoration, because if the bounded cosine alone did most of the work I would be overcomplicating things. The check is the `g = 1` case I already worked: a length-75 row cannot put even 10% on one entry even in the perfect-separation case, and typical rows are flatter. To make this concrete with a less extreme row, take a top entry at cosine 0.3 against 74 competitors at cosine −0.1. At `g = 1` the top probability is `e^0.3 / (e^0.3 + 74·e^-0.1) = 1.350 / (1.350 + 66.95) = 0.0198` — under 2%. At `g = 12.5` the same row gives `e^3.75 / (e^3.75 + 74·e^-1.25) = 42.5 / (42.5 + 21.2) = 0.667` — two thirds of the mass on the right token. So the scale-up moves a realistic row from near-uniform to clearly concentrated; cosine-normalizing without the learned re-scaling is not a half-measure, it removes the head's ability to make sharp selections. The scale-up is the indispensable complement to the bound.

There is a useful head-count consideration too. Because I normalize *per head*, the head dimension `d_k = d / M` only enters through the cosine, and a cosine over a 16-dimensional slice is just as well-defined as over a 256-dimensional one — I am not relying on `d_k` being large for the initial scale to be right, because I am not relying on the `sqrt(d_k)` variance argument anymore. That does not prove head-count invariance; it only says the mechanism should not break just because `d_k` becomes smaller, and I would test it by sweeping the number of heads down to small head dimensions like 16. And where this sits relative to the rest of the recipe: I keep PreNorm (residual path stays an identity map, so the residual gradient does not pick up exploding/vanishing terms from the normalization, which is what makes warmup-light low-resource training stable) and FixNorm (unit-length word embeddings, so a word's norm does not drown out its direction at the *embedding* layer — note this is the very same magnitude-vs-direction principle I just used inside attention, applied at the input side). The query-key normalization slots in as the attention-internal member of the same family. I can use vanilla LayerNorm on the residual stream because this attention-internal normalization is not meant to replace residual-stream normalization; it is aimed at the score computation.

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

Let me lay out the causal chain so I am sure it holds together. Heads on low-resource data were collapsing into winner-take-all even where diffuse attention should help; the cause is that softmax sees only score *differences* and the scaled dot product is unbounded, so query/key norm growth can make a slight, even accidental lead saturate the softmax — `[760, 752, 750]` and `[12, 4, 2]` both collapse to `[0.99962, 0.00034, 0.00005]` because their gaps coincide. The `1/sqrt(d_k)` factor only normalizes the score variance *at initialization* (I checked: `q · k` over 64-dim Gaussians has variance ≈64 and std ≈8 = `sqrt(d_k)`) under independence assumptions that decay as training moves the projections; it divides by a constant and bounds nothing. The output-softmax of language models shows the general disease — in a dot product `||q|| ||k|| cos`, the magnitudes drown out the angular agreement — so the fix is to discard the magnitudes and keep the cosine, i.e. `l2`-normalize the queries and keys, which by Cauchy–Schwarz bounds every score to `[−1, 1]` for all of training. This must be done *per head along the head dimension* — I checked that normalizing the full embedding instead leaves the head slices at norm ≈0.35, not 1 — and to `q` and `k` only (the values are averaged, not scored). The bound kills saturation-by-magnitude but overshoots: with `g = 1` the best-case top probability in a length-75 row is `e^2/(e^2 + 74) ≈ 0.091`, so long-sequence heads cannot make sharp selections. The missing piece is a *learnable* per-module scalar `g`, initialized by the empirical `log2(L^2 − L)` matrix-size heuristic (≈12.5 for these lengths, well above the ≈4.45 the bound says suffices for a sharp row), which buys back sharpness — a length-75 row at `g = 12.5` reaches ≈1 in the best case — while keeping the bound. The result, `softmax(g · q_hat k_hat^T) v`, sits inside attention and complements rather than replaces the residual-stream normalization, pairs naturally with PreNorm and FixNorm, and has a clean head-count rationale because the per-head score remains a cosine even when `d_k` changes.
