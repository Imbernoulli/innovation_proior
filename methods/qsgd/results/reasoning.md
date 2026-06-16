Let me start from what actually hurts when I scale training across machines. I have `K` workers, each holding a full copy of the parameter vector `x` in `R^n`, each computing a stochastic gradient `g` on its own slice of data, and on every iteration they all have to agree on the aggregate gradient before stepping. The way that agreement happens is communication: each worker puts its gradient on the wire and pulls in everyone else's. If the gradient is dense, that's `n` floating-point numbers — `32n` bits — sent and received per peer, every single iteration. For a small model nobody notices. But the models I care about have tens of millions of parameters, and the moment I profile a real run I see the ugly truth: as the model grows and as I add GPUs, the wall-clock time of an iteration is increasingly *not* the gradient computation, it's the gradient *exchange*. The compute bar shrinks with more parallelism, but the communication bar swells, until on a heavily parallel run most of the iteration is just shoving floats across the network. The bottleneck is bytes, not flops. So the whole game is: put fewer bits on the wire per gradient, without wrecking the optimization.

The obvious lever is to compress the gradient before sending it and decompress on the other side. People already do versions of this in production frameworks, and there's a very aggressive one I know works in practice: squash each gradient coordinate down to a single bit — basically its sign relative to a threshold of zero — reconstruct it with a couple of recomputed scaling values, and ship that. One bit per coordinate instead of thirty-two. The catch is that if I just do this raw, it diverges. The reconstruction is a lousy stand-in for the true gradient and the errors don't wash out, they accumulate against me. The trick people use to rescue it is error feedback: keep a residual buffer `Delta`, and instead of quantizing the bare gradient `G(t)`, quantize `G(t) + Delta(t-N)` where `Delta` is whatever got lopped off last time this coordinate was sent, and update `Delta(t) = G(t) - Q_inverse(G_quant(t))`. The whole gradient does eventually get folded into the model, just smeared across iterations. And empirically it holds up — it scaled speech networks beautifully.

But let me be honest about why I'm uneasy with it, because the discomfort is the seed of everything that follows. First, there is no convergence guarantee. None — not even under strong assumptions. It's a heuristic that happens to work under conditions nobody can fully characterize, and "happens to work" is exactly the thing that bites you on a model you haven't tried. Second, it's stuck at roughly one bit per coordinate; there's no dial to spend a few more bits to buy back stability, or to push compression even further. Third, the residual buffer is a whole extra copy of the model in memory, which on a GPU is not free. And I keep coming back to the first one. Why is there no guarantee? What is it about this scheme that puts it outside the reach of the convergence theory I already trust for plain SGD?

So let me write down that theory and stare at it, because the answer is going to be hiding in it. For a convex, `L`-smooth `f`, SGD with access to stochastic gradients `g` satisfying `E[g] = grad f` and a second-moment bound `E[||g||^2] <= B` converges: with a good constant step size, after `T` steps the averaged iterate obeys

  E[ f( (1/T) sum_t x_t ) ] - min f  <=  R * sqrt(2 sigma^2 / T)  +  L R^2 / T,

with `R^2 = sup ||x - x_0||^2`. To hit error `epsilon` in the regime where the first, statistical term dominates, I need

  T = O( R^2 * sigma^2 / (K epsilon^2) )

iterations — and the `1/K` is just because averaging `K` independent worker gradients is a minibatch of size `K`, dividing the variance by `K`. Now look hard at the dependence on the noise. The iteration count is *linear in the variance* of the stochastic gradient. That's the whole content of the bound from my point of view. Nothing else about the gradient matters to the theorem — not its sparsity, not its representation, not how many bits it took to send — only its variance, and only because it's unbiased so I can call it a stochastic gradient at all.

And there it is. The reason 1-bit-with-feedback has no guarantee is that its quantizer is *biased*: the expected reconstruction is not the true gradient, so it is not a stochastic gradient for `f`, so the theorem simply doesn't apply to it. Error feedback is the patch that tries to recover, over many steps, the unbiasedness it broke in one step. So flip the whole thing around. What if I never break unbiasedness in the first place? Suppose I compress with a randomized map `Q` that is *exactly unbiased in expectation*, `E[Q(g)] = g`, for every gradient. Then `Q(g)` is itself a perfectly legitimate stochastic gradient for `f` — its mean is right — and the *only* thing it does to the SGD bound is inflate the variance. If `Q` multiplies the second moment by some factor `c`, then by the bound above it multiplies the iteration count by about `c`, and changes nothing else. No error feedback. No new proof machinery. No residual buffer. I just plug the inflated variance into a theorem I already have. The bias was the enemy the entire time; the right move is to compress *unbiasedly* and pay only in variance.

So the design problem sharpens to: build a randomized quantizer `Q` that (a) is unbiased, `E[Q(v)] = v`, (b) produces values cheap to encode in few bits, and (c) has the smallest variance blowup I can manage for a given bit budget. Let me build the very simplest unbiased quantizer I can and see what it costs, because I suspect the simplest one will teach me where the pain is.

Take a single coordinate `v_i` of the gradient. I want to randomly round it to something coarse but get it right on average. The magnitude information I can keep cheaply is the norm `||v||_2` — that's one float, shared across all `n` coordinates. So normalize: `|v_i| / ||v||_2` is in `[0, 1]`. The crudest unbiased thing is to round each normalized magnitude stochastically to either `0` or `1`. Set

  Q(v_i) = ||v||_2 * sgn(v_i) * xi_i,  where xi_i = 1 with probability |v_i|/||v||_2, else 0.

Is it unbiased? `E[xi_i] = |v_i|/||v||_2`, so `E[Q(v_i)] = ||v||_2 * sgn(v_i) * |v_i|/||v||_2 = sgn(v_i)|v_i| = v_i`. Clean. Each coordinate becomes a sign bit plus a Bernoulli that's usually zero, and a single shared float for the norm. So I send the norm, plus, for each coordinate that came up `1`, its location and its sign. Beautiful and almost free.

Now what does it cost me in variance, and in bits? Variance first, because that's what the SGD bound charges me for. Compute the second moment:

  E[ ||Q(v)||^2 ] = sum_i E[ ||v||_2^2 * xi_i^2 ] = ||v||_2^2 * sum_i E[xi_i^2] = ||v||_2^2 * sum_i (|v_i|/||v||_2) = ||v||_2 * sum_i |v_i| = ||v||_2 * ||v||_1,

using `xi_i^2 = xi_i` for a 0/1 variable so `E[xi_i^2] = E[xi_i] = |v_i|/||v||_2`. Now `||v||_1 <= sqrt(n) * ||v||_2` by Cauchy-Schwarz (the `L1` norm of an `n`-vector is at most `sqrt(n)` times the `L2` norm). So

  E[ ||Q(v)||^2 ] <= sqrt(n) * ||v||_2^2.

The second-moment blowup is at most `sqrt(n)`. By the SGD bound, that means up to `sqrt(n)` times as many iterations to reach the same error. And the bits? The expected number of nonzeros is `E[ sum_i xi_i ] = sum_i |v_i|/||v||_2 = ||v||_1/||v||_2 <= sqrt(n)`, so on average I send about `sqrt(n)` locations and signs plus a float — roughly `O(sqrt(n) log n)` bits, versus `32n`. For a model with `n` around ten million that's a colossal bandwidth saving: instead of ~`n` floats I send on the order of `sqrt(n)` integers.

So I have a provably-convergent compressor, sublinear bits, and I derived it from one principle. I should be thrilled. But stare at the variance: `sqrt(n)`. For `n = 10^7` that's about three thousand. Three thousand times as many iterations to get to the same place. Even with the communication per iteration cut to ribbons, paying a `~3000x` factor on iteration count is a catastrophe — I'd never make it back. The bandwidth-per-step is great and the steps-to-converge is ruinous. I'm not happy. The simplest unbiased scheme is too crude: rounding everything to just `{0, 1}` throws away so much magnitude resolution that the variance explodes with the dimension. Wall.

What's the actual source of the blowup? Each `xi_i` is a coin flip between `0` and `1`, so for a coordinate whose normalized magnitude sits at, say, `0.5`, the quantizer slams it to either `0` or the full `||v||_2`, a huge swing around its true value — that per-coordinate variance, summed over `n` coordinates, is what piles up into `sqrt(n)`. The fix has to be: don't make each coordinate choose between `0` and `1`. Give it *intermediate* landing spots so the random rounding only has to bridge a small gap. If I lay down `s` evenly spaced levels in `[0, 1]` instead of two, then a normalized magnitude only ever rounds between two *adjacent* levels `1/s` apart, and the per-coordinate jump shrinks by a factor of `s`. That should crush the variance, at the cost of more bits to name which of the `s` levels I landed on. There's my missing knob: `s`, the number of quantization levels, smoothly trading bits for variance.

Let me build it carefully and keep unbiasedness exact. For a coordinate with normalized magnitude `a = |v_i|/||v||_2 in [0,1]`, find the integer `ell` with `0 <= ell < s` such that `a in [ell/s, (ell+1)/s]` — the quantization interval `a` falls into. I'll round to `ell/s` or `(ell+1)/s`, and the probabilities have to be set so the expectation is exactly `a`. Let `p(a,s)` be the probability of rounding *up* to `(ell+1)/s`. Then

  E[xi_i] = (ell/s)(1 - p) + ((ell+1)/s) p = ell/s + p/s.

I want this to equal `a`, so `p/s = a - ell/s`, i.e.

  p(a, s) = a*s - ell.

That's just the fractional part of `a*s` — the distance from `a` (scaled by `s`) to the lower level — which is exactly right: the closer `a` is to the upper level, the more likely I round up. And it's a valid probability, in `[0,1]`, because `a in [ell/s, (ell+1)/s]` means `a*s in [ell, ell+1]`. So define

  Q_s(v_i) = ||v||_2 * sgn(v_i) * xi_i(v, s),  xi_i = (ell+1)/s with prob p(a,s), else ell/s.

Unbiased by construction: `E[Q_s(v_i)] = ||v||_2 * sgn(v_i) * a = v_i`. And my old two-level scheme is just `s = 1`: levels `0` and `1`, `p(a,1) = a`, exactly the Bernoulli I started with. Good — it's a strict generalization, which is what I want, one dial that includes the crude version at one end.

Now the payoff has to show up in the variance. Each `xi_i` only randomizes between two adjacent levels `1/s` apart, so its variance is the variance of a (scaled) Bernoulli on a gap of size `1/s`:

  E[xi_i^2] = E[xi_i]^2 + Var(xi_i) = a^2 + (1/s^2) p(a,s)(1 - p(a,s)).

The Bernoulli is between `ell/s` and `(ell+1)/s`, a span of `1/s`, so its variance is `(1/s^2) p(1-p)` — that's where the `1/s^2` enters, and it's the whole point: more levels, smaller gap, quadratically smaller per-coordinate variance. Bound `p(1-p) <= p`, so `E[xi_i^2] <= a^2 + p/s^2`. Sum over coordinates, remembering `Q_s(v_i) = ||v||_2 * sgn * xi_i` and `a_i = |v_i|/||v||_2`:

  E[ ||Q_s(v)||^2 ] = ||v||_2^2 sum_i E[xi_i^2] <= ||v||_2^2 sum_i ( a_i^2 + p(a_i,s)/s^2 )
                    = ||v||_2^2 ( sum_i a_i^2 + (1/s^2) sum_i p(a_i,s) )
                    = ||v||_2^2 ( 1 + (1/s^2) sum_i p(a_i,s) ),

since `sum_i a_i^2 = sum_i v_i^2/||v||_2^2 = 1`. Now I need to control `sum_i p(a_i, s)`, and there are two different bounds on each `p`, which is what produces the `min` in the final answer. On the one hand `p <= 1` always, so `sum_i p <= n`, giving the term `n/s^2`. On the other hand `p(a,s) = a*s - ell <= a*s`, so `sum_i p <= s sum_i a_i = s * ||v||_1/||v||_2 <= s * sqrt(n)`, and then `(1/s^2) sum_i p <= sqrt(n)/s`. Take whichever is smaller:

  E[ ||Q_s(v)||^2 ] <= ( 1 + min( n/s^2, sqrt(n)/s ) ) ||v||_2^2,

and therefore the variance of the compression itself is

  E[ ||Q_s(v) - v||^2 ] <= min( n/s^2, sqrt(n)/s ) ||v||_2^2.

Let me sanity-check both ends. At `s = 1` the `min` is `min(n, sqrt(n)) = sqrt(n)` — exactly the blowup I derived for the crude scheme, good, it's consistent. For `1 <= s <= sqrt(n)`, the smaller branch is `sqrt(n)/s`, because `sqrt(n)/s <= n/s^2` is the same as `s <= sqrt(n)`. So the useful part of the dial moves the added variance down like `1/s` until the two bounds meet. At `s = sqrt(n)`, both branches equal `1`, so the added compression variance is at most `||v||_2^2` and the second moment is at most `2 ||v||_2^2`. That is the sweet spot. With `s = sqrt(n)` levels I've taken the dimension-dependent `sqrt(n)` penalty all the way down to a constant `2` in the second moment, meaning at most a constant-factor iteration cost. The crisis is resolved by spending more levels — exactly the bit-for-variance trade I went looking for, now made quantitative.

But wait — if I need `s = sqrt(n)` levels, naively each coordinate's level index takes `log(s) = (1/2) log n` bits, and I have `n` of them, so that's `~(n/2) log n` bits. That's worse than the `32n` I started with for big `n`. The variance is fixed but the bits ballooned. So I have to be clever about the encoding, not just dump `log s` bits per coordinate. And here's the structural fact I can exploit: the integer level indices `s * xi_i` are *not* uniformly distributed. A large integer level means a coordinate carries a large fraction of the total norm, and `||v||_2 = 1` (after normalizing) can only support a few large coordinates — most coordinates are small and round to small level indices. Large integers are rare; small integers are common. That's exactly the situation a universal integer code is built for.

So encode each level index with Elias recursive (omega) coding, where `|Elias(k)| <= (1 + o(1)) log k + 1` bits: it writes `k`'s binary representation, prepends the length of that representation, and recurses on the length, so small `k` is cheap and large `k` pays only a logarithmic premium, with linear-time decode and no need to know `k`'s size in advance. Let me actually bound the total. I'll send the tuple `(||v||_2, sigma, zeta)`: the norm in `32` bits, the signs `sigma`, and the integer levels `zeta_i = s * xi_i`. I'll walk the vector, encoding for each nonzero the *distance* to the next nonzero (so I locate them without a full bitmap), then its sign, then `Elias(zeta_i)`.

The key technical lemma I need is: for a length-`m` vector `q` of positive integers with `||q||_p^p <= rho`, the total Elias cost is

  sum_i |Elias(q_i)| <= ( (1 + o(1))/p * log(rho/m) + 1 ) * m.

Let me prove it, because the whole bit bound rides on it. Using `|Elias(q_i)| <= (1+o(1)) log q_i + 1`,

  sum_i |Elias(q_i)| <= (1+o(1)) sum_i log q_i + m
                     = ((1+o(1))/p) sum_i log(q_i^p) + m
                     <= ((1+o(1))/p) * m * log( (1/m) sum_i q_i^p ) + m,

where the last step is Jensen applied to the concave `log`: the average of `log(q_i^p)` is at most `log` of the average of `q_i^p`. Since `sum_i q_i^p = ||q||_p^p <= rho`, the average is at most `rho/m`, giving `((1+o(1))/p) m log(rho/m) + m`. That's the lemma. The two costs in my encoding both fall out of it. The positions: the gaps between consecutive nonzeros form a positive-integer vector whose entries sum to at most `n` (they tile the index range), so with `m = ||zeta||_0`, `p = 1`, and `rho = n`, the position bits are at most `((1+o(1)) log(n/m) + 1)m`. The values: per nonzero a sign bit plus `Elias(s * xi_i)`; the squared `L2` norm of the integer level vector controls the cost with `p = 2`, giving value bits `((1+o(1))/2) m log( s^2 ||zeta||_2^2 / m )` plus the sign and constant terms.

To turn those into a clean number I need two more facts: how many nonzeros there are, and how big the integer levels are. The density: how many coordinates survive quantization in expectation? Let `u_i = |v_i|/||v||_2`, so `||u||_2 = 1`, and split the coordinates by whether `u_i` exceeds `1/s`. For the large coordinates with `u_i > 1/s`, there can't be many: `1 = sum_i u_i^2 >= sum_{u_i > 1/s} u_i^2 > (number of them)/s^2`, so at most `s^2` of them. The small coordinates with `u_i <= 1/s` round between `0` and `1/s` and become nonzero only when they round up, with probability `u_i s`; their expected count is at most `s sum_i u_i = s ||u||_1 <= s sqrt(n)`. Together the expected number of nonzeros is at most on the order of `s^2 + s sqrt(n)`, i.e. `O(s(s + sqrt(n)))`, and at `s = 1` that's `O(sqrt(n))`, matching the crude scheme's density. The integer-level magnitude: each `zeta_i = s xi_i` and `xi_i <= u_i + 1/s` (it never exceeds the upper level of its interval), so `||zeta||_2^2 = s^2 sum_i xi_i^2 <= s^2 sum_i (u_i + 1/s)^2 <= 2 s^2 sum_i (u_i^2 + 1/s^2) = 2(s^2 + n)`, using `(a+b)^2 <= 2(a^2 + b^2)`.

Now feed these in. The function `x log(C/x)` is concave and peaks at `x = C/2`, so Jensen lets me push the expectation over `||zeta||_0` inside, replacing `||zeta||_0` by its bound. After substituting the density `O(s(s+sqrt n))` and the level-norm `2(s^2+n)`, the per-iteration expected bit count for `Q_s` comes out as

  ( 3 + (3/2 + o(1)) * log( 2(s^2 + n) / (s(s + sqrt(n))) ) ) * s(s + sqrt(n)) + 32.

Let me read off the two regimes I care about. The sparse end, `s = 1`: density `O(sqrt n)`, and the cost is `O(sqrt(n) log n)` bits per iteration, with the `sqrt(n)` variance blowup — great bandwidth, ruinous iteration count, the place I started. The dense end, `s = sqrt(n)`: the levels are no longer sparse, every coordinate is nonzero, so transmitting positions buys nothing — I can just send the value of every coordinate in sequence. Drop the position encoding and code each coordinate, including zeros, with a shifted Elias code `Elias'(k) = Elias(k+1)`. The dense coding calculation has the form

  E[bits] <= F + ( 2 + (1/2 + o(1)) log( 1 + (s^2 + min(n, s sqrt(n))) / n ) ) n.

The same two ingredients are doing the work: the shifted Elias length is controlled by a logarithm of the integer level, and Jensen moves the expectation inside the concave `log(1+x)`. The variance computation gives `E[||zeta||_2^2] <= s^2(1 + min(n/s^2, sqrt(n)/s)) = s^2 + min(n, s sqrt(n))`, which is the term inside the logarithm. Now plug in `s = sqrt(n)`: `min(n, s sqrt(n)) = n`, so the logarithm sees `1 + (n+n)/n = 3`. Since `2 + (1/2)log_2 3 = 2.792...`, the dense-regime cost is at most

  2.8 n + 32  bits per iteration,

against `32 n` for full precision — about a `5.7x` bandwidth reduction — while keeping the second-moment blowup at most `2`. That's the headline tradeoff, and now it's a theorem, not a hope. And it's essentially optimal: any scheme that keeps the variance blowup bounded by a constant has to send `Omega(n)` bits per round, because doing better would beat the communication lower bound for distributed mean estimation. So I can't asymptotically improve this tradeoff; I've hit the wall, which is a good place to be.

Now the convergence guarantee follows with no extra work — the payoff for insisting on unbiasedness. Let `alpha = min(n/s^2, sqrt(n)/s)`. If the original stochastic gradients have second moment at most `B`, then the quantized gradients have second moment at most `B_q = (1 + alpha)B`; equivalently, the compression itself contributes at most `alpha B` extra mean-square noise. I just substitute this inflated second-moment bound into the standard SGD theorem. Running parallel `Q_s`-SGD on `K` workers to reach error `epsilon` needs

  T = O( R^2 * max( 2 B_q / (K epsilon^2), L/epsilon ) )

iterations, with the per-round bit cost from the coding bound above (and `2.8n + 32` at `s = sqrt n`). The convergence is identical in form to full-precision SGD; only the variance/second-moment quantity changes. And it isn't limited to convex objectives: the same unbiasedness-as-variance argument rides on top of any SGD analysis. For smooth non-convex `f`, the stationarity guarantee of Ghadimi-Lan applies with the inflated second moment — there's a random stopping iterate `R` and constant step `eta = O(1/L)` such that

  (1/L) E[ ||grad f(x_R)||^2 ] <= O( sqrt(L (f(x_1) - f*)) / N  +  (1 + alpha) B / L ),

the compression entering only through the same noise-scale term. The neural networks I want to train are non-convex, so this is the version that actually matters: the quantization doesn't change the shape of the guarantee, it just scales the noise floor, which I can shrink by spending more levels.

Now let me get from the clean theory to something I'd actually run, because the gap between the full-vector `s = sqrt(n)` construction and a real GPU is where the practical version lives.

The variance bound `min(n/s^2, sqrt n/s) ||v||^2` is tied to the full dimension `n` of the tensor, and `n` is huge, so to get the blowup down to a constant I needed `s = sqrt(n)` — thousands of levels, more bits than I'd like for a 2-bit or 4-bit budget. But there's a cheap way to decouple variance from dimension: don't normalize and quantize the whole tensor at once. Flatten the gradient to a 1-D vector and chop it into buckets of `d` consecutive entries, then quantize each bucket independently with its own norm and its own `s` levels. Every bucket is just a `d`-dimensional vector, so all my lemmas apply to it with `d` in place of `n`: the per-bucket variance blowup is `min(d/s^2, sqrt d/s)`. Now the dimension in the bound is `d`, which I choose, not the unfixable `n`. Concretely, with a bucket size `d = 512` and `4` bits — so `s = 2^4 = 16` levels — the blowup is at most `sqrt(512)/16 ≈ 1.41`, while spending only 4 bits per coordinate instead of `(1/2)log n`. Bucketing costs me one extra scaling float per bucket, and it hands me a clean knob: `d = 1` is no quantization, `d = n` is the full-tensor scheme, and anything in between trades scale overhead against variance.

There's a second practical choice in the normalization. The theory normalizes by `||v||_2`, which is what gives the sparsity guarantees, but I can also normalize by the maximum absolute value of a bucket when I care more about dense fixed-width behavior than about sparse Elias coding. Dividing by the max maps the largest coordinate to exactly `1` and keeps every normalized magnitude as large as possible, so random rounding spends its levels on the range actually occupied by the bucket. The trade is clear: max-normalization forfeits the sparsity proof, while `L2` normalization is the version whose density and Elias bit accounting I just proved.

So let me write the operator I'd actually ship — the compress/decompress pair that fills the empty slot in the data-parallel exchange. Flatten; take the norm; for each coordinate, scale its absolute value into level space `s |v_i|/||v||`; the integer floor is the lower level, and I round up with probability equal to the fractional part. Carry the sign on the integer level. The payload on the wire is the vector of signed integer levels plus the single norm float; decompression is just `(||v|| / s) *` level. Memory is exactly one tensor of levels plus a scalar — no residual buffer, because I never needed error feedback.

```python
import torch

from grace_dl.torch import Compressor


class QSGDCompressor(Compressor):

    def __init__(self, quantum_num):
        super().__init__()
        self.quantum_num = quantum_num

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        norm = tensor.norm()
        norm = norm.flatten()
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = previous_level + is_next_level

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(
            torch.int8 if self.quantum_num < 128 else torch.half
        )
        tensor_compressed = tensor_compressed, norm

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, norm = tensor_compressed

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed
```

Let me trace the whole causal chain back, because every piece earned its place. The pain was that gradient communication, not computation, dominates data-parallel training as models and worker counts grow, so I have to put fewer bits on the wire. The aggressive 1-bit scheme already does that but it's a biased heuristic with no guarantee, propped up by error feedback. Reading the standard SGD bound told me why it has no guarantee — the bound charges only for variance, and only because the gradient is unbiased — and that handed me the central move: compress unbiasedly, `E[Q(g)] = g`, so `Q(g)` is just another stochastic gradient and the only cost is inflated variance, which plugs straight into the existing theorem with no error feedback at all. The crudest unbiased quantizer — round each normalized magnitude to `{0,1}` — gave sublinear bits but a `sqrt(n)` variance blowup, ruinous on high-dimensional models. Adding `s` intermediate levels with stochastic rounding to the neighboring level (probability equal to the fractional part, forced by demanding exact unbiasedness) shrank the per-coordinate jump and the added variance bound to `min(n/s^2, sqrt n/s)`, reaching added variance `1` and second-moment factor `2` at `s = sqrt(n)`. The integer levels are mostly small, so Elias coding of position-gaps and values, bounded via Jensen, brought the bits down to `O(sqrt n log n)` in the sparse regime and `2.8n + 32` in the dense regime — about `5.7x` under full precision, and provably near the `Omega(n)` lower bound. Substituting the inflated second moment into the convex SGD theorem and the Ghadimi-Lan non-convex bound gave convergence guarantees identical in form to full-precision SGD. Bucketing can decouple the variance from the unfixable dimension `n`, and max-normalization is a dense-code variant that trades away the sparsity proof. The core implementation remains one stochastic-rounding kernel, one vector of signed levels, and one scale, with no residual buffer.
