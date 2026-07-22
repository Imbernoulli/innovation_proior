I want to keep the one part of the earlier solver that is clearly doing the right kind of work: it refuses to run global attention over all mesh points. A million-point car or aircraft mesh is not a sensible attention sequence. The slice operator already has the right skeleton: assign each point to a small set of physical states, average the point features into `M` state tokens, run attention among those tokens, and send the updated states back to the points. That gives me `O(N M C + M^2 C)` instead of `O(N^2 C)`, and with `M` fixed at a few dozen the point count appears only linearly. So I am not looking for a new backbone. I am looking for the precise reason this skeleton breaks when the mesh becomes huge.

The first place I distrust is the weights. The old slice weights are `w = softmax(Linear(x) / tau0)`. If the weights are too flat, every state token becomes nearly the same weighted average of all points. Then the later `M`-token attention is only attention among copies of a global mean, and the whole physical-state story collapses. The old softmax does help by making a distribution over states, but it still has one global sharpness scale in the formula. That single scale is what I want to interrogate. A point deep inside a slowly changing region should commit strongly to one state; a point on a sharp physical transition can honestly belong to a mixture of states. Can one temperature express both?

Let me actually look at what the temperature does to a fixed set of logits, so I am not arguing from intuition. Take logits `[2.0, 0.5, -0.5, -1.0, 0.3]` over five states and sweep `tau`, measuring the entropy of the resulting softmax (max entropy for five states is `log 5 = 1.609`):

```text
tau = 0.05 : argmax-mass = 1.000  entropy = 0.000
tau = 1.0  : argmax-mass = 0.650  entropy = 1.080
tau = 20   : argmax-mass = 0.218  entropy = 1.608
```

So the sharpness genuinely lives in `tau`: small `tau` drives a near one-hot assignment (all mass on one state, zero entropy), large `tau` drives the near-uniform mixture (entropy almost at the `1.609` ceiling). That confirms the lever I want is the temperature itself, and that a single global `tau` forces the same entropy budget on every point. The clean-region point and the transition-region point above would be reading off the same column of this table, which is exactly the wrong coupling.

The fix that follows is to make the temperature local. For point `i`, I keep the base temperature `tau0`, but I add a learned pointwise correction from the point feature: `tau_i = tau0 + Linear(x_i)`. In implementation I make this a small per-head predictor, `dim_head -> slice_num -> 1`, add a learned per-head bias initialized at `0.5`, and clamp the result below by `0.01`. The clamp matters because the temperature is a denominator; if it reaches zero, the logits and gradients can blow up. I can read the limiting behavior straight off the sweep above: a constant predicted `tau_i` reproduces the old constant-temperature operator (it just picks one row of the table for everyone), `tau_i` near `0.05` gives a sharp choice, and a large `tau_i` gives the uniform mixture that is only appropriate where the local physics is genuinely mixed. So the local temperature buys per-point control over precisely that entropy axis.

But a locally chosen softmax temperature is still only a softened assignment. I want the assignment process itself to behave like a draw from a categorical distribution, because each mesh point should usually choose one state rather than weakly vote for many states. Direct `argmax` is not usable because it blocks gradients. The standard escape is the Gumbel reparameterization: perturb the logits with Gumbel noise and take a tempered softmax,

```text
w_i = softmax((Linear(x_i) + g_i) / tau_i),  g_i = -log(-log(u_i)),  u_i ~ Uniform(0,1).
```

I want to be careful here, because I always have to stop and think about the sign of `g`. It is easy to write `g = log(-log(u))`, and I cannot tell by staring at it whether the negative sign belongs out front. The claim that makes this useful is that the argmax of `logits + g` is distributed as a categorical with probabilities `softmax(logits)` — so let me just check it by sampling rather than trusting the algebra. With logits `[2, 1, 0, -1]`, `softmax(logits) = [0.6439, 0.2369, 0.0871, 0.0321]`. Drawing 400k Gumbel perturbations and recording which index wins the argmax:

```text
target softmax probs : [0.6439, 0.2369, 0.0871, 0.0321]
empirical, g = -log(-log u) : [0.6436, 0.2375, 0.0867, 0.0322]   (max abs diff 0.0007)
empirical, g = +log(-log u) : [0.7010, 0.2413, 0.0516, 0.0061]   (max abs diff 0.0571)
```

So `g = -log(-log(u))` reproduces the categorical probabilities to within sampling noise, and flipping the sign quietly distorts the tails — it over-weights the dominant state and starves the small ones by almost a factor of five on the last entry. That settles the sign: the negative is part of the sample, and `-log(-log(u))` is the Gumbel variable, not `log(-log(u))`. With low temperature this relaxation approaches a one-hot draw (the `tau = 0.05` row above), with higher temperature it stays soft. That gives me sharp state choices in clean regions and multi-state hedging around fast-changing boundaries, with gradients flowing through the softmax the whole time.

Now I check the state computation. With weights `w_ij`, the state token is still a weighted mean,

```text
s_j = sum_i w_ij x_i / (sum_i w_ij + 1e-5).
```

The `1e-5` is only a numerical stabilizer in the denominator. It does not change the intended estimator when a state has nontrivial mass, but it prevents division by zero for an empty or nearly empty state. The deslice is still `x'_i = sum_j w_ij s'_j`, so the weights are used in both directions. If `M = 1`, every point maps to the same state token and the operator is global pooling plus a pointwise broadcast, so it loses correlations. If `M` is too large, cost increases and states can fragment into small noisy groups. The practical range of 32 or 64 states is the compromise: enough states to separate regimes, few enough for cheap state attention.

There is a second efficiency issue hiding in the old implementation. It projects each point into one feature stream that decides the slice and another feature stream that supplies the content to average. That separation is defensible when the slice weights are weak, but it doubles pointwise projection memory at exactly the scale where memory is scarce. Once the state assignment is made sharper and more local, I can let one stream do both jobs: project `x` once to `x_mid`, use `x_mid` to compute the slice logits and temperature, and also use `x_mid` as the value averaged into states. The old content stream `f` is no longer needed. To see this is not cosmetic, count the per-point projected activations at width `inner = 256`: two streams store `2 * 256 = 512` floats per point per layer, one stream stores `256`. At `N = 1.2e6` points that is the difference between `6.1e8` and `3.1e8` activation floats for the projection alone, before any of the attention tensors — a genuine factor-of-two on the dominant per-point term, plus halving that Linear's parameters.

The multi-GPU case is the place where the sums must be exact, and it is the part I am least willing to assert by hand. Suppose the point set is sharded over GPUs, with GPU `k` holding `N_k` local points and local weights `w^(k)`. Each GPU can compute its local state mass `sum_i w_ij^(k)` and local numerator `sum_i w_ij^(k) x_i^(k)`. The tempting shortcut is to have each GPU normalize its own shard into local states and then average those local states across GPUs — that keeps the communicated object the same shape. But that gives each shard equal influence regardless of how much state mass it holds, so I should check whether it actually reproduces the single-GPU weighted mean. I built a tiny case: `N = 10` points, `C = 3` features, `M = 2` states, random `X` and random per-point masses `W`, sharded unevenly into groups of `2, 1, 7` points. Against the single-GPU reference `s_ref = (W^T X) / (sum W + 1e-5)`:

```text
sum numerators, sum masses, divide after  ->  max |s - s_ref| = 1.1e-16
per-shard normalize, then average states   ->  max |s - s_ref| = 0.97
```

So the shortcut is not a small approximation — it lands on a completely different state token (off by ~0.97 on a quantity of order one), because the tiny one-point shard gets the same vote as the seven-point shard. The only form that reconstructs the true mean is to reduce the numerator and the mass separately as SUMs and divide once at the end:

```text
s_j =
  AllReduce_sum_k sum_i w_ij^(k) x_i^(k)
  ------------------------------------------------.
  AllReduce_sum_k sum_i w_ij^(k) + 1e-5
```

Both reductions are SUM reductions, the denominator is reduced before normalization, and the numerator is reduced before the division — which is what makes the `1.1e-16` agreement above hold. After that, every GPU has the same `M` global states, runs the same small state attention, and deslices only to its local points with its local weights. In the one-GPU case the all-reduces are identity operations, so this is strictly a generalization of the single-device operator and cannot change its outputs there. In the multi-GPU case the communication is `O(#gpu * M * (C + 1))`: `M` masses plus `M*C` numerator entries, independent of `N`.

I also need to check the attention scaling. The state attention is ordinary scaled dot-product attention over tensors shaped like `B, H, M, dim_head`. If I call `F.scaled_dot_product_attention(q, k, v)`, I should not manually multiply by `dim_head ** -0.5`; PyTorch applies the scaled-dot-product convention internally. I can store that scale as a harmless attribute, but I should not apply it twice, or I would shrink the logits by `1/dim_head` and over-smooth the state attention.

One subtle code issue is worth spelling out rather than hiding. The distributed formula above is only true if the code actually holds the reduced mass and reduced numerator, not merely calls that look like reductions. In the distributed functional API I am using, the all-reduce returns the reduced tensor rather than mutating in place, so the executable form has to bind the return values: `slice_norm = dist_nn.all_reduce(slice_norm, ...)` and `slice_token = dist_nn.all_reduce(slice_token, ...)`. If I dropped the assignment, `slice_norm` and `slice_token` would still hold the local shard sums and the division above would silently fall back to per-shard means — the `0.97`-off case. So the return binding is load-bearing, not stylistic.

The last piece is memory inside each residual block. The operator and the feed-forward sublayer are wrapped with activation checkpointing in training, so the forward activations are recomputed during backward instead of stored. This does not change the function being represented; it only re-derives the same intermediates on the backward pass. It trades compute for memory, which is exactly the right trade at million-point scale. At evaluation, the block calls the attention and MLP directly, with no recomputation.

Stepping back, every change I kept survived a check rather than an argument: the temperature sweep showed one global `tau` cannot give different points different assignment entropy, so the temperature became per-point; the sampling test fixed the Gumbel sign and showed the relaxation samples the intended categorical; the activation count showed the single content stream really halves the per-point projection memory; and the tiny sharded example showed only sum-before-normalize reconstructs the true state mean while the shape-preserving shortcut is wrong by order one. What is left is still a slice-attention neural operator — points to weights to state means to state attention to deslice — but with weights that are locally adaptive and near-categorical, lower per-point projection memory, and a distributed state computation that communicates only the small state statistics and matches the single-device result exactly.
