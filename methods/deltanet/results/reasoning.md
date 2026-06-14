Let me start from what actually hurts. I want to train autoregressive language models on long sequences, and softmax attention is killing me on two fronts at once: the score matrix is `L x L`, so compute is `O(L^2 d)`, and at inference I have to keep every past key and value around — the KV cache grows linearly in the sequence length and there's no bound on it. For a fixed model I'd love a layer whose per-step state is a fixed-size object, so inference is constant-memory and training is linear in `L`. That's the whole appeal of linear attention: Katharopoulos and colleagues noticed that if you replace the exponential kernel `exp(k_i^T q_t)` with a plain dot product of feature-mapped vectors `phi(k_i)^T phi(q_t)`, the causal sum re-associates. Write it out: `o_t = sum_{i<=t} (phi(q_t)^T phi(k_i)) v_i / sum_{j<=t} phi(q_t)^T phi(k_j)`, and because the scalar `phi(q_t)^T phi(k_i)` commutes, I can pull `phi(q_t)` out to the right of a running sum, `o_t = (sum_{i<=t} v_i phi(k_i)^T) phi(q_t) / ((sum_j phi(k_j))^T phi(q_t))`. The numerator's running sum is a matrix `S_t = sum_{i<=t} v_i phi(k_i)^T`, the denominator's is a vector `z_t = sum_{i<=t} phi(k_i)`, and the whole layer collapses to a linear RNN: `S_t = S_{t-1} + v_t phi(k_t)^T`, `z_t = z_{t-1} + phi(k_t)`, read out by `o_t = S_t phi(q_t)/(z_t^T phi(q_t))`. The entire past is compressed into the fixed-size `S_t`. No KV cache. Constant-memory inference. The denominator `z_t^T phi(q_t)` is a known source of numerical trouble — it can get tiny and blow the ratio up — so I'll follow the common simplification and drop the normalizer, and take `phi` to be the identity, leaving the bare recurrence `S_t = S_{t-1} + v_t k_t^T`, `o_t = S_t q_t`. Clean.

But there's a reason this hasn't replaced softmax attention, and I need to stare at the failure before I patch it. The update `S_t = S_{t-1} + v_t k_t^T` is purely *additive* — every token writes its outer product into `S` and nothing is ever removed. This is a Hebbian, outer-product associative memory; in fast-weight-programmer language `S` is a "fast weight" matrix written by outer products and read by a matrix-vector product. The trouble is capacity. Suppose I've written `S = sum_i v_i k_i^T` and now I try to read with a stored key `k_j`: `S k_j = sum_i v_i (k_i^T k_j) = v_j (k_j^T k_j) + sum_{i != j} (k_i^T k_j) v_i`. The first term is what I wanted; the second is cross-talk from every key that isn't orthogonal to `k_j`. In `d` dimensions I can have at most `d` mutually orthogonal vectors, so the moment my sequence is longer than the key dimension, the keys *cannot* all be orthogonal — I'm in an overcapacity regime and the retrieval is contaminated by interference that only grows as I write more. That's not a hand-wavy worry; it's exactly why additive linear attention underperforms softmax attention on language modeling, and underperforms it *badly* on recall-intensive tasks where I need to fetch a specific value I stored earlier. The deep problem is that the additive rule has no way to *deallocate*. If a key already lives in memory and I see it again with a new value, I'd want to clear out the old association and write the new one — and crucially, whether and how much to clear should depend on what's *already in `S` for that key*, not on some blanket schedule.

So how do people try to add forgetting? The mainstream answer is gating: multiply the state by a data-dependent decay before the additive write, `S_t = S_{t-1} ⊙ M_t + v_t k_t^T`. That's the template behind a whole zoo — gated linear attention with a `beta_t alpha_t^T` decay, RetNet with a fixed scalar `gamma`, Mamba's selective state space, RWKV-6, HGRN2, mLSTM. They're cheap because `⊙` is elementwise, `O(dn)` per step, and they're competitive with transformers on plain language modeling. But look at what the decay actually does: it's elementwise — scalar, or diagonal per channel — so it forgets *globally* or *per coordinate*. It cannot say "the specific association I previously stored for *this* key collides with the new key, erase *that* and nothing else." Targeted, content-addressed removal would require the previous state `S_{t-1}` to enter the *write itself*, interacting with the incoming key — not merely to be scaled by a multiplicative gate. An elementwise gate is structurally the wrong tool for the collision problem. That's why these models still trail softmax attention precisely on the recall tasks I care about.

What write rule *does* let the old contents shape the removal? Go back to the classical adaptive-filtering idea: the delta rule, Widrow and Hoff's least-mean-squares. Treat `S` as a little regressor that's supposed to map `k_t` to `v_t`, and instead of blindly adding, take one gradient step on the squared prediction error. The loss is `L_t(S) = 1/2 || S k_t - v_t ||^2`. Its gradient with respect to `S` is `(S k_t - v_t) k_t^T` — the outer product of the residual with the key. So one SGD step with learning rate `beta_t` is `S_t = S_{t-1} - beta_t (S_{t-1} k_t - v_t) k_t^T`. This is exactly what I want: the write is proportional to the *error* `v_t - S_{t-1} k_t`, so if `S_{t-1}` already maps `k_t` close to `v_t`, almost nothing happens, and if it maps `k_t` to a stale value, the residual is large and the correction is strong. Read it the other way and the content-dependence is even clearer: retrieve the old value `v_t^old = S_{t-1} k_t`, blend it with the new one `v_t^new = beta_t v_t + (1-beta_t) v_t^old`, and swap — `S_t = S_{t-1} - v_t^old k_t^T + v_t^new k_t^T`, removing the old and writing the new for that key. The scalar `beta_t = sigma(W_beta x_t) in (0,1)` is a *dynamic writing strength*: at `beta_t = 1` the old value is fully overwritten, at `beta_t = 0` the memory is untouched. Contrast the loss with additive linear attention's, which is the linear `L'_t = -<S k_t, v_t>`: its gradient is `-v_t k_t^T`, constant regardless of how wrong the prediction is, which is exactly the no-error-correction behavior that drove the cross-talk. The quadratic loss gives gradients that scale with the error, so it self-corrects, and that's the intuition for why it recalls better. People have known for decades that the delta-rule fast weight has higher capacity than the Hebbian one, and applied as a sequence layer it does improve recall and small-scale language modeling. Good — this is the write rule I want.

So why isn't everyone already training this at scale? I hit the training wall. I need to think carefully about the *training* algorithm, because that's where additive linear attention won its real advantage, and the delta rule looks like it throws that advantage away. For additive linear attention, the value I write at step `t` is just `v_t` — it doesn't depend on the running state at all. So I can stack all the values into a matrix `V` ahead of time and compute the whole output in one shot with the parallel form `O = (Q K^T ⊙ M_L) V`, a couple of big matmuls, masked causally. That's `O(L^2 d)` FLOPs but it runs in `O(1)` sequential steps, pure matrix multiply, so it saturates the tensor cores and keeps the GPU busy. Or I can run the recurrent form `S_t = S_{t-1} + v_t k_t^T` in `O(Ld^2)` FLOPs but strictly one step at a time on the elementwise units. In practice nobody does pure parallel or pure recurrent at length; they chunk. Split the sequence into `L/C` chunks of size `C`, carry a chunk-level state `S_[t]` from chunk to chunk, and within each chunk do the heavy lifting with the parallel form while propagating between chunks with the recurrence. For additive linear attention the chunkwise identities are `S_[t+1] = S_[t] + V_[t]^T K_[t]` and `O_[t] = Q_[t] S_[t]^T + (Q_[t] K_[t]^T ⊙ M_C) V_[t]`. Cost is `O(LCd + Ld^2)`, sequential depth `O(L/C)`, and with `C` a small constant like 64 or 128 it's both subquadratic and matmul-rich — `C = L` recovers the parallel form, `C = 1` the recurrent form, and you tune `C` to trade FLOPs against sequence-parallelism. This is the machine that makes linear attention trainable at scale.

Now try to put the delta rule into that machine and watch it jam. The value I write at step `t` is no longer `v_t` — it's tangled up with the old value `v_t^old = S_{t-1} k_t`, which depends on the *running state*. So I can't stack the writes into a matrix ahead of time and matmul them, because each one needs `S_{t-1}`, which is the output of all the previous writes. The naive way to compute the writes is to literally roll the recurrence forward, materializing `S_{t-1}` (a `d x d` matrix) at every step just to get `v_t^old` — that's `O(d^2)` memory per step and a strictly sequential, elementwise loop, exactly the hardware-inefficient procedure the original delta-rule layer was stuck with. It runs one step at a time, never touches a tensor core, and that's the entire reason the delta rule hasn't scaled. I need to break the state-dependence — find a way to compute all the writes for a chunk *without* unrolling the state matrix step by step.

Let me look hard at the update written as a transition on `S`. Substitute `v_t^old = S_{t-1} k_t` and `v_t^new = beta_t v_t + (1-beta_t) S_{t-1} k_t` into `S_t = S_{t-1} - v_t^old k_t^T + v_t^new k_t^T`:

```
S_t = S_{t-1} - beta_t (S_{t-1} k_t) k_t^T + beta_t v_t k_t^T
    = S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T.
```

There it is. The delta update is a *matrix multiplication* of the previous state by `I - beta_t k_t k_t^T` plus a rank-one additive term. That transition matrix is identity-plus-rank-one, a generalized Householder transformation. So the state isn't just being added to — it's being hit on the right by a structured `d x d` matrix at every step. That structure is the lever. Two thoughts collide here. First: a sum of "decayed" rank-one writes. If I unroll the recurrence, `S_t = sum_{i=1}^t beta_i v_i k_i^T prod_{j=i+1}^t (I - beta_j k_j k_j^T)` — each write `beta_i v_i k_i^T` gets multiplied on the right by the product of all the transition matrices that came after it. Second, and this is the one I want to chase: those transition matrices are exactly the kind whose *products* have a beautiful compact form.

Before the product, let me handle the additive structure of `S` itself, because if I can keep `S` in the same additive shape as vanilla linear attention, I get to reuse all of its machinery for free. Claim: `S_t = sum_{i=1}^t u_i k_i^T` for some vectors `u_i` — same outer-product-sum shape as `S_t = sum v_i k_i^T`, just with a different "pseudo value" `u_i` in place of `v_i`. Let me prove it by induction and *discover* what `u_i` has to be. Base case: `S_1 = beta_1 v_1 k_1^T` (start from `S_0 = 0`), so `u_1 = beta_1 v_1`. Inductive step: assume `S_{t-1} = sum_{i<t} u_i k_i^T`. Then

```
S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    = sum_{i<t} u_i k_i^T  -  (sum_{i<t} u_i k_i^T) beta_t k_t k_t^T  +  beta_t v_t k_t^T
    = sum_{i<t} u_i k_i^T  -  beta_t (sum_{i<t} u_i (k_i^T k_t)) k_t^T  +  beta_t v_t k_t^T
    = sum_{i<t} u_i k_i^T  +  [ beta_t ( v_t - sum_{i<t} u_i (k_i^T k_t) ) ] k_t^T.
```

The bracket is the new pseudo value, so `u_t = beta_t ( v_t - sum_{i<t} u_i (k_i^T k_t) )`, and `S_t = sum_{i<=t} u_i k_i^T`. Sanity-check it against the value-blend interpretation: `u_t` should be `v_t^new - v_t^old = beta_t(v_t - v_t^old)`, and `v_t^old = S_{t-1} k_t = (sum_{i<t} u_i k_i^T) k_t = sum_{i<t} u_i (k_i^T k_t)` — yes, identical. So the delta-rule layer is just vanilla linear attention with `v_i` replaced by the pseudo value `u_i`. Once I have the `u_i`'s stacked into `U`, the output is `O = (Q K^T ⊙ M) U`, the very same parallel/chunkwise machinery as before. The per-token matrix state never has to be materialized; the remaining object is the vector sequence `U` plus the chunk boundary state. That's a real win: I've reduced the whole problem to *computing the `u_i`'s*.

But the `u_i` recurrence is still sequential — `u_t` depends on `u_1, ..., u_{t-1}` through the `sum_{i<t} u_i (k_i^T k_t)` term — and computing all `L` of them this way is `O(L^2 d)` and can't be parallelized the way `V` could. So I've moved the bottleneck but not removed it. Within a chunk of length `C` this is a `C`-step sequential dependency, and I want it as matmuls. Hold that thought.

Now the product of transition matrices, because the chunkwise form needs it. The "decay factor" applied to a chunk-initial state as I sweep through the chunk is `P_n = prod_{t=1}^n (I - beta_t k_t k_t^T)`. I claim this also collapses to identity-minus-an-outer-product-sum: `P_n = I - sum_{t=1}^n w_t k_t^T`. This is the WY representation of a product of Householder-type matrices, and again I'll induct to find `w_t`. Base: `P_1 = I - beta_1 k_1 k_1^T`, so `w_1 = beta_1 k_1`. Step: assume `P_{n-1} = I - sum_{i<n} w_i k_i^T`. Then

```
P_n = P_{n-1}(I - beta_n k_n k_n^T)
    = (I - sum_{i<n} w_i k_i^T)(I - beta_n k_n k_n^T)
    = I - sum_{i<n} w_i k_i^T  -  beta_n k_n k_n^T  +  (sum_{i<n} w_i k_i^T) beta_n k_n k_n^T
    = I - sum_{i<n} w_i k_i^T  -  [ beta_n k_n - beta_n sum_{i<n} w_i (k_i^T k_n) ] k_n^T
    = I - sum_{t=1}^n w_t k_t^T,
```

with `w_n = beta_n ( k_n - sum_{i<n} w_i (k_i^T k_n) )`. Look at that — it's the *exact same recurrence* as `u_t`, only with `k_t` standing in for `v_t`. So `w_t` is the pseudo-value of `k`, and `u_t` is the pseudo-value of `v`, computed by one and the same triangular recurrence. That's not a coincidence; both are "remove the part of the new vector that's already explained by the earlier writes, then scale by `beta`." Good. Now I have both the running state (`I - sum w_t k_t^T` form for the decay, `sum u_t k_t^T` form for the writes) stored in `O(d)` vectors.

Assemble the chunkwise recurrence. Unroll `S_t = sum_{i<=t} beta_i v_i k_i^T prod_{j=i+1}^t (I - beta_j k_j k_j^T)` and define the chunk-local pieces: within chunk `[t]`, the decay from the chunk start to position `r` is `P_[t]^r = I - sum_{i<=r} w_[t]^i k_[t]^i^T`, and the accumulated writes are `H_[t]^r = sum_{i<=r} u_[t]^i k_[t]^i^T`. The state at the `r`-th element of chunk `[t]` is the carried-in chunk state decayed plus the local writes: `S_[t]^r = S_[t]^0 P_[t]^r + H_[t]^r`. Substitute the two compact forms:

```
S_[t]^r = S_[t]^0 (I - sum_{i<=r} w_[t]^i k_[t]^i^T) + sum_{i<=r} u_[t]^i k_[t]^i^T
        = S_[t]^0 + sum_{i<=r} ( u_[t]^i - S_[t]^0 w_[t]^i ) k_[t]^i^T.
```

The bracket `u_[t]^i - S_[t]^0 w_[t]^i` is the *effective* write for position `i`: the pseudo value, corrected for the part of the carried-in state that this position's pseudo-key would have decayed. Stack a chunk's vectors into `C x d` matrices `Q_[t], K_[t], V_[t], U_[t], W_[t]` and the chunk recurrence and output become, exactly mirroring vanilla linear attention's chunk identities,

```
S_[t+1] = S_[t] + (U_[t] - W_[t] S_[t]^T)^T K_[t],
O_[t]   = Q_[t] S_[t]^T + (Q_[t] K_[t]^T ⊙ M)(U_[t] - W_[t] S_[t]^T).
```

This is the goal shape: chunk-to-chunk propagation through `S_[t]` (a `d x d` matrix carried across `L/C` chunks, never the per-token states), and intra-chunk work that's all matmuls — `Q_[t] K_[t]^T`, the masked product against the corrected writes, the outer-product update. Compare to additive linear attention's `O_[t] = Q_[t] S_[t]^T + (Q_[t] K_[t]^T ⊙ M) V_[t]`: I've literally replaced `V_[t]` with `U_[t] - W_[t] S_[t]^T`. Everything I built for linear attention transfers.

Everything except the one piece I flagged: building `U_[t]` and `W_[t]` still means running the triangular recurrence `u_[t]^r = beta_[t]^r (v_[t]^r - sum_{i<r} u_[t]^i (k_[t]^i^T k_[t]^r))` (and the same for `w` with `k` in place of `v`), and as written that's fully sequential within the chunk — it can't use tensor cores. If I leave it like this, I've reduced the I/O cost and the FLOPs but I've still got a `C`-deep elementwise dependency chain in the hot loop, and the whole point was to be matmul-rich. I need to turn this recurrence into a closed form.

Stare at the recurrence as a *linear system*, because it is one. Writing the rows `r = 1..C` of `W_[t]` (the `C x d` stack of `w_[t]^r`), the recurrence says

```
W_[t][r,:] = beta_[t]^r K_[t][r,:] - beta_[t]^r sum_{i<r} W_[t][i,:] (K_[t][i,:] K_[t][r,:]^T).
```

That's lower-triangular: row `r` depends only on earlier rows `i < r`. Let `B = diag(beta_[t])` and `L = tril(B K_[t] K_[t]^T, -1)` — the strictly-lower-triangular matrix whose `(r,i)` entry, for `i < r`, is `beta_[t]^r (K_[t][r,:] K_[t][i,:]^T)`, i.e. `beta`-weighted key-key similarities below the diagonal. Then the recurrence is exactly `W_[t] + L W_[t] = B K_[t]`, because the `L W_[t]` term reconstructs the `sum_{i<r} beta_[t]^r (k_r^T k_i) w_i` correction. Solve it:

```
(I + L) W_[t] = B K_[t]   =>   W_[t] = (I + L)^{-1} B K_[t] = T_[t] K_[t],   with   T_[t] = (I + L)^{-1} B = (I + tril(diag(beta_[t]) K_[t] K_[t]^T, -1))^{-1} diag(beta_[t]),
```

and identically `U_[t] = T_[t] V_[t]`, since the `u` recurrence is the same system with `V_[t]` on the right-hand side. The sequential dependency is gone — it's been absorbed into a single matrix `T_[t]`, the same `T_[t]` for both `U` and `W`. And `I + L` is unit lower-triangular, so its inverse is cheap and *also* matmul-friendly: solve by forward substitution, which is the UT transform for accumulating Householder products. Concretely, to invert `I + L` you build it up row by row — but each row's correction is a small matmul of already-computed rows against the strictly-lower part, so a `C x C` triangular inverse becomes a short loop of matmuls over `C` (and `C` is 64 or 128, fitting in fast memory). Now *every* part of the algorithm is a matrix multiply: `T_[t]` by forward substitution, `W_[t] = T_[t] K_[t]`, `U_[t] = T_[t] V_[t]`, the masked intra-chunk products, the chunk-state update. The delta rule, which looked irreducibly sequential because each write depends on the running state, is now a chunked sequence of dense matmuls with only `L/C` sequential steps between chunks.

Tally the cost: `O(LCd + Ld^2)` FLOPs, `O(L/C)` sequential steps — same asymptotics as chunkwise linear attention, and rich in matmul so it actually uses the hardware. I can shave memory further: don't store the chunk-level states `S_[t]` after the forward pass (that'd be `O((L/C) d^2)`); recompute them in the backward pass, the same trick FlashLinearAttention uses. For completeness there's also a fully parallel form — combining the pieces gives an attention-like matrix `A = (Q K^T ⊙ M) T` with `A_{ij} = k_j^T P_{j+1}^i q_i`, which is interesting for interpretability — but computing `T` over the *whole* sequence is an `L x L` triangular inverse that scales cubically in `L` without the chunking, so for training I stick with the chunkwise form.

Now the architectural choices, which I shouldn't just inherit — each one has to earn its place, and the most important one is stability of the recurrence. The transition matrix is `M_t = I - beta_t k_t k_t^T`. For the recurrence not to blow up or vanish, I need the eigenvalues of `M_t` to stay in the unit disk. What are they? `M_t` acts as the identity on everything orthogonal to `k_t` (eigenvalue `1`, multiplicity `d-1`), and on the `k_t` direction it scales by `1 - beta_t k_t^T k_t = 1 - beta_t ||k_t||^2` (multiplicity `1`). So stability needs `|1 - beta_t ||k_t||^2| <= 1`, i.e. `0 <= beta_t ||k_t||^2 <= 2`. With `beta_t in (0,1)` from the sigmoid, I need to control `||k_t||`. The original delta-rule layer used an L1-style normalization of the key/query; since `||k_t||_2 <= ||k_t||_1`, an L1-normalized key keeps `0 <= 1 - beta_t ||k_t||_2^2 <= 1`. But L2 normalization makes the spectral story exact. If I set `k_t = SiLU(W_K x_t)/||SiLU(W_K x_t)||_2` so that `||k_t||_2 = 1`, then the contractive eigenvalue is exactly `1 - beta_t`, which for `beta_t in (0,1)` lands neatly in `[0,1]` — always stable, no special cases. And there's a lovely interpretation at the boundary: when `beta_t = 1`, `M_t = I - k_t k_t^T` with a unit `k_t` is an orthogonal *projection* — it annihilates the one-dimensional subspace spanned by `k_t` and leaves the other `d-1` dimensions completely untouched. That's *targeted forgetting*: a full write erases exactly the direction being overwritten and preserves everything else, which is precisely the content-addressed deallocation the additive rule couldn't do and the elementwise gates couldn't localize. So L2 norm isn't just a stability hack; it's what makes the erase surgical. I'll L2-normalize both `q` and `k`.

The feature map: the original layer followed linear attention in using `elu(.)+1` to keep the features positive. But I'm no longer carrying the normalizer `z_t` that needed positivity, and after L2 normalization I mainly want a smooth, gated nonlinearity before the direction is projected onto the unit sphere. SiLU `x sigma(x)` keeps sign information, suppresses small negative noise smoothly, and avoids the hard zeroing of ReLU, so I'll use SiLU on `q` and `k` before the L2 normalization. Writing strength `beta_t = sigma(W_beta x_t)`, one sigmoid-scalar per head — negligible parameters, since `W_beta` is just `d -> num_heads`. For stable training of linear-attention-style layers it's well established that you want a normalization right before the output projection, so I'll put an RMSNorm on the per-head output before projecting back. And one more piece that recent linear recurrent models find important in practice: a lightweight short (depthwise) convolution, kernel size 4, applied to the `q`, `k`, `v` projections before the recurrence — it generalizes the shift operation, lets the layer do precise local token comparisons that pure content-based addressing is bad at, and it's cheap. I'll fold it in.

One implementation detail I want to get right because it bit linear attention too: numerical scaling. When I form the intra-chunk `Q K^T` products, I scale the queries by `d_k^{-1/2}` exactly as softmax attention does, to keep the products in a sane range before they hit the masked matmul. And I fold `beta` into the key and value at the very start — `k_beta = beta * k`, `v_beta = beta * v` — so the implementation can build `A = (I + tril(diag(beta) K K^T, -1))^{-1}` and then compute `U = A @ v_beta`, `W = A @ k_beta`. In equation form that's the same as `T = A diag(beta)`, `U = T V`, `W = T K`; in code the diagonal factor is already inside the right-hand sides.

For the readable reference version, I can keep a head-first layout. The structure is: chunk everything; pre-multiply `beta` into `k` and `v`; start from the negative strictly lower part `-tril(diag(beta) K K^T, -1)` and use forward substitution to form `A = (I + tril(diag(beta) K K^T, -1))^{-1}`; form `U = A V_beta`, `W = A K_beta`; then sweep chunks carrying the transposed `d_k x d_v` state `S`, computing the corrected writes `u_i - w_i S`, the inter-chunk read `q_i S`, the intra-chunk masked read, and the outer-product state update.

```python
import torch
from einops import rearrange


def delta_rule_chunkwise(q, k, v, beta, chunk_size=64):
    # q,k,v: [b, h, L, d] (q,k already SiLU + L2-normalized); beta: [b, h, L] in (0,1)
    b, h, L, d_k = q.shape
    q = q * (d_k ** -0.5)                  # softmax-style scaling before QK^T products
    v_beta = v * beta[..., None]           # fold writing-strength into the value  -> V_beta
    k_beta = k * beta[..., None]           # and into the key                      -> K_beta
    assert L % chunk_size == 0

    # split into [b, h, n_chunks, C, d]
    q, k, v_beta, k_beta = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), (q, k, v_beta, k_beta))

    # Build A = (I + tril(diag(beta) K K^T, -1))^{-1}; beta is already in K_beta/V_beta.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)       # row r, col i: -beta_r k_r^T k_i
    for i in range(1, chunk_size):                                    # forward substitution
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone()
                                               * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)   # this is A

    u = attn @ v_beta                       # U = A V_beta : pseudo-values
    w = attn @ k_beta                       # W = A K_beta : pseudo-keys (decay vectors)

    S = k.new_zeros(b, h, d_k, v_beta.shape[-1])  # carried transposed chunk state [d_k, d_v]
    o = torch.zeros_like(v_beta)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for i in range(L // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0)    # intra-chunk causal Q K^T
        u_i = u[:, :, i] - w[:, :, i] @ S                            # effective write in transposed-state layout
        o_inter = q_i @ S                                            # read carried-in state
        o[:, :, i] = o_inter + attn @ u_i                           # + intra-chunk read
        S = S + k_i.transpose(-1, -2) @ u_i                         # outer-product state update
    return rearrange(o, 'b h n c d -> b h (n c) d'), S
```

In the production-shaped layer I do not hand-roll that kernel. I project into `[B,T,H,D]`, use the short-convolution modules to apply the SiLU local mixer, let the FLA kernels apply L2 normalization inside the chunk/recurrent op when requested, select the recurrent kernel for very short sequences and the chunk kernel otherwise, and then RMS-normalize each head before the output projection:

```python
import torch
import torch.nn as nn
from einops import rearrange
from fla.modules import RMSNorm, ShortConvolution
from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule


class DeltaRuleTokenMixer(nn.Module):
    def __init__(self, hidden_size, num_heads, mode="chunk", conv_size=4, norm_eps=1e-5):
        super().__init__()
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.head_dim = hidden_size // num_heads
        self.mode = mode
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)        # writing strength beta_t
        self.q_conv1d = ShortConvolution(hidden_size=hidden_size, kernel_size=conv_size, bias=False, activation="silu")
        self.k_conv1d = ShortConvolution(hidden_size=hidden_size, kernel_size=conv_size, bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(hidden_size=hidden_size, kernel_size=conv_size, bias=False, activation="silu")
        self.o_norm = RMSNorm(self.head_dim, eps=norm_eps, dtype=torch.float32)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, recurrent_state=None, use_cache=False):
        q, _ = self.q_conv1d(x=self.q_proj(x), cache=None, output_final_state=False)
        k, _ = self.k_conv1d(x=self.k_proj(x), cache=None, output_final_state=False)
        v, _ = self.v_conv1d(x=self.v_proj(x), cache=None, output_final_state=False)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b t h d', h=self.num_heads), (q, k, v))
        beta = self.b_proj(x).sigmoid()                                    # [B,T,H], beta_t in (0,1)
        mode = "fused_recurrent" if x.shape[1] <= 64 else self.mode
        op = fused_recurrent_delta_rule if mode == "fused_recurrent" else chunk_delta_rule
        o, recurrent_state = op(
            q=q, k=k, v=v, beta=beta,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
        )
        o = self.o_norm(o)
        return self.o_proj(rearrange(o, 'b t h d -> b t (h d)')), recurrent_state
```

Let me trace the causal chain one more time so I'm sure it holds together. Softmax attention is `O(L^2)` with an unbounded KV cache, so I went to linear attention, which compresses the past into a fixed-size matrix state and runs as a linear RNN — constant-memory inference, linear training. But its additive Hebbian write has bounded capacity: once the sequence outruns the key dimension, non-orthogonal keys cross-talk and recall degrades, and there's no way to deallocate. Gated variants add forgetting but only elementwise, which can't localize the erase to the colliding association. The delta rule fixes the write at the source — one SGD step on the squared retrieval error, `S_t = S_{t-1} - beta_t(S_{t-1}k_t - v_t)k_t^T`, so the correction scales with the error and `beta_t` is a learned writing strength; rewritten, it's a state multiplied by the generalized Householder `I - beta_t k_t k_t^T` plus a rank-one write. That state-dependence is what made the original delta-rule layer trainable only by a strictly sequential recurrence — the value written depends on the running state, so it can't be matmul'd ahead like additive linear attention's. I broke the dependence in two moves: an induction showing `S_t = sum u_i k_i^T` with a pseudo-value `u_t = beta_t(v_t - sum_{i<t} u_i(k_i^T k_t))`, which reduces the whole layer to computing the `u_i` and reuses linear attention's chunkwise machinery (`V` replaced by `U - W S^T`); and the WY representation `prod(I - beta_t k_t k_t^T) = I - sum w_t k_t^T` with the same triangular recurrence for `w_t`. Both recurrences are one lower-triangular linear system, so I solved them in closed form as `T = (I + tril(diag(beta) K K^T, -1))^{-1} diag(beta)` with `U = T V`, `W = T K`, inverting the unit-triangular `I + L` by forward substitution — the UT transform — which turns the entire algorithm into dense matmuls with only `L/C` sequential steps, recomputing chunk states in the backward pass to save memory. Stability comes from L2-normalizing the keys so the only non-unit eigenvalue is `1 - beta_t in [0,1]`, which at `beta_t = 1` makes the transition an orthogonal projection — exactly the targeted, content-addressed forgetting the whole search was for — and SiLU plus a short convolution, an output RMSNorm, and a sigmoid writing strength round out a layer that drops straight into a pre-norm transformer block.
