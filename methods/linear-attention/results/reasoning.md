OK, let me think this through from scratch. The thing that hurts is the cost of self-attention, so let me stare at exactly where the cost comes from before reaching for any clever architecture. I have queries, keys, values $Q, K, V$, each a sequence of $N$ vectors, and the output is

$$ V' = \text{softmax}\!\left(\frac{QK^T}{\sqrt{D}}\right) V . $$

The middle object is $QK^T$, which is $N \times N$. To build it I do $N^2$ dot products — $\mathcal{O}(N^2 D)$ — and then to multiply the softmaxed matrix by $V$ I do another $\mathcal{O}(N^2 M)$. And it's not just time: I have to *keep* that $N\times N$ matrix around, because backprop through the softmax needs it. So memory is $\mathcal{O}(N^2)$ too. Everything quadratic traces back to one decision: I materialize an $N\times N$ array of pairwise scores.

And there's a second, sharper version of the pain. When I generate autoregressively — one token feeding the next — position $i$ attends to all $j \le i$. So at step $i$ the new query $Q_i$ has to be compared against every past key, which is $\mathcal{O}(i)$ work, and over a whole sequence that's $\sum_i i = \mathcal{O}(N^2)$ again. Even if I cache the past keys and values so I don't recompute them, the per-step cost still *grows* with how much I've already generated. There's no constant-cost step. Generating an image pixel by pixel — three thousand-plus steps for one CIFAR image — is agonizing for exactly this reason.

So what would a fix even look like? I keep coming back to the contrast with an RNN. An RNN carries a fixed-size hidden state, and each step it does a constant amount of work to fold the new input into the state and read off an output. Constant per step, memory independent of how far along the sequence I am. That is precisely the cost profile I want at generation time. The reason attention doesn't have it is the $N\times N$ matrix — the moment I form all pairwise scores, the past won't compress into a fixed-size summary. Can I avoid forming it?

Let me look at what's forcing me to form it. It's the softmax. Write out the entry-wise rule. The standard people-write-it-this-way form hides the structure; let me write attention for a single output position with a general similarity, the way the kernel-smoother reading of attention does:

$$ V'_i = \frac{\sum_{j=1}^N \text{sim}(Q_i, K_j)\, V_j}{\sum_{j=1}^N \text{sim}(Q_i, K_j)} . $$

This is just a weighted average of the value vectors, weight $\propto$ similarity between query $i$ and key $j$. If I set $\text{sim}(q,k) = \exp(q^Tk/\sqrt{D})$ I recover softmax attention exactly — the denominator is the softmax normalizer, the numerator the weighted sum. Good. Now, what does $\text{sim}$ actually have to satisfy for this to be a sensible attention? It's a normalized average of values, so I need the weights non-negative and not all zero — otherwise I could get negative "weights" or divide by something that isn't a positive total. Non-negativity is the only real constraint. So $\text{sim}$ can be *any* non-negative function — any positive-definite kernel $k(x,y) \ge 0$ would do. softmax's exponential is one choice among many; polynomial and RBF kernels have been used for attention and work about as well. The exponential is not sacred.

That reframing is liberating, but by itself it changes nothing about cost — I still have an $N\times N$ block of $\text{sim}(Q_i,K_j)$ values. The question is whether a *clever* choice of $\text{sim}$ lets me dodge the matrix. What structural property would I need?

The reason I can't pull $Q_i$ out and reuse work across the $N$ output positions is that $\exp(Q_i^T K_j)$ tangles $Q_i$ and $K_j$ together inside the nonlinearity — it does not factor into "something depending only on $Q_i$" times "something depending only on $K_j$". So every $(i,j)$ pair is its own irreducible computation.

So suppose $\text{sim}$ *did* factor — suppose I could write $\text{sim}(q,k) = \phi(q)^T \phi(k)$ for some feature map $\phi$ mapping into a non-negative space (so the dot product is $\ge 0$ and it's a valid attention). This is exactly the trick people use to linearize the softmax in large-vocabulary classification: approximate $\exp(u^Tv)$ by $\phi(u)^T\phi(v)$ so the expensive normalization factors and can be handled cheaply. Let me just substitute it and see what happens to the sum:

$$ V'_i = \frac{\sum_{j=1}^N \phi(Q_i)^T \phi(K_j)\, V_j}{\sum_{j=1}^N \phi(Q_i)^T \phi(K_j)} . $$

Now look at the numerator. $\phi(Q_i)^T \phi(K_j)$ is a *scalar*, and it multiplies the vector $V_j$. A scalar times a vector — and the scalar is a dot product whose left factor $\phi(Q_i)$ does not depend on $j$. So I can pull $\phi(Q_i)^T$ outside the sum over $j$. Writing $V_j$ as a row and being careful with the outer product, $\phi(Q_i)^T \phi(K_j) V_j = \phi(Q_i)^T \big(\phi(K_j) V_j^T\big)$, where $\phi(K_j)\in\mathbb{R}^{C}$ is a column, $V_j^T\in\mathbb{R}^{1\times M}$ a row, so $\phi(K_j)V_j^T$ is a $C\times M$ matrix. Summing over $j$ before contracting with $\phi(Q_i)$:

$$ V'_i = \frac{\phi(Q_i)^T \left(\sum_{j=1}^N \phi(K_j) V_j^T\right)}{\phi(Q_i)^T \left(\sum_{j=1}^N \phi(K_j)\right)} . $$

Stare at this for a second — there it is. The two sums, $\sum_j \phi(K_j) V_j^T$ (a $C\times M$ matrix) and $\sum_j \phi(K_j)$ (a $C$-vector), **don't depend on $i$ at all**. I compute each of them *once*, by a single pass over the keys and values, and then for every query I just do a couple of small contractions: $\phi(Q_i)^T$ against the $C\times M$ matrix and against the $C$-vector. In vectorized form the whole thing is the statement that I get to choose the order of the matrix products,

$$ \big(\phi(Q)\,\phi(K)^T\big)\,V \;=\; \phi(Q)\,\big(\phi(K)^T V\big), $$

and the right-hand grouping never forms the $N\times N$ thing in the middle. The left grouping is the old $\mathcal{O}(N^2)$; the right grouping is associativity buying me linearity. Cost: forming $\sum_j \phi(K_j)V_j^T$ is $\mathcal{O}(NCM)$, forming $\sum_j \phi(K_j)$ is $\mathcal{O}(NC)$, and contracting each query is $\mathcal{O}(CM)$ over $N$ queries, again $\mathcal{O}(NCM)$. Linear in $N$. And memory is linear too — I'm holding a $C\times M$ matrix and a $C$-vector, not an $N\times N$ array — which also means there's no giant matrix to stash for the backward pass. The quadratic is gone, and it's gone purely because the similarity factored and matrix multiplication is associative.

So the whole thing now hinges on the choice of $\phi$. What do I want from it? It has to make $\phi(q)^T\phi(k) \ge 0$ so attention stays well-defined. It should be cheap — the feature dimension $C$ now sits in the cost $\mathcal{O}(NCM)$, so a blown-up $C$ eats the win. And ideally it's finite-dimensional, because the obvious "just match softmax exactly" idea runs straight into a wall here: the feature map that makes $\phi(q)^T\phi(k)$ equal to $\exp(q^Tk)$ is *infinite*-dimensional. There's no finite $\phi$ that reproduces exact softmax, so I cannot linearize softmax itself — I have to pick a different, genuinely-finite kernel and accept it's a new attention, not a cheap softmax.

What are my finite options? The polynomial kernel has an exact finite feature map; a degree-2 polynomial transformer comes out to $\mathcal{O}(ND^2M)$, which beats quadratic-in-$N$ as soon as $N > D^2$ — fine for very long sequences but heavy when $N$ is only moderate, since $C$ scales like $D^2$. For the sequence lengths I actually care about here I'd rather keep $C = D$. So let me look for the cheapest possible positive feature map that doesn't inflate the dimension at all: just an elementwise function that keeps $C = D$.

The simplest positive elementwise map: push each coordinate through something non-negative. $\text{relu}(x)$ is non-negative and dirt cheap — but it zeroes everything below $0$, and crucially it zeroes the *gradient* there too, so any query/key coordinate that goes negative gets a dead gradient and stops learning. That's a real failure mode for something I'm going to train through. I want non-negative output but a live gradient on the negative side. The exponential linear unit does exactly that: $\text{elu}(x) = x$ for $x\ge 0$ and $\alpha(e^x - 1)$ for $x<0$, smooth, with nonzero slope everywhere. It bottoms out at $-\alpha$ (with $\alpha=1$, at $-1$). So shift it up by one:

$$ \phi(x) = \text{elu}(x) + 1 . $$

Now $\phi(x) \ge 0$ for all $x$ (since $\text{elu}(x) \ge -1$), so $\phi(q)^T\phi(k)$ is a sum of products of non-negatives, hence non-negative — valid attention, and empirically the attention converges normally with a positive similarity like this. It's $\mathcal{O}(D)$ to apply, keeps $C = D$ so the layer is $\mathcal{O}(NDM)$, and the gradient is alive everywhere, including the negative regime where relu would have killed it. That's the feature map. $\phi$ applied rowwise to $Q$ and to $K$, then the associativity rearrangement above.

Now the part I actually care about: causal masking, because the autoregressive case is where the pain was sharpest. With masking, position $i$ only sees $j \le i$:

$$ V'_i = \frac{\sum_{j=1}^i \text{sim}(Q_i, K_j)\, V_j}{\sum_{j=1}^i \text{sim}(Q_i, K_j)} . $$

The usual way to do this with softmax is to build the full $N\times N$ score matrix and then add $-\infty$ above the diagonal before the softmax. But that *rebuilds the very matrix I just got rid of* — it would throw the whole linear win away. I need masking that lives inside the factored form. So apply the same feature-map substitution and associativity, but now the sums run to $i$ instead of $N$:

$$ V'_i = \frac{\phi(Q_i)^T \sum_{j=1}^i \phi(K_j) V_j^T}{\phi(Q_i)^T \sum_{j=1}^i \phi(K_j)} . $$

The sums now depend on $i$ — they're not shared across all queries anymore, because each query $i$ has its own cutoff. At first that looks like I've lost the reuse. But these are *prefix* sums: the sum up to $i$ is the sum up to $i-1$ plus the one new term. Define

$$ S_i = \sum_{j=1}^i \phi(K_j) V_j^T, \qquad Z_i = \sum_{j=1}^i \phi(K_j), $$

so the masked output is just

$$ V'_i = \frac{\phi(Q_i)^T S_i}{\phi(Q_i)^T Z_i}, $$

and the $S_i, Z_i$ obey

$$ S_i = S_{i-1} + \phi(K_i) V_i^T, \qquad Z_i = Z_{i-1} + \phi(K_i), $$

starting from $S_0 = 0$, $Z_0 = 0$. So I sweep through the sequence once, and at each step I do a constant amount of work — add a rank-one $C\times M$ update to $S$, add a $C$-vector to $Z$, then contract $\phi(Q_i)$ against the current $S_i$ and $Z_i$ to read off $V'_i$. One pass, $\mathcal{O}(N)$ total, constant work per step. Causal masking that costs nothing extra asymptotically.

And now look at what I've actually written down. $S$ and $Z$ are a *fixed-size state*. Each step folds the current input into that state and reads an output off it. Let me write the full transformer layer this way, putting the projections back in. With $x_i$ the input at step $i$ for this layer, $f_l$ the position-wise feed-forward with its residual:

$$
\begin{aligned}
 s_0 &= 0, \quad z_0 = 0, \\
 s_i &= s_{i-1} + \phi(x_i W_K)\,(x_i W_V)^T, \\
 z_i &= z_{i-1} + \phi(x_i W_K), \\
 y_i &= f_l\!\left(\frac{\phi(x_i W_Q)^T s_i}{\phi(x_i W_Q)^T z_i} + x_i\right).
\end{aligned}
$$

This is an RNN. Literally — a model that takes an input, updates an internal state $(s, z)$, and emits an output, with the state of fixed dimension regardless of how long the sequence is. The "attention" I started from, once it's causal and linearized, *is* a recurrent network whose hidden state is the pair $(s, z)$: $s$ a running attention memory and $z$ a running normalizer memory. The recurrence is over *time*, the sequence index — not over depth. So the autoregressive cost problem I opened with just dissolves: generation is $\mathcal{O}(1)$ per step with constant memory, because all I keep between steps is $(s, z)$, the same way an LSTM keeps its cell. For every image the softmax model laboriously crawls through, this thing can stream out thousands, and it can do it on a CPU because each step is so cheap the bottleneck is just the unavoidable loop over positions. And note the recurrence is general — it never used anything about $\phi$ besides it being a feature map, so *any* attention, even softmax in principle, can be cast as this kind of recurrence; it's just that softmax's $\phi$ is infinite-dimensional so its state wouldn't be finite.

There's a genuine asymmetry worth holding onto between training and inference. At training time the whole ground-truth sequence is known, so I don't have to run the recurrence step by step — I can compute all the prefix sums in parallel across positions, the way teacher-forced transformers parallelize, and take full advantage of the accelerator. At inference, the output feeds the next input, so it's inherently sequential and I run the actual recurrence, $\mathcal{O}(1)$ a step. Best of both: parallel like a transformer to train, recurrent like an RNN to generate.

But there's a trap waiting in the *training* of the causal version, and if I don't deal with it the linear-memory promise is hollow. If I implement $V'_i = \phi(Q_i)^T S_i / (\phi(Q_i)^T Z_i)$ naively in an autodiff framework, the framework will, to get gradients, store every intermediate $S_i$ — and each $S_i$ is a $C\times M$ matrix. That's $N$ of them: memory back up by a factor of $\max(C, M)$ versus just holding values, which defeats the point exactly when sequences get long or models get deep. So I need the backward pass to also run as a single linear-memory sweep, never stashing all the $S_i$. That means deriving the gradients by hand as cumulative sums.

Let me do it. Focus on the numerator — call it $\bar V$ — and let the denominator and the final division be handled by autograd, since those are cheap (a dot of $\phi(Q_i)$ with a running sum of $\phi(K)$). Absorb $\phi$ into $Q, K$ to keep notation clean, so $Q\in\mathbb{R}^{N\times D}$, $K\in\mathbb{R}^{N\times D}$, $V\in\mathbb{R}^{N\times M}$ and

$$ \bar V_i = Q_i^T \sum_{j=1}^i K_j V_j^T . $$

I want $\nabla_Q \mathcal{L}$, $\nabla_K \mathcal{L}$, $\nabla_V \mathcal{L}$ given $\nabla_{\bar V}\mathcal{L}$. Write a single scalar entry without vector notation, summing the dimension index $d$ over $D$ and pushing the prefix sum inside:

$$ \bar V_{ie} = \sum_{d=1}^D Q_{id} \sum_{j=1}^i K_{jd} V_{je} = \sum_{d=1}^D \sum_{j=1}^i Q_{id} K_{jd} V_{je} . $$

Now the gradient with respect to a query entry $Q_{lt}$. The key observation is that $Q_{lt}$ appears in $\bar V_{ie}$ only when $i = l$ — the query at position $l$ only affects the output at position $l$. So I don't sum over output positions; I only need $i = l$:

$$ \frac{\partial \mathcal{L}}{\partial Q_{lt}} = \sum_{e=1}^M \frac{\partial \mathcal{L}}{\partial \bar V_{le}} \frac{\partial \bar V_{le}}{\partial Q_{lt}} = \sum_{e=1}^M \frac{\partial \mathcal{L}}{\partial \bar V_{le}} \left( \sum_{j=1}^l K_{jt} V_{je} \right), $$

because differentiating $\bar V_{le} = \sum_d \sum_{j\le l} Q_{ld} K_{jd} V_{je}$ in $Q_{lt}$ picks out $d = t$ and leaves $\sum_{j\le l} K_{jt} V_{je}$. Recognize the inner double-structure: $\sum_{j\le l} K_{jt} V_{je}$ is exactly the $(t,e)$ entry of $\sum_{j\le l} K_j V_j^T = S_l$. So in vector form

$$ \nabla_{Q_i}\mathcal{L} = \nabla_{\bar V_i}\mathcal{L}\,\left(\sum_{j=1}^i K_j V_j^T\right)^{\!T} . $$

That's a *forward* cumulative sum — the same prefix sum $S_i$ I built in the forward pass, summed from $1$ to $N$. Nice: $\nabla_Q$ reuses the forward-direction running state.

Now $K$, which is where it gets interesting, because $K_l$ does *not* only affect one output. $K_l$ enters $\bar V_{ie}$ for every $i \ge l$ — every position at or after $l$ has $l$ inside its prefix sum. So now I *do* sum over output positions, and only over $i \ge l$:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial K_{lt}} &= \sum_{e=1}^M \sum_{i=l}^N \frac{\partial \mathcal{L}}{\partial \bar V_{ie}} \frac{\partial \bar V_{ie}}{\partial K_{lt}} \\
&= \sum_{e=1}^M \sum_{i=l}^N \frac{\partial \mathcal{L}}{\partial \bar V_{ie}} \, Q_{it} V_{le},
\end{aligned}
$$

where differentiating $\bar V_{ie} = \sum_d \sum_{j\le i} Q_{id} K_{jd} V_{je}$ with respect to $K_{lt}$ picks out $d = t$ and $j = l$ (valid only when $l \le i$), leaving $Q_{it} V_{le}$. Group it: $V_{le}$ doesn't depend on $i$, so pull it out, and $\sum_{i\ge l} \frac{\partial \mathcal{L}}{\partial \bar V_{ie}} Q_{it}$ is the $(t,e)$ entry of $\sum_{j\ge l} Q_j (\nabla_{\bar V_j}\mathcal{L})^T$. So

$$ \nabla_{K_i}\mathcal{L} = \left(\sum_{j=i}^N Q_j \left(\nabla_{\bar V_j}\mathcal{L}\right)^{\!T}\right) V_i . $$

This sum runs from $i$ to $N$ — a *reverse* cumulative sum, accumulated from the end of the sequence backward, exactly like backpropagation through time in an RNN. And by the same bookkeeping the value gradient falls out from the same reverse running matrix:

$$ \nabla_{V_i}\mathcal{L} = \left(\sum_{j=i}^N Q_j \left(\nabla_{\bar V_j}\mathcal{L}\right)^{\!T}\right)^{\!T} \phi(K_i), $$

— $V_i$ contributes to $\bar V_{ie}$ for all $i \ge l$ in the same pattern as $K$, so it shares the reverse-cumsum matrix and just contracts the other way. So the picture is symmetric and clean: $\nabla_Q$ comes from a forward cumulative sum (sum $1\to N$), and $\nabla_K, \nabla_V$ come from a reverse cumulative sum (sum $N\to 1$). Two passes, each $\mathcal{O}(N)$, each holding only a running $D\times M$ matrix — never the full set of $S_i$. Constant memory in $N$ for the backward pass, matching the forward. Overall the causal layer is $\mathcal{O}(NCM)$ time and $\mathcal{O}(N\max(C,M))$ memory.

Let me write out the forward/backward as the explicit loops, because the structure is the whole point. Forward: keep a running $S$, and at each step add the rank-one update then read the numerator off.

```
S <- 0
for i = 1..N:
    S <- S + phi(K_i) V_i^T        # prefix-sum state, the S_i recurrence
    Vbar_i <- phi(Q_i) S           # numerator at position i
```

Backward: first a forward sweep that rebuilds the same $S$ to get $\nabla_Q$, then a reverse sweep accumulating $\sum_{j\ge i}\phi(Q_j)G_j^T$ (with $G = \nabla_{\bar V}\mathcal{L}$) to get $\nabla_V$ and $\nabla_K$:

```
S <- 0
for i = 1..N:
    S <- S + phi(K_i) V_i^T
    grad_phiQ_i <- G_i S^T          # forward cumsum  -> eq for grad of Q
S <- 0
for i = N..1:
    S <- S + phi(Q_i) G_i^T          # reverse cumsum
    grad_V_i   <- S^T phi(K_i)        # -> grad of V
    grad_phiK_i <- S V_i              # -> grad of K
```

Both directions touch only a running matrix; nothing scales with $N$ in memory. In practice the inner rank-one updates and contractions are tight enough to write as a small custom kernel.

So, putting it together as code. Three pieces, matching how I derived them: the cheap positive feature map, the unmasked linear attention that just builds the two key/value sums once, and the causal version that runs the prefix-sum state (with the hand-derived gradient as a custom autograd op), plus the recurrent form for generation that is the same state updated one step at a time.

```python
import torch
from torch.nn import Module

# phi(x) = elu(x) + 1 : non-negative (so phi(q).phi(k) >= 0 is valid
# attention), O(D) cheap, keeps C = D, and unlike relu+1 its gradient is
# alive for x < 0 instead of dead.
def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    """Unmasked: V'_i = phi(Q_i)^T (sum_j phi(K_j) V_j^T) / (phi(Q_i)^T sum_j phi(K_j)).
    The two key sums are computed once and reused for every query -> O(N)."""
    def __init__(self, feature_map=elu_feature_map, eps=1e-6):
        super().__init__()
        self.feature_map = feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)            # (N, L, H, D)
        K = self.feature_map(keys)               # (N, S, H, D)
        # KV = sum_j phi(K_j) V_j^T : the (D x M) state, formed once
        KV = torch.einsum("nshd,nshm->nhmd", K, values)
        # normalizer denom: phi(Q_i)^T sum_j phi(K_j)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        # V'_i = phi(Q_i)^T KV * (1/denom)
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)
        return V.contiguous()


class CausalDotProduct(torch.autograd.Function):
    """Numerator of causal linear attention with the hand-derived
    cumulative-sum gradients, so forward AND backward are O(N) time and
    constant memory in N (never store every S_i)."""
    @staticmethod
    def forward(ctx, Q, K, V):
        ctx.save_for_backward(Q, K, V)
        N, H, L, _ = Q.shape
        M = V.shape[-1]
        out = Q.new_zeros((N, H, L, M))
        # forward sweep: S_i = S_{i-1} + phi(K_i) V_i^T ; Vbar_i = phi(Q_i) S_i
        for n in range(N):
            for h in range(H):
                S = Q.new_zeros((Q.shape[-1], M))
                for i in range(L):
                    S = S + torch.ger(K[n, h, i], V[n, h, i])
                    out[n, h, i] = S.t().mv(Q[n, h, i])
        return out

    @staticmethod
    def backward(ctx, G):
        Q, K, V = ctx.saved_tensors
        gQ, gK, gV = (torch.zeros_like(Q), torch.zeros_like(K),
                      torch.zeros_like(V))
        N, H, L, _ = Q.shape
        for n in range(N):
            for h in range(H):
                # forward cumsum -> grad of Q :  grad_Q_i = G_i (sum_{j<=i} K_j V_j^T)^T
                S = Q.new_zeros((Q.shape[-1], V.shape[-1]))
                for i in range(L):
                    S = S + torch.ger(K[n, h, i], V[n, h, i])
                    gQ[n, h, i] = S.mv(G[n, h, i])
                # reverse cumsum -> grads of K and V
                S = Q.new_zeros((Q.shape[-1], V.shape[-1]))
                for i in range(L - 1, -1, -1):
                    S = S + torch.ger(Q[n, h, i], G[n, h, i])
                    gV[n, h, i] = S.t().mv(K[n, h, i])   # grad of V
                    gK[n, h, i] = S.mv(V[n, h, i])       # grad of K
        return gQ, gK, gV


def causal_linear(Q, K, V):
    return CausalDotProduct.apply(Q, K, V)


class CausalLinearAttention(Module):
    """Causal mask without ever forming the N x N matrix: prefix-sum state."""
    def __init__(self, feature_map=elu_feature_map, eps=1e-6):
        super().__init__()
        self.feature_map = feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)
        Q = Q.permute(0, 2, 1, 3).contiguous()        # (N, H, L, D)
        K = K.permute(0, 2, 1, 3).contiguous()
        V = values.permute(0, 2, 1, 3).contiguous()
        # denominator: phi(Q_i)^T Z_i with Z_i = cumsum of phi(K)
        Z = 1 / (torch.einsum("nhli,nhli->nhl", Q, K.cumsum(2)) + self.eps)
        Vbar = causal_linear(Q, K, V)                  # numerator, prefix-sum
        out = Vbar * Z[:, :, :, None]
        return out.permute(0, 2, 1, 3).contiguous()


class RecurrentLinearAttention(Module):
    """The same causal mechanism as an RNN: carry (S, Z), update per step in
    O(1), read off the output. This is the generation-time form."""
    def __init__(self, feature_map=elu_feature_map, eps=1e-6):
        super().__init__()
        self.feature_map = feature_map
        self.eps = eps

    def forward(self, query, key, value, state=None):
        Q = self.feature_map(query)                    # (N, H, D)
        K = self.feature_map(key)
        N, H, D = Q.shape
        M = value.shape[-1]
        if state is None:
            S = query.new_zeros((N, H, D, M))          # attention memory s_i
            Z = query.new_zeros((N, H, D))             # normalizer memory z_i
        else:
            S, Z = state
        Z = Z + K                                      # z_i = z_{i-1} + phi(K_i)
        S = S + torch.einsum("nhd,nhm->nhdm", K, value)  # s_i = s_{i-1} + phi(K_i) V_i^T
        denom = 1 / (torch.einsum("nhd,nhd->nh", Q, Z) + self.eps)
        V = torch.einsum("nhd,nhdm,nh->nhm", Q, S, denom)
        return V, [S, Z]
```

The chain end to end: softmax attention is quadratic and its autoregressive generation has no constant-cost step because it forms an $N\times N$ score matrix; reading attention as a kernel smoother shows the similarity need only be non-negative, not exponential; replacing $\exp(q^Tk)$ by a factored $\phi(q)^T\phi(k)$ lets associativity move the value sum next to the keys so the two key-side sums are computed once — linear time and memory; choosing $\phi = \text{elu}(\cdot)+1$ keeps it positive, cheap, and gradient-alive; causal masking turns the two sums into prefix sums with a one-step recurrence, which is exactly a fixed-size-state RNN running over time, giving $\mathcal{O}(1)$-per-step generation; and deriving the numerator's gradients as a forward and a reverse cumulative sum keeps even training at linear time and constant memory.
