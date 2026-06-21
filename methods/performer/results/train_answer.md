The thing that actually hurts is one operation inside the Transformer. Given queries, keys, and values $Q, K, V \in \mathbb{R}^{L\times d}$, softmax attention forms the score matrix $A = \exp(QK^\top/\sqrt d)$, normalizes each row by $D = \mathrm{diag}(A\,\mathbf{1}_L)$, and returns $D^{-1} A V$. The entire cost lives in $A$: it is $L\times L$, so building it is $O(L^2 d)$ time and holding it is $O(L^2 + Ld)$ memory. For sequences of a few thousand tokens — protein chains of eight thousand residues, images flattened to twelve thousand pixels, book-length text — that quadratic term is the whole bill, and the model runs out of memory or time. I want attention that scales *linearly* in $L$ while still approximating full dense softmax, not some sparse, local, or low-rank substitute that quietly changes what attention computes.

The existing fast-attention options all give up the target to buy the speed. Sparse and local schemes (Sparse Transformer, Longformer, Routing Transformer) restrict each query to a fixed or learned subset of keys, so $A$ has $O(L)$ nonzeros — but that is a structurally different, sparser mechanism whose pattern is a prior that may not fit the data, with no rigorous bound on what representation power is lost, and it usually needs hand-written kernels. Reformer buckets high-dot-product pairs with locality-sensitive hashing for $O(L\log L)$, but it must tie $Q=K$, carries no unbiasedness guarantee, and bakes in the assumption that only a few large scores matter. Linformer projects the $L$ keys and values down to $k\ll L$ rows, assuming the attention matrix is low rank; it is a *biased* estimator with large mean-squared error and is defined only for the bidirectional case. The Linear Transformer replaces $\exp(q_i^\top k_j)$ outright with $\varphi(q_i)^\top\varphi(k_j)$ for a hand-picked positive map $\varphi(x)=\mathrm{elu}(x)+1$, which gives genuine linear cost but does *not* approximate the softmax kernel — it changes the attention function — and is observed to train unstably with exploding gradients.

The crucial observation is that the $L^2$ is not fundamental to what attention means; it comes from the *order of operations*. I build $A$ in full and only then multiply by $V$. Stare at one unnormalized output row, $(AV)_i = \sum_j \exp(q_i^\top k_j)\, v_j$: the reason nothing factors is that $\exp(q_i^\top k_j)$ couples $q_i$ and $k_j$ jointly. But if the score were a plain dot product of per-token features, $A(i,j) = \varphi(q_i)^\top\varphi(k_j)$ with $\varphi:\mathbb{R}^d\to\mathbb{R}^m$, then
$$(AV)_i = \sum_j \big(\varphi(q_i)^\top\varphi(k_j)\big)\,v_j = \varphi(q_i)^\top\Big(\sum_j \varphi(k_j)\,v_j^\top\Big),$$
and the inner sum $\sum_j \varphi(k_j)\,v_j^\top$ is an $m\times d$ matrix independent of $i$ — computed once in $O(Lmd)$, then reused by every query. The same reassociation handles the normalizer, $D(i) = \sum_j A(i,j) = \varphi(q_i)^\top(\sum_j\varphi(k_j))$. Stacking the feature rows into $Q'$ and $K'$, the whole estimate is
$$\widehat{\mathrm{Att}} = \hat D^{-1}\big(Q'\,((K')^\top V)\big),\qquad \hat D = \mathrm{diag}\big(Q'\,((K')^\top \mathbf{1}_L)\big),$$
where the parentheses are the point: form $(K')^\top V$ first, an $m\times d$ object, then multiply by $Q'$. This costs $O(Lmd)$ time and $O(Lm + Ld + md)$ memory; the $L\times L$ matrix is never created. So the entire problem reduces to one question: can I write $\exp(q_i^\top k_j)$ as a dot product $\varphi(q_i)^\top\varphi(k_j)$?

What I propose is exactly that, done right; I call it the Performer, with its attention mechanism FAVOR+ — Fast Attention Via positive Orthogonal Random features. The kernel $\exp(x^\top y)$ is not a finite dot product, but it is a positive-definite kernel and therefore an inner product in expectation, which is enough. Folding the $\sqrt d$ into $q,k$, the score is the softmax kernel $\mathrm{SM}(x,y)=\exp(x^\top y)$. The textbook move is Rahimi–Recht random features: pull out the norms, $\exp(x^\top y) = \exp(\|x\|^2/2)\,K_{\mathrm{gauss}}(x,y)\,\exp(\|y\|^2/2)$, and approximate the shift-invariant Gaussian kernel by trigonometric features $\varphi(x)=\tfrac{1}{\sqrt m}(\sin\omega_1^\top x,\cos\omega_1^\top x,\dots)$ with $\omega_i\sim N(0,I)$, since $\mathbb{E}_\omega[\cos(\omega^\top(x-y))] = \exp(-\|x-y\|^2/2)$. This is unbiased and linear-cost, but it is the wrong feature map, and the reason is the whole story. Trigonometric features take values in $[-1,1]$, so individual estimated scores can be *negative*, and then $\hat D(i)$, a sum of them, can be negative or near zero, at which point $\hat D^{-1}$ flips sign or explodes and the "convex combination of values" becomes garbage. Quantifying it, with $z=x+y$ and $\Delta=x-y$,
$$\mathrm{MSE}(\widehat{\mathrm{SM}}_{\mathrm{trig}}) = \frac{1}{2m}\,\exp(\|x+y\|^2)\,\mathrm{SM}(x,y)^{-2}\,\big(1-\exp(-\|x-y\|^2)\big)^2,$$
where the $\mathrm{SM}^{-2}$ factor comes from $\exp(\|x\|^2+\|y\|^2) = \exp(\|z\|^2)\,\mathrm{SM}(x,y)^{-2}$. As $\mathrm{SM}(x,y)\to 0$ this MSE goes to *infinity*. The attention matrix is dominated by small entries — most token pairs are low-relevance — so the estimator is least accurate precisely where the matrix has most of its mass, and those amplified errors flow straight into $\hat D^{-1}$. That is the wall: the reassociation is correct and the cost is linear, but $\sin/\cos$ features make it untrainable.

So I need a feature map that is unbiased for $\mathrm{SM}$, strictly *non-negative* so $\hat D$ stays a positive normalizer, and low-variance exactly *as the kernel goes to zero*. Non-negativity rules out $\sin$ and $\cos$ and points at $\exp$. To get there, complete the square aiming at the exponential form directly. The awkward factor in $\exp(x^\top y) = \exp(-\|x\|^2/2)\exp(\|x+y\|^2/2)\exp(-\|y\|^2/2)$ is the middle one; turn it into a Gaussian expectation using $(2\pi)^{-d/2}\int \exp(-\|\omega-c\|^2/2)\,d\omega = 1$ with $c=x+y$. Expanding the exponent, $-\tfrac12\|\omega-(x+y)\|^2 + \tfrac12\|x+y\|^2 = -\tfrac12\|\omega\|^2 + \omega^\top(x+y)$, the $\|x+y\|^2$ terms cancel exactly, giving $\exp(\|x+y\|^2/2) = \mathbb{E}_{\omega\sim N(0,I)}[\exp(\omega^\top x)\exp(\omega^\top y)]$. Substituting back,
$$\mathrm{SM}(x,y) = \exp(x^\top y) = \mathbb{E}_{\omega\sim N(0,I)}\big[\exp(\omega^\top x - \tfrac12\|x\|^2)\,\exp(\omega^\top y - \tfrac12\|y\|^2)\big].$$
The two factors are *separable* — one depends only on $x$, the other only on $y$ — and each is an exponential, hence strictly positive. The feature map is therefore
$$\varphi_+(u) = \frac{\exp(-\|u\|^2/2)}{\sqrt m}\big(\exp(\omega_1^\top u),\dots,\exp(\omega_m^\top u)\big),\qquad \omega_i\sim N(0,I),$$
and $\varphi_+(x)^\top\varphi_+(y)$ is an unbiased, non-negative estimator of $\mathrm{SM}(x,y)$. Every entry is positive, so $\hat D$ is a genuine positive normalizer and $\hat D^{-1}$ never blows up. (One honest caveat: the *row-normalized* output $\hat D^{-1}\hat A V$ is a ratio estimator, so the clean unbiasedness claim is about the kernel/attention-matrix entries *before* row normalization, not about the normalized output.) Now the variance near zero, where trig died. The same calculation gives, with $z=x+y$,
$$\mathrm{MSE}(\widehat{\mathrm{SM}}_+) = \frac{1}{m}\,\exp(\|x+y\|^2)\,\mathrm{SM}(x,y)^2\,\big(1-\exp(-\|x+y\|^2)\big),$$
using $\exp(-(\|x\|^2+\|y\|^2))\exp(\|z\|^2) = \exp(2x^\top y) = \mathrm{SM}(x,y)^2$. Where trig had $\mathrm{SM}^{-2}$ out front, this has $\mathrm{SM}^2$: as $\mathrm{SM}(x,y)\to 0$ the MSE *vanishes*. The positive estimator is most accurate exactly where the trigonometric one was least accurate — on the small entries that dominate the matrix and get amplified by the normalizer. That asymmetry is the whole point; positivity is not a cosmetic fix for the sign, it fixes the variance in the regime that was killing training. A symmetrization sharpens it further for free: since $\omega$ and $-\omega$ are equidistributed, averaging $\tfrac12(e^{\omega^\top z}+e^{-\omega^\top z})$ stays positive and unbiased while the negative covariance between the two terms cuts the variance, giving $\mathrm{MSE}(\widehat{\mathrm{SM}}_{\mathrm{hyp}+}) = \tfrac12(1-\exp(-\|x+y\|^2))\,\mathrm{MSE}(\widehat{\mathrm{SM}}_+)$, which is below the plain positive estimator with twice the features.

The second lever is *orthogonal* sampling. Drawing $m$ independent Gaussian directions is wasteful — in high dimension independent directions cluster and waste samples. Forcing $\omega_1,\dots,\omega_m$ to be exactly orthogonal covers the sphere more evenly, and because $N(0,I)$ is isotropic I can Gram–Schmidt a Gaussian block without changing any marginal (each $\omega_i$ keeps its $N(0,I)$-directional law with chi-distributed length), so the estimator stays unbiased; for $m>d$ I stack independent $d\times d$ orthogonal blocks plus a partial final block. To see *why* it helps, write each kernel of interest as $F(z)=\mathbb{E}_{\omega\sim\Omega}[g(\omega^\top z)]$ with $\Omega$ isotropic and $g$ entire with *non-negative* power-series coefficients (for softmax, $g=\exp$, $a_s = 1/s!\ge 0$). Both estimators are unbiased, so the MSE difference is entirely in the cross terms, $\mathrm{MSE}(\hat F^{\mathrm{iid}})-\mathrm{MSE}(\hat F^{\mathrm{ort}}) = (1-\tfrac1m)\,(\mathbb{E}[X_1^{\mathrm{iid}}X_2^{\mathrm{iid}}] - \mathbb{E}[X_1^{\mathrm{ort}}X_2^{\mathrm{ort}}])$ with $X_i=g(\omega_i^\top z)$. Expanding in the power series reduces this to comparing mixed moments $\mathbb{E}[(\omega_1^\top z)^{d_1}(\omega_2^\top z)^{d_2}]$ under independence versus orthogonality. Splitting $\omega_i=\|\omega_i\|\,\hat u_i$ and noting that orthonormal directions are a random rotation of the basis (equivalently a random rotation of $z$), the comparison collapses to a single ratio of $\chi_d$-moments, $\tau(d_1,\dots,d_m)=\prod_i\mu_d(d_i)/\mu_d(\sum_i d_i)$. Monomials with any odd exponent or supported on a single index vanish; among the survivors, $\tau$ is maximized at $d_i=d_j=2$ where it equals exactly $d/(d+2)<1$. Because every coefficient $a_s a_t$ is non-negative — this is where positivity is load-bearing, since negative coefficients could cancel or flip the sign — every surviving term in the difference is non-negative, so orthogonality never hurts, and keeping the leading term gives the explicit gap
$$\mathrm{MSE}(\widehat{\mathrm{SM}}_{\mathrm{ort}+}) \le \mathrm{MSE}(\widehat{\mathrm{SM}}_+) - \frac{2(m-1)}{m(d+2)}\,\big(\mathrm{SM}(x,y) - \exp(-\tfrac12(\|x\|^2+\|y\|^2))\big)^2.$$
This holds for *every* $d>0$, not just asymptotically — orthogonal features help even in low dimension, which earlier orthogonal-feature analyses could not claim. The number of features needed for uniform entrywise accuracy is $m=\Theta\big((d/\delta^2)\log(4d^{3/4}R/\delta)\big)$ with query/key norms bounded by $R$: it depends on $d$, $R$, and precision, never on the sequence length $L$.

Two practical pieces close it out. For numerical range, instead of $N(0,I)$ I can draw $\omega$ uniformly from the sphere of radius $\sqrt d$, giving a regularized softmax kernel $\mathrm{SMREG}$; the same orthogonality results carry over, a term-by-term Taylor comparison shows $\mathrm{SMREG}\le\mathrm{SM}$ (a clean universal lower bound), and a Poisson tail argument gives $\mathrm{SMREG}(x,y)/\mathrm{SM}(x,y)\ge 1 - 2/d^{1/3} + o(1/d^{1/3})$, so it tracks $\mathrm{SM}$ within a vanishing relative gap while staying bounded. And for the causal (autoregressive) case, the global sum becomes a prefix sum: define the running accumulator $G_i = \sum_{j\le i}\varphi(k_j)\,[v_j,1]^\top$ (the appended $1$ computes the normalizer alongside $AV$), so $\mathrm{output}_i = \varphi(q_i)^\top G_i$; updating $G_i$ from $G_{i-1}$ adds one outer product, the whole pass is $O(Lmd)$, and the lower-triangular $L\times L$ mask is never formed. Finally, nothing in the reassociation actually required the softmax kernel — only $\varphi(x)\ge 0$ — so a *generalized* kernelizable attention takes $\varphi(x)=f(x)+\varepsilon$ (optionally $f(Wx)+\varepsilon$ with random projections) for any non-negative $f$; the default is deterministic $\mathrm{ReLU}$ features applied straight to the data with $\varepsilon=10^{-3}$ and no projection or softmax-norm correction, with the $f=\exp$ softmax estimator above as the other option, both at linear cost.

In implementation the softmax path builds non-negative random features with $d^{-1/4}$ normalization, an orthogonal Gaussian projection (redrawn by default), a per-query/per-key max subtraction for range safety, and a denominator stabilizer; the generalized path defaults to deterministic ReLU features.

```python
import math
import jax
from jax import random
import jax.numpy as jnp

SOFTMAX_DEFAULTS = dict(
    renormalize_attention=True,
    numerical_stabilizer=1e-6,
    nb_features=256,
    ortho_features=True,
    ortho_scaling=0,
    redraw_features=True,
)

GENERALIZED_DEFAULTS = dict(
    renormalize_attention=True,
    numerical_stabilizer=0.0,
    nb_features=256,
    features_type="deterministic",
    kernel_fn=jax.nn.relu,
    kernel_epsilon=1e-3,
    redraw_features=False,
)

def gaussian_orthogonal_random_matrix(key, nb_rows, nb_columns, scaling=0):
    blocks = []
    rng = key
    for _ in range(nb_rows // nb_columns):
        rng, block_key = random.split(rng)
        q, _ = jnp.linalg.qr(random.normal(block_key, (nb_columns, nb_columns)))
        blocks.append(q.T)
    remaining = nb_rows - len(blocks) * nb_columns
    if remaining:
        rng, block_key = random.split(rng)
        q, _ = jnp.linalg.qr(random.normal(block_key, (nb_columns, nb_columns)))
        blocks.append(q.T[:remaining])

    matrix = jnp.vstack(blocks)
    if scaling == 0:
        multiplier = jnp.linalg.norm(random.normal(key, (nb_rows, nb_columns)), axis=1)
    elif scaling == 1:
        multiplier = math.sqrt(float(nb_columns)) * jnp.ones((nb_rows,))
    else:
        raise ValueError("scaling must be 0 or 1")
    return jnp.diag(multiplier) @ matrix

def softmax_features(data, projection_matrix, is_query, attention_axes=(-2,), eps=1e-6):
    data_normalizer = data.shape[-1] ** -0.25
    ratio = projection_matrix.shape[0] ** -0.5
    data_dash = jnp.einsum("...d,md->...m", data_normalizer * data, projection_matrix)
    diag = jnp.sum(data ** 2, axis=-1, keepdims=True) * (data_normalizer ** 2) / 2.0
    if is_query:
        max_term = jnp.max(data_dash, axis=-1, keepdims=True)
    else:
        max_term = jnp.max(data_dash, axis=(-1,) + attention_axes, keepdims=True)
    return ratio * (jnp.exp(data_dash - diag - max_term) + eps)

def generalized_features(data, projection_matrix=None, kernel_fn=jax.nn.relu, eps=1e-3):
    if projection_matrix is None:
        return kernel_fn(data) + eps
    return kernel_fn(jnp.einsum("...d,md->...m", data, projection_matrix)) + eps

def noncausal_favor(q_prime, k_prime, value, eps=1e-6):
    z = jnp.einsum("...lm,...ld->...md", k_prime, value)
    w = jnp.einsum("...lm,...md->...ld", q_prime, z)
    normalizer = jnp.einsum("...lm,...m->...l", q_prime, k_prime.sum(axis=-2))
    normalizer = normalizer + 2 * eps * (jnp.abs(normalizer) <= eps)
    return w / normalizer[..., None]

def causal_favor(q_prime, k_prime, value, eps=1e-6):
    kv_prefix = jnp.cumsum(jnp.einsum("...lm,...ld->...lmd", k_prime, value), axis=-3)
    k_prefix = jnp.cumsum(k_prime, axis=-2)
    w = jnp.einsum("...lm,...lmd->...ld", q_prime, kv_prefix)
    normalizer = jnp.einsum("...lm,...lm->...l", q_prime, k_prefix)
    normalizer = normalizer + 2 * eps * (jnp.abs(normalizer) <= eps)
    return w / normalizer[..., None]
```
