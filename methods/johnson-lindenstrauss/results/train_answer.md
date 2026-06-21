We have $n$ points $v_1,\dots,v_n$ sitting in $\mathbb{R}^d$ with $d$ enormous, and we want a map $f:\mathbb{R}^d\to\mathbb{R}^k$ into a far smaller space that keeps every pairwise squared distance inside a multiplicative $(1\pm\varepsilon)$ window,
$$(1-\varepsilon)\,\|v_i-v_j\|^2 \;\le\; \|f(v_i)-f(v_j)\|^2 \;\le\; (1+\varepsilon)\,\|v_i-v_j\|^2 \qquad\text{for every pair }i,j,$$
with $k$ as small as possible and — this is the real prize — independent of the ambient $d$. Every distance-based computation done downstream (nearest neighbors, clustering, regression) pays for the working dimension, so a $k$ that depends only on how hard the *point set* is, not on the accidental coordinates the data arrives in, is what makes the whole approach worth anything. The difficulty is the simultaneity: a single map must keep all $\binom{n}{2}$ inter-point distances in band at once, and these events are coupled through the shared map. The obvious data-fitting reduction is exactly wrong here. Take the top-$k$ principal directions via SVD and project onto them: the best rank-$k$ approximation $A_k$ satisfies $\|A-A_k\|\le\|A-D\|$ for every rank-$k$ $D$ in Frobenius norm, but that objective controls only the *total* squared displacement summed over points. A global optimum is free to collapse one inter-point distance to nearly zero if that buys a larger reduction elsewhere; there is no floor under any individual pair, and the $k$ it needs is set by the data's spectrum, not by $n$ and $\varepsilon$. We want the opposite of fitting the data: a map that ignores the data entirely and treats every direction even-handedly. One can also project onto a uniformly random $k$-dimensional subspace (Frankl–Maehara sharpened the sufficient $k$ to about $9(\varepsilon^2-2\varepsilon^3/3)^{-1}\ln n+1$), but constructing a true orthonormal $k$-frame means orthogonalizing $k$ vectors in $\mathbb{R}^d$ — clean in principle, heavier than necessary and awkward to implement, especially in systems restricted to simple aggregate operations.

I propose the Johnson–Lindenstrauss embedding: a single random *linear* projection, with the target dimension forced to grow only like $\log n/\varepsilon^2$. Two moves carry it. First, insisting the map be linear is what tames the $\binom{n}{2}$ coupled constraints: if $f$ is linear then $f(v_i)-f(v_j)=f(v_i-v_j)$, so preserving the distance between two points is *literally* preserving the length of the single difference vector $x=v_i-v_j$. The all-pairs requirement collapses into $\binom{n}{2}$ statements of the form "this one fixed vector keeps its length," and I only have to reason about one such event and then count how many I am asking to hold at once. Second, making the map random and rotation-blind is what lets it ignore the data; the engine is a distributional fact about how well one fixed vector's norm survives. Let $R$ be a $k\times d$ matrix with i.i.d. mean-zero, variance-one entries scaled by $1/\sqrt{k}$ — Gaussian entries are the cleanest choice — and set $f(X)=XR^\top$. For a fixed $x$, each row's inner product $\langle U_i,x\rangle$ is exactly $N(0,\|x\|^2)$ by the 2-stability of the Gaussian, so $\sum_i\langle U_i,x\rangle^2/\|x\|^2$ is a $\chi^2_k$ variable, the same for *every* $x$, and the $1/\sqrt{k}$ makes $\mathbb{E}\|f(x)\|^2=\|x\|^2$. The concentration runs off the one-coordinate moment generating function $\mathbb{E}[e^{sX^2}]=(1-2s)^{-1/2}$ for $s<\tfrac12$. Applying Markov to $e^{\pm\lambda S}$ for $S=\sum_i\langle U_i,x\rangle^2$ and optimizing the exponent gives, with $\beta$ the deviation factor, the single closed form
$$\Pr[\,\|f(x)\|^2\le\beta\,]\le\exp\!\Big(\tfrac{k}{2}\big(1-\beta+\ln\beta\big)\Big)\ (\beta<1),\qquad \Pr[\,\|f(x)\|^2\ge\beta\,]\le\exp\!\Big(\tfrac{k}{2}\big(1-\beta+\ln\beta\big)\Big)\ (\beta>1).$$
The ambient $d$ has vanished — when the projected squared length is written as a ratio concentrating about its mean $k/d$ and rescaled by $\sqrt{d/k}$, $d$ only ever sets the mean, never the spread of the relative error. The two tails are not symmetric, and which log bound I use matters for the final constant. For the lower tail, $\beta=1-\varepsilon$, the inequality $\ln(1-x)\le -x-\tfrac{x^2}{2}$ gives $\varepsilon+\ln(1-\varepsilon)\le-\tfrac{\varepsilon^2}{2}$, hence $\exp(-k\varepsilon^2/4)$. For the upper tail, $\beta=1+\varepsilon$, I need an *upper* bound on $\ln(1+\varepsilon)$ that still leaves a negative quadratic: the one-term $\ln(1+x)\le x$ leaves no decay and the alternating two-term truncation is a lower bound near zero and so unusable, which forces the three-term bound $\ln(1+x)\le x-\tfrac{x^2}{2}+\tfrac{x^3}{3}$, giving $\exp\!\big(-\tfrac{k}{2}(\tfrac{\varepsilon^2}{2}-\tfrac{\varepsilon^3}{3})\big)$. For $\varepsilon\in(0,1)$ this upper tail, with its extra $\tfrac{\varepsilon^3}{6}$ chipping away at the exponent, is the *binding* one and dictates $k$; the $\varepsilon^3/3$ correction is exactly what sharpens the constant, and being lazy with a two-term log would cost dimension. Now the union bound closes it: to make each pair's two-sided failure below $2/n^2$ — so the sum over $\binom{n}{2}<n^2/2$ pairs stays below $1$ — set the binding tail to $\exp(-2\ln n)$, which yields
$$k \;\ge\; \frac{4\ln n}{\varepsilon^2/2-\varepsilon^3/3}\qquad(\approx 8\ln n/\varepsilon^2\ \text{for small }\varepsilon).$$
With $k=\lceil 4(\varepsilon^2/2-\varepsilon^3/3)^{-1}\ln n\rceil$ the union over all pairs fails with probability at most $(n-1)/n<1$, so a single random matrix preserves every distance with probability at least $1/n$ — a good map exists by the probabilistic method, and repeating $O(n)$ draws boosts success to a constant.

What makes the construction not just provable but cheap is that the concentration needed almost nothing from the Gaussian. Each coordinate of $f(x)$ is $\langle r,x\rangle^2$, and as long as $r$ has independent mean-zero, unit-variance entries, $\mathbb{E}[\langle r,x\rangle^2]=\sum_i x_i^2\,\mathbb{E}[r_i^2]=\|x\|^2$ — the cross terms die because the entries are independent and mean-zero, so only the diagonal survives. Every coordinate is therefore an unbiased estimator of $\|x\|^2$, and $\|f(x)\|^2$ averages $k$ of them; the Gaussianity gave a comfortable exact $\chi^2$ but unbiasedness asked only for mean zero, variance one, independence. So I can replace the Gaussian by Rademacher $\pm1$ entries: the mean is unchanged, and computing $\langle r,x\rangle$ becomes additions and subtractions of coordinates — integer arithmetic, one bit per entry, "add a random half of the attributes, subtract the other half" in a database. The worry is that without spherical symmetry the law of $\|f(x)\|^2$ now depends on $x$, so the $\chi^2$ argument no longer applies directly. The fix is moment domination: expanding even moments, every monomial with an odd exponent vanishes, and term by term $\mathbb{E}[r_i^{2\ell}]=1\le(2\ell-1)!!$, so $\mathbb{E}[e^{hQ^2}]\le(1-2h)^{-1/2}$ for $Q=\langle r,x\rangle$ and every $x$ — the upper tail matches the Gaussian exactly. The lower tail needs its own check, since the positive MGF alone does not bound it: using $e^{-z}\le1-z+z^2/2$ with $\mathbb{E}[Q^2]=1,\ \mathbb{E}[Q^4]\le3$ gives $\mathbb{E}[e^{-hQ^2}]\le1-h+\tfrac32h^2$, and optimizing leaves the lower-tail exponent at least $\tfrac{\varepsilon^2}{4}-\tfrac{\varepsilon^3}{6}$, no weaker than the binding upper tail. The same logic pushes sparsity: with density $p$, entries $\pm1/\sqrt{p}$ keep variance one but have fourth moment $1/p$, and since the Gaussian fourth moment is $3$, moment domination forbids $p<\tfrac13$. At the edge $p=\tfrac13$ the entries are $\sqrt3\cdot\{+1\text{ w.p. }\tfrac16,\,0\text{ w.p. }\tfrac23,\,-1\text{ w.p. }\tfrac16\}$, with $\mathbb{E}[r_i^{2\ell}]=3^{\ell-1}\le(2\ell-1)!!$ (equality at $\ell=1,2$), so two-thirds of the matrix is zero — a threefold speedup — at no cost in the bound. If one wants an explicit failure probability at most $n^{-\eta}$, the union-bound target shifts and $k_0=(4+2\eta)(\varepsilon^2/2-\varepsilon^3/3)^{-1}\log n$. In the implementation below `density="auto"` means $1/3$ and `density=1` is the dense Rademacher choice.

```python
import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.random import sample_without_replacement

def min_embedding_dim(n_samples, *, eps=0.1):
    eps = np.asarray(eps)
    n_samples = np.asarray(n_samples)
    if np.any(eps <= 0.0) or np.any(eps >= 1):
        raise ValueError("eps must be in (0, 1)")
    if np.any(n_samples <= 0):
        raise ValueError("n_samples must be positive")
    denominator = (eps**2 / 2) - (eps**3 / 3)
    bound = np.ceil(4 * np.log(n_samples) / denominator).astype(np.int64)
    return np.maximum(bound, 1)

def _check_density(density, n_features):
    if density == "auto":
        density = 1.0 / 3.0
    try:
        density = float(density)
    except (TypeError, ValueError) as exc:
        raise ValueError("density must be 'auto', 1/3, or 1") from exc
    if np.isclose(density, 1.0):
        return 1.0
    if np.isclose(density, 1.0 / 3.0):
        return 1.0 / 3.0
    raise ValueError("density must be 'auto', 1/3, or 1")

def _check_input_size(n_components, n_features):
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if n_features <= 0:
        raise ValueError("n_features must be positive")

def _dense_random_matrix(n_components, n_features, random_state=None):
    _check_input_size(n_components, n_features)
    rng = check_random_state(random_state)
    return rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(n_components),
        size=(n_components, n_features),
    )

def _sparse_random_matrix(n_components, n_features, density="auto", random_state=None):
    _check_input_size(n_components, n_features)
    density = _check_density(density, n_features)
    rng = check_random_state(random_state)
    if density == 1:
        components = rng.binomial(1, 0.5, (n_components, n_features)) * 2 - 1
        return components / np.sqrt(n_components)

    indices = []
    offset = 0
    indptr = [offset]
    for _ in range(n_components):
        nnz = rng.binomial(n_features, density)
        indices_i = sample_without_replacement(n_features, nnz, random_state=rng)
        indices.append(indices_i)
        offset += nnz
        indptr.append(offset)
    indices = np.concatenate(indices).astype(np.intp, copy=False)
    data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1
    indptr = np.asarray(indptr, dtype=np.intp)
    components = sp.csr_matrix((data, indices, indptr), shape=(n_components, n_features))
    return np.sqrt(1 / density) / np.sqrt(n_components) * components

class RandomProjection:
    def __init__(self, n_components="auto", eps=0.1, kind="gaussian",
                 density="auto", dense_output=False, random_state=None):
        self.n_components = n_components
        self.eps = eps
        self.kind = kind
        self.density = density
        self.dense_output = dense_output
        self.random_state = random_state

    def fit(self, X):
        n_samples, n_features = X.shape
        random_state = check_random_state(self.random_state)
        if self.n_components == "auto":
            k = min_embedding_dim(n_samples=n_samples, eps=self.eps)
        else:
            k = self.n_components
        self.n_components_ = int(k)
        if self.kind == "gaussian":
            self.components_ = _dense_random_matrix(
                self.n_components_, n_features, random_state=random_state
            )
        elif self.kind == "sparse":
            self.density_ = _check_density(self.density, n_features)
            self.components_ = _sparse_random_matrix(
                self.n_components_, n_features, density=self.density_,
                random_state=random_state
            )
        else:
            raise ValueError("kind must be 'gaussian' or 'sparse'")
        return self

    def transform(self, X):
        return safe_sparse_dot(X, self.components_.T, dense_output=self.dense_output)
```
