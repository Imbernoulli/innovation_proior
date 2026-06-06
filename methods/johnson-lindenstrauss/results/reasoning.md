OK, let me think this through from scratch. I have $n$ points sitting in $\mathbb{R}^d$, $d$ enormous, and I want to push them down into $\mathbb{R}^k$ with $k$ as small as I can get, while keeping every pairwise distance to within $(1\pm\varepsilon)$. Everything distance-based I'd do afterward — nearest neighbors, clustering, regression — pays for the ambient dimension, so the prize is a $k$ that depends only on how hard the *point set* is, not on the accidental $d$ the data happens to arrive in. The question I actually care about: how small can $k$ be, and does it have to grow with $d$ at all?

First, is this even possible? My gut says "to keep distances you need the dimension the data really lives in, and that can be huge." Take the worst case for *exactness*: $n$ mutually equidistant points, a regular simplex. To realize $n$ equal distances exactly I need $n-1$ dimensions — no squeezing that into fewer without breaking a distance. So if I demand zero error, $k$ is forced up near $n$ and there's nothing to do. The whole hope lives in the $\varepsilon$ of slack. I should keep my eye on the fact that it's *squared* distances with a *multiplicative* band $(1\pm\varepsilon)$ that I'm allowed to be sloppy about.

Now what kind of map? The instinct from data analysis is to fit the data: take the top-$k$ principal directions, the SVD, project there. That minimizes a sum of squared residuals — the best rank-$k$ approximation $A_k$ beats every rank-$k$ matrix $D$ in Frobenius norm, $\|A-A_k\|\le\|A-D\|$. But stare at what that objective actually controls: it's the *total* displacement summed over points. A global optimum can sacrifice one pair completely — shrink one inter-point distance to nearly zero — if that buys a bigger reduction elsewhere in the sum. There's no floor under any individual distance. And the $k$ it needs is dictated by the spectrum of *this* data; nothing promises it's free of $d$. So the data-fitting instinct is exactly wrong for an all-pairs, $d$-independent guarantee. I want the opposite: a map that doesn't look at the data at all and treats every direction even-handedly.

A map that treats every direction the same — that screams *random and rotation-invariant*. Project onto a uniformly random $k$-dimensional subspace. I want this map to be **linear**. If $f$ is linear, then $f(v_i)-f(v_j)=f(v_i-v_j)$, so preserving the distance between points $v_i,v_j$ is *literally the same statement* as preserving the length of the single vector $x = v_i - v_j$. The $\binom{n}{2}$ pairwise-distance requirements collapse into $\binom{n}{2}$ single-vector length-preservation requirements about the difference vectors. Suddenly I don't have to reason about a coupled set of pairwise constraints; I have to reason about *one* event — "this fixed vector keeps its length" — and then control how many such events I'm asking to hold at once.

So the engine I need is: for a fixed vector $x$, when I apply my random linear map, how concentrated is $\|f(x)\|^2$ around $\|x\|^2$? If I can show $\|f(x)\|^2$ lands in $(1\pm\varepsilon)\|x\|^2$ except with some tiny probability $p$, then by a union bound the chance that *any* of the $\binom{n}{2}$ difference vectors fails is at most $\binom{n}{2}\,p$. If I can drive $p$ below, say, $1/n^2$, that product is below $1$, and by the probabilistic method a good map *exists* — in fact a random draw works with decent probability. The entire problem reduces to a single tail bound, and the price of "all pairs" is just the $\log$ of their count showing up through the union bound. That's where I'd expect $\log n$ to enter $k$.

Let me set up the concentration. Take $x$ a unit vector without loss of generality (everything scales). I project onto a random $k$-subspace. Reasoning about a random subspace directly is annoying, but there's a symmetry that flips it into something I can compute: projecting a *fixed* unit vector onto a *random* $k$-subspace has exactly the same distribution as projecting a *random* unit vector onto a *fixed* $k$-subspace — say the span of the first $k$ coordinates. And a random unit vector is easy to build: let $X_1,\dots,X_d$ be i.i.d. $N(0,1)$ and set $Y = X/\|X\|$; then $Y$ is uniform on the sphere $S^{d-1}$. Project $Y$ to its first $k$ coordinates, call the result $Z\in\mathbb{R}^k$, and let $L=\|Z\|^2 = (X_1^2+\dots+X_k^2)/(X_1^2+\dots+X_d^2)$. The expected squared length is $\mathbb{E}[L]=k/d$ by symmetry — each coordinate carries an equal $1/d$ share. So the "right" scaling to make the map length-preserving is to multiply by $\sqrt{d/k}$; then the expected squared length is $1$, matching $\|x\|^2$.

Now I need the tails of $L$ around $\mu=k/d$. I want $\Pr[L \le (1-\varepsilon)\mu]$ and $\Pr[L\ge(1+\varepsilon)\mu]$ both tiny. $L$ is a ratio of sums of squared Gaussians — a Chernoff/MGF argument is the natural tool. Let me check the one-coordinate MGF rather than just quote it: for $X\sim N(0,1)$,

$$\mathbb{E}[e^{sX^2}] = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} e^{-(1/2-s)x^2}\,dx = (1-2s)^{-1/2},\qquad s<\tfrac12.$$

Let me do the lower tail, $\beta = 1-\varepsilon < 1$, i.e. bound $\Pr[L\le\beta\mu]$. Write $A=X_1^2+\dots+X_k^2$ and $B=X_{k+1}^2+\dots+X_d^2$. Clearing the denominator, $L\le \beta k/d$ means $dA \le k\beta(A+B)$, or $k\beta(A+B)-dA\ge0$. Exponentiate with a parameter $t>0$ and apply Markov:

$$\Pr[\,L\le\beta k/d\,] \le \mathbb{E}\big[\exp\{t(k\beta(A+B)-dA)\}\big].$$

The first $k$ coordinates appear with coefficient $t(k\beta-d)$ on $X_i^2$, the remaining $d-k$ with coefficient $tk\beta$. By independence this factors:

$$= \big(\mathbb{E}[e^{tk\beta X^2}]\big)^{d-k}\,\big(\mathbb{E}[e^{t(k\beta - d)X^2}]\big)^{k} = (1-2tk\beta)^{-(d-k)/2}\,(1-2t(k\beta-d))^{-k/2}.$$

Call this $g_-(t)$. The MGF demands $tk\beta < \tfrac12$ (the other constraint $t(k\beta-d)<\tfrac12$ is automatic since $t>0$ and $k\beta<d$ in this regime), so $0<t<1/(2k\beta)$. I minimize $\log g_-(t)$:

$$\frac{d}{dt}\log g_-(t)=\frac{(d-k)k\beta}{1-2tk\beta}+\frac{k(k\beta-d)}{1-2t(k\beta-d)}.$$

Setting this to zero leaves $2\beta t(d-k\beta)=1-\beta$, so

$$t_0 = \frac{1-\beta}{2\beta(d-k\beta)},$$

which lies in the allowed interval. Plugging $t_0$ back gives $1-2t_0k\beta=(d-k)/(d-k\beta)$ and $1-2t_0(k\beta-d)=1/\beta$, so

$$\Pr[L\le \beta\mu] \;\le\; \beta^{k/2}\Big(1+\frac{(1-\beta)k}{d-k}\Big)^{(d-k)/2} \;\le\; \exp\!\Big(\tfrac{k}{2}\big(1-\beta+\ln\beta\big)\Big),$$

where the last step uses $1+y\le e^y$ on the middle factor. For the upper tail, $\beta>1$, the event is $dA\ge k\beta(A+B)$. If $k\beta\ge d$ the threshold is at least the maximum possible value of $L$, so the interesting case is $k\beta<d$. The same Markov move now uses $dA-k\beta(A+B)\ge0$:

$$\Pr[\,L\ge\beta k/d\,]\le (1+2tk\beta)^{-(d-k)/2}(1+2t(k\beta-d))^{-k/2}.$$

This is the previous $g_-$ evaluated at $-t$, and the positive optimizer is $t_+=(\beta-1)/(2\beta(d-k\beta))$. Substitution gives the same closed form

$$\Pr[L\ge \beta\mu] \;\le\; \beta^{k/2}\Big(1+\frac{(1-\beta)k}{d-k}\Big)^{(d-k)/2} \;\le\; \exp\!\Big(\tfrac{k}{2}\big(1-\beta+\ln\beta\big)\Big).$$

Good — the signs changed in the Chernoff step, but one expression governs both tails.

Now substitute the actual $\beta$'s and squeeze the $\ln$. Lower tail, $\beta=1-\varepsilon$:

$$\tfrac k2\big(1-(1-\varepsilon)+\ln(1-\varepsilon)\big) = \tfrac k2\big(\varepsilon + \ln(1-\varepsilon)\big).$$

I need an upper bound on $\ln(1-\varepsilon)$. The inequality $\ln(1-x)\le -x-\tfrac{x^2}{2}$ holds for $0\le x<1$. So $\varepsilon+\ln(1-\varepsilon) \le \varepsilon - \varepsilon - \tfrac{\varepsilon^2}{2} = -\tfrac{\varepsilon^2}{2}$, giving

$$\Pr[L\le(1-\varepsilon)\mu] \le \exp\!\big(-\tfrac{k\varepsilon^2}{4}\big).$$

Upper tail, $\beta=1+\varepsilon$:

$$\tfrac k2\big(1-(1+\varepsilon)+\ln(1+\varepsilon)\big) = \tfrac k2\big(-\varepsilon+\ln(1+\varepsilon)\big).$$

I want to *lower*-bound the exponent's magnitude, so I need an upper bound on $\ln(1+\varepsilon)$ that still leaves a negative quadratic term. The one-term bound $\ln(1+x)\le x$ gives no decay at all, and the alternating two-term truncation $x-\tfrac{x^2}{2}$ is a lower bound near zero, so it cannot be used here. The useful upper bound is the three-term one, $\ln(1+x)\le x-\tfrac{x^2}{2}+\tfrac{x^3}{3}$ (valid for $x\ge0$):

$$-\varepsilon+\ln(1+\varepsilon) \le -\varepsilon + \varepsilon - \tfrac{\varepsilon^2}{2}+\tfrac{\varepsilon^3}{3} = -\big(\tfrac{\varepsilon^2}{2}-\tfrac{\varepsilon^3}{3}\big),$$

so

$$\Pr[L\ge(1+\varepsilon)\mu] \le \exp\!\Big(-\tfrac k2\big(\tfrac{\varepsilon^2}{2}-\tfrac{\varepsilon^3}{3}\big)\Big).$$

Stare at the two exponents. Lower tail: $-k\varepsilon^2/4$. Upper tail: $-\tfrac k2(\tfrac{\varepsilon^2}{2}-\tfrac{\varepsilon^3}{3}) = -k(\tfrac{\varepsilon^2}{4}-\tfrac{\varepsilon^3}{6})$. For $\varepsilon\in(0,1)$ the upper-tail exponent is the *smaller in magnitude* — it has that extra positive $\tfrac{\varepsilon^3}{6}$ chipping away at it — so the upper tail is the **binding** one. It dictates how big $k$ must be, and the $\varepsilon^3/3$ correction I kept is precisely what sharpens the final constant; if I'd been lazy and used the two-term log bound, I'd have paid for it in $k$. So the controlling tail is $\exp\!\big(-\tfrac k2(\tfrac{\varepsilon^2}{2}-\tfrac{\varepsilon^3}{3})\big)$.

Now wire in the union bound. I want each difference vector's *two-sided* failure probability below $2/n^2$, so that summed over $\binom{n}{2}<n^2/2$ pairs the total failure is below $1$. Each tail below $1/n^2 = \exp(-2\ln n)$ suffices. Setting the binding (upper) tail to $\exp(-2\ln n)$:

$$\tfrac k2\big(\tfrac{\varepsilon^2}{2}-\tfrac{\varepsilon^3}{3}\big) \ge 2\ln n \;\;\Longleftrightarrow\;\; k \ge \frac{4\ln n}{\varepsilon^2/2 - \varepsilon^3/3}.$$

And the lower tail with this $k$ is even smaller (its exponent coefficient $\varepsilon^2/4$ exceeds $(\varepsilon^2/2-\varepsilon^3/3)/2 = \varepsilon^2/4-\varepsilon^3/6$ for $\varepsilon\in(0,1)$), so it's also $\le 1/n^2$. There it is: $k = O(\log n/\varepsilon^2)$, and — this is the payoff — **$d$ has completely vanished**. It dropped out the moment I worked with $L$ as a ratio that concentrates around $k/d$ and then rescaled by $\sqrt{d/k}$; the ambient dimension only ever set the *mean*, never the spread of the relative error. With $k = \lceil 4(\varepsilon^2/2-\varepsilon^3/3)^{-1}\ln n\rceil$, each pair fails with probability $\le 2/n^2$, the union over $\binom{n}{2}$ pairs fails with probability at most $\binom{n}{2}\cdot 2/n^2 = 1 - 1/n$, so a single random projection succeeds with probability at least $1/n$. That's a positive probability — a good map exists — and repeating the draw $O(n)$ times boosts the success probability to a constant. Set $f(v_i) = \sqrt{d/k}\,v_i'$ where $v_i'$ is the projection, and we're done with the existence statement.

Now I want to *implement* this, and the "uniformly random $k$-subspace" is awkward — it means orthonormalizing $k$ vectors in $\mathbb{R}^d$. Do I actually need a true orthonormal frame? Let me be lazy on purpose: instead of orthonormalizing, just take $k$ independent vectors with i.i.d. $N(0,1)$ coordinates. For a fixed $x$, $\langle U_i,x\rangle = \sum_j x_j U_{ij}$ is exactly $N(0,\|x\|^2)$ by 2-stability, so $\sum_i\langle U_i,x\rangle^2/\|x\|^2$ is $\chi^2_k$. The same $\chi^2_k$ appears for *every* $x$ — all vectors look alike under a Gaussian row — so the concentration bound applies directly, with no ratio and no orthogonality bookkeeping: for $S=\sum_i G_i^2$, the MGF above gives $\Pr[S\le\beta k]\le\exp(\tfrac{k}{2}(1-\beta+\ln\beta))$ by applying Markov to $e^{-\lambda S}$ with $\lambda=(1-\beta)/(2\beta)$, and $\Pr[S\ge\beta k]\le\exp(\tfrac{k}{2}(1-\beta+\ln\beta))$ by applying it to $e^{\lambda S}$ with $\lambda=(\beta-1)/(2\beta)$. The scale is forced by unbiasedness: without scaling, the expected squared length is $k\|x\|^2$, so I fold in $1/\sqrt k$. That's the form I can actually code: a $k\times d$ matrix whose entries have standard deviation $1/\sqrt k$, and $f(X)=XR^\top$.

Let me reframe what's really happening, because it suggests how to make the map cheaper. Each *column* of $R$ — no wait, each *row* of $R$, i.e. each random vector — produces one coordinate of $f(x)$ by taking an inner product with $x$. That coordinate squared is an estimator of $\|x\|^2$: it's $\langle r, x\rangle^2$, and as long as $r$ has independent mean-zero, unit-variance entries, $\mathbb{E}[\langle r,x\rangle^2] = \sum_{i,j} x_i x_j\,\mathbb{E}[r_i r_j] = \sum_i x_i^2\,\mathbb{E}[r_i^2] = \|x\|^2$ — the cross terms die because the entries are independent with mean zero, and only the diagonal survives with variance $1$. So *every* coordinate of $f(x)$ is an unbiased estimator of $\|x\|^2$, and $\|f(x)\|^2 = \tfrac1k\sum_{i=1}^k \langle r_i,x\rangle^2$ is the average of $k$ i.i.d. unbiased estimators. By the law of large numbers it concentrates; how many I need is set purely by the *variance* of one estimator. Notice what this argument did and did not use: it used mean zero, variance one, and independence. It did **not** use Gaussianity. The Gaussian was a comfort — it gave me an exact $\chi^2$ and the cleanest possible concentration — but unbiasedness, the thing that pins the mean to $\|x\|^2$, asks for almost nothing.

So can I replace the Gaussian entries with something dumber and faster? What if each $r_{ij}$ is just $\pm1$ with probability $\tfrac12$ each? Mean zero, variance one — the expectation calculation above goes through unchanged, so $\mathbb{E}\|f(x)\|^2=\|x\|^2$ still holds exactly. The map becomes: pick a random $\pm1$ matrix, and computing $\langle r,x\rangle$ is just additions and subtractions of the coordinates of $x$ — no floating-point multiplies, far fewer random bits, and in a database it's literally "add up a random half of the attributes, subtract the other half." That's a huge practical win *if* the concentration survives. And here's the worry: without spherical symmetry, the distribution of $\|f(x)\|^2$ now *depends on $x$*. For a Gaussian $R$ every $x$ gave the same $\chi^2$; for a $\pm1$ matrix, some adversarial $x$ might make $\langle r,x\rangle$ much more variable than others. I can't just wave the chi-square argument anymore.

Let me look at the concentration directly. The binding event was the upper tail, and the Chernoff bound there came down to controlling $\mathbb{E}[\exp(h Q^2)]$ where $Q = \langle r, x\rangle$ for a single row $r$ (then raised to the $k$-th power by independence across rows). So the whole question reduces to: for which entry-distributions is $\mathbb{E}[\exp(hQ^2)]$ — or equivalently all the even moments $\mathbb{E}[Q^{2\ell}]$ — no larger than in the Gaussian case? If I can show the $\pm1$ moments are dominated by the Gaussian moments for *every* $x$, then the Chernoff bound, and hence $k$, is no worse.

Which $x$ is worst? Intuitively the adversary wants $x$ aligned so that $\langle r,x\rangle$ swings as wildly as possible. A spike like $x=(1,0,\dots,0)$ is actually *tame*: then $\langle r,x\rangle = r_1 = \pm1$, perfectly bounded, no tail at all. The dangerous $x$ is the *spread-out* one, $w=\tfrac{1}{\sqrt d}(1,1,\dots,1)$, where $\langle w,r\rangle = \tfrac{1}{\sqrt d}\sum_i r_i$ is a sum of $d$ independent signs — the configuration with the fattest fluctuations. So I'd conjecture the all-ones vector (and its sign-flips) is the worst case, and a convexity/balancing argument confirms it: among unit vectors, the even moments $\mathbb{E}[Q(x)^{2\ell}]$ are maximized when the squared coordinates are all equal. Concretely, take any $x$ with two unequal squared coordinates $x_1^2\ne x_2^2$ and replace both by their average $\gamma=\sqrt{(x_1^2+x_2^2)/2}$; one shows $\mathbb{E}[Q(x)^{2\ell}]\le \mathbb{E}[Q(\tilde x)^{2\ell}]$ for the more balanced $\tilde x$, and iterating drives $x$ to $w$. The key lemma behind one balancing step: writing $Q$ as $T + a r_1 + b r_2$ (the contribution of the other coordinates lumped into $T$), one checks $\mathbb{E}[(T+ar_1+br_2)^{2\ell}] \le \mathbb{E}[(T+\gamma r_1 + \gamma r_2)^{2\ell}]$ with $\gamma=\sqrt{(a^2+b^2)/2}$, because the relevant difference expands, via the binomial theorem, into a sum of terms $\binom{2k}{2j}T^{2(k-j)}D_{2j}$ with each $D_{2j}\ge0$ — using $(a+b)^2+(a-b)^2 = 2a^2+2b^2$ and $(x+y)^j\ge x^j+y^j$ for $x,y\ge0$, $j\ge1$. Balancing only ever raises the even moments, so $w$ is extremal.

Now I just need: at the worst case $w$, are the $\pm1$ even moments still $\le$ the Gaussian ones? At $w$, $Q(w) = \tfrac1{\sqrt d}\sum_i Y_i$ with $Y_i=\pm1$ i.i.d., versus the Gaussian comparison $T = \tfrac1{\sqrt d}\sum_i G_i\sim N(0,1)$ with $G_i\sim N(0,1)$. Match them term by term in the moment expansion: any monomial $\mathbb{E}[Y_{i_1}\cdots Y_{i_{2k}}]$ is nonzero only when every index appears an even number of times (odd powers integrate to zero by symmetry, for both $Y$ and $G$), so everything reduces to comparing single-coordinate even moments $\mathbb{E}[Y_1^{2\ell}]$ vs. $\mathbb{E}[G_1^{2\ell}]$. For the sign, $\mathbb{E}[Y_1^{2\ell}] = 1$. For the Gaussian, $\mathbb{E}[G_1^{2\ell}] = (2\ell-1)!! = (2\ell)!/(\ell!\,2^\ell) \ge 1$. So $\mathbb{E}[Y_1^{2\ell}]\le\mathbb{E}[G_1^{2\ell}]$ trivially, the domination propagates up through every multi-index, and $\mathbb{E}[Q(w)^{2k}]\le\mathbb{E}[T^{2k}]$. Chaining: for *any* unit $x$, $\mathbb{E}[Q(x)^{2k}]\le\mathbb{E}[Q(w)^{2k}]\le\mathbb{E}[T^{2k}]$. The sign matrix's moments are everywhere below the Gaussian's. Summing into the MGF, $\mathbb{E}[\exp(hQ^2)]\le (1-2h)^{-1/2}$ for $h\in[0,1/2)$ — exactly the $\chi^2_1$ MGF — and $\mathbb{E}[Q^4]\le 3$. The Chernoff bound therefore reproduces the *same* tail $\exp(-\tfrac k2(\tfrac{\varepsilon^2}{2}-\tfrac{\varepsilon^3}{3}))$, and hence the *same* $k\ge 4(\varepsilon^2/2-\varepsilon^3/3)^{-1}\ln n$. Spherical symmetry bought me comfort, not necessity; concentration is the only thing that mattered, and the sign matrix concentrates at least as well.

Can I push sparsity further? The estimator only needs mean zero and variance one, so I have a free parameter: let each entry be $0$ with some probability and a larger spike otherwise, keeping variance one. Try $r_{ij} \in \{+\sqrt3, 0, -\sqrt3\}$ with probabilities $\{\tfrac16, \tfrac23, \tfrac16\}$: mean zero, and variance $3\cdot\tfrac13 = 1$. Now two-thirds of the matrix is zero, so applying it touches only a third of the coordinates — a threefold speedup and even fewer operations. Does the worst-case-is-all-ones argument survive? The same balancing and moment-domination go through — the single-coordinate even moments are $\mathbb{E}[Y_1^{2\ell}] = 3^\ell\cdot\tfrac13 = 3^{\ell-1} \le (2\ell)!/(\ell!\,2^\ell)$, still dominated by the Gaussian — so the positive-probability union-bound dimension is unchanged. If I want the explicit high-probability form with failure at most $n^{-\beta_{\rm prob}}$, I only change the union-bound target from $2/n^2$ to $2/n^{2+\beta_{\rm prob}}$, which gives $k_0=\frac{4+2\beta_{\rm prob}}{\varepsilon^2/2-\varepsilon^3/3}\log n$. There's a sharp limit though: in this simple three-value, variance-one family, pushing the zero-probability past $\tfrac23$ breaks the domination needed for arbitrary vectors. So $\tfrac23$ is exactly the most sparsity I can take for free with this argument.

So the method is settled. Sample a $k\times d$ matrix $R$ with i.i.d. mean-zero, variance-one entries — Gaussian, or the database-friendly $\pm1$ / $\{\pm\sqrt3,0\}$ signs — fold in a $1/\sqrt k$ so the map is unbiased, set $k = \lceil 4(\varepsilon^2/2-\varepsilon^3/3)^{-1}\ln n\rceil$, and project $f(X)=XR^\top$. Let me write it grounded in how this is actually built.

```python
import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.random import sample_without_replacement

def johnson_lindenstrauss_min_dim(n_samples, *, eps=0.1):
    eps = np.asarray(eps)
    n_samples = np.asarray(n_samples)
    if np.any(eps <= 0.0) or np.any(eps >= 1):
        raise ValueError("eps must be in (0, 1)")
    if np.any(n_samples <= 0):
        raise ValueError("n_samples must be positive")
    denominator = (eps**2 / 2) - (eps**3 / 3)
    return (4 * np.log(n_samples) / denominator).astype(np.int64)

def _check_density(density, n_features):
    if density == "auto":
        density = 1 / np.sqrt(n_features)
    elif density <= 0 or density > 1:
        raise ValueError("density must be in (0, 1]")
    return density

def _check_input_size(n_components, n_features):
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if n_features <= 0:
        raise ValueError("n_features must be positive")

def _gaussian_random_matrix(n_components, n_features, random_state=None):
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
    indices = np.concatenate(indices)
    data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1
    components = sp.csr_array((data, indices, indptr), shape=(n_components, n_features))
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
            k = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=self.eps)
        else:
            k = self.n_components
        self.n_components_ = int(k)
        if self.kind == "gaussian":
            self.components_ = _gaussian_random_matrix(
                self.n_components_, n_features, random_state=random_state
            )
        else:
            self.density_ = _check_density(self.density, n_features)
            self.components_ = _sparse_random_matrix(
                self.n_components_, n_features, density=self.density_,
                random_state=random_state
            )
        return self

    def transform(self, X):
        return safe_sparse_dot(X, self.components_.T, dense_output=self.dense_output)
```

So the causal chain: insist on a *linear* map so each pairwise distance becomes the length of one difference vector; make the map *random and rotation-blind* so it ignores the data and treats every direction alike, which a data-fitting SVD can't; reduce the all-pairs requirement to a single tail bound, $\Pr[\,|\|f(x)\|^2-\|x\|^2|>\varepsilon\|x\|^2\,]$, and prove it via the squared-Gaussian MGF, where the ambient $d$ only sets the mean and cancels, leaving the relative error governed by $k$ alone; sharpen the binding upper tail with the three-term log bound to get $k = 4(\varepsilon^2/2-\varepsilon^3/3)^{-1}\ln n$; union-bound over $\binom{n}{2}$ pairs so a random draw works with probability $\ge 1/n$; then notice the proof used only mean zero, variance one, and independence — so swap Gaussians for $\pm1$ or sparse $\{\pm\sqrt3,0\}$ entries, with moment domination at the all-ones worst case certifying the same $k$ — buying integer arithmetic, sparsity, and far fewer random bits at no cost in the bound.
