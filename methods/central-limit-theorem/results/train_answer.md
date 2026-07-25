The problem is to understand why sums of many small independent random effects converge to the same bell-shaped normal distribution, and to find a criterion that explains this beyond the special binomial case. De Moivre's classical argument for coin flips succeeds because explicit binomial coefficients can be approximated by a Gaussian, but that machinery collapses as soon as the summands have different distributions. Earlier attempts that assume uniformly bounded summands or finite higher moments capture special situations, yet they hide the true boundary: the real obstacle is not the size of individual variables in isolation, but whether any single summand (including its tail) carries a visible fraction of the total variance after normalization.

The right representation turns out to be the characteristic function phi_X(t) = E exp(itX). It always exists, it is bounded, and it is defined for every real t, so no tail integrability beyond a second moment is required. Most importantly, independence converts sums into products of transforms, which means the algebra of the proof happens at the level of multiplying many factors near one rather than convolving many distributions directly. A finite second moment gives the local expansion phi_X(t) = 1 - t^2 sigma^2 / 2 + o(t^2) around zero, and multiplying many expressions of the form 1 - small quadratic naturally produces an exponential of a negative quadratic. That exponential is exactly the characteristic function of the standard normal law, so the target limit is built into the variance structure from the start. The remaining challenge is to control the Taylor remainder uniformly across heterogeneous summands, and this is where a variance-tail condition becomes essential.

The method is the Central Limit Theorem, proved here in the Lindeberg-Feller form using characteristic functions. Consider a triangular array of independent centered random variables X_{n,1}, ..., X_{n,k_n} with variances sigma_{n,j}^2 such that the total variance sum_j sigma_{n,j}^2 tends to 1. The Lindeberg condition requires that for every fixed epsilon > 0, the sum of tail second moments sum_j E[X_{n,j}^2 1{|X_{n,j}| > epsilon}] tends to zero. This condition precisely captures the idea that no individual summand or tail contributes visible variance at the final scale. Under this assumption, the normalized sum sum_j X_{n,j} converges in distribution to the standard normal N(0,1). The Lindeberg condition is not merely sufficient; together with the variance normalization it is essentially the right boundary, because any visible tail variance would produce a non-Gaussian atom in the limit.

The proof works by comparing the true characteristic factor E exp(itX_{n,j}) with the quadratic proxy 1 - t^2 sigma_{n,j}^2 / 2. Inside the threshold |X_{n,j}| <= epsilon, Taylor's theorem bounds the error by a small multiple of X_{n,j}^2, and the Lindeberg condition makes the aggregate tail contribution vanish. Because independence turns the sum's characteristic function into a product of factors, the total error between the true product and the product of quadratic proxies goes to zero. The product of the proxies is prod_j (1 - t^2 sigma_{n,j}^2 / 2), and since the variances sum to one, its logarithm tends to -t^2 / 2. Levy's continuity theorem then converts pointwise convergence of characteristic functions into convergence in distribution to the standard normal.

This framework also explains the classical iid case and Lyapunov's condition as easy corollaries. If X_j are iid with finite variance, scaling by sqrt(n) makes every tail variance vanish, so Lindeberg's condition holds automatically. If a higher moment sum_j E|X_{n,j}|^{2+delta} tends to zero, Markov's inequality forces the Lindeberg tail sum to zero as well, showing that Lyapunov's condition is sufficient but not necessary. The essence of the theorem is therefore not a particular moment of order above two, but the disappearance of visible individual contributions under normalization.

Stated precisely, this is the deliverable. Let $X_{n,1},\dots,X_{n,k_n}$ be, for each $n$, a row of independent random variables with $E[X_{n,j}]=0$, variances $\sigma_{n,j}^2$, and

$$\sum_{j=1}^{k_n}\sigma_{n,j}^2 \longrightarrow 1.$$

Assume the Lindeberg condition: for every $\varepsilon>0$,

$$\sum_{j=1}^{k_n} E\!\left[X_{n,j}^2\,\mathbf{1}\{|X_{n,j}|>\varepsilon\}\right] \longrightarrow 0.$$

Then

$$\sum_{j=1}^{k_n} X_{n,j} \;\Longrightarrow\; N(0,1).$$

The proof is exactly the comparison carried out above: writing $\varphi_{n,j}(t)=E\big[e^{itX_{n,j}}\big]$ for each factor's characteristic function, the Taylor-versus-tail split shows that for every fixed $t$,

$$\sum_{j} \left|\varphi_{n,j}(t) - \left(1-\frac{t^2\sigma_{n,j}^2}{2}\right)\right| \longrightarrow 0,$$

so the true characteristic function of the row sum, $\prod_j \varphi_{n,j}(t)$, has the same limit as $\prod_j\left(1-\frac{t^2\sigma_{n,j}^2}{2}\right)$. Because $\sum_j\sigma_{n,j}^2\to1$ and the Lindeberg condition forces $\max_j\sigma_{n,j}^2\to0$, the logarithm of that proxy product tends to $-t^2/2$, so the characteristic function of the sum converges pointwise to $e^{-t^2/2}$, and Lévy's continuity theorem turns this into convergence in distribution to $N(0,1)$. The classical iid theorem drops out as the row $X_{n,j}=X_j/\sqrt{n}$ for iid, mean-zero, unit-variance $X_j$: $\sigma_{n,j}^2=1/n$ sums to exactly $1$, and dominated convergence sends the Lindeberg tail sum to $0$ for every fixed $\varepsilon$, so a finite second moment alone already supplies the condition. Lyapunov's criterion is the convenient sufficient special case, since for $\delta>0$,

$$\sum_{j} E\!\left[X_{n,j}^2\,\mathbf{1}\{|X_{n,j}|>\varepsilon\}\right] \le \varepsilon^{-\delta}\sum_j E\big|X_{n,j}\big|^{2+\delta},$$

so $\sum_j E|X_{n,j}|^{2+\delta}\to0$ forces the Lindeberg sum to $0$ as well. This is the criterion in its full and final form: it is the vanishing of visible tail variance at the normalizing scale, not any particular higher moment, that separates the rows that converge to the Gaussian law from the rows that do not.
