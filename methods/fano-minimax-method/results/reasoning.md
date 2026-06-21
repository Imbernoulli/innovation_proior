I start with the quantifier that makes the lower bound hard. The risk I want to lower bound is an infimum over all estimators, so I cannot pick a bad estimator and criticize it. I need an argument that follows any estimator around and shows that, no matter how it is designed, it must fail somewhere.

The first tempting idea is to use two parameter values. If two data distributions are close, then no test can distinguish them reliably, and any estimator that separates the two parameters would give such a test. That is clean, but I immediately see the wall: two points only carry one bit of difficulty. In high-dimensional and nonparametric problems, the hard part is often not one direction but the number of separated possibilities. I need the lower bound to remember volume, sparsity choices, or metric entropy.

So I replace two worlds by many worlds. I choose parameters $\theta_1,\ldots,\theta_M$ whose pairwise distance is at least $2\epsilon$ under the estimation metric. This choice has to balance three things at once. The set must be large, because $\log M$ is the amount of uncertainty I will ask the data to resolve. The induced distributions must be close, because otherwise the data really can reveal the index. The parameters must be separated, because otherwise confusing them does not imply much loss.

I then randomize the hard set. I let $V$ be uniform on $\{1,\ldots,M\}$ and generate the data from the law indexed by $V$. This looks like a Bayesian move, but I am only using it as a lower-bound device. The worst-case risk over the full model is at least the average risk over any finite subset, so proving that the average over this artificial experiment is large is enough.

Now I need the bridge from estimation to testing. Given any estimator $\hat\theta(X)$, I round it to the nearest point in my finite set:

$$
\hat V(X)=\arg\min_j \rho(\hat\theta(X),\theta_j).
$$

If the true index is $V=v$ and the estimator lands within $\epsilon$ of $\theta_v$, then no other $\theta_j$ can be as close, because all other selected parameters are at least $2\epsilon$ away. Therefore accurate estimation forces correct identification. Taking the contrapositive, if the induced test gets the index wrong, then the estimator has made at least $\epsilon$ error:

$$
\{\hat V\ne V\}\subseteq \{\rho(\hat\theta(X),\theta_V)\ge \epsilon\}.
$$

This is the point where the original minimax quantifier disappears. The argument has not used anything special about $\hat\theta$. Every estimator induces some test, and every such test is subject to the same information constraint.

I can now apply the classical Fano inequality. For a uniform index over $M$ alternatives and any estimator of that index,

$$
\mathbb P(\hat V\ne V)\ge 1-\frac{I(V;X)+\log 2}{\log M}.
$$

The denominator is the uncertainty in the hidden choice. The numerator is the information that the sample carries about that choice, up to the standard binary-entropy slack. If the selected worlds are numerous but statistically close, the error probability stays bounded away from zero.

Combining the rounding step and the Fano step gives the basic minimax lower bound. For a nondecreasing loss transform $\Phi$,

$$
\inf_{\hat\theta}\sup_{\theta\in\Theta}\mathbb E_\theta[\Phi(\rho(\hat\theta,\theta))]
\ge
\Phi(\epsilon)\left(1-\frac{I(V;X)+\log 2}{\log M}\right),
$$

provided the hard set is a $2\epsilon$-packing. If the separation is written as an $\epsilon$-packing instead, the scale becomes $\Phi(\epsilon/2)$. I have to keep this half-distance constant straight, because it is exactly where the nearest-neighbor reduction enters.

The remaining work in any application is not the inequality itself. It is the construction and the information bound. I usually upper bound the mutual information by KL divergence, for example

$$
I(V;X)\le \frac{1}{M}\sum_{v=1}^M D(P_v\|Q)
$$

for a reference distribution $Q$, or by the pairwise average

$$
I(V;X)\le \frac{1}{M^2}\sum_{v,w}D(P_v\|P_w).
$$

With independent samples, these divergences typically add over the sample size $n$. The lower-bound rate comes from choosing the separation scale so that the information remains a small fraction of $\log M$.

The distance-based version sharpens the same idea. Instead of forcing exact recovery of the index, I can ask whether the estimated index lies within a radius $t$ under an index distance $\rho_V$. Then the effective number of distinguishable regions is not $|V|$ but roughly $|V|/N_t^{\max}$, where $N_t^{\max}$ is the largest number of index points inside a radius-$t$ neighborhood. Duchi and Wainwright's corollary turns this directly into

$$
M_n\ge
\Phi(\delta(t)/2)
\left(1-\frac{I(X_1^n;V)+\log 2}{\log(|V|/N_t^{\max})}\right),
$$

where $\delta(t)$ is the parameter separation guaranteed whenever two index values are farther than $t$. This avoids an extra discretize-then-reduce step in some problems and also explains the classical packing proof as the special case $t=0$.

I also need to keep the limitations visible. This method often proves a constant-probability or weak-converse lower bound, and it can be loose if the packing is poorly chosen or the KL bound is too crude. In adaptive problems, the mutual information terms can become difficult to control. But when the geometry supplies many separated alternatives and the data laws remain close, the method cleanly proves that the obstacle is not an inadequate estimator. The sample simply does not contain enough information to identify the truth at the requested resolution.
