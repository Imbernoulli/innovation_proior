I keep running into the same wall from different directions, so let me just stare at it. I have a model $f(x\mid\phi)$ for some complete data $x$, but I never see all of $x$. What reaches me is $y$ — a coarsening of $x$ through a known many-to-one map $x\mapsto y(x)$. Counts collapsed across categories. Lifetimes I only know to exceed some censoring time. Survey cells left blank. Points whose component-of-origin label was never recorded. The likelihood I am entitled to maximize is therefore the marginal,

$$g(y\mid\phi)=\int_{\mathcal X(y)} f(x\mid\phi)\,dx,$$

where $\mathcal X(y)$ is all the complete data consistent with what I saw. And I want $\hat\phi=\arg\max_\phi \log g(y\mid\phi)$.

If I just go at it directly — write $L(\phi)=\log g(y\mid\phi)$ and set $\partial_\phi L=0$ — the integral over the hidden coordinates is trapped inside a logarithm. Take the case I care about most, a finite mixture: $g(y\mid\phi)=\prod_i \sum_k \pi_k\, p_k(y_i\mid\theta_k)$, so $L=\sum_i \log\sum_k \pi_k p_k(y_i)$. A sum of logs of sums. Differentiate and the $\pi_k$, $\theta_k$ of every component show up coupled through a denominator $\sum_j \pi_j p_j(y_i)$ that is itself the thing I'm trying to estimate. No closed form. Pearson back in 1894 gave up on likelihood here and went to moments precisely because of this.

And yet — this is the thing nagging at me — if I *had* seen the complete data, none of this would be hard. If for each point $y_i$ I also knew its label $z_i$, the likelihood would factor: each component would just be fit to the points assigned to it, the mixing weights would be the label frequencies, done, closed form. The difficulty is entirely manufactured by the marginalization. Hartley saw this in 1958 for multinomial counts: he took observed counts that were collapses of a finer multinomial, split them back into latent sub-counts, used the current parameter to apportion each observed count into *expected* sub-counts, then re-estimated by ordinary multinomial maximum likelihood as if those filled-in counts were real, and looped. Hasselblad and Day and Wolfe each independently did the mixture version — compute each point's current probability of belonging to each component, re-fit each component with those probabilities as weights, repeat — and each of them *reported* that the likelihood went up every step. Baum and his collaborators did the hidden-Markov-chain version and actually proved improvement for their model with some auxiliary function. Orchard and Woodbury even gave the slogan: a "missing information principle," replace the missing data by its conditional expectation and iterate.

So everybody keeps reinventing the same loop — fill in the hidden part using the current guess, re-estimate as if the filled-in data were observed, repeat — in their own corner, with their own notation, and mostly without a proof that it works. The loop is obviously *there*. What I don't have, and what I want, is two things: the loop written once at a level of generality that doesn't care whether it's multinomial counts or Gaussian mixtures or censored survival times, and a reason — a real theorem — that the observed-data likelihood $L(\phi)$ goes up every single iteration, not just "in our experiments."

Let me first get honest about what "fill in the hidden part" means, because "replace the missing data by its conditional expectation" is doing a lot of quiet work and I suspect it's only accidentally right. The conditional density of the complete data given what I observed is

$$k(x\mid y,\phi)=\frac{f(x\mid\phi)}{g(y\mid\phi)},$$

defined on $\mathcal X(y)$ — this is the posterior over the hidden coordinates at the current parameter $\phi$. Now the identity I keep circling: from $f=g\cdot k$,

$$\log f(x\mid\phi)=\log g(y\mid\phi)+\log k(x\mid y,\phi),$$

so

$$\log g(y\mid\phi)=\log f(x\mid\phi)-\log k(x\mid y,\phi).$$

The left side has no $x$ in it. So I'm free to take the expectation of both sides over $x$ under *any* distribution on $\mathcal X(y)$ I like and the left side just sits there unchanged. That freedom is the whole game; let me use it deliberately. Take the conditional expectation under the posterior at some parameter $\phi'$ — that is, $E[\,\cdot\mid y,\phi']$:

$$\log g(y\mid\phi)=E\big[\log f(x\mid\phi)\,\big|\,y,\phi'\big]-E\big[\log k(x\mid y,\phi)\,\big|\,y,\phi'\big].$$

Give the two pieces names, because they're the two halves I'll be juggling forever:

$$Q(\phi\mid\phi')=E\big[\log f(x\mid\phi)\,\big|\,y,\phi'\big],\qquad H(\phi\mid\phi')=E\big[\log k(x\mid y,\phi)\,\big|\,y,\phi'\big],$$

so that

$$L(\phi)=Q(\phi\mid\phi')-H(\phi\mid\phi')\qquad\text{for every }\phi'.$$

$Q$ is the expected *complete-data* log-likelihood, with the expectation taken over the hidden coordinates as the current guess $\phi'$ believes them to be. $H$ is the expected log of the posterior — the leftover term.

Now, what would I love to do? I'd love to just maximize $\log f(x\mid\phi)$ over $\phi$, because that's the easy complete-data problem. I can't, because I don't know $x$. But I can maximize its *expectation* $Q(\phi\mid\phi')$, where the expectation softens the missing $x$ into something I can compute from the current $\phi'$. So the move suggests itself: hold $\phi'$ fixed at my current estimate, maximize $Q(\phi\mid\phi')$ over $\phi$, call the maximizer my new estimate, and repeat. At iteration $p$, with current value $\phi^{(p)}$, I first compute $Q(\phi\mid\phi^{(p)})=E[\log f(x\mid\phi)\mid y,\phi^{(p)}]$, which is the expectation that fills in the hidden coordinates by their posterior under $\phi^{(p)}$; then I choose $\phi^{(p+1)}=\arg\max_\phi Q(\phi\mid\phi^{(p)})$. That is exactly Hartley's and Baum's and Hasselblad's loop, now stated without reference to any particular model.

But I have to *earn* this. Maximizing $Q$ is not the same as maximizing $L$ — there's that extra $-H$ term floating around. Why on earth should pushing up $Q$ push up $L$? If it doesn't, this whole loop is just folklore.

So let me chase the increment in $L$ across one step and see what controls it. Go from $\phi^{(p)}$ to any new $\phi$. Using $L(\phi)=Q(\phi\mid\phi^{(p)})-H(\phi\mid\phi^{(p)})$ and the same identity at $\phi=\phi^{(p)}$, $L(\phi^{(p)})=Q(\phi^{(p)}\mid\phi^{(p)})-H(\phi^{(p)}\mid\phi^{(p)})$, subtract:

$$L(\phi)-L(\phi^{(p)})=\underbrace{\big[Q(\phi\mid\phi^{(p)})-Q(\phi^{(p)}\mid\phi^{(p)})\big]}_{\text{change in } Q}+\underbrace{\big[H(\phi^{(p)}\mid\phi^{(p)})-H(\phi\mid\phi^{(p)})\big]}_{\text{change in } -H}.$$

The first bracket is exactly what the M-step makes non-negative: I *chose* $\phi=\phi^{(p+1)}$ to maximize $Q(\cdot\mid\phi^{(p)})$, so $Q(\phi^{(p+1)}\mid\phi^{(p)})\ge Q(\phi^{(p)}\mid\phi^{(p)})$, and that bracket is $\ge 0$. Good. But the second bracket is not under my control at all — it's a side effect. If it could be negative and large, it could swamp the $Q$ gain and $L$ could go *down*. So everything hangs on the sign of $H(\phi^{(p)}\mid\phi^{(p)})-H(\phi\mid\phi^{(p)})$. I need this to be $\ge 0$, i.e. I need

$$H(\phi\mid\phi^{(p)})\le H(\phi^{(p)}\mid\phi^{(p)})\quad\text{for all }\phi.$$

In words: among all $\phi$, the function $\phi\mapsto H(\phi\mid\phi^{(p)})=E[\log k(x\mid y,\phi)\mid y,\phi^{(p)}]$ is *maximized at $\phi=\phi^{(p)}$ itself*. Is that true? Let me just compute the difference and stop hoping:

$$H(\phi\mid\phi^{(p)})-H(\phi^{(p)}\mid\phi^{(p)})=E\!\left[\log\frac{k(x\mid y,\phi)}{k(x\mid y,\phi^{(p)})}\,\Bigg|\,y,\phi^{(p)}\right].$$

This is an expectation of a log-ratio under $k(\cdot\mid y,\phi^{(p)})$. That's exactly the shape Jensen's inequality eats. $\log$ is concave, so $E[\log U]\le \log E[U]$. Here $U=k(x\mid y,\phi)/k(x\mid y,\phi^{(p)})$ and the expectation is under the *denominator's* distribution $k(\cdot\mid y,\phi^{(p)})$. So

$$E\!\left[\log\frac{k(x\mid y,\phi)}{k(x\mid y,\phi^{(p)})}\,\Bigg|\,y,\phi^{(p)}\right]\le \log E\!\left[\frac{k(x\mid y,\phi)}{k(x\mid y,\phi^{(p)})}\,\Bigg|\,y,\phi^{(p)}\right]=\log\int_{\mathcal X(y)} \frac{k(x\mid y,\phi)}{k(x\mid y,\phi^{(p)})}\,k(x\mid y,\phi^{(p)})\,dx.$$

The $k(\cdot\mid y,\phi^{(p)})$ inside the integral cancels against the denominator, leaving $\int_{\mathcal X(y)} k(x\mid y,\phi)\,dx=1$, because $k(\cdot\mid y,\phi)$ is a probability density on $\mathcal X(y)$. So the right side is $\log 1=0$. Therefore

$$H(\phi\mid\phi^{(p)})-H(\phi^{(p)}\mid\phi^{(p)})\le 0,$$

with equality, by the equality condition in Jensen, exactly when $U$ is constant a.e., i.e. $k(x\mid y,\phi)=k(x\mid y,\phi^{(p)})$ almost everywhere. That's it — that's the load-bearing fact. The leftover term doesn't just fail to hurt; it *helps*, or at worst is neutral. (Another way to see the same thing: that log-ratio expectation is $-\mathrm{KL}\big(k(\cdot\mid y,\phi^{(p)})\,\big\|\,k(\cdot\mid y,\phi)\big)$, and a Kullback–Leibler divergence is $\ge 0$ with equality iff the two posteriors agree. Same inequality, same equality case.)

Put the two brackets back together. For the actual EM step $\phi^{(p+1)}=\arg\max Q(\cdot\mid\phi^{(p)})$,

$$L(\phi^{(p+1)})-L(\phi^{(p)})=\underbrace{\big[Q(\phi^{(p+1)}\mid\phi^{(p)})-Q(\phi^{(p)}\mid\phi^{(p)})\big]}_{\ge\,0\ \text{(M-step)}}+\underbrace{\big[H(\phi^{(p)}\mid\phi^{(p)})-H(\phi^{(p+1)}\mid\phi^{(p)})\big]}_{\ge\,0\ \text{(Jensen)}}\ \ge\ 0.$$

The observed-data log-likelihood is non-decreasing at every iteration. And I didn't even need the M-step to find the *maximizer* of $Q$ — I only used $Q(\phi^{(p+1)}\mid\phi^{(p)})\ge Q(\phi^{(p)}\mid\phi^{(p)})$, that the M-step makes $Q$ no smaller. So any update that merely *increases* $Q$ — a partial M-step, a single Newton step on $Q$, whatever — already inherits the monotonicity. That's a free generalization: call the map $\phi\mapsto M(\phi)$ a generalized EM step if $Q(M(\phi)\mid\phi)\ge Q(\phi\mid\phi)$, and the increase theorem holds verbatim. Equality in the whole thing forces equality in *both* brackets: $Q(\phi^{(p+1)}\mid\phi^{(p)})=Q(\phi^{(p)}\mid\phi^{(p)})$ and the posterior unchanged. At a fixed point $\phi^{(p+1)}=\phi^{(p)}=\phi^\ast$, both hold trivially, and then — since $\phi^\ast$ maximizes $Q(\cdot\mid\phi^\ast)$ — the gradient of $Q$ in its first slot vanishes at $\phi^\ast$; and because $H(\cdot\mid\phi^\ast)$ is maximized at $\phi^\ast$ its gradient vanishes there too, so the gradient of $L=Q-H$ vanishes at $\phi^\ast$. Fixed points of the loop are stationary points of the actual likelihood. Not necessarily the global maximum — mixtures notoriously have several — but stationary, which is exactly the honest claim.

Let me make sure I believe the geometry of why this works, not just the algebra, because the algebra came out almost too cleanly. Rearrange the identity $L=Q-H$ at a *fixed* current parameter $\phi'$ but think of it as a function of a *trial distribution* $q$ over the hidden coordinates rather than of the posterior specifically. For any density $q$ on $\mathcal X(y)$,

$$\log g(y\mid\phi)=\underbrace{E_q[\log f(x\mid\phi)]+\mathcal H(q)}_{\;\equiv\,F(q,\phi)\;}+\;\mathrm{KL}\big(q\,\big\|\,k(\cdot\mid y,\phi)\big),$$

where $\mathcal H(q)=-E_q[\log q(x)]$ is the entropy of $q$ — to check this, expand the KL: $\mathrm{KL}(q\|k)=E_q[\log q]-E_q[\log k(x\mid y,\phi)]=-\mathcal H(q)-E_q[\log f(x\mid\phi)]+E_q[\log g(y\mid\phi)]$, and $E_q[\log g]=\log g$ since $g$ has no $x$, so $F(q,\phi)+\mathrm{KL}=\log g$, as claimed. Since $\mathrm{KL}\ge 0$, this says

$$L(\phi)=F(q,\phi)+\mathrm{KL}\big(q\,\big\|\,k(\cdot\mid y,\phi)\big)\ \ge\ F(q,\phi),$$

so $F(q,\phi)=E_q[\log f(x\mid\phi)]+\mathcal H(q)$ is a *lower bound* on the log-likelihood for every $q$ — and the bound is **tight**, $F=L$, exactly when the KL vanishes, i.e. when $q$ is the posterior $k(\cdot\mid y,\phi)$. Now the two steps are revealed as the two coordinate maximizations of the *same* object $F(q,\phi)$. Maximize $F$ over $q$ with $\phi$ fixed: that's $-\mathrm{KL}$ as large as possible, achieved at $q=k(\cdot\mid y,\phi)$ — and at that $q$ the bound touches the true likelihood. That is the E-step: setting $q$ to the posterior is precisely "fill in the hidden coordinates by their conditional distribution," and what it *buys* is that the lower bound becomes exact at the current $\phi$. Then maximize $F$ over $\phi$ with $q$ fixed: the entropy $\mathcal H(q)$ doesn't depend on $\phi$, so this is just maximizing $E_q[\log f(x\mid\phi)]=Q(\phi\mid\phi^{(p)})$. That is the M-step. EM is coordinate ascent on $F$.

And now the monotonicity is visible without any algebra. After the E-step the bound is tight: $F(q^{(p)},\phi^{(p)})=L(\phi^{(p)})$. The M-step raises $F$: $F(q^{(p)},\phi^{(p+1)})\ge F(q^{(p)},\phi^{(p)})$. And $F$ is always below $L$: $L(\phi^{(p+1)})\ge F(q^{(p)},\phi^{(p+1)})$. Chain them: $L(\phi^{(p+1)})\ge F(q^{(p)},\phi^{(p+1)})\ge F(q^{(p)},\phi^{(p)})=L(\phi^{(p)})$. The likelihood climbs because each E-step re-anchors a lower bound *to* the current likelihood value, and the M-step then climbs that bound — and you can never climb the bound past the thing it lower-bounds. This is why filling in by the *posterior* specifically, and not by some point guess or hard assignment, is the non-negotiable choice: only the posterior makes $\mathrm{KL}=0$, only then is the bound tight, only then does an increase in the bound certify an increase in the real likelihood. A hard label or a MAP plug-in leaves $\mathrm{KL}>0$, the bound stays slack, and the guarantee evaporates. That's the content hiding inside "replace the missing data by its conditional expectation": it's right, but for the sharper reason that the posterior is the unique $q$ closing the gap, not because expectations are a natural thing to plug in.

Now let me make this concrete on the mixture, both to check it produces a real closed-form algorithm and to see *why* the awful log-of-sum dissolves. The complete data is $x=(y,z)$ where for each point $y_i$, $z_i$ is the one-hot indicator vector telling me which of the $R$ components generated it. Complete-data log-likelihood:

$$\log f(x\mid\phi)=\sum_{i=1}^n\sum_{k=1}^R z_{ik}\Big[\log \pi_k+\log p_k(y_i\mid\theta_k)\Big].$$

The thing to notice is that this is *linear* in the hidden indicators $z_{ik}$. That linearity is what makes the E-step trivial: taking $E[\cdot\mid y,\phi^{(p)}]$ only requires $E[z_{ik}\mid y_i,\phi^{(p)}]$, and since $z_{ik}\in\{0,1\}$, its expectation is just its posterior probability of being one,

$$r_{ik}\;=\;E[z_{ik}\mid y_i,\phi^{(p)}]\;=\;\Pr(z_{ik}=1\mid y_i,\phi^{(p)})\;=\;\frac{\pi_k^{(p)}\,p_k\!\big(y_i\mid\theta_k^{(p)}\big)}{\sum_{j=1}^R \pi_j^{(p)}\,p_j\!\big(y_i\mid\theta_j^{(p)}\big)}.$$

These are the responsibilities — exactly the quantities Hasselblad and Day and Wolfe were computing, now identified as the posterior $E[z\mid y,\phi^{(p)}]$, the E-step. So

$$Q(\phi\mid\phi^{(p)})=\sum_{i=1}^n\sum_{k=1}^R r_{ik}\Big[\log\pi_k+\log p_k(y_i\mid\theta_k)\Big].$$

Look at what happened to the logarithm. In $L$ I had $\sum_i\log\sum_k \pi_k p_k(y_i)$ — the log wrapped *around* the sum over $k$, gluing all components together. In $Q$ the log sits *inside*, with the responsibilities $r_{ik}$ as fixed weights out front. The components have decoupled. The M-step is now $R$ separate, ordinary weighted maximum-likelihood fits.

For Gaussian components $p_k(y\mid\theta_k)=\mathcal N(y\mid\mu_k,\Sigma_k)$, maximize $Q$ in closed form. Let $N_k=\sum_i r_{ik}$, the effective number of points soft-assigned to component $k$. The mixing weights are constrained by $\sum_k\pi_k=1$, so I isolate the part of $Q$ that depends on them:

$$\sum_{i,k}r_{ik}\log\pi_k=\sum_k N_k\log\pi_k.$$

With the Lagrangian $\mathcal L_\pi=\sum_k N_k\log\pi_k+\lambda(\sum_k\pi_k-1)$, the derivative is $N_k/\pi_k+\lambda=0$, hence $\pi_k=-N_k/\lambda$. Summing over $k$ gives $1=-\sum_k N_k/\lambda=-n/\lambda$, so $\lambda=-n$ and $\pi_k=N_k/n$, the average responsibility. Differentiating the weighted Gaussian terms gives the responsibility-weighted mean and covariance:

$$\pi_k^{(p+1)}=\frac{N_k}{n},\qquad \mu_k^{(p+1)}=\frac{\sum_i r_{ik}\,y_i}{N_k},\qquad \Sigma_k^{(p+1)}=\frac{\sum_i r_{ik}\,(y_i-\mu_k^{(p+1)})(y_i-\mu_k^{(p+1)})^{\mathsf T}}{N_k}.$$

These are exactly the formulas you'd write down for fitting a single Gaussian to a dataset — but with each point weighted by how much it belongs to component $k$. The hard mixture maximization has become a sequence of trivial weighted single-Gaussian fits, and by the theorem above, $\sum_i\log\sum_k\pi_k\mathcal N(y_i\mid\mu_k,\Sigma_k)$ is guaranteed to not decrease across each E-then-M sweep. The general reason this M-step coincides with the ordinary complete-data estimator: for an exponential-family complete-data model, $Q(\phi\mid\phi^{(p)})=-\log a(\phi)+\phi\, t^{(p)\mathsf T}+\text{const}$ with $t^{(p)}=E[t(x)\mid y,\phi^{(p)}]$, whose maximizer solves $E(t\mid\phi)=t^{(p)}$ — the very same likelihood equation as for fully observed data, only with the observed sufficient statistic replaced by its conditional expectation. So the M-step reuses, verbatim, whatever complete-data fitting machinery already exists; that reuse is the whole practical appeal, and the reason this one loop subsumes Hartley's multinomial counts, Baum's Markov chains, the censored-data and variance-component and factor-analysis cases all at once.

So the causal chain, start to finish: the observed-data likelihood is hard because marginalizing the hidden coordinates buries them inside a log; the complete-data likelihood would be easy; for *any* distribution $q$ over the hidden coordinates, $F(q,\phi)=E_q[\log f(x\mid\phi)]+\mathcal H(q)$ is a lower bound on $\log g(y\mid\phi)$, tight exactly when $q$ is the posterior $k(\cdot\mid y,\phi)$ (the gap is $\mathrm{KL}(q\|k)\ge 0$, by Jensen); the E-step maximizes the bound over $q$ by setting it to the posterior — which fills in the hidden coordinates by their conditional expectation and snaps the bound onto the current likelihood — and the M-step maximizes the bound over $\phi$ by maximizing the expected complete-data log-likelihood $Q(\phi\mid\phi')$, which for exponential families is the ordinary complete-data fit with sufficient statistics replaced by conditional expectations; because each E-step re-tightens the bound and each M-step raises it, and a tight bound that goes up drags the true likelihood up with it, the observed-data likelihood increases monotonically and fixed points are stationary points of that likelihood. For a Gaussian mixture this is: compute responsibilities, then re-fit each component as a responsibility-weighted Gaussian and the weights as average responsibilities.

```python
import numpy as np

def logsumexp(a, axis=1, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)

def estimate_log_gaussian_prob_full(X, means, covariances):
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    log_prob = np.empty((n_samples, n_components), dtype=X.dtype)
    for k in range(n_components):
        chol = np.linalg.cholesky(covariances[k])
        diff = X - means[k]
        whitened = np.linalg.solve(chol, diff.T).T
        mahalanobis = np.sum(whitened * whitened, axis=1)
        log_det = np.sum(np.log(np.diag(chol)))
        log_prob[:, k] = -0.5 * (
            n_features * np.log(2.0 * np.pi) + mahalanobis
        ) - log_det
    return log_prob

def estimate_log_prob_resp(X, weights, means, covariances):
    # E-step in log space: log r_ik = log pi_k + log N_ik - log sum_j pi_j N_ij.
    weighted_log_prob = estimate_log_gaussian_prob_full(X, means, covariances)
    weighted_log_prob = weighted_log_prob + np.log(weights)
    log_prob_norm = logsumexp(weighted_log_prob, axis=1, keepdims=True)
    log_resp = weighted_log_prob - log_prob_norm
    return np.squeeze(log_prob_norm, axis=1), log_resp

def estimate_gaussian_parameters_full(X, resp, reg_covar=1e-6):
    # M-step: the weighted complete-data MLE, with the small diagonal stabilizer
    # used in practical Gaussian-mixture code.
    n_components = resp.shape[1]
    n_features = X.shape[1]
    nk = resp.sum(axis=0) + 10.0 * np.finfo(resp.dtype).eps
    means = (resp.T @ X) / nk[:, None]
    covariances = np.empty((n_components, n_features, n_features), dtype=X.dtype)
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = ((resp[:, k] * diff.T) @ diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
    return nk, means, covariances

def m_step_full(X, log_resp, reg_covar=1e-6):
    resp = np.exp(log_resp)
    weights, means, covariances = estimate_gaussian_parameters_full(
        X, resp, reg_covar=reg_covar
    )
    weights = weights / weights.sum()
    return weights, means, covariances

def em_gmm_full(X, weights, means, covariances, max_iter=100, tol=1e-3, reg_covar=1e-6):
    lower_bounds = []
    lower_bound = -np.inf
    for _ in range(max_iter):
        previous = lower_bound
        log_prob_norm, log_resp = estimate_log_prob_resp(X, weights, means, covariances)
        weights, means, covariances = m_step_full(X, log_resp, reg_covar=reg_covar)
        lower_bound = float(np.mean(log_prob_norm))
        lower_bounds.append(lower_bound)
        if np.isfinite(previous) and abs(lower_bound - previous) < tol:
            break
    return weights, means, covariances, lower_bounds
```
