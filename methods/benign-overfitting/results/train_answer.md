The problem is to understand why a predictor that interpolates noisy training labels can still generalize well. In classical low-dimensional statistics, forcing zero training error on noisy data is a mistake: the fitted model memorizes the noise and predicts poorly. Yet modern overparameterized models routinely reach zero training error while maintaining useful test accuracy. The naive explanations fall short. Saying the model is "too big" is not enough, because some large models interpolate badly. Saying "double descent" occurs is only a description of the empirical curve; it does not say which data distributions make the second descent possible. Ridge regression is a controlled baseline, but its zero-regularization limit is itself an exact interpolant, so the question is really about that limit. The missing ingredient is a precise way to measure whether the covariate geometry gives the interpolant a safe place to hide the noise.

The key observation is that prediction risk is not Euclidean parameter error; it is covariance-weighted parameter error. An error in a high-variance covariate direction is expensive, while an error in a low-variance direction is almost invisible. If the covariate distribution has a broad tail of weak directions, the minimum-norm interpolant can absorb the training noise into those directions without paying much in future prediction error. The dividing line is therefore spectral, not merely dimensional.

The method is Benign Overfitting, analyzed through the minimum-norm least-squares interpolant in overparameterized linear regression. Given a design matrix X with full row rank and more columns than rows, the estimator is the pseudoinverse solution that has the smallest Euclidean norm among all exact fits. This is the zero-regularization limit of ridge regression. The fitted parameter splits naturally into a signal component and a noise component. The signal component projects the true parameter onto the row space of the sample, and the noise component pushes the training residuals back into parameter space through the pseudoinverse. The excess risk decomposes into a covariance-weighted bias term and a covariance-weighted variance term.

To make the geometry explicit, diagonalize the covariate covariance. The contribution of each eigen-direction to the noise cost is controlled by the eigenvalue and by how the sample Gram matrix concentrates. The crucial spectral quantities are two effective ranks. The first effective rank asks whether the low-variance tail is wide enough, relative to its largest eigenvalue, to make the tail Gram matrix behave like a scalar multiple of the identity. The second effective rank asks whether the tail is balanced enough to dilute noise evenly rather than concentrating it in a few directions. When the tail starts early, is wide enough, and is balanced enough, the cost of exact interpolation becomes small. When it does not, interpolation is harmful. This gives matching upper and lower bounds on the excess risk in terms of these effective ranks, rather than a loose capacity argument.

The practical message is that overparameterization is benign when the covariance spectrum provides many cheap directions before the sample runs out of dimensions to control them. A large but flat tail of tiny eigenvalues is friendly; a spectrum that is square or dominated by a few large eigenvalues is not. This distinction explains why the same interpolation threshold can look catastrophic in one setting and harmless in another.

Making this exact means separating the two costs cleanly and giving each a name. Let $P = X^\top(XX^\top)^{-1}X$ be the projection onto the row space the sample actually observes, and set

$$B = (I-P)\Sigma(I-P), \qquad C = (XX^\top)^{-1}X\Sigma X^\top(XX^\top)^{-1}.$$

The excess risk of the minimum-norm interpolant then satisfies, up to universal constants,

$$R(\hat\theta) \le 2\,{\theta^*}^\top B\,\theta^* + 2\,\epsilon^\top C\,\epsilon, \qquad \mathbb{E}_\epsilon R(\hat\theta) \ge {\theta^*}^\top B\,\theta^* + \sigma^2\,\mathrm{tr}(C).$$

$B$ prices the signal directions the sample never observed; $\mathrm{tr}(C)$ prices exactly fitting the training noise, and it is this single scalar that decides whether interpolation is safe.

For covariance eigenvalues $\lambda_1 \ge \lambda_2 \ge \cdots$, define the two effective ranks

$$r_k(\Sigma) = \frac{\sum_{i>k}\lambda_i}{\lambda_{k+1}}, \qquad R_k(\Sigma) = \frac{\big(\sum_{i>k}\lambda_i\big)^2}{\sum_{i>k}\lambda_i^2},$$

and let $k^* = \min\{k \ge 0 : r_k(\Sigma) \ge bn\}$ for a fixed constant $b$, read as $\infty$ if no such $k$ exists. $r_k$ certifies that the spectral tail past index $k$ is wide enough, relative to its own leading eigenvalue, that the tail's sample Gram matrix behaves like a scalar multiple of the identity on the $n$-dimensional sample; $R_k$ certifies that the same tail is balanced enough that no single eigen-direction absorbs a disproportionate share of the noise.

For constants that depend only on the subgaussian norm of the design, and assuming $\log(1/\delta) < n/c$, these two numbers settle the question completely. If $k^* \ge n/c_1$ — including $k^* = \infty$ — the reservoir never gets going before the sample runs out of room to control it, and interpolation is harmful no matter how large $p$ is:

$$\mathbb{E}\,R(\hat\theta) \ge \sigma^2/c.$$

If instead $k^* < n/c_1$, then with probability at least $1-\delta$,

$$R(\hat\theta) \le c\,\|\theta^*\|^2\|\Sigma\| \max\!\Big\{\sqrt{\tfrac{r_0(\Sigma)}{n}},\ \tfrac{r_0(\Sigma)}{n},\ \sqrt{\tfrac{\log(1/\delta)}{n}}\Big\} + c\log(1/\delta)\,\sigma_y^2\Big(\tfrac{k^*}{n} + \tfrac{n}{R_{k^*}(\Sigma)}\Big),$$

and the noise term in that bound is not an artifact of the proof — it is necessary up to constants,

$$\mathbb{E}\,R(\hat\theta) \ge (\sigma^2/c)\Big(\tfrac{k^*}{n} + \tfrac{n}{R_{k^*}(\Sigma)}\Big).$$

A covariance sequence is therefore benign exactly when three limits hold together: $r_0(\Sigma_n)/n \to 0$ controls the signal scale, $k_n^*/n \to 0$ says the reservoir starts early, and $n/R_{k_n^*}(\Sigma_n) \to 0$ says the reservoir is balanced. The signal-side condition is sharp only up to a logarithmic gap — a matching lower bound holds whenever $r_0(\Sigma)/(n\log(1+r_0(\Sigma)))$ is large, for some $\theta^*$ of the same norm — but the two tail conditions above are matched exactly, upper bound against lower bound, by the same pair of effective ranks.

This is why the same interpolation threshold can be catastrophic in one model and invisible in another: what buys safety is not the count of extra directions but whether the spectrum supplies a reservoir that is both early and flat. The margin is thin in fixed infinite dimension — for $\lambda_k = k^{-\alpha}\log^{-\beta}(k+1)$ the benign regime is exactly $\alpha = 1$, $\beta > 1$ — and wide in growing finite dimension with a small isotropic floor, where a large flat tail drives $R_{k^*}$ up almost for free even under a rapidly decaying signal spectrum. Either way, the two effective ranks $r_{k^*}$ and $R_{k^*}$ are the only numbers that matter.
