I keep coming back to the same wall, so let me sit with it properly. I have a state that evolves in discrete time, $x_k = f_{k-1}(x_{k-1}, w_{k-1})$, and I never see it — I see noisy measurements $y_k = h_k(x_k, v_k)$. The noises $w$ and $v$ have known densities, not necessarily Gaussian, and $f, h$ can be anything. I want the posterior of the current state given everything I've seen, $p(x_k \mid D_k)$ with $D_k = \{y_1,\dots,y_k\}$, and I want it recursively, one cheap step per measurement.

The exact recursion is not the problem; I can write it down in two lines and it's model-free. If I have $p(x_{k-1}\mid D_{k-1})$, I push it forward through the dynamics,
$$p(x_k\mid D_{k-1}) = \int p(x_k\mid x_{k-1})\, p(x_{k-1}\mid D_{k-1})\, dx_{k-1},$$
and then I fold in the new measurement by Bayes,
$$p(x_k\mid D_k) = \frac{p(y_k\mid x_k)\,p(x_k\mid D_{k-1})}{\int p(y_k\mid x_k)\,p(x_k\mid D_{k-1})\,dx_k}.$$
Predict, then update. That's the whole thing. And those two integrals are precisely where it dies. The transition density itself, $p(x_k\mid x_{k-1})$, is already an integral: since $x_k = f_{k-1}(x_{k-1},w_{k-1})$ is a deterministic map once $x_{k-1}$ and the noise are fixed, $p(x_k\mid x_{k-1}) = \int \delta\big(x_k - f_{k-1}(x_{k-1},w_{k-1})\big)\,p(w_{k-1})\,dw_{k-1}$. Stack that inside the prediction integral, then the normalising integral in the update, and for general $f, h$ none of it has a closed form.

There is exactly one case where it closes, and I should be honest about why it closes, because that tells me what I'm fighting. If $f, h$ are linear and the noise is Gaussian — $x_k = Ax_{k-1}+w$, $y_k = Hx_k+v$, $w\sim N(0,Q)$, $v\sim N(0,R)$ — then a Gaussian prior stays Gaussian through both steps, so the entire posterior is pinned down by a mean and a covariance, and the recursion becomes the Kalman filter: predict $\hat x_k^- = A\hat x_{k-1}$, $P_k^- = AP_{k-1}A^\top + Q$; correct $K_k = P_k^- H^\top(HP_k^-H^\top+R)^{-1}$, $\hat x_k = \hat x_k^- + K_k(y_k - H\hat x_k^-)$, $P_k = (I-K_kH)P_k^-$. Beautiful, finite, exact. But it's exact *only because the Gaussian family is closed under linear maps and Bayes with Gaussian likelihood*. The representation — two moments — is the whole trick, and it's also the whole limitation. The instant $f$ or $h$ bends, or the noise has tails, or the posterior wants two bumps, a Gaussian can't hold the truth, and propagating a mean and covariance is propagating a lie.

I know how people patch this and I know each patch fails. The extended Kalman filter linearises $f, h$ about the predicted state and runs Kalman on the linearisation — but the posterior is still forced to be one Gaussian, and I've watched it diverge: in bearings-only tracking, where I only observe an angle and not range, the true posterior over position is a curved ridge, sometimes two-lobed, and a single Gaussian sits off the ridge and grows over-confident until it loses the target. The Gaussian sum filter stacks a mixture of Gaussians and runs a bank of EKFs, but the number of components multiplies every step and has to be pruned by hand, and each component is still a local linear-Gaussian fib. The grid filters give up on parametric forms entirely and just evaluate the density on a fixed lattice of points in state space — honest, but the lattice doesn't move with the mass, so I either waste most points in near-zero-probability regions or starve the region that matters, and the point count explodes with dimension. A four-dimensional state with even ten nodes per axis is already ten thousand points, every one needing a full integral every step. None of these is the right object.

So let me throw out the function-on-state-space idea altogether and ask what the cheapest faithful representation of a distribution actually is. Here's the thing I keep underusing: a density and a sample from it are two views of the same object. The density makes the sample; a sample of size $N$ recreates the density — histogram, kernel estimate, empirical CDF — and as $N\to\infty$ they carry identical information. And crucially, if I want any summary — the mean, the covariance, a percentile, a highest-posterior-density region, the probability the state lies in some set — I read it straight off the sample: the mean is the sample mean, the probability of a region is the fraction of points in it. The Monte Carlo error decays like $1/\sqrt N$ *no matter the dimension* of the space, which is exactly the property the grid lacks. So what if the thing I propagate isn't a function over $\mathbb{R}^n$ at all, but a cloud of $N$ points — call them particles — $\{x_k(i)\}_{i=1}^N$, whose empirical distribution *is* my approximation to $p(x_k\mid D_k)$? No Gaussian assumption, no grid; the points naturally pile up where the density is high, because that's where they were drawn. That feels right. The question is whether I can carry that cloud through the predict step and the update step and keep it being a sample from the right thing.

Prediction first, because I suspect it's free. Suppose right now $\{x_{k-1}(i)\}$ is a sample from $p(x_{k-1}\mid D_{k-1})$. I want a sample from the predictive $p(x_k\mid D_{k-1})$. Look at what that integral *is*: it's the law of $x_k$ when $x_{k-1}\sim p(x_{k-1}\mid D_{k-1})$ and then $x_k = f_{k-1}(x_{k-1},w_{k-1})$ with $w_{k-1}\sim p(w_{k-1})$, independently. So I don't need to do the integral analytically — I just *realise* it. For each particle, draw a fresh noise $w_{k-1}(i)\sim p(w_{k-1})$ and push it through the map: $x_k^*(i) = f_{k-1}(x_{k-1}(i), w_{k-1}(i))$. Each $x_k^*(i)$ is then a draw from the predictive density. The delta-function form of the transition is precisely what makes this exact — sampling $(x_{k-1}, w_{k-1})$ jointly and applying the deterministic map is sampling from the marginal of $x_k$. Prediction costs one noise draw and one function evaluation per particle, and it never touches a density evaluation. So the dynamics propagate the cloud for free, for *any* $f$ and any noise I can sample. Good.

Now the update, and this is where I expect to fight. I have a predictive sample $\{x_k^*(i)\}$ from $p(x_k\mid D_{k-1})$, the new measurement $y_k$ arrives, and Bayes says the posterior is the predictive *times the likelihood*, renormalised: $p(x_k\mid D_k) \propto p(y_k\mid x_k)\, p(x_k\mid D_{k-1})$. Multiplying a sample by a function is exactly the situation importance sampling handles. If I attach to each particle a weight proportional to the likelihood, $\tilde\omega(i) = p(y_k\mid x_k^*(i))$, then the *weighted* cloud $\{x_k^*(i), W(i)\}$, with $W(i) = \tilde\omega(i)/\sum_j \tilde\omega(j)$, represents the posterior: for any function $\varphi$, $\int \varphi\, p(x_k\mid D_k)\,dx_k \approx \sum_i W(i)\,\varphi(x_k^*(i))$. And note I never had to compute the normaliser $\int p(y_k\mid x_k)p(x_k\mid D_{k-1})\,dx_k$ — the weights are only needed up to proportionality, the division by $\sum_j\tilde\omega(j)$ does the normalising for me. That single fact, that I escape the intractable normalising integral, is most of why this is going to work.

But wait. If I just attach weights, the next prediction step is in trouble. Prediction needs an *unweighted* sample to push through $f$ — the clean argument I just made assumed $x_{k-1}(i)$ were draws from the posterior, not weighted draws. I could carry the weights forward, sure, push the weighted particles through $f$ and multiply by the next likelihood, but let me think about what happens to those weights over time before I commit to that, because I have a bad feeling about it.

The cleaner alternative is sitting in a result I half-remember from Smith and Gelfand — "Bayesian statistics without tears." They ask exactly my question in its static form: I have a sample $\{\theta_i\}$ from an easy density $g$, and I want a sample from a target density $h$ known only up to a positive unnormalised function $f$, so $h(\theta)=f(\theta)/\int f$. Can I make the second sample from the first without computing $\int f$? The weighted bootstrap says yes: form the importance ratios $c_i=f(\theta_i)/g(\theta_i)$, normalise $q_i=c_i/\sum_j c_j$, and draw $\theta^*$ from the discrete distribution that puts mass $q_i$ on $\theta_i$. Let me actually check the convergence, because the whole update is going to ride on it. Take $\theta$ univariate and ask for the CDF of the resampled $\theta^*$:
$$\Pr(\theta^* \le a)=\frac{\frac1N\sum_i \frac{f(\theta_i)}{g(\theta_i)}\,\mathbf 1_{(-\infty,a]}(\theta_i)}{\frac1N\sum_i \frac{f(\theta_i)}{g(\theta_i)}}.$$
The $\theta_i$ are from $g$, so the law of large numbers sends the numerator to $\int_{-\infty}^a \frac{f(\theta)}{g(\theta)}g(\theta)\,d\theta=\int_{-\infty}^a f(\theta)\,d\theta$ and the denominator to $\int f(\theta)\,d\theta$. Therefore $\Pr(\theta^*\le a)\to \int_{-\infty}^a h(\theta)\,d\theta$. So $\theta^*$ does converge in distribution to $h$. If the target is a tilted version of the sampling density, $h\propto Lg$, then $f=Lg$ and the ratio collapses to $c_i=L(\theta_i)$. The resampling turns the weighted sample back into an *unweighted* one. And it needs $h$ only up to proportionality — perfect for Bayes.

Now I map it onto my filter and it clicks. Set the sampling density $g$ to the predictive prior $p(x_k\mid D_{k-1})$, which is exactly what my predicted particles $\{x_k^*(i)\}$ are a sample from. Set the tilt $L$ to the likelihood $p(y_k\mid x_k)$, which I can evaluate. Then $h\propto Lg = p(y_k\mid x_k)p(x_k\mid D_{k-1})$, which is the posterior $p(x_k\mid D_k)$. So the Bayes update *is* a weighted bootstrap: compute $q_i = p(y_k\mid x_k^*(i))/\sum_j p(y_k\mid x_k^*(j))$, resample $N$ times with replacement from the predicted particles using those probabilities, and the resampled cloud $\{x_k(i)\}$ is an (unweighted) sample from the posterior — ready to be pushed through $f$ again next step. Bayes' rule, implemented as a resampling. The likelihood acts as a survival probability: particles that explain the measurement get copied, particles that don't get culled.

That gives me the whole loop, and it's almost embarrassingly simple. Initialise by drawing $N$ particles from the known prior $p(x_1)$; those particles feed directly into the first update with $y_1$. After that, repeat: predict — push every particle through the dynamics with a fresh noise draw; update — weight by the likelihood of $y_k$, normalise, and resample. Read estimates off the cloud at any time. The only things I need from the model are: I can sample $p(x_1)$, I can evaluate the likelihood $p(y_k\mid x_k)$ as a known function, and I can sample the process noise $p(w_k)$. No linearisation, no Gaussianity, no grid, nothing about $f$ or $h$ beyond being able to apply them. And it's trivially parallel — every particle is independent through predict and weight.

Before I trust it, I owe myself the thing I had a bad feeling about: *why resample at all?* It cost me an idea — I threw away the clean weighted representation and replaced it with a noisier resampled one. Resampling literally injects extra randomness: instead of $N$ distinct weighted points I now have $N$ points with duplicates. Surely keeping the weights is better? Let me follow the no-resampling road and see where it goes, because if it's fine I should take it.

No resampling means I carry weights across all time. This is sequential importance sampling, and I should write the weight recursion out exactly rather than wave at it. Think of the whole trajectory $x_{1:k}$ and a target proportional to $\gamma_k(x_{1:k}) = p(x_{1:k}, y_{1:k})$, with normaliser $Z_k = p(y_{1:k})$, so the posterior is $\pi_k(x_{1:k}) = \gamma_k/Z_k$. I build the proposal sequentially, one coordinate at a time: $q_k(x_{1:k}) = q_1(x_1)\prod_{j=2}^k q_j(x_j\mid x_{1:j-1})$. The unnormalised importance weight is $w_k = \gamma_k/q_k$, and because both $\gamma$ and $q$ factor across time it telescopes:
$$w_k(x_{1:k}) = \frac{\gamma_k(x_{1:k})}{q_k(x_{1:k})} = \frac{\gamma_{k-1}(x_{1:k-1})}{q_{k-1}(x_{1:k-1})}\cdot\frac{\gamma_k(x_{1:k})}{\gamma_{k-1}(x_{1:k-1})\,q_k(x_k\mid x_{1:k-1})} = w_{k-1}(x_{1:k-1})\cdot \alpha_k,$$
with the incremental weight $\alpha_k = \dfrac{\gamma_k(x_{1:k})}{\gamma_{k-1}(x_{1:k-1})\,q_k(x_k\mid x_{1:k-1})}$. For filtering, $\gamma_k/\gamma_{k-1} = f(x_k\mid x_{k-1})\,g(y_k\mid x_k)$, so $\alpha_k = \dfrac{g(y_k\mid x_k)\,f(x_k\mid x_{k-1})}{q_k(x_k\mid x_{1:k-1})}$. So the weight at time $k$ is a *product* of $k$ incremental factors, $w_k = w_1\prod_{j=2}^k \alpha_j$. There's the smell. A product of $k$ random factors, each with some spread, will have a variance that compounds. Let me make that quantitative on the simplest possible toy, because "compounds" isn't an argument.

Take the cleanest case: the target factorises, $\pi_k(x_{1:k}) = \prod_{j=1}^k N(x_j;0,1)$, so $\gamma_k = \prod_j e^{-x_j^2/2}$ and $Z_k = (2\pi)^{k/2}$. Propose independently with $q_k = \prod_j N(x_j;0,\sigma^2)$. Compute the relative variance of the importance-sampling estimate of the normaliser $\hat Z_k = \frac1N\sum_i w_k(x_{1:k}^i)$. For one coordinate,
$$\frac{1}{Z_1^2}\int \frac{\gamma_1(x)^2}{q_1(x)}\,dx
=\frac{1}{2\pi}\int \frac{e^{-x^2}}{(2\pi\sigma^2)^{-1/2}e^{-x^2/(2\sigma^2)}}\,dx
=\frac{\sigma^2}{\sqrt{2\sigma^2-1}},$$
which is finite only when $\sigma^2>\tfrac12$. Independence raises that one-step factor to the $k$th power, so the relative variance of $\hat Z_k$ is
$$\frac{\mathbb{V}[\hat Z_k]}{Z_k^2} = \frac1N\left[\left(\frac{\sigma^4}{2\sigma^2-1}\right)^{k/2} - 1\right],$$
and on the finite side, $\sigma^2>\tfrac12$, the base is
$$\frac{\sigma^4}{2\sigma^2-1}=1+\frac{(\sigma^2-1)^2}{2\sigma^2-1},$$
which is greater than $1$ for every $\sigma^2\ne1$. So the bracket grows like (something $>1$)$^{k/2}$ — the variance blows up *exponentially* in the number of steps for any imperfect Gaussian scale in this toy. Put numbers on it: $\sigma^2 = 1.2$, a genuinely good proposal, gives $\frac{\sigma^4}{2\sigma^2-1}\approx 1.103$, and at $k=1000$ that's $1.103^{500}\approx 1.9\times 10^{21}$. To pull the relative variance down to $0.01$ I'd need $N\approx 2\times 10^{23}$ particles. That is not a tuning problem, it's a wall.

I should say concretely what this does to the cloud, because the variance number is abstract. As the weight product compounds, almost all of the normalised weight must pile onto a tiny handful of particles and the rest go to essentially zero — that is what an exponentially growing weight variance *is*, a few enormous weights among $N-$few negligible ones. Picture the weight histogram for the prior-proposal filter (weight $\propto$ likelihood) on a stochastic-volatility model without resampling: it starts spread across the $N$ particles, but the same compounding that drives the variance must concentrate it, so within a handful of steps only tens of particles carry meaningful weight and soon a *single* particle holds almost all of it. The cloud has collapsed — I'm paying for $N$ particles but representing the posterior with one. The filtered mean then drifts off the truth and the reported spread shrinks to a sliver because the surviving particle thinks it's certain. So keeping the weights is not fine. Sequential importance sampling degenerates, and it degenerates fast.

I want a single number to watch for this so I'm not eyeballing histograms. The right one falls out of asking: my $N$ weighted particles are worth how many equally-weighted ones? If the weights are all equal, $W(i) = 1/N$, they're worth all $N$. If one weight is $1$ and the rest $0$, they're worth $1$. The quantity that interpolates is the inverse sum of squared normalised weights,
$$\text{ESS} = \Big(\sum_{i=1}^N W(i)^2\Big)^{-1},$$
the effective sample size: it equals $N$ when the weights are uniform and $1$ when one particle dominates, and it tracks how many "real" samples the weighted cloud is equivalent to. Degeneracy is exactly ESS crashing toward $1$. Now I have a diagnostic *and* a trigger.

The disease is the unbounded product $w_k = \prod_j \alpha_j$ — the weight history accumulates without limit, and nothing in sequential importance sampling ever truncates it. So truncate it: resampling *cuts the product*. When I resample, I throw away the weights: every surviving particle gets weight reset to $1/N$, and the low-weight particles are deleted (with high probability) while the high-weight ones are duplicated. The duplication is doing real work — it spends my $N$ slots on the regions the data actually support, instead of dragging along particles that will never matter again. After resampling the next step starts from $w \equiv 1/N$, so its weight is just the *next* incremental factor $\alpha_{k+1}$, not the whole compounded history. The product can never run away because I keep resetting it.

I should check that resampling doesn't bias me and quantify what it buys, because it isn't free. The standard scheme draws offspring counts $N_k^{1:N}\sim\text{Multinomial}(N, W_k^{1:N})$, giving each surviving particle weight $1/N$. Then $\mathbb{E}[N_k^i \mid W] = N\,W_k^i$, so the resampled empirical measure is an *unbiased* estimate of the weighted one — on average I've changed nothing, I've only re-expressed it. What it costs is immediate variance: replacing $N$ distinct weighted points by $N$ points with duplicates adds Monte Carlo noise *at this step*. So resampling is a trade — a little extra variance now in exchange for not carrying a runaway product into the future. To see the payoff cleanly, look at what it does to that exponential. The asymptotic variance of the resampling scheme, instead of carrying the single global proposal $q_k(x_{1:k})$ through, replaces it at each step $k$ by the *resampled* proposal $\pi_{k-1}(x_{1:k-1})q_k(x_k\mid x_{1:k-1})$ — the system gets "reset" to the target at every resampling. On the same toy, that turns the variance from a product into a sum:
$$\frac{\mathbb{V}_{\text{SMC}}[\hat Z_k]}{Z_k^2}\approx \frac{k}{N}\left[\left(\frac{\sigma^4}{2\sigma^2-1}\right)^{1/2} - 1\right],$$
linear in $k$ instead of exponential. With $\sigma^2 = 1.2$, $k = 1000$: the resampling version needs $N\approx 10^4$ where sequential importance sampling needed $\approx 2\times 10^{23}$ — nineteen orders of magnitude. And this isn't special to the factorised toy; the deeper reason it works in real filtering is that the state-space model *forgets*: under mild mixing, what happened at time $j$ is almost irrelevant to the marginal at time $k$ once $k - j$ is large, so the per-step errors that resampling resets don't accumulate explosively, and the variance of the filtering marginal grows on the order of $\frac{C\cdot k}{N}$. Resampling is what lets me exploit that forgetting.

There is one honest cost I should name so I'm not surprised by it. Resampling reduces the number of *distinct* values for the early states: every time I resample, some particles representing $x_1$ get deleted, so after enough steps the whole cloud traces back to a handful of time-$1$ ancestors. That means the joint, path-space distribution $p(x_{1:k}\mid D_k)$ degenerates — fine for *filtering*, where I only care about the current marginal $p(x_k\mid D_k)$, which keeps being re-populated by fresh noise at each prediction, but a real problem if I ever wanted to smooth. For the filtering task in front of me, resampling is purely a win, and the path degeneracy is a price I'm happy to pay. So I won't resample blindly every step — I'll watch the ESS and resample only when it drops below a threshold, say $N/2$, so I don't pay the immediate variance when the weights are already healthy.

Now, my one design freedom is the proposal $q_k(x_k\mid x_{1:k-1})$ — where do I draw the new particle from before weighting? I made the simplest choice without thinking: propose from the dynamics, $q_k = f(x_k\mid x_{k-1})$, i.e. just push through the system model. Let me see what that does to the incremental weight. With $q_k = f$, the $f$ cancels and $\alpha_k = \dfrac{g(y_k\mid x_k)\,f(x_k\mid x_{k-1})}{f(x_k\mid x_{k-1})} = g(y_k\mid x_k)$ — the weight collapses to *just the likelihood*. That's why the update was so clean: proposing from the prior makes the importance weight equal to the likelihood, which is exactly the weighted-bootstrap mass I wrote down. The whole method needs only the ability to sample $f$ and evaluate $g$. That's the bootstrap filter.

Is the prior the *best* proposal, though? Let me find the variance-minimising one to know what I'm giving up. The incremental weight $\alpha_k = g(y_k\mid x_k)f(x_k\mid x_{k-1})/q_k(x_k\mid x_{k-1})$ has zero conditional variance — it becomes independent of the new $x_k$ — exactly when $q_k(x_k\mid x_{k-1}) \propto g(y_k\mid x_k)f(x_k\mid x_{k-1})$, i.e. when I propose from $p(x_k\mid y_k, x_{k-1}) = \dfrac{g(y_k\mid x_k)f(x_k\mid x_{k-1})}{p(y_k\mid x_{k-1})}$, the posterior of the new state given the *new measurement and the old state*. That optimal proposal folds the measurement into the proposal so particles are born already explaining $y_k$, and the incremental weight reduces to $p(y_k\mid x_{k-1})$, independent of $x_k$. The trouble is that proposing from it requires sampling that conditional and computing its normaliser $p(y_k\mid x_{k-1})$, which for a general nonlinear $h$ is itself intractable — the very integral I started out unable to do. So in general I can't have it. The prior proposal is the workhorse fallback: it ignores $y_k$ when proposing, which means if the likelihood is sharp and lands far out in the tail of the predictive prior, only a few particles fall under it and the weights are immediately uneven — the overlap between prior and likelihood is small, ESS drops, and I resample harder. That's the precise weakness of the bootstrap choice, and it tells me exactly when I'll be in trouble: when a new measurement is very informative relative to the dynamics, or when the prior and likelihood barely overlap, as at the start of a track or when a target flies past the observer.

That last failure mode deserves one more fix, because I can see it biting even with resampling. Suppose the system noise is tiny and the likelihood is sharp. After an update, the resampled cloud is full of *exact duplicates* — the few high-weight particles copied many times — and if the dynamics barely perturb them, those duplicates stay on top of each other forever. The cloud has the right number of points but only a few distinct locations; it's impoverished. In the extreme of no process noise the whole cloud can collapse onto a single value within a few steps. The cure is to break the ties by hand: after resampling, add a small independent jitter to each particle — roughening. I draw $\epsilon(i)\sim N(0, J_k)$ and set $x_k(i)\leftarrow x_k(i)+\epsilon(i)$. How big should the jitter be? I want it to scale like the local spacing of the particles. If a component of the state spans a range $E$ (its max minus min across the cloud before roughening), and I imagine the $N$ particles laid on an equivalent regular grid in $d$ dimensions, the node spacing goes like $E\,N^{-1/d}$, so I set the jitter standard deviation $\sigma = K\,E\,N^{-1/d}$ with a small tuning constant $K$ (something like $0.2$). The inverse-$d$-th-root of $N$ is the right normalisation — it makes the jitter shrink as I add particles, at the rate the grid tightens — and $K$ is a compromise: too large and I blur the posterior, too small and I leave tight clusters around the duplicates. When the system noise is already healthy I don't even need this; the dynamics roughen the cloud for free. There's a second, optional trick for the small-overlap case — prior editing: before committing, peek one step ahead, and if a particle is so far from the next likelihood that it's certain to get zero weight, reject and redraw it, boosting the number of useful particles where the measurement actually has mass. It's a crude one-step smoothing of the proposal, used only to generate samples, and I'd report estimates from the un-edited cloud.

Let me also be clear about the honest limits, since the justification is asymptotic. The weighted-bootstrap convergence is a statement as $N\to\infty$; for finite $N$ I can't give a clean guarantee, and "how many particles do I need" depends on at least three things — the dimension of the state (I expect $N$ to climb fast with dimension unless the components are nearly independent), the typical overlap between prior and likelihood (small overlap wastes particles), and how many time steps I run. Those aren't bugs to hide; they're the knobs that decide whether the cloud stays alive.

Putting it together, here is the bootstrap particle filter.

```python
import numpy as np

def systematic_resample(weights):
    # one uniform offset, N equally spaced points up the weight CDF;
    # low-variance, O(N), the standard choice for particle filters
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    N = weights.size
    positions = (np.random.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions, side="left")


class BootstrapParticleFilter:
    """p(x_k | y_1:k) as a cloud of N particles.
    Needs only: sample the prior, sample the dynamics, evaluate the likelihood."""

    def __init__(self, model, N, ess_threshold=0.5, roughen_K=0.2):
        self.model = model          # sample_prior, propagate (= f with fresh noise), log_likelihood (= g)
        self.N = N
        self.ess_threshold = ess_threshold * N
        self.K = roughen_K
        self.x = None               # (N, d) particles
        self.w = None               # (N,) normalised weights

    def initialise(self):
        # N draws from the known prior p(x_1); equal weights
        self.x = self.model.sample_prior(self.N)
        self.w = np.full(self.N, 1.0 / self.N)

    def predict(self):
        # Chapman-Kolmogorov, realised: push each particle through f with a fresh noise draw.
        # {x*(i)} is then a sample from the predictive p(x_k | D_{k-1}).
        self.x = self.model.propagate(self.x)

    def update(self, y):
        # Bayes update as a weighted bootstrap (Smith-Gelfand): weight by the likelihood,
        # because proposing from the prior makes the incremental weight = g(y | x).
        logw = np.log(self.w + 1e-300) + self.model.log_likelihood(y, self.x)
        max_logw = np.max(logw)
        if not np.isfinite(max_logw):
            raise ValueError("all particle likelihoods are zero")
        w = np.exp(logw - max_logw)        # stabilise
        total = w.sum()
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("particle weights are not normalisable")
        self.w = w / total                 # normalise (the intractable Bayes denominator drops out)

        ess = 1.0 / np.sum(self.w ** 2)    # effective sample size = (sum W_i^2)^{-1}
        if ess < self.ess_threshold:       # resample only when the cloud is degenerating
            idx = systematic_resample(self.w)
            self.x = self.x[idx]
            self.w = np.full(self.N, 1.0 / self.N)   # cut the weight product: reset to 1/N
            self._roughen()                # break duplicate ties when dynamics noise is too small

    def _roughen(self):
        # jitter ~ N(0, sigma^2) per component, sigma = K * (sample range) * N^{-1/d}:
        # scales like the spacing of an equivalent N-point grid; vanishes as N grows.
        d = self.x.shape[1]
        rng = self.x.max(axis=0) - self.x.min(axis=0)
        sigma = self.K * rng * self.N ** (-1.0 / d)
        self.x = self.x + np.random.randn(*self.x.shape) * sigma

    def estimate(self):
        # any summary read straight off the (weighted) cloud
        return np.average(self.x, axis=0, weights=self.w)


def run_bootstrap_filter(model, observations, N=1000):
    pf = BootstrapParticleFilter(model, N)
    pf.initialise()
    out = []
    for k, y in enumerate(observations):
        if k > 0:
            pf.predict()  # propagate from x_{k-1} to x_k
        pf.update(y)      # reweight by likelihood, resample if degenerate, roughen
        out.append(pf.estimate())
    return out
```

So the causal chain is: the filtering posterior has no closed form off the linear-Gaussian knife-edge, so represent it not as a function or a grid but as a cloud of particles whose empirical measure *is* the density; prediction is then a free push of each particle through the dynamics with fresh noise, an exact Monte Carlo realisation of the Chapman–Kolmogorov integral; the Bayes update is a weighted bootstrap, weighting each predicted particle by the likelihood and resampling when the weights degenerate, which sidesteps the intractable normaliser; but carrying weights forward sequentially makes their variance — a product of per-step factors — explode exponentially in time, collapsing the cloud onto one particle, which the effective sample size $(\sum W_i^2)^{-1}$ makes visible; resampling cures this by cutting that product, deleting low-weight particles and duplicating high-weight ones at the cost of a little immediate variance and some path degeneracy I don't care about for filtering, turning exponential variance growth into linear; proposing from the prior is what makes the weight equal the likelihood, the workhorse fallback for the optimal-but-intractable measurement-aware proposal; and roughening plus optional prior editing keep the cloud from impoverishing when the dynamics are too quiet or the prior and likelihood barely overlap.
