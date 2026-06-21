The setting is a deep, non-convex network — many layers, somewhere between $10^4$ and $10^6$ parameters — trained on crowdsourced data full of people's records, and then shipped, sometimes onto a phone where anyone can read its weights. High-capacity networks memorize: reconstruction attacks pull recognizable training images out of a face recognizer with only query access, and distributed training systems already worry about adversaries who see sanitized parameter updates. So I have to plan against the worst adversary — one who knows my entire training procedure, can read the final parameters, and may even control every training record except the single one I am trying to protect. I want a real, worst-case mathematical guarantee against that adversary, and the honest currency for that is $(\varepsilon, \delta)$-differential privacy: a randomized mechanism $M$ is private if for any two datasets $d, d'$ differing by adding or removing one record, and any output set $S$, $\Pr[M(d) \in S] \le e^{\varepsilon}\Pr[M(d') \in S] + \delta$. The inequality buys deniability — whatever the released model reveals, it would have revealed almost as readily had any one person not been in the data. The prize is a *small*, single-digit $\varepsilon$ on a genuine deep net, and that is exactly what no one has had: people get small $\varepsilon$ on convex models with a handful of parameters, or they train real networks but at a privacy loss so large the guarantee says nothing.

The first instinct is to treat training as a black box and privatize only the output: run ordinary SGD, get $\theta_{\text{final}}$, and add Gaussian noise to it before release. The Gaussian mechanism tells me exactly how much: for a vector query $f$ with $L_2$-sensitivity $S_f = \max_{d \sim d'}\|f(d) - f(d')\|_2$, the release $f(d) + \mathcal{N}(0, S_f^2 \sigma^2 I)$ is $(\varepsilon,\delta)$-private for $\sigma$ on the order of $\sqrt{2\ln(1.25/\delta)}/\varepsilon$. The trouble is the sensitivity of "the final weights of SGD on a non-convex loss." For a convex problem the minimizer is a stable function of the data, and that is precisely what the private-ERM line (Chaudhuri–Monteleoni–Sarwate, Bassily–Smith–Thakurta) exploits to get tight excess-risk bounds like $\tilde{O}(\sqrt{p}/\varepsilon)$. But my loss is non-convex with millions of parameters; SGD is a long chaotic trajectory, and one changed example, amplified through thousands of steps, could in principle send $\theta_{\text{final}}$ anywhere. I have no tight characterization of that dependence, so the only honest sensitivity is a hopeless worst case, and noise calibrated to it would obliterate the model. Output perturbation is dead for non-convex deep nets. The remaining option is to privatize the *process* rather than the endpoint — control the influence of the data at every step — because the one object the data touches at each step is something I actually understand: the gradient.

I propose DP-SGD, differentially private stochastic gradient descent. It rests on two coupled pieces: a per-step mechanism that clips and noises gradients, and a tight accountant — the moments accountant — that makes the composed privacy budget small enough to matter. Take the mechanism first. At step $t$ I form a gradient estimate and step $\theta \leftarrow \theta - \eta g$; if I add noise to that gradient and can bound how much one example moves it, each step is a Gaussian mechanism with a sensitivity I can compute, and composition handles the rest. The obstacle is that there is no a-priori bound on $\|\nabla_\theta L(\theta, x)\|$ — an outlier or misclassified example early in training can produce an enormous gradient — so the sum $\sum_x \nabla_\theta L(\theta, x)$ has unbounded sensitivity and I cannot calibrate noise to infinity. I *make* the sensitivity finite by force, capping each example's contribution. The cleanest cap in $L_2$, the norm the Gaussian mechanism wants, is to rescale:
$$\bar{g}(x) = g(x) \big/ \max\!\Big(1, \tfrac{\|g(x)\|_2}{C}\Big).$$
This is the identity when $\|g\|_2 \le C$ and otherwise scales the gradient down to norm exactly $C$, preserving its direction so the step distortion is minimal. The load-bearing design choice is *where* to clip: I must clip per example, before averaging. The non-private habit is to clip the *averaged* batch gradient just to keep a step from exploding, but that bounds the norm of the whole batch gradient and tells me nothing about any individual's contribution — a single giant per-example gradient can hide inside a batch whose average is small. Privacy needs to bound each person's influence on the released sum, so the clip must happen per example. That costs me the per-example gradients, which plain batched autodiff does not hand over, though they can be computed efficiently. With per-example clipping in place the sensitivity is clean: under the add/remove convention $d = d' \cup \{\text{one example}\}$, the two summed aggregates $\sum_i \bar{g}(x_i)$ differ by exactly one clipped vector of norm at most $C$, so the sum has $L_2$-sensitivity $C$. The Gaussian mechanism then adds $\mathcal{N}(0, \sigma^2 C^2 I)$ to the sum, and dividing by the lot size $L$ to form the mean is pure post-processing, giving per-coordinate noise of standard deviation $\sigma C / L$. I keep $\sigma$ and $C$ deliberately decoupled: $C$ is the sensitivity, set by the geometry of the gradients; $\sigma$ is a unitless noise multiplier, calibrated once to the privacy budget. Examples are drawn not as the full dataset but as a random *lot* of expected size $L = qN$ with per-example inclusion probability $q = L/N$. This is partly the ordinary SGD reason — a subsample estimates the gradient and its variance falls with $L$ — but the privacy reason is at least as important: privacy amplification by subsampling means a mechanism that is $(\varepsilon, \delta)$-private on the full data is roughly $(O(q\varepsilon), q\delta)$-private on a uniform $q$-fraction, because an example unlikely to even be looked at is, in expectation, better hidden. Subsampling also lets the noise-averaging lot $L$ differ from the hardware batch — compute gradients in small batches to fit memory, group several into a lot before noising; the analysis only sees the lot.

The piece that actually decides whether $\varepsilon$ is single-digit or vacuous is the accounting. I run $T$ subsampled-Gaussian steps and need the total privacy of the composition. The standard tool is advanced (strong) composition (Dwork–Rothblum–Vadhan): composing $k$ steps that are each $(\varepsilon, \delta)$-private gives $(\tilde\varepsilon, k\delta + \delta')$ with $\tilde\varepsilon \approx \varepsilon\sqrt{2k\ln(1/\delta')}$, the $\sqrt{k}$ being the only reason iterative private algorithms are feasible at all. But plugging in $q = 0.01$, $\sigma = 4$, $\delta = 10^{-5}$, $T = 10^4$ gives $\varepsilon \approx 9.3$ — too close to ten, and far worse than the Gaussian structure should demand. The reason is that strong composition is a bound for *arbitrary* $(\varepsilon, \delta)$ mechanisms: it knows only each step's tail and is blind to the shape of the noise I compose, even though every step composes the same very well-understood object. The fix is to compose the privacy-loss *distribution* directly. The privacy loss at output $o$ for neighbors $d, d'$ is the log-likelihood ratio $c(o) = \log\big(\Pr[M(d) = o]/\Pr[M(d') = o]\big)$, a random variable over the mechanism's coins, and $(\varepsilon, \delta)$-DP is exactly a tail statement about it. Moment generating functions are the right shape for composing a sum of per-step losses, so I define the log-MGF of the loss at order $\lambda$,
$$\alpha_M(\lambda) = \max_{\mathrm{aux},\,d,\,d'} \log \mathbb{E}_{o \sim M(\mathrm{aux}, d)}\big[\exp(\lambda\, c(o))\big],$$
where the auxiliary input carries the public past so the definition handles *adaptive* steps — the current $\theta$ is a function of all earlier noisy gradients. Two properties make $\alpha$ an accountant. It composes linearly, $\alpha_M(\lambda) \le \sum_i \alpha_{M_i}(\lambda)$: the joint density factors into conditionals, so the total loss is the sum of per-step losses each conditioned on the realized past, and peeling the MGF from the last step backward — conditioning on the realized prefix, where the only fresh randomness is $M_k$'s draw and the prefix is exactly the allowed auxiliary input — bounds each conditional MGF by $\exp(\alpha_{M_k}(\lambda))$; the product of those bounds telescopes, and taking logs gives the sum. I never need the unconditioned losses to be independent, only each fresh-noise conditional MGF bounded, which is why it survives adaptivity. And it converts back to $(\varepsilon, \delta)$ by Markov on the exponentiated loss: $\Pr_{o \sim M(d)}[c(o) \ge \varepsilon] \le \exp(\alpha_M(\lambda) - \lambda\varepsilon)$, and splitting any output set by the bad event $\{c \ge \varepsilon\}$ turns that into the DP inequality with $\delta = \min_\lambda \exp(\alpha_M(\lambda) - \lambda\varepsilon)$.

The heart of it is bounding $\alpha(\lambda)$ for a single subsampled-Gaussian step. Work in units of $C$ so the sensitivity is $1$ and the noise has std $\sigma$; with one differing coordinate the problem is one-dimensional. Let $\mu_0$ be the density of $\mathcal{N}(0, \sigma^2)$ and $\mu_1$ that of $\mathcal{N}(1, \sigma^2)$ — the differing example shifts one coordinate's mean by its unit clipped gradient. Because the example is included only with probability $q$, the output under $d$ is the mixture $\mu = (1-q)\mu_0 + q\mu_1$ while under $d'$ it is $\mu_0$, and $\alpha(\lambda) = \log\max(E_1, E_2)$ over the two likelihood-ratio directions, each of the form $\mathbb{E}_{z \sim \nu_1}[(\nu_0/\nu_1)^{\lambda+1}]$. Expanding $\nu_0/\nu_1 = 1 + (\nu_0 - \nu_1)/\nu_1$ binomially, the $t=0$ term is $1$ and the $t=1$ term is $\int(\nu_0 - \nu_1) = 0$ — the first-order term vanishes, which is exactly why the bound is $O(q^2)$ rather than $O(q)$. The leading $t=2$ term, in the harder direction $\nu_0 = \mu_0, \nu_1 = \mu$, uses $\mu \ge (1-q)\mu_0$ and $\mu_0 - \mu = q(\mu_0 - \mu_1)$ to reduce to $\tfrac{q^2}{1-q}\mathbb{E}_{z \sim \mu_0}[((\mu_0 - \mu_1)/\mu_0)^2]$, and since $\mu_1/\mu_0 = \exp((2z-1)/2\sigma^2)$, that Gaussian expectation collapses cleanly via $\mathbb{E}_{z\sim\mu_0}[\exp(az/\sigma^2)] = \exp(a^2/2\sigma^2)$ to $\exp(1/\sigma^2) - 1 \approx 1/\sigma^2$. Multiplying by $\binom{\lambda+1}{2} = \lambda(\lambda+1)/2$ gives the second-order contribution $q^2\lambda(\lambda+1)/((1-q)\sigma^2)$. The higher terms $t \ge 3$, bounded region-by-region with the Gaussian absolute-moment bound and the three pointwise comparisons for $|\mu_0 - \mu_1|$, decrease geometrically once $q < 1/(16\sigma)$ and $\lambda \le \sigma^2\ln(1/(q\sigma))$, so the whole tail is dominated by the $t=3$ term and
$$\alpha(\lambda) \le \frac{q^2\lambda(\lambda+1)}{(1-q)\sigma^2} + O\!\Big(\frac{q^3\lambda^3}{\sigma^3}\Big).$$
Summing over $T$ steps and optimizing $\lambda$ in the tail bound — needing $\alpha(\lambda) \le \lambda\varepsilon/2$ and $\exp(-\lambda\varepsilon/2) \le \delta$ within the validity range — gives the calibration theorem: there exist constants $c_1, c_2$ such that for any $\varepsilon < c_1 q^2 T$, DP-SGD is $(\varepsilon, \delta)$-private whenever
$$\sigma \ge c_2 \cdot \frac{q\sqrt{T\log(1/\delta)}}{\varepsilon}.$$
Strong composition would instead demand $\sigma = \Omega\big(q\sqrt{T\log(1/\delta)\log(T/\delta)}/\varepsilon\big)$, a $\sqrt{\log(T/\delta)}$ factor worse, and never sheds the $Tq\delta$ accumulation in the $\delta$ part. On the same $q = 0.01, \sigma = 4, \delta = 10^{-5}, T = 10^4$ numbers the moments accountant reports $\varepsilon \approx 1.26$ versus $\approx 9.34$ — the entire gap is composing the loss distribution's moments instead of just its tail. In practice I can numerically integrate $E_1$ and $E_2$ for each $\lambda$ for tightness, with the analytic $q^2\lambda(\lambda+1)/((1-q)\sigma^2)$ as the fallback that proves the theorem; the optimal $\lambda$ is small so a grid up to about $32$ suffices; and because I fix $T$ and the privacy parameters ahead of time and read off the spent $\varepsilon$ at the end, I invert the accountant to choose $\sigma$ before training rather than retargeting mid-run.

So the full algorithm is: at each step subsample a lot at rate $q$, compute per-example gradients, clip each in $L_2$ to $C$, sum, add $\mathcal{N}(0, \sigma^2 C^2 I)$, divide by the expected lot size, and take an ordinary SGD step on the result — with $\sigma$ calibrated to the budget up front by inverting the moments accountant. The mechanism fills the one empty slot, the transformation from a lot's per-example gradients to the single noised aggregate the optimizer steps on: flatten each parameter's per-example gradient, combine the per-parameter norms into one per-example $L_2$ norm, form the clip factor $\min(1, C/(\|g\| + 10^{-6}))$, apply that same factor to every parameter tensor for that example, sum the clipped gradients per parameter, add Gaussian noise of std $\sigma C$ to the sum, and only then divide by the expected lot size when the loss reduction is a mean. The ordering matters — the Gaussian mechanism is calibrated to the sensitivity of the sum, and the mean is post-processing — and the tiny denominator floor only shrinks the clip factor slightly, so it cannot increase sensitivity.

```python
import torch


class PrivateGradientMechanism:
    """Per-example L2 clipping to max_grad_norm, then Gaussian noise of
    std (noise_multiplier * max_grad_norm) on the summed clipped gradients.
    """

    def __init__(self, max_grad_norm, noise_multiplier, expected_lot_size,
                 loss_reduction="mean", generator=None):
        self.max_grad_norm = max_grad_norm          # C: per-example L2 clip threshold = sensitivity
        self.noise_multiplier = noise_multiplier    # sigma: noise multiplier (privacy knob)
        self.expected_lot_size = expected_lot_size
        self.loss_reduction = loss_reduction
        self.generator = generator

    def _generate_noise(self, reference):
        return torch.normal(
            mean=0.0,
            std=self.noise_multiplier * self.max_grad_norm,
            size=reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=self.generator,
        )

    def aggregate(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]

        # per-example flat L2 norm across all parameters
        per_param_norms = [g.reshape(batch_size, -1).norm(2, dim=1) for g in per_sample_grads]
        norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)        # [B]

        # clip factor min(1, C / (||g|| + small floor))
        clip_factor = (self.max_grad_norm / (norms + 1e-6)).clamp(max=1.0)

        noised_grads = []
        for g in per_sample_grads:
            clip_factor_on_device = clip_factor.to(g.device).to(g.dtype)
            summed = torch.einsum("i,i...", clip_factor_on_device, g)      # clipped sum
            # Gaussian mechanism: sum has sensitivity C, so std on the sum is sigma*C
            grad = summed + self._generate_noise(summed)
            if self.loss_reduction == "mean":
                grad = grad / self.expected_lot_size
            noised_grads.append(grad)

        return noised_grads

    def get_effective_sigma(self, step, epoch):
        return self.noise_multiplier
```

```python
import math


def compute_epsilon(steps, sigma, q, delta, orders=None):
    """Accumulated epsilon from the sampled-Gaussian log-moment bound."""
    if orders is None:
        orders = range(1, 64)
    best_eps = float("inf")
    for lam in orders:
        if lam <= 0:
            continue
        per_step_alpha = q * q * lam * (lam + 1) / ((1.0 - q) * sigma * sigma)
        total_alpha = steps * per_step_alpha                              # composes linearly
        eps = (total_alpha + math.log(1.0 / delta)) / lam                  # Markov tail bound
        best_eps = min(best_eps, eps)
    return best_eps


def calibrate_noise_to_epsilon(target_epsilon, steps, q, delta, tol=1e-3):
    """Smallest sigma whose composed budget spends at most target_epsilon (binary search)."""
    lo, hi = 0.01, 100.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if compute_epsilon(steps, mid, q, delta) > target_epsilon:
            lo = mid                                                      # more epsilon -> more noise
        else:
            hi = mid
    return (lo + hi) / 2
```
