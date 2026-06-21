A pretrained diffusion model is a rich, reusable prior over some signal space, $p_0(x_0)$, baked into a denoiser $D_\theta$. What I want is to reuse that prior at test time to impose a condition expressed as a differentiable loss $l_y(x_0)$ on clean signals — a measurement-consistency term $\|A(x_0)-y\|^2$ for an inverse problem, a classifier's negative log-probability, a CLIP score, a path-following-and-obstacle-avoidance penalty for a motion model — without retraining anything. That is exactly sampling from the posterior $p_0^{(l)}(x_0\mid y)\propto p_0(x_0)\,\exp(-l_y(x_0))$, reading $\exp(-l_y(x_0))$ as an unnormalized likelihood $p_0(y\mid x_0)$ with normalizer $Z=\int p_0\exp(-l)\,dx_0$ that I carry along and watch drop out. The obstacle is that a diffusion sampler integrates a reverse process driven only by the score of the noised marginal at each level, so I need the conditional score at every noise level $t$, not just at $t=0$. Bayes' rule on the noised variable splits it cleanly,

$$\nabla_{x_t}\log p_t(x_t\mid y)=\nabla_{x_t}\log p_t(x_t)+\nabla_{x_t}\log p_t(y\mid x_t).$$

The first term is free: Tweedie gives $D_\theta(x_t,t)=\mathbb{E}[x_0\mid x_t]=x_t+\sigma_t^2\nabla_{x_t}\log p_t(x_t)$, so $\nabla_{x_t}\log p_t(x_t)=(D_\theta(x_t,t)-x_t)/\sigma_t^2$ from one network call. The whole game is the second term, the guidance $\nabla_{x_t}\log p_t(y\mid x_t)$, which I have to manufacture cheaply and accurately at every noise level. Writing it out, since $y$ depends only on the clean signal and so $y\perp x_t\mid x_0$,

$$p_t(y\mid x_t)=\int p(x_0\mid x_t)\,p_0(y\mid x_0)\,dx_0=\mathbb{E}_{x_0\sim p(x_0\mid x_t)}\big[\exp(-l_y(x_0))\big],$$

an expectation of the clean-data likelihood over the denoising posterior $p(x_0\mid x_t)$. That posterior is the hard object — accurately sampling or evaluating it would take many diffusion steps, the very cost I am trying to avoid — so with a single network evaluation the integral is intractable and must be approximated. The existing plug-and-play option, DPS, takes the cheapest approximation: collapse $p(x_0\mid x_t)$ to a delta at the Tweedie estimate $\hat x_t=D_\theta(x_t,t)$, giving $\nabla_{x_t}\log p_t(y\mid x_t)\approx-\nabla_{x_t}l_y(\hat x_t)$, a backprop of the clean-data loss gradient through the denoiser. This is genuinely useful — one network call, plug-and-play, handles nonlinear $A$ and measurement noise — and its single-network-call structure is worth keeping. But replacing $\mathbb{E}[\exp(-l)]$ by $\exp(-l(\mathbb{E}[x_0]))$ is a Jensen swap, exact only when the loss is affine or the posterior is a point, and $p(x_0\mid x_t)$ is emphatically not a point, least of all at high noise. The other baselines are worse fits: classifier guidance needs a separately trained classifier on noisy images at every level (paired data, not plug-and-play, no arbitrary test-time loss); diffusion-as-plug-and-play-prior optimizes to a slow single point that can stall in a local minimum and cannot inherit fast samplers; and the closed-form reconstruction/linear solvers only handle linear operators with least-squares losses.

To see whether the delta's bias is benign or structural, build the smallest tractable example: $x_0$ one-dimensional, $p_0$ a mixture of two well-separated Gaussians ($N(-1,0.2^2)$ and $N(+1,0.2^2)$, equal weights), $y\in\{0,1\}$ the label. Then every $p_t(x_t)$ is a Gaussian mixture and $p_t(y\mid x_t)$ is a closed-form logistic, so the true guidance can be held up against the delta at any $\sigma_t$. The clean-data guidance gradient is large and steep right at the decision boundary $x_0\approx 0$ and goes flat deep inside either mode. Trace the delta through the schedule. At high noise the MMSE estimate $\hat x_t=\mathbb{E}[x_0\mid x_t]$ of the symmetric mixture is pulled to the middle, $\hat x_t\approx0$, landing the delta on the steep part — so it reports a large guidance — whereas the true expectation averages $\exp(-l)$ over a posterior smeared across both modes, where the two sides partly cancel and the truth is small. At low noise $\hat x_t$ has committed to a mode, sitting in a flat region, so the delta reports a small guidance, whereas the true posterior is tight and the truth is large. The delta has it backwards at both ends: too large at high noise, too small at low noise. This is structural — evaluating a curved gradient at the posterior mean is not averaging it over the spread, and the spread is enormous at high noise. The tell that the field already feels this is the standard patch of dividing the delta's guidance by the denoiser's negative log-likelihood times a constant, which shrinks the guidance at high noise and inflates it at low noise — exactly my correction direction, but bolted onto the wrong object. The honest fix is to stop collapsing the posterior to a point.

I propose Loss-Guided Diffusion, LGD (in the Monte Carlo form, LGD-MC). The idea is to keep DPS's single-network-call structure but replace the delta with a spread-out Gaussian surrogate $q(x_0\mid x_t)=\mathcal{N}(\mu(x_t),r_t^2 I)$ for the denoising posterior, and to estimate $\mathbb{E}_{x_0\sim q}[\exp(-l(x_0))]$ honestly by Monte Carlo. Two design choices fix $q$: where to center it and how wide to make it. For the center, take the variational view and maximize the expected log-likelihood of the true clean signals under $q$, $\max_q\mathbb{E}_{x_0\sim p(x_0\mid x_t)}[\log q(x_0\mid x_t)]$. At fixed covariance $\log q=-\|x_0-\mu\|^2/(2r_t^2)+\text{const}$, so this reduces to $\min_\mu\mathbb{E}[\|\mu-x_0\|^2]$, whose minimizer is the mean: $\mu(x_t)=\mathbb{E}[x_0\mid x_t]=\hat x_t$. So the Tweedie estimate is not merely convenient — it is the provably optimal mean of any fixed-covariance Gaussian surrogate, and it comes from the one network call I am already making. The difference from DPS is therefore entirely in giving $q$ a width. For the width, reason from the noising model $x_t=x_0+\sigma_t\varepsilon$ with a roughly unit-scale prior on $x_0$: the posterior over $x_0$ is a Gaussian-conjugate balance of the prior precision ($\approx 1$) and the likelihood precision ($1/\sigma_t^2$), giving posterior precision $(1+\sigma_t^2)/\sigma_t^2$ and hence

$$r_t=\frac{\sigma_t}{\sqrt{1+\sigma_t^2}}.$$

The limits are exactly right: at high noise $\sigma_t\to\infty$, $r_t\to 1$ — as wide as the prior, since a very noisy observation says almost nothing about $x_0$ — and at low noise $\sigma_t\to 0$, $r_t\to\sigma_t\to 0$, so the surrogate collapses to the DPS delta precisely where the point estimate was accurate. I am not discarding DPS; I am fixing it where it is broken (high noise, wide posterior) and recovering it where it was fine. That this is the right direction, and not wishful thinking, follows from a total-variation bound on the bias. With $M=\max_{x_0}p_0(y\mid x_0)$ (the loss makes $\exp(-l)$ bounded),

$$\Big|\mathbb{E}_p[p_0(y\mid x_0)]-\mathbb{E}_q[p_0(y\mid x_0)]\Big|=\Big|\int\big(p(x_0\mid x_t)-q(x_0\mid x_t)\big)\,p_0(y\mid x_0)\,dx_0\Big|\le M\int|p-q|\,dx_0=2M\cdot\mathrm{TV}(p,q).$$

The DPS delta is maximally far from a continuous spread-out posterior — a point mass and an absolutely continuous distribution have $\mathrm{TV}=1$ — whereas a Gaussian with the right mean and right width is built to be closer in TV, and whenever it is closer the worst-case bias bound is smaller. This is an asymptotic statement about the choice of $q$ (the bias when $\mathbb{E}_q$ is evaluated exactly), separate from how many samples I will need.

To compute $\mathbb{E}_{x_0\sim q}[\exp(-l(x_0))]$ and its gradient I Monte Carlo it, but carefully, because the expectation sits inside the log. The right estimator replaces the expectation by the sample mean and then takes the log — a log-mean-exp of the negative loss over $n$ samples,

$$\mathrm{MC}_n(x_t,y)=\nabla_{x_t}\log\!\Big(\tfrac1n\sum_{i=1}^n\exp(-l_y(x^{(i)}))\Big)=-\sum_i w_i\,\nabla_{x_t}l_y(x^{(i)}),\qquad w_i=\frac{\exp(-l_y(x^{(i)}))}{\sum_j\exp(-l_y(x^{(j)}))}.$$

Differentiating the log-mean-exp yields a softmin-weighted average of the per-sample loss gradients, with smaller-loss samples weighted more. The naive alternative of averaging the loss gradients $\tfrac1n\sum_i\nabla l(x^{(i)})$ estimates $\nabla\mathbb{E}[l]$, the same Jensen confusion one level down — the expectation must wrap $\exp(-l)$, not $l$. The exact DPS formula reappears only when $r_t\to 0$ and $q$ collapses to the delta; with positive $r_t$, $n=1$ is a one-sample Gaussian estimate, not the DPS delta, and increasing $n$ only reduces Monte Carlo error around the surrogate expectation. For the gradient to flow I reparameterize, $x^{(i)}=\hat x_t+r_t\varepsilon^{(i)}$, $\varepsilon^{(i)}\sim\mathcal{N}(0,I)$, so every sample is a deterministic function of the shared $\hat x_t=D_\theta(x_t,t)$ plus fixed noise. The crucial efficiency fact is that all $n$ samples branch off the same denoiser output, so the gradient path holds exactly one $D_\theta$: aggregate the cheap per-sample loss gradients into a cotangent $g$, then take a single vector-Jacobian product $\texttt{autograd.grad}(\hat x_t,x_t,g)$ through the denoiser. The expensive backward through the diffusion network happens once regardless of $n$; only the loss and forward-operator gradients scale with $n$.

For the inverse-problem implementation, the forward operator exposes squared-residual gradients and a squared-residual loss, and the practical loop takes the arithmetic mean of the per-sample residual gradients before the one VJP, $g=\tfrac1n\sum_i\nabla_{x^{(i)}}\|A(x^{(i)})-y\|^2$. This is a mean-gradient realization specialized to the least-squares interface — not the exact softmin-weighted log-mean-exp, which would carry the $w_i$ — but it preserves the posterior spread through the sampled $x^{(i)}$, which is the whole point. The step is then stabilized exactly as DPS stabilizes its own: the raw squared-residual guidance $\rho\nabla\|y-A(\hat x)\|^2$ has a notoriously unstable magnitude because $\nabla\|\cdot\|^2=2(\cdot)\nabla(\cdot)$ carries the residual magnitude, so I normalize by the residual norm. With $\text{loss}=\|A(x)-y\|^2$, multiplying its gradient by $0.5/\sqrt{\text{loss}}$ gives $\nabla\|A(x)-y\|^2/(2\|A(x)-y\|)=\nabla\|A(x)-y\|=\nabla\sqrt{\text{loss}}$, the gradient of the root loss — a far gentler, empirically stable scale. On top of that sits one overall $\texttt{guidance\_scale}$ multiplier in the classifier-guidance spirit, since $s\cdot\nabla\log p(y\mid x)=\nabla\log(p(y\mid x)^s/Z)$ sharpens the conditional toward the loss's modes, tuned per problem since its natural value tracks the operator's loss scale. Assembling the reverse step in the EDM-scaled form: with schedule $\sigma$, scaling $s$, step $\Delta t$, the precomputed coefficients are $\texttt{scaling\_factor}=1-(\dot s/s)\Delta t$ and $\texttt{factor}=2s^2\dot\sigma\sigma\Delta t$, the score is $(\text{denoised}-x_{\text{cur}}/s)/\sigma^2/s$, the unconditional SDE step is $x_{\text{next}}=x_{\text{cur}}\cdot\texttt{scaling\_factor}+\texttt{factor}\cdot\text{score}+\sqrt{\texttt{factor}}\,\varepsilon$ (the ODE drops the noise and halves the score), and finally I subtract $\texttt{ll\_grad}\cdot\texttt{scale}$ — subtracting the loss-gradient update is the same as adding the estimated $\nabla_{x_t}\log p_t(y\mid x_t)$, since the score already points up the prior and the loss gradient points toward worse fit. The result is a plug-and-play loss update computed from one network call per step, with posterior spread restored exactly where the point estimate was mis-scaled.

```python
import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np

import wandb


class LGD(Algo):
    def __init__(self,
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 guidance_scale,
                 num_samples=10,
                 batch_grad=True,
                 sde=True):
        super(LGD, self).__init__(net, forward_op)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde
        self.num_samples = num_samples
        self.batch_grad = batch_grad

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution,
                                self.net.img_resolution, device=device) * self.scheduler.sigma_max
        x_next = x_initial
        x_next.requires_grad = True
        pbar = tqdm(range(self.scheduler.num_steps))

        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)

            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], \
                self.scheduler.scaling_factor[i]
            rt = sigma / np.sqrt(1 + sigma ** 2)

            denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))

            samples = denoised + torch.randn((self.num_samples, *denoised.shape[1:]), device=device) * rt

            if self.batch_grad:
                gradient, loss_scale = self.forward_op.gradient(samples, observation, return_loss=True)
                gradients = gradient
                avg_loss = loss_scale
            else:
                gradients = torch.empty((self.num_samples, *denoised.shape[1:]), device=device)
                losses = np.empty(self.num_samples)
                for j in range(self.num_samples):
                    gradient, loss_scale = self.forward_op.gradient(samples[j:j+1], observation, return_loss=True)
                    gradients[j] = gradient
                    losses[j] = loss_scale
                avg_loss = losses.mean()

            avg_grad = torch.mean(gradients, dim=0, keepdim=True).detach()

            ll_grad = torch.autograd.grad(denoised, x_cur, avg_grad)[0]
            ll_grad = ll_grad * 0.5 / torch.sqrt(avg_loss)

            score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
            pbar.set_description(
                f'Iteration {i + 1}/{self.scheduler.num_steps}. '
                f'Data fitting loss: {torch.sqrt(loss_scale)}')
            if wandb.run is not None:
                wandb.log({'data_fitting_loss': torch.sqrt(loss_scale)})

            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5

            x_next -= ll_grad * self.scale

        return x_next
```
