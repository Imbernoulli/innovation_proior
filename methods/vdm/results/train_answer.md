The leaderboard fact that started all of this is simple and a little embarrassing: on bits per dimension for CIFAR-10 and downsampled ImageNet, autoregressive models keep winning, while the Gaussian-corruption generative models — the ones that add noise to an image and learn to walk it back — produce gorgeous samples but mediocre likelihood. Either that model class is fundamentally a bad density estimator, or we are training and evaluating it with the wrong objective. I do not believe the first. The standard recipes give us most of the apparatus we want — a forward process $q(z_t|x) = \mathcal N(\alpha_t x, \sigma_t^2 I)$ with signal-to-noise ratio $\mathrm{SNR}(t)=\alpha_t^2/\sigma_t^2$ strictly decreasing, a denoiser to reverse it — but they fall short in three specific ways that each cost likelihood. First, the objective people actually minimize is a quality-tuned surrogate: the discrete-time bound equals a weighted sum of denoising-score-matching losses with a definite per-noise-level weighting, and the perceptual recipes drop that weighting (set it to one on the noise MSE), which improves FID but is no longer a bound on the data log-likelihood. Second, the noise schedule — the shape of $\mathrm{SNR}(t)$ — is hand-designed (β-linear, cosine) and frozen, even though different schedules give very different bounds. Third, the bound is numerically fragile: naive implementations compute intermediate quantities very close to $1$, exactly where floating point is worst, so practitioners fall back to 64-bit arithmetic; and at the lowest noise levels the marginal $q(z_t)$ is sharply peaked because 8-bit pixels are discrete, and a smooth convolutional denoiser struggles with that fine-scale detail that likelihood — unlike FID — punishes acutely.

I propose Variational Diffusion Models (VDM), which treat a corruption model as exactly what it is — a VAE whose inference network is fixed (the known Gaussian corruption) and whose latent is an infinitely deep Markov chain of progressively noisier $z_t$ — and derive the right bound honestly from that view. The negative ELBO of such a chain decomposes the way these always do, into a prior term, a reconstruction term, and a diffusion term: $$-\mathrm{VLB}(x) = \underbrace{\mathrm{KL}\big(q(z_1|x)\,\|\,\mathcal N(0,I)\big)}_{\text{prior}} + \underbrace{\mathbb E_{q(z_0|x)}[-\log p(x|z_0)]}_{\text{reconstruction}} + L_T.$$ The first two are standard VAE terms; everything subtle lives in $L_T = \sum_{i=1}^T \mathbb E_{q(z_t|x)}\,\mathrm{KL}\big(q(z_s|z_t,x)\,\|\,p(z_s|z_t)\big)$ over the chain of transitions with $s=(i-1)/T,\,t=i/T$. The key design choice is the model conditional: I set $p(z_s|z_t)=q\big(z_s|z_t,\,x=\hat x_\theta(z_t,t)\big)$, the true posterior with the unknown clean data replaced by a denoiser's prediction. The obvious alternative — a freely-parameterized Gaussian mean and variance — is worse because it leaves the variance unconstrained; matching the posterior makes the two distributions inside each KL share an *identical* variance, so the Gaussian KL collapses to a pure squared error $\mathrm{KL}=\frac{1}{2\sigma_Q^2}\|\mu_Q-\mu_\theta\|^2$, and because the posterior mean is affine in $x$ and $z_t$ with the $z_t$ part common to both, even the $z_t$ term cancels, leaving $\mu_Q-\mu_\theta = (\alpha_s\sigma_{t|s}^2/\sigma_t^2)(x-\hat x_\theta)$. Grinding the coefficient — using $\sigma_{t|s}^2=\sigma_t^2-\alpha_{t|s}^2\sigma_s^2$ and $\alpha_s^2\alpha_{t|s}^2=\alpha_t^2$ — everything messy cancels and the per-transition KL becomes half the SNR drop times the denoising error: $$\mathrm{KL}=\tfrac12\big(\mathrm{SNR}(s)-\mathrm{SNR}(t)\big)\,\|x-\hat x_\theta\|^2.$$ That is the load-bearing identity. The whole diffusion loss sees the schedule only through SNR: $L_T=\frac T2\,\mathbb E_{\varepsilon,i}\big[(\mathrm{SNR}(s)-\mathrm{SNR}(t))\,\|x-\hat x_\theta(z_t,t)\|^2\big]$ with $i\sim\mathrm{Uniform}\{1,\dots,T\}$ as an unbiased estimator.

What makes the bound clean is the next step. Comparing $L_{2T}$ to $L_T$ by inserting a midpoint, the difference reduces to $\frac12\mathbb E_\varepsilon\sum_i(\mathrm{SNR}(s)-\mathrm{SNR}(t'))(\|x-\hat x_\theta(z_{t'},t')\|^2-\|x-\hat x_\theta(z_t,t)\|^2)$; since $z_{t'}$ is less noisy than $z_t$, predicting $x$ from it is strictly easier, the bracket is negative, and $L_{2T}-L_T<0$ for any decent denoiser. More steps always help — the discrete loss is an upper Riemann sum of a decreasing integrand — so the best depth is infinite and I take the continuous-time limit directly. Writing $L_T$ with $\tau=1/T$, the difference quotient $(\mathrm{SNR}(t-\tau)-\mathrm{SNR}(t))/\tau\to-\mathrm{SNR}'(t)$ and the uniform index becomes $t\sim\mathrm{Uniform}[0,1]$: $$L_\infty=-\tfrac12\,\mathbb E_{\varepsilon,\,t\sim U[0,1]}\big[\mathrm{SNR}'(t)\,\|x-\hat x_\theta(z_t,t)\|^2\big].$$ Now the real payoff: $\mathrm{SNR}$ is strictly monotonic, hence invertible, so change variables to $v\equiv\mathrm{SNR}(t)$ with $dv=\mathrm{SNR}'(t)\,dt$; the minus sign and the swapped limits combine into a clean positive integral $L_\infty=\frac12\,\mathbb E_\varepsilon\int_{\mathrm{SNR}_{\min}}^{\mathrm{SNR}_{\max}}\|x-\tilde x_\theta(z_v,v)\|^2\,dv$, with $\mathrm{SNR}_{\min}=\mathrm{SNR}(1)$ and $\mathrm{SNR}_{\max}=\mathrm{SNR}(0)$. The integrand touches the schedule only through $z_v$, and since $v=\alpha_v^2/\sigma_v^2$ gives $\sigma_v=\alpha_v/\sqrt v$, $$z_v=\alpha_v x+\sigma_v\varepsilon=\alpha_v\big(x+\varepsilon/\sqrt v\big),$$ whose informative content $x+\varepsilon/\sqrt v$ depends on $v$ alone. So once the two endpoints are fixed, the continuous-time bound is *completely independent of the schedule's interior shape* — the whole thing everyone hand-tunes enters only through its two endpoints, and everything between $t=0$ and $t=1$ is gauge. This also unifies the variance-preserving and variance-exploding processes: with the same endpoints their latents differ only by a deterministic rescaling $z_v^A=(\alpha_v^A/\alpha_v^B)z_v^B$, so defining the denoiser to absorb that factor makes the loss, the reverse conditionals, and hence $p(x)$ identical — they are the same continuous-time model.

That invariance is not just pretty, it is operationally useful, and exploiting it is the second pillar of the method. Because the bound's value is fixed by the endpoints, I am free to use the interior shape for something the bound does *not* care about: the variance of the Monte Carlo estimator, which controls how noisy the gradients are and hence how fast optimization goes, and which very much does depend on how my samples of $t$ are spread across SNR levels. So I learn the schedule shape to minimize estimator variance while the endpoints carry the actual bound. Concretely I parameterize $\gamma_\eta(t)\equiv-\log\mathrm{SNR}(t)$ (monotone increasing since SNR decreases), built from linear layers with non-negative weights so monotonicity — needed for the change of variables — is guaranteed, with a wide middle layer for flexibility; then I rescale it to pin the endpoints, $\gamma_\eta(t)=\gamma_0+(\gamma_1-\gamma_0)(\tilde\gamma_\eta(t)-\tilde\gamma_\eta(0))/(\tilde\gamma_\eta(1)-\tilde\gamma_\eta(0))$, so $\gamma_0=-\log\mathrm{SNR}_{\max}$ and $\gamma_1=-\log\mathrm{SNR}_{\min}$ are fixed no matter what the interior weights do. The endpoints are trained against the VLB; the interior $\eta$ against variance. The cute trick is how: do SGD on the *squared* Monte Carlo loss, because $\mathbb E[(L^{MC})^2]=L_\infty^2+\mathrm{Var}[L^{MC}]$ and $L_\infty^2$ is interior-independent, so $\nabla_\eta\mathbb E[(L^{MC})^2]=\nabla_\eta\mathrm{Var}[L^{MC}]$ — gradient descent on the squared loss is exactly variance minimization, for free, with no separate variance estimator and no second backward pass through the denoiser (the needed $dL^{MC}/d\mathrm{SNR}$ is a byproduct of the main backward). I also sample $t$ antithetically: draw one $u_0\sim U[0,1]$ and set $t^i=\mathrm{mod}(u_0+i/k,1)$, so each $t^i$ is marginally uniform but the batch tiles $[0,1]$ evenly, further cutting variance.

To make the objective look like the field's and be numerically sane, I parameterize the denoiser as noise prediction $\hat\varepsilon_\theta$ (the well-behaved target, fixed unit scale across noise levels), with $\hat x_\theta=(z_t-\sigma_t\hat\varepsilon_\theta)/\alpha_t$ and the score being just another linear face $s_\theta=-\hat\varepsilon_\theta/\sigma_t$. In the variance-preserving spec, $\alpha_t^2=\mathrm{sigmoid}(-\gamma)$, $\sigma_t^2=\mathrm{sigmoid}(\gamma)$, $\mathrm{SNR}=e^{-\gamma}$, and substituting $x-\hat x_\theta=-\sigma_t(\varepsilon-\hat\varepsilon_\theta)/\alpha_t$ turns $\mathrm{SNR}'(t)\|x-\hat x\|^2$ into $-\gamma'(t)\|\varepsilon-\hat\varepsilon\|^2$, so $$L_\infty=\tfrac12\,\mathbb E_{\varepsilon,t}\big[\gamma'_\eta(t)\,\|\varepsilon-\hat\varepsilon_\theta(z_t,t)\|^2\big],$$ and the discrete version is $L_T=\frac T2\,\mathbb E_{\varepsilon,i}\big[(e^{\gamma(t)-\gamma(s)}-1)\,\|\varepsilon-\hat\varepsilon_\theta\|^2\big]$. I deliberately write the coefficient as $e^{\gamma(t)-\gamma(s)}-1$: for fine time steps $\gamma(t)-\gamma(s)$ is small positive, the naive $\exp$ rounds toward $1$ and loses its significant digits, so I use the stable primitive $\mathrm{expm1}$, which lets the whole thing train in fp32 (or bf16) where naive implementations needed fp64. The same care appears in the sampler's posterior, using $\mathrm{expm1}$ and $\mathrm{softplus}$ to stay away from the bad region near $1$. The reconstruction term handles the discrete 8-bit data with a decoder $p(x_i|z_{0,i})\propto q(z_{0,i}|x_i)$ normalized over the 256 pixel values — a discretized-Gaussian categorical head, tight because $\mathrm{SNR}(0)$ is large — and the prior term is the closed-form diagonal-Gaussian KL $\frac12\sum((1-\mathrm{var}_1)x^2+\mathrm{var}_1-\log\mathrm{var}_1-1)$. Consistency is guaranteed by denoising score matching: predicting $\varepsilon$ is predicting $\nabla\log q(z_t|x)$ up to $-1/\sigma_t$, and for any positive weighting the marginal and conditional score-matching losses agree up to a $\theta$-independent constant, so at the optimum $s_\theta^*(z_t)=\nabla\log q(z_t)$ at every noise level — the bound I derived chases the right target. Finally, the one wall the clean theory ignores — fine-scale detail — I close with Fourier features: append channels $\sin(2^n\pi z),\cos(2^n\pi z)$ for high integer $n\in\{7,8\}$, which amplify tiny scalar changes in $z$ before they enter the smooth network, letting the denoiser carry the low-noise detail that likelihood demands so $\mathrm{SNR}_{\max}$ can be pushed high. (In the released code the band appears as $\mathrm{range}(6,8)$ because the multiplier is $2\pi$, giving the same two frequencies since $2^6\cdot2\pi=2^7\pi$.) This also explains the leaderboard gap directly: writing the weighted bound $L_\infty(x,w)=\frac12\mathbb E\int w(v)\|x-\tilde x_\theta\|^2dv$, the VLB is $w=1$, while the "simple" noise-MSE objective is $w=1/\gamma'(t)$ — more weight on low SNR, good for FID, bad for likelihood. The class was never a bad density estimator; people were training a quality-weighted surrogate with a hand-fixed schedule. Take $w=1$, learn the endpoints, add Fourier features, and the corruption model becomes a first-rate likelihood model whose $-\mathrm{VLB}$ at finite $T$ is also the bits-back codelength for near-optimal lossless compression.

```python
import jax, jax.numpy as jnp, numpy as np
import flax.linen as nn


class VDM(nn.Module):
    config: object

    def setup(self):
        self.encdec = EncDec(self.config)
        self.score_model = ScoreUNet(self.config)     # predicts eps_hat(z_t, gamma_t)
        if self.config.gamma_type == "learnable_nnet":
            self.gamma = NoiseSchedule_NNet(self.config)
        elif self.config.gamma_type == "fixed":
            self.gamma = NoiseSchedule_FixedLinear(self.config)
        elif self.config.gamma_type == "learnable_scalar":
            self.gamma = NoiseSchedule_Scalar(self.config)
        else:
            raise ValueError("unknown gamma_type")

    def __call__(self, images, conditioning, deterministic=True):
        g_0, g_1 = self.gamma(0.), self.gamma(1.)
        var_0, var_1 = nn.sigmoid(g_0), nn.sigmoid(g_1)
        x = images
        n_batch = x.shape[0]
        f = self.encdec.encode(x)

        # 1) reconstruction loss  -log p(x | z_0)
        eps_0 = jax.random.normal(self.make_rng("sample"), f.shape)
        z_0_rescaled = f + jnp.exp(0.5 * g_0) * eps_0     # = z_0 / sqrt(1 - var_0)
        loss_recon = - self.encdec.logprob(x, z_0_rescaled, g_0)

        # 2) prior loss  KL( q(z_1|x) || N(0, I) )
        mean1_sqr = (1. - var_1) * jnp.square(f)
        loss_klz = 0.5 * jnp.sum(mean1_sqr + var_1 - jnp.log(var_1) - 1., axis=(1, 2, 3))

        # 3) diffusion loss
        rng1 = self.make_rng("sample")
        if self.config.antithetic_time_sampling:
            t0 = jax.random.uniform(rng1)
            t = jnp.mod(t0 + jnp.arange(0., 1., step=1. / n_batch), 1.)   # antithetic / low-discrepancy
        else:
            t = jax.random.uniform(rng1, shape=(n_batch,))
        T = self.config.sm_n_timesteps
        if T > 0:
            t = jnp.ceil(t * T) / T
        g_t = self.gamma(t)
        var_t = nn.sigmoid(g_t)[:, None, None, None]
        eps = jax.random.normal(self.make_rng("sample"), f.shape)
        z_t = jnp.sqrt(1. - var_t) * f + jnp.sqrt(var_t) * eps         # alpha_t x + sigma_t eps
        eps_hat = self.score_model(z_t, g_t, conditioning, deterministic)
        loss_diff_mse = jnp.sum(jnp.square(eps - eps_hat), axis=[1, 2, 3])

        if T == 0:                                                     # continuous time
            _, g_t_grad = jax.jvp(self.gamma, (t,), (jnp.ones_like(t),))   # gamma'(t)
            loss_diff = .5 * g_t_grad * loss_diff_mse
        else:                                                          # discrete time
            s = t - (1. / T)
            g_s = self.gamma(s)
            loss_diff = .5 * T * jnp.expm1(g_t - g_s) * loss_diff_mse

        return loss_recon, loss_klz, loss_diff

    def sample(self, i, T, z_t, conditioning, rng):
        eps = jax.random.normal(jax.random.fold_in(rng, i), z_t.shape)
        t = (T - i) / T; s = (T - i - 1) / T
        g_s, g_t = self.gamma(s), self.gamma(t)
        eps_hat = self.score_model(z_t, g_t * jnp.ones((z_t.shape[0],), g_t.dtype),
                                   conditioning, deterministic=True)
        a = nn.sigmoid(-g_s)
        c = - jnp.expm1(g_s - g_t)
        sigma_t = jnp.sqrt(nn.sigmoid(g_t))
        return jnp.sqrt(nn.sigmoid(-g_s) / nn.sigmoid(-g_t)) * (z_t - sigma_t * c * eps_hat) \
               + jnp.sqrt((1. - a) * c) * eps


class NoiseSchedule_NNet(nn.Module):
    config: object
    n_features: int = 1024

    def setup(self):
        g0, g1 = self.config.gamma_min, self.config.gamma_max
        self.l1 = DenseMonotone(1, kernel_init=constant_init(g1 - g0), bias_init=constant_init(g0))
        self.l2 = DenseMonotone(self.n_features)
        self.l3 = DenseMonotone(1, use_bias=False)

    @nn.compact
    def __call__(self, t):
        t = jnp.reshape(t, (-1, 1)) if not jnp.isscalar(t) and len(jnp.shape(t)) > 0 \
            else t * jnp.ones((1, 1))
        h = self.l1(t)
        _h = self.l3(2 * (nn.sigmoid(self.l2(2. * (t - .5))) - .5)) / self.n_features
        return jnp.squeeze(h + _h, axis=-1)


class DenseMonotone(nn.Dense):
    """Dense with non-negative weights -> monotone increasing."""
    @nn.compact
    def __call__(self, x):
        kernel = jnp.abs(self.param('kernel', self.kernel_init, (x.shape[-1], self.features)))
        y = x @ kernel
        if self.use_bias:
            y = y + self.param('bias', self.bias_init, (self.features,))
        return y


class Base2FourierFeatures(nn.Module):
    start: int = 6; stop: int = 8; step: int = 1
    @nn.compact
    def __call__(self, z):
        freqs = jnp.asarray(range(self.start, self.stop, self.step), z.dtype)
        w = jnp.tile((2. ** freqs * 2 * jnp.pi)[None, :], (1, z.shape[-1]))
        h = jnp.repeat(z, len(freqs), axis=-1) * w
        return jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)


def constant_init(value, dtype='float32'):
    return lambda key, shape, dtype=dtype: value * jnp.ones(shape, dtype)


def loss_fn(model, params, batch, rng):
    loss_recon, loss_klz, loss_diff = model.apply({'params': params}, **batch, rngs={"sample": rng})
    to_bpd = 1. / (np.prod(batch["images"].shape[1:]) * np.log(2.))
    return (loss_recon.mean() + loss_klz.mean() + loss_diff.mean()) * to_bpd
```
