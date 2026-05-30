# Variational Diffusion Models (VDM)

## Problem

Gaussian-corruption generative models produce excellent samples but trail autoregressive models on
likelihood (bits/dim). VDM makes such a model a state-of-the-art *likelihood* model by (1) optimizing the
exact variational bound rather than a quality-weighted surrogate, (2) learning the noise schedule, and
(3) adding Fourier input features for fine-scale detail.

## Key ideas

1. **Diffusion as an infinitely-deep VAE.** Forward process `q(z_t|x) = N(α_t x, σ_t² I)` with
   `SNR(t) = α_t²/σ_t²` strictly decreasing. The generative model inverts it. The negative ELBO splits as
   `−VLB = KL(q(z_1|x)‖N(0,I)) + E_q[−log p(x|z_0)] + L_T` (prior + reconstruction + diffusion loss).

2. **The diffusion loss collapses to SNR.** Choosing `p(z_s|z_t) = q(z_s|z_t, x=x̂_θ(z_t,t))` gives equal
   variances inside each KL, so each transition KL `= ½(SNR(s) − SNR(t))‖x − x̂_θ‖²`. Hence
   `L_T = (T/2) E_{ε,i}[(SNR(s) − SNR(t))‖x − x̂_θ(z_t,t)‖²]`.

3. **More steps is always better → continuous time.** `L_2T − L_T < 0` for a good denoiser (upper Riemann
   sum of a decreasing integrand). Taking `T → ∞`:
   `L_∞ = −½ E_{ε,t}[SNR'(t)‖x − x̂_θ(z_t,t)‖²]`.

4. **SNR-endpoint invariance.** Change variables `v = SNR(t)`:
   `L_∞ = ½ E_ε ∫_{SNR_min}^{SNR_max} ‖x − x̃_θ(z_v,v)‖² dv`, with `SNR_min=SNR(1)`, `SNR_max=SNR(0)`.
   Since `z_v = α_v(x + ε/√v)`, the integrand depends on the schedule only through `v`; the continuous-time
   bound depends on the schedule **only through its two endpoints**, invariant to the shape between. This also
   proves variance-preserving and variance-exploding processes are the same continuous-time model (up to a
   trivial latent rescaling).

5. **Parameterizations.** `x̂ = (z_t − σ_t ε̂)/α_t`; score `s_θ = (α_t x̂ − z_t)/σ_t² = −ε̂/σ_t`. Predict
   noise `ε̂_θ`. With VP spec and `γ(t) = −log SNR(t)`:
   `L_∞ = ½ E_{ε,t}[γ'(t)‖ε − ε̂_θ‖²]`, `L_T = (T/2) E_{ε,i}[(exp(γ(t)−γ(s)) − 1)‖ε − ε̂_θ‖²]`
   (use `expm1` → stable in fp32). Consistency holds via denoising score matching: at the optimum
   `s_θ^* = ∇ log q(z_t)`.

6. **Learn the schedule for variance, not loss.** Endpoints `γ_0,γ_1` are trained against the VLB; the
   interior of the monotone schedule `γ_η(t)` is trained to minimize the estimator variance via SGD on the
   **squared** loss (since `E[L^{MC}²] = L² + Var` and `L²` is interior-independent,
   `∇_η E[L^{MC}²] = ∇_η Var`), plus low-discrepancy/antithetic `t` sampling.

7. **Fourier features.** Append `sin(2^n π z), cos(2^n π z)` (`n∈{6,7}`, i.e. `range(6,8)` in the implementation) so the denoiser can fit fine pixel
   detail; lets `SNR_max` go high and improves likelihood substantially.

8. **Weighted loss / connections.** `L_∞(x,w)=½E∫ w(v)‖x−x̃_θ‖² dv`; `w=1` is the VLB. The "simple"
   noise-MSE objective corresponds to `w=1/γ'(t)` (more weight on low SNR → better FID, worse likelihood).
   For a fixed `T`, the discrete model is a hierarchical latent model whose `−VLB` is the bits-back codelength,
   giving near-optimal lossless compression.

## Code

Faithful to the JAX/Flax implementation.

```python
import jax, jax.numpy as jnp, numpy as np
import flax.linen as nn


class VDM(nn.Module):
    config: object

    def setup(self):
        self.encdec = EncDec(self.config)
        self.score_model = ScoreUNet(self.config)     # predicts eps_hat(z_t, gamma_t)
        self.gamma = NoiseSchedule_NNet(self.config)  # monotone gamma(t) = -log SNR(t)

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
        t0 = jax.random.uniform(rng1)
        t = jnp.mod(t0 + jnp.arange(0., 1., step=1. / n_batch), 1.)   # antithetic / low-discrepancy
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

`EncDec` encodes integer pixels to `[-1,1]` and decodes via a discretized-Gaussian categorical head
(`logprob` = log-softmax of `-0.5((z - x_vals)·e^{-g_0/2})²` over the value grid). `ScoreUNet` is a U-Net
that concatenates `Base2FourierFeatures(z)` to `z`, conditions on `γ_t` via a sinusoidal embedding, and
predicts `ε̂` (with a `+z` base-measure skip). The diffusion-loss branches implement
`½γ'(t)‖ε−ε̂‖²` (continuous) and `½T·expm1(γ_t−γ_s)‖ε−ε̂‖²` (discrete).
