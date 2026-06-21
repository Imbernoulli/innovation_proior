Denoising diffusion models have reached sample quality comparable to GANs using a stable, likelihood-based, non-adversarial training objective, but their sampling procedure is impractically slow. The generative chain is constructed as the reverse of a forward noising Markov chain that may use a thousand steps, and producing a single image requires running every one of those steps sequentially, each step a full forward pass through a large network. On a single GPU this can be roughly twenty hours to draw fifty thousand 32x32 images, versus under a minute for a GAN. The slowness appears structural: each true reverse conditional is only close to Gaussian when the corresponding forward step is small, which forces many small steps. The standard ancestral sampler is therefore faithful but rigid, and its injected per-step noise also prevents the terminal latent from being a deterministic encoding of the generated image.

The key observation is that this rigidity is not actually enforced by the training loss. After the usual variance-reduction algebra, the noise-prediction objective is a sum of per-timestep terms, each of which feeds the network a draw from the marginal q(x_t|x_0) = N(sqrt(alpha_t) x_0, (1-alpha_t) I). The loss never references the joint q(x_{1:T}|x_0); it only asks the network to denoise single marginal draws. With independent parameters per timestep, the minimizer is the same for any positive weighting of these terms. Therefore a trained noise predictor is committed only to the marginals, not to any particular Markovian joint. Any inference process with the same marginals is equally valid, and the same trained network optimizes it without retraining.

The method is Denoising Diffusion Implicit Models, or DDIM. It constructs the full family of generally non-Markovian inference processes that preserve the fixed marginals q(x_t|x_0). Parameterize the reverse conditional q(x_{t-1}|x_t,x_0) as a Gaussian whose mean is affine in x_0 and x_t and whose covariance is sigma_t^2 I. Requiring the marginal of x_{t-1} to be N(sqrt(alpha_{t-1}) x_0, (1-alpha_{t-1}) I) pins down the mean coefficient, giving q_sigma(x_{t-1}|x_t,x_0) = N(sqrt(alpha_{t-1}) x_0 + sqrt(1-alpha_{t-1}-sigma_t^2) (x_t - sqrt(alpha_t) x_0)/sqrt(1-alpha_t), sigma_t^2 I). A downward induction proves that every marginal is preserved for any choice of sigma_t with 0 <= sigma_t^2 <= 1-alpha_{t-1}. The corresponding variational objective is again a weighted noise-prediction MSE, so its minimizer coincides with the already-trained unweighted epsilon-MSE solution.

At sampling time, x_0 is unknown, but the noise predictor gives a predicted clean value f_theta(x_t) = (x_t - sqrt(1-alpha_t) epsilon_theta(x_t))/sqrt(alpha_t). Plugging this estimate into the reverse conditional yields the DDIM sampling step x_{t-1} = sqrt(alpha_{t-1}) f_theta(x_t) + sqrt(1-alpha_{t-1}-sigma_t^2) epsilon_theta(x_t) + sigma_t z, where z is fresh Gaussian noise. Writing sigma_t(eta) = eta sqrt((1-alpha_{t-1})/(1-alpha_t)) sqrt(1-alpha_t/alpha_{t-1}) gives a single stochasticity dial. Setting eta=1 recovers the original stochastic ancestral sampler on the full adjacent grid; setting eta=0 removes the noise term entirely and makes the generative map a deterministic pushforward of the initial latent x_T.

The deterministic eta=0 case is particularly valuable. It makes x_T a true latent code, so the same initial noise always yields the same image, enabling spherical interpolation in latent space and deterministic encoding of real images by running the procedure backward. It also removes any injected noise that a short chain would otherwise have to clean up, making wider steps safer. Because the objective is blind to forward chain length, the update can be run on a sub-sequence tau of the original T levels with S much smaller than T, giving the desired 10x-50x speedup using the frozen model. In the small-step limit the deterministic update is an Euler discretization of the probability-flow ODE in the rescaled coordinate x_bar = x/sqrt(alpha), confirming that fewer steps simply coarsen a smooth deterministic map rather than introduce new stochastic error.

```python
import torch

def alpha_at(alphas, t):
    # cumulative coefficient with alpha_0 := 1 (t = -1 maps to alpha_0)
    a = torch.cat([alphas.new_ones(1), alphas], dim=0)
    return a.index_select(0, t + 1).view(-1, 1, 1, 1)

def q_sample(x0, t, alphas, eps):
    a = alpha_at(alphas, t)
    return a.sqrt() * x0 + (1 - a).sqrt() * eps

def training_loss(model, x0, alphas):
    t = torch.randint(0, len(alphas), (x0.size(0),), device=x0.device)
    eps = torch.randn_like(x0)
    xt = q_sample(x0, t, alphas, eps)
    return ((model(xt, t.float()) - eps) ** 2).mean()

@torch.no_grad()
def sample(model, alphas, x, seq, eta=0.0):
    """
    model(x_t, t) -> predicted noise eps.
    alphas: cumulative signal coefficients alpha_1..alpha_T.
    x: initial latent at the largest selected noise level; usually x_T ~ N(0, I).
    seq: increasing zero-based indices into alphas; len(seq) = S sampling steps.
    eta: 0 -> deterministic DDIM; 1 -> stochastic ancestral endpoint.
    """
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = torch.ones(n, device=x.device) * i
        t_next = torch.ones(n, device=x.device) * j
        a = alpha_at(alphas, t.long())
        a_next = alpha_at(alphas, t_next.long())

        xt = xs[-1].to(x.device)
        eps = model(xt, t)
        x0 = (xt - eps * (1 - a).sqrt()) / a.sqrt()
        x0_preds.append(x0)

        c1 = eta * (((1 - a / a_next) * (1 - a_next) / (1 - a)).sqrt())
        c2 = ((1 - a_next) - c1 ** 2).sqrt()
        xt_next = a_next.sqrt() * x0 + c1 * torch.randn_like(xt) + c2 * eps
        xs.append(xt_next)
    return xs, x0_preds

def make_seq(num_timesteps, num_sampling_steps, kind="uniform"):
    if kind == "uniform":
        skip = max(num_timesteps // num_sampling_steps, 1)
        return list(range(0, num_timesteps, skip))
    if kind == "quad":
        return [int(s) for s in (
            torch.linspace(0, (0.8 * num_timesteps) ** 0.5, num_sampling_steps) ** 2
        ).tolist()]
    raise NotImplementedError(kind)

# x_T = torch.randn(batch, C, H, W)
# xs, x0_preds = sample(model, alphas, x_T, make_seq(1000, 50), eta=0.0)
# x_0 = xs[-1]
```

Training remains the standard unweighted epsilon-MSE; only the sampler changes. The same trained network serves the full ancestral chain, the deterministic few-step DDIM chain, and any intermediate eta, because all members of the marginal-preserving family share the same noise-prediction optimum. Choosing a short sub-sequence and eta=0 turns a thousand sequential network passes into a few tens while keeping the likelihood-based training and sample quality intact.
