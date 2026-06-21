I am handed a fixed batch of logged trajectories from a continuous-control system — a MuJoCo locomotion agent — with no ability to interact further, unknown dynamics $s' = f(s,a)$, and the task of emitting a good action at whatever state I find myself in. The textbook recipe is to fit a single-step dynamics model $\hat f(s_t, a_t)$ to the data and then hand it to a trajectory optimizer that searches for an action sequence maximizing $\sum_t r(s_t, a_t)$. I keep watching this fail the same way. To plan over a horizon I have to roll the learned model forward autoregressively, feeding each prediction back as the next input, so per-step errors compound and by the end of a long rollout the plan has drifted somewhere the real system would never go. Worse, the optimizer is strong and the model is differentiable and imperfect, so the search walks the plan straight into the regions where $\hat f$ is confidently wrong — exactly where the model reports high reward. The "optimal" plan comes back looking like an adversarial example, not a trajectory. The disease is structural: separating model from planner creates a seam between the model and reality, and a single-step accuracy objective is the wrong thing to train when the model will be used for long multi-step rollouts.

The reflex fixes each give up something I want. Weakening the planner to a gradient-free search (random shooting, cross-entropy) keeps it from exploiting the model but throws away the long-horizon reasoning that was the point. Going fully model-free — regressing a conservative policy or value function straight from the batch, as the offline-RL value methods do — avoids model exploitation but discards the trajectory-level structure in the data; conditioning on an arbitrary future goal or composing a new objective at test time stops being natural, and long-horizon credit assignment falls back on bootstrapped values rather than explicit lookahead. The sequence-model route — tokenize a trajectory and fit an autoregressive Transformer, generate left-to-right conditioned on a target return — fights the very thing I want to escape, because decision-making is anti-causal: the action I take now depends on where I am trying to end up, so if I want $p(s_1 \mid s_0, s_T)$ then $s_1$ depends on a *future* state, and a strictly causal decoder has the conditioning flowing the wrong way. Left-to-right generation also re-imports compounding error, since each token conditions the next. So before reaching for machinery I name what I actually want: producing a plan should *be* sampling from a model of trajectories, so there is no separate exploitable search bolted on top; the whole plan should be produced at once rather than unrolled in time, so neither rollout error nor causal ordering bites; the model should be trained for the quality of the whole trajectory it emits rather than single-step accuracy; and its base should be reward-free — just "what do plausible trajectories of this system look like" — so it stays on the data manifold by construction, since off-manifold plans are the failure I am fighting.

The method I propose is an unconditional, trajectory-level denoising diffusion planner, where sampling from the model *is* planning. Diffusion is the right engine because it generates continuous data well and does so *iteratively* rather than in one forward pass — and that iterative sampler, read through the score-based lens, behaves like Langevin ascent on the data density, so it naturally lands on high-density, in-distribution points. Let me lay the engine out from the ground up. The forward process slowly destroys a data point $x^0$: at each step add a little Gaussian noise and shrink the signal, $q(x^i \mid x^{i-1}) = \mathcal N(x^i; \sqrt{1-\beta_i}\,x^{i-1}, \beta_i I)$. The shrink by $\sqrt{1-\beta_i}$ rather than pure additive noise is what keeps the per-step scale controlled: with $x^{i-1}$ at unit coordinatewise variance, $\mathrm{Var}(x^i) = (1-\beta_i)\cdot 1 + \beta_i = 1$, so the signal stays at unit scale all the way down to $\mathcal N(0, I)$. Writing $\alpha_i = 1-\beta_i$ and $\bar\alpha_i = \prod_{j\le i}\alpha_j$, composing all forward steps telescopes — sums of independent Gaussians merge — into a closed form that lets me jump to any noise level in one shot,
$$q(x^i \mid x^0) = \mathcal N\big(x^i;\ \sqrt{\bar\alpha_i}\,x^0,\ (1-\bar\alpha_i) I\big), \qquad x^i = \sqrt{\bar\alpha_i}\,x^0 + \sqrt{1-\bar\alpha_i}\,\varepsilon,\ \ \varepsilon\sim\mathcal N(0,I),$$
so I never simulate the chain during training: pick a level $i$, draw one $\varepsilon$, corrupt $x^0$ directly.

The model is the reverse chain $p_\theta(x^{i-1}\mid x^i) = \mathcal N(x^{i-1}; \mu_\theta(x^i, i), \Sigma^i)$ from a standard-normal prior, fit by maximizing a variational bound on $\log p_\theta(x^0)$. Conditioning the forward posterior on $x^0$ via Bayes makes the bound's $q(x^i\mid x^0)/q(x^{i-1}\mid x^0)$ ratios telescope, leaving a sum whose middle terms are each a KL between two Gaussians — closed-form and low-variance. Those terms match $p_\theta(x^{i-1}\mid x^i)$ to the tractable forward posterior $q(x^{i-1}\mid x^i, x^0) = \mathcal N(\tilde\mu_i, \tilde\beta_i I)$ with
$$\tilde\beta_i = \frac{1-\bar\alpha_{i-1}}{1-\bar\alpha_i}\beta_i, \qquad \tilde\mu_i(x^i, x^0) = \frac{\sqrt{\bar\alpha_{i-1}}\,\beta_i}{1-\bar\alpha_i}\,x^0 + \frac{\sqrt{\alpha_i}\,(1-\bar\alpha_{i-1})}{1-\bar\alpha_i}\,x^i.$$
Fixing the reverse variance to an untrained constant $\Sigma^i = v_i I$ leaves only the mean to learn, and the KL collapses to $\tfrac{1}{2 v_i}\lVert \tilde\mu_i - \mu_\theta\rVert^2 + C$. The clean move is not to predict the mean directly but to reparameterize through the noise. Substituting $x^0 = (x^i - \sqrt{1-\bar\alpha_i}\,\varepsilon)/\sqrt{\bar\alpha_i}$ into $\tilde\mu_i$ collapses it to a function of $x^i$ and $\varepsilon$ alone, $\tilde\mu_i = \tfrac{1}{\sqrt{\alpha_i}}\big(x^i - \tfrac{\beta_i}{\sqrt{1-\bar\alpha_i}}\varepsilon\big)$, so I have the network output an estimate of the added noise and *define*
$$\mu_\theta(x^i, i) = \frac{1}{\sqrt{\alpha_i}}\Big(x^i - \frac{\beta_i}{\sqrt{1-\bar\alpha_i}}\,\varepsilon_\theta(x^i, i)\Big).$$
Because the shared $\tfrac{1}{\sqrt{\alpha_i}}$ and $\tfrac{\beta_i}{\sqrt{1-\bar\alpha_i}}$ factors cancel between $\tilde\mu_i$ and $\mu_\theta$, the per-term loss becomes a plain denoising regression, predict the noise that was mixed in. There is a deeper reason to derive in $\varepsilon$-space: $\varepsilon$ points from the clean data toward $x^i$, so up to scale it is the score, the gradient of the log density of the noised data — the network is a multi-scale score estimator and the reverse chain is the Langevin-like density ascent I wanted. The exact per-term weight $\beta_i^2/(2 v_i \alpha_i (1-\bar\alpha_i))$ is fiddly and varies with $i$; dropping it gives the simplified objective at a uniformly random level,
$$L_{\text{simple}}(\theta) = \mathbb E_{i,\,x^0,\,\varepsilon}\Big[\big\lVert \varepsilon - \varepsilon_\theta\big(\sqrt{\bar\alpha_i}\,x^0 + \sqrt{1-\bar\alpha_i}\,\varepsilon,\ i\big)\big\rVert^2\Big], \quad i\sim\mathrm{Uniform}\{1,\dots,N\},$$
and discarding the weight is not merely convenient but helpful: it is largest for the small-$i$, nearly-clean easy tasks, so dropping it shifts capacity toward the harder large-$i$ tasks where most structure must be rebuilt. Predicting $x^0$ directly is the same algebraic family — recover $\varepsilon = (x^i - \sqrt{\bar\alpha_i}\,x^0)/\sqrt{1-\bar\alpha_i}$ — so I can derive everything in $\varepsilon$-space and still use an $x^0$ target when the implementation chooses that branch.

The substance of turning this generator into a planner is the choice of what $x$ *is*. Modeling states and predicting actions separately would just reproduce the model-then-controller split I am killing. So I put states and actions into the same object and generate them jointly: a trajectory is a 2D array $\tau$ with one column per planning timestep $t = 0,\dots,H-1$ and, within a column, the $\mathrm{obs\_dim}$ state features stacked on the $\mathrm{act\_dim}$ action features, width $\mathrm{obs\_dim}+\mathrm{act\_dim}$. Now $x^0 = \tau$ and the diffusion chain denoises the entire plan at once — no left-to-right unrolling, so the anti-causal worry vanishes because every timestep sees every other through the network at each denoise step, and because actions are denoised under the same objective the model is trained for the controller it yields, not just for state prediction. The two axes of this array are *not* symmetric, and respecting that asymmetry is the whole architectural question. Along the horizon, the data is a time series — a plan can start anywhere, the same local dynamics pattern can occur anywhere in the window, nearby timesteps constrain each other — so that axis wants translation-equivariance and locality, i.e. convolution. Along the feature axis the coordinates are heterogeneous: "the third state dimension" and "the first action dimension" are fixed, different things with no translation structure. So I convolve along the horizon axis only and treat the features as channels: a 1D temporal U-Net with $\mathrm{obs\_dim}+\mathrm{act\_dim}$ channels. A 2D image convolution would be wrong precisely because it would impose equivariance across the feature axis, mixing state-dim-3 with action-dim-1 as if they were neighboring pixels. Global coherence over a long horizon comes from two compounding mechanisms despite each convolution being local: the U-Net's multi-scale down/up-sampling gives deep blocks a receptive field spanning the whole plan, and — the part special to an iterative sampler — each of the $N$ denoising steps need only enforce *local* consistency, which applied repeatedly and propagated becomes global coherence. Concretely each block is a temporal-convolution residual block (two `Conv1d` along the horizon, GroupNorm — chosen because batch sizes are modest and I want batch-independent normalization — and Mish, with a residual), conditioned on the noise level through a sinusoidal embedding of the diffusion index $i$ passed through a small MLP and added in each block; kernel size 5 for a bit more receptive field, and no attention, since for these low-dimensional locomotion trajectories convolutional locality plus iterative sampling suffices. The repeated downsampling and skip concatenations require the horizon to be a power of two, which a locomotion $H = 32$ satisfies.

Two further choices finish it. The chain is short. For megapixel images people run $N = 1000$ linear-schedule steps, but a locomotion window is far lower-dimensional and smoother, with much less structure to destroy and rebuild, so on the order of $N = 20$ steps suffice — and with a chain that short the schedule matters more, because a linear schedule tuned for $1000$ steps would wreck the signal-to-noise ratio over $20$. I therefore use a normalized cosine schedule, carried directly in cumulative $(\alpha, \sigma)$ form where $x^i = \alpha_i x^0 + \sigma_i \varepsilon$,
$$\alpha_i = \frac{\cos\!\big(\tfrac{\mathrm{clip}(t_i,0,0.9946)+s}{1+s}\cdot\tfrac{\pi}{2}\big)}{\cos\!\big(\tfrac{s}{1+s}\cdot\tfrac{\pi}{2}\big)},\ \ s = 0.008, \qquad \sigma_i = \sqrt{1-\alpha_i^2},$$
which keeps the SNR gracefully spread over a short chain and avoids an exactly zero signal scale at the endpoint. The other choice is the only conditioning an unguided planner cannot do without: every plan must begin at the *current* state. When I stand at observation $s$, the sampled plan must have $s_0 = s$; I am not free to imagine a different start. That is exactly inpainting — fix the known coordinates, let the model fill the rest consistently — so I build a `fix_mask` that is $1$ on $\tau[0, :\mathrm{obs\_dim}]$ and clamp those coordinates to the current observation at *every* step of the reverse chain, and since they are given rather than predicted I zero their training loss too. This is a hard start-state constraint, not reward — no value, no return, no preference for high-reward plans enters; the base model is the dynamically-plausible, reward-agnostic plan distribution, and I want it clean. One small loss weighting: in receding-horizon control I sample a plan, execute only its first action $\tau[0, \mathrm{obs\_dim}:]$, then replan, so that one coordinate block is the only thing actually run and deserves to be reconstructed especially well; I up-weight its loss by a factor of $10$ and leave the rest of the array at unit weight.

Training is then: stack a logged $(\mathrm{obs}, \mathrm{act})$ window into $x^0$, draw a random level $i$ per example, corrupt in one shot $x^i = \alpha_i x^0 + \sigma_i \varepsilon$, re-clamp the masked start-state coordinates to their true values, run the network, and regress to $\varepsilon$ if predicting noise or to $x^0$ if predicting data — same corruption, same mask, same weighted squared error, only the target changes — multiplying by the per-coordinate loss weight and by $(1-\texttt{fix\_mask})$ to kill the clamped coordinates, then step and update an EMA of the weights (decay $0.9999$) for a stable sampler. To match the CleanDiffuser configuration I take the $x^0$-prediction route, `predict_noise=False`, and recover the noise in closed form $\varepsilon_\theta = (x^i - \alpha_i\,x^0_\theta)/\sigma_i$. Planning is sampling: allocate the array, set its start-state coordinates to the current observation, initialize the rest to Gaussian noise scaled by a temperature, and run the reverse chain. From the prediction I get $\varepsilon_\theta$ (directly, or via $\varepsilon_\theta = (x^i - \alpha_i\,x^0_\theta)/\sigma_i$), and the ancestral DDPM step in $(\alpha, \sigma)$ form from level $i$ to $i-1$ is
$$x^{i-1} = \frac{\alpha_{i-1}}{\alpha_i}\big(x^i - \sigma_i\varepsilon_\theta\big) + \sqrt{\sigma_{i-1}^2 - \mathrm{std}_i^2}\,\varepsilon_\theta + \mathrm{std}_i\,z,\qquad \mathrm{std}_i = \frac{\sigma_{i-1}}{\sigma_i}\sqrt{1-\Big(\frac{\alpha_i}{\alpha_{i-1}}\Big)^2},$$
where the first term is $\alpha_{i-1}$ times the implied clean trajectory $(x^i - \sigma_i\varepsilon_\theta)/\alpha_i = x^0_\theta$, the second adds back the correlated noise needed for the marginal at level $i-1$ to match the forward process, and the fresh noise $\mathrm{std}_i\,z$ is added at every step except the last ($z = 0$ at $i = 1$, since the final step should land on a clean trajectory). After each step I re-clamp the start-state coordinates, holding the inpainting constraint through the whole chain. With no reward guidance there is no classifier gradient to inject and no candidate re-ranking: I draw a single plan and read off its first action, $\tau[0, \mathrm{obs\_dim}:]$, clipped to the valid action range, and execute that. The slot in the harness is therefore a temporal U-Net wrapped by the discrete diffusion SDE, exposing only `update(obs, act)` and `plan(current_obs)`.

```python
import torch
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import JannerUNet1d


class PlannerModel:
    """Trajectory-level diffusion planner with start-state inpainting and no reward guidance."""
    def __init__(self, obs_dim, act_dim, horizon, device,
                 model_dim=32, dim_mult=(1, 2, 2, 2), diffusion_steps=20,
                 predict_noise=False, action_loss_weight=10.0,
                 ema_rate=0.9999, lr=2e-4):
        self.obs_dim, self.act_dim, self.horizon, self.device = obs_dim, act_dim, horizon, device
        self.diffusion_steps = diffusion_steps
        if horizon & (horizon - 1) != 0:
            raise ValueError("JannerUNet1d requires a power-of-two horizon.")

        self.net = JannerUNet1d(
            obs_dim + act_dim,
            model_dim=model_dim,
            emb_dim=model_dim,
            dim_mult=list(dim_mult),
            timestep_emb_type="positional",
            attention=False,
            kernel_size=5,
        )

        self.fix_mask = torch.zeros((horizon, obs_dim + act_dim))
        self.fix_mask[0, :obs_dim] = 1.0
        self.loss_weight = torch.ones((horizon, obs_dim + act_dim))
        self.loss_weight[0, obs_dim:] = action_loss_weight

        self.agent = DiscreteDiffusionSDE(
            nn_diffusion=self.net,
            nn_condition=None,
            fix_mask=self.fix_mask,
            loss_weight=self.loss_weight,
            classifier=None,
            ema_rate=ema_rate,
            optim_params={"lr": lr, "weight_decay": 1e-5},
            diffusion_steps=diffusion_steps,
            noise_schedule="cosine",
            predict_noise=predict_noise,
            device=device,
        )
        self.optimizer = self.agent.optimizer

    def update(self, obs, act):
        x = torch.cat([obs, act], dim=-1).to(self.device)
        return self.agent.update(x)

    @torch.no_grad()
    def plan(self, current_obs, temperature=0.5, sample_steps=None):
        current_obs = current_obs.to(self.device)
        n = current_obs.shape[0]
        prior = torch.zeros((n, self.horizon, self.obs_dim + self.act_dim), device=self.device)
        prior[:, 0, :self.obs_dim] = current_obs
        traj, _ = self.agent.sample(
            prior=prior,
            solver="ddpm",
            n_samples=n,
            sample_steps=sample_steps or self.diffusion_steps,
            use_ema=True,
            temperature=temperature,
            w_cfg=0.0,
            w_cg=0.0,
        )
        return traj[:, 0, self.obs_dim:].clip(-1.0, 1.0)
```
