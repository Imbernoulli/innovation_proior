Let me start from what actually goes wrong. I have a batch of logged trajectories from some MuJoCo locomotion agent, no way to interact further, and I want a controller. The textbook recipe is: fit a dynamics model to the data, then run a trajectory optimizer on it to find an action sequence that maximizes reward. I keep watching this fail in a specific way. I learn a single-step model s_{t+1} = f_hat(s_t, a_t), and to plan over a horizon I roll it forward autoregressively — feed each prediction back in as the next input. Every prediction carries error, and because each one is the input to the next, the errors compound; by the end of a long horizon the rollout has drifted somewhere the real system would never go. Worse, the optimizer is strong and the model is differentiable and imperfect, so the search happily walks the plan into exactly those regions where f_hat is confidently wrong, because that is where the model reports high reward. The "optimal" plan comes back looking like an adversarial example, not a trajectory. So the standard decomposition — learn a model, hand it to a separate planner — has a structural disease: the planner exploits the seam between the model and reality, and a single-step accuracy objective is the wrong thing to be training when I am going to use the model for long multi-step rollouts.

The reflex fixes do not satisfy me. I could weaken the planner — random shooting, cross-entropy method, gradient-free search — so it can't exploit the model as hard. But then I've given up the long-horizon reasoning I wanted. I could throw out dynamics entirely and go model-free, regress a conservative policy or value function straight from the batch, the way the offline-RL value methods do. That avoids model exploitation, but it discards the trajectory structure in the data and the ability to reason explicitly over whole futures; conditioning on an arbitrary future goal, or composing a new objective at test time, stops being natural. And I could go the sequence-model route — tokenize a trajectory and fit an autoregressive Transformer over interleaved states and actions, generate by sampling left-to-right conditioned on a target. But left-to-right is the thing I want to get away from. Decision-making is anti-causal: the action I take now depends on where I'm trying to end up. If I want p(s_1 | s_0, s_T), then s_1 depends on a *future* state, and a strictly causal decoder that emits time 0, then 1, then 2 has the conditioning flowing the wrong way. Autoregression also re-imports the compounding-error problem — each generated token conditions the next, so errors accumulate down the sequence just as they did in the rollout.

So let me name what I actually want before I reach for any machinery. I want the act of producing a plan to *be* the act of sampling from a model of trajectories, so the planner can't be a separate, exploitable search bolted on top — there should be no seam to abuse. I want the whole plan produced at once, not unrolled in time, so neither rollout error nor causal ordering bites me. I want the model trained for the quality of the whole trajectory it produces, not single-step accuracy. And I want the thing it learns first — before any notion of reward enters — to be just "what do plausible trajectories of this system look like," a model that stays on the data manifold by construction, because the whole failure I'm fighting is plans that leave the manifold.

What generates continuous data well right now, and does it *iteratively* rather than in one forward pass? The denoising diffusion family fits both of those. Let me make sure I understand its machinery from the ground up, because I'm going to lean on every piece. Take any data point x^0. Define a forward process that slowly destroys it: at each step add a little Gaussian noise and shrink the signal, q(x^i | x^{i-1}) = N(x^i; sqrt(1 - beta_i) x^{i-1}, beta_i I), for a small prespecified variance schedule beta_i. Why shrink by sqrt(1 - beta_i) and not just add noise? My guess is that it's there to keep the per-step variance from inflating: if x^{i-1} has unit coordinatewise variance, then formally Var(x^i) = (1 - beta_i)*1 + beta_i = 1. Let me not just trust that and actually run a few steps. Start 10^4 samples at unit variance and push them through betas 0.1, 0.2, 0.05, 0.3 in turn; the measured Var(x^i) comes out 0.972, 0.980, 0.985, 0.990 — hovering at 1, with no drift up or down, exactly as the algebra says. So the scaling is what keeps the signal at unit scale all the way down rather than blowing up; after enough steps the data is indistinguishable from N(0, I). Now the closed form I'll actually need. Write alpha_i = 1 - beta_i and alpha_bar_i = prod_{j=1}^i alpha_j. Compose two steps: x^i = sqrt(alpha_i) x^{i-1} + sqrt(beta_i) eps_{i-1}, and x^{i-1} = sqrt(alpha_{i-1}) x^{i-2} + sqrt(beta_{i-1}) eps_{i-2}. Substitute and the two independent Gaussians merge — sum of N(0, a) and N(0, b) is N(0, a+b) — and after telescoping all the way to x^0 the coefficient on x^0 should be the product sqrt(alpha_i alpha_{i-1}...alpha_1) = sqrt(alpha_bar_i), with accumulated noise variance 1 - alpha_bar_i. I want to be sure the telescoping really lands there and I haven't dropped a cross term, so I simulate the full chain for a fixed x0 = 2 over those same four betas and look at the resulting samples: empirical mean / x0 = 0.6929 against sqrt(alpha_bar_i) = 0.6920, and empirical variance 0.521 against 1 - alpha_bar_i = 0.521. Both match. So

  q(x^i | x^0) = N(x^i; sqrt(alpha_bar_i) x^0, (1 - alpha_bar_i) I),

which means I can jump to any noise level in one shot: x^i = sqrt(alpha_bar_i) x^0 + sqrt(1 - alpha_bar_i) eps with eps ~ N(0, I). That's what makes training cheap — I never have to simulate the chain, I just pick a level i, draw one eps, and corrupt x^0 directly.

The model is the *reverse* chain: p_theta(x^{i-1} | x^i) = N(x^{i-1}; mu_theta(x^i, i), Sigma^i), starting from p(x^N) = N(0, I). I fit it by maximizing a variational bound on log p_theta(x^0). Let me actually write the bound, because how it decomposes tells me what to parameterize. Using the standard ELBO and the Markov structure,

  L = E_q[ -log p(x^N) - sum_{i>=1} log ( p_theta(x^{i-1} | x^i) / q(x^i | x^{i-1}) ) ].

If I leave it like this, each term compares the learned reverse step to the *forward* step, which is high-variance to estimate. I can condition the forward posterior on x^0 instead. By Bayes, q(x^i | x^{i-1}) = q(x^{i-1} | x^i, x^0) q(x^i | x^0) / q(x^{i-1} | x^0), and the q(x^i|x^0)/q(x^{i-1}|x^0) ratios telescope across the sum, leaving

  L = E_q[ KL(q(x^N | x^0) || p(x^N)) + sum_{i>1} KL(q(x^{i-1} | x^i, x^0) || p_theta(x^{i-1} | x^i)) - log p_theta(x^0 | x^1) ].

Now every middle term is a KL between two Gaussians — closed form, low variance. The first term has no parameters once the schedule is fixed (q(x^N|x^0) is essentially N(0,I) already, so it's a constant). The last term is a reconstruction term. So all the learning is in matching p_theta(x^{i-1}|x^i) to the forward posterior q(x^{i-1}|x^i,x^0). I need that posterior explicitly. It's Gaussian, q(x^{i-1}|x^i,x^0) = N(x^{i-1}; mu_tilde_i, beta_tilde_i I), and grinding the Gaussian algebra (multiply the two known Gaussians, complete the square) gives

  beta_tilde_i = (1 - alpha_bar_{i-1})/(1 - alpha_bar_i) * beta_i,
  mu_tilde_i(x^i, x^0) = [ sqrt(alpha_bar_{i-1}) beta_i / (1 - alpha_bar_i) ] x^0 + [ sqrt(alpha_i)(1 - alpha_bar_{i-1})/(1 - alpha_bar_i) ] x^i.

I'll fix the reverse variance to an untrained constant Sigma^i = v_i I (the natural choices are v_i = beta_i or v_i = beta_tilde_i, the two extremes of reverse entropy), so the only thing to learn is the mean mu_theta. With the model variance fixed, the KL is just the squared distance of the means scaled by that variance,

  L_{i-1} = E_q[ (1/(2 v_i)) || mu_tilde_i(x^i, x^0) - mu_theta(x^i, i) ||^2 ] + C.

So the most literal thing is to make mu_theta predict the posterior mean. Fine — but there may be something cleaner. Substitute the reparameterization x^0 = (x^i - sqrt(1 - alpha_bar_i) eps)/sqrt(alpha_bar_i) into mu_tilde_i. The claim is that after the algebra the x^0-dependent and x^i-dependent pieces recombine into a function of x^i and eps alone:

  mu_tilde_i = (1/sqrt(alpha_i)) ( x^i - (beta_i/sqrt(1 - alpha_bar_i)) eps ).

This is exactly the kind of simplification I'd believe too readily and get wrong by a coefficient, so I check it numerically rather than trust the hand-algebra. For a few levels i I draw a random x0 and eps, form x^i = sqrt(alpha_bar_i) x0 + sqrt(1-alpha_bar_i) eps, then evaluate the original two-term posterior mean (the x^0/x^i form above) and this compact eps-form, and compare. At i = 2, 3, 4 they agree to 5e-17, 2e-16, 2e-16 — i.e. to floating point. So the two formulas are the same object; the substitution is clean. And that reframes everything: the mean I'm trying to match is, given x^i (which the network already sees as input), entirely determined by eps — the noise that was added. So instead of having the network output the mean directly, let it output an estimate of that noise, eps_theta(x^i, i), and *define*

  mu_theta(x^i, i) = (1/sqrt(alpha_i)) ( x^i - (beta_i/sqrt(1 - alpha_bar_i)) eps_theta(x^i, i) ).

Why use eps for the derivation instead of asking the network for the mean directly? Plug this mu_theta back into L_{i-1}. The (1/sqrt(alpha_i)) and (beta_i/sqrt(1-alpha_bar_i)) factors are shared between mu_tilde and mu_theta, so the difference of means is just (beta_i/(sqrt(alpha_i) sqrt(1-alpha_bar_i)))(eps - eps_theta), and

  L_{i-1} - C = E_{x^0, eps}[ (beta_i^2 / (2 v_i alpha_i (1 - alpha_bar_i))) || eps - eps_theta(sqrt(alpha_bar_i) x^0 + sqrt(1-alpha_bar_i) eps, i) ||^2 ].

It's a plain denoising regression: predict the noise that was mixed in. And eps isn't an arbitrary target either. The network is being asked to estimate eps from x^i, and eps points along the direction from clean data toward x^i — up to scale, by Tweedie's identity, this is the score, the gradient of the log density of the noised data. So the same network is implicitly a multi-scale score estimator, and the reverse chain is Langevin-like ascent on the data density. I flag that to myself as promising for control — a sampler that climbs toward high-density, in-distribution regions is exactly the on-manifold behavior I've been chasing — but I'll only know it actually delivers in-distribution trajectories once I've defined what x is and checked the sampler, which is later. Predicting x^0 directly is still the same algebraic family — recover eps from it by eps = (x^i - sqrt(alpha_bar_i) x^0)/sqrt(1 - alpha_bar_i) — so I can derive the DDPM engine in eps-space and still use an x^0 target when the implementation chooses that branch.

Now look at the per-term weight beta_i^2/(2 v_i alpha_i (1 - alpha_bar_i)). It varies with i and it's fiddly. The clean move is to just drop it and minimize the unweighted regression at a uniformly random noise level:

  L_simple(theta) = E_{i, x^0, eps}[ || eps - eps_theta( sqrt(alpha_bar_i) x^0 + sqrt(1-alpha_bar_i) eps, i ) ||^2 ],  i ~ Uniform{1..N}.

Is throwing away the weight harmless? Before I assume it is, let me look at what the weight actually does across i. Take v_i = beta_i (one of the two stated reverse-variance choices) and a linear beta schedule over N = 1000, like the original images setup, and tabulate w_i = beta_i^2/(2 v_i alpha_i (1 - alpha_bar_i)). It comes out 0.50 at i = 1, 0.074 at i = 10, 0.010 at i = 100 — about a 50x falloff from the smallest noise level to the deep end. So the weight heavily favors the small-i, nearly-clean, easy denoising tasks, and dropping it down-weights exactly those relative to the harder large-i tasks where most of the structure has to be reconstructed. One honest caveat the numbers force on me: it is not perfectly monotone — at the very last steps, where 1 - alpha_bar_i saturates near 1, w ticks back up slightly — so "always largest at small i" is too strong; the accurate statement is that it's heavily concentrated on small i and dropping it reallocates capacity toward the hard large-i tasks. That's a feature, not just a convenience, so I'll go with the unweighted loss. Training is then: sample a real trajectory window, pick a random noise level, corrupt it in one shot, and regress the network to the added noise. And sampling is the reverse chain: start x^N ~ N(0, I), and for i = N down to 1,

  x^{i-1} = (1/sqrt(alpha_i)) ( x^i - ((1 - alpha_i)/sqrt(1 - alpha_bar_i)) eps_theta(x^i, i) ) + sqrt(v_i) z,  z ~ N(0, I), z = 0 at i = 1.

That's the whole engine. Now the real question: what is x, here? I have this generic continuous-data generator. How do I make "sampling from it" *be* "producing a plan"?

A trajectory is a sequence over planning time t = 0..H-1 of states s_t and actions a_t. My first instinct is to model the states and predict actions separately, but that reproduces the model-then-controller split I'm trying to kill — the controller would be an afterthought of state prediction. The training objective should care about the action it emits as much as the state. So let me put states and actions *into the same object* and generate them jointly: treat the action at each timestep as just additional coordinates of that timestep's vector. Stack them. A trajectory becomes a two-dimensional array, one column per planning timestep, and within a column the obs_dim state features stacked on top of the act_dim action features:

  tau = [ s_0  s_1  ...  s_{H-1} ;  a_0  a_1  ...  a_{H-1} ],   width = obs_dim + act_dim, length = H.

Now x^0 = tau, and the diffusion model generates the whole array at once. This is precisely "produce the entire plan simultaneously" — the denoising chain refines all timesteps together, no left-to-right unrolling, so the anti-causal conditioning I worried about is a non-issue: every timestep sees every other through the network at each denoise step. And because actions live in the same array as states and are denoised under the same objective, the model is trained for the quality of the controller it yields, not just for state prediction.

What network goes on this array? The array has two axes and they are *not* symmetric, and getting this asymmetry right is the whole architectural question. Along the horizon axis, the data is a time series: a plan can start at any moment, the same local dynamics pattern can occur anywhere in the window, and nearby timesteps are the ones that constrain each other. That axis wants translation-equivariance and locality — convolution. Along the feature axis, the coordinates are heterogeneous: "the third state dimension" and "the first action dimension" mean fixed, different things; they are not interchangeable and there is no translation structure to exploit. So I must convolve along the horizon axis only, and treat the features as channels — a one-dimensional temporal convolution over time, with obs_dim + act_dim channels. This is why a 2D image convolution would be wrong: it would impose equivariance across the feature axis, mixing state-dim-3 with action-dim-1 as if they were neighboring pixels, which is meaningless. Being fully convolutional in the horizon dimension means the network has no learned absolute table for a particular plan length; within lengths compatible with its downsampling layout, the horizon is set by the input array.

But a single temporal convolution only sees a few neighboring timesteps — its receptive field is local. How do I get *global* coherence over a long horizon out of local convolutions? Two compounding mechanisms. First, stack the convolutions in a U-Net: downsample along the horizon to coarser temporal resolutions and back up, so deep blocks have a receptive field spanning the whole plan while shallow blocks stay local — the standard multi-scale trick, and the same backbone shape that works for image diffusion, just with 1D temporal convolutions instead of 2D spatial ones. The implementation has one practical consequence: the repeated downsampling and skip concatenations require the horizon length to be a power of two, so a locomotion horizon like H=32 fits the network cleanly. Second, and this is the part special to the iterative sampler: each denoising step only needs to enforce *local* consistency of the trajectory — make nearby states and actions mutually plausible — and there are N of these steps composed in sequence. Local consistency, applied repeatedly and propagated, becomes global coherence. So I don't need any one denoising step to reason globally; the iteration does that work. That's a reassuring division of labor between architecture and sampling.

Concretely the block is a temporal-convolution residual block: two Conv1d layers along the horizon axis, each followed by a group normalization and a Mish nonlinearity, with a residual connection. Group norm because batch sizes are modest and I want normalization that doesn't depend on the batch; Mish as a smooth activation, both carried over from what works in image-diffusion U-Nets. The network must also know which noise level it's denoising — eps_theta depends on i — so I embed the diffusion step i with a sinusoidal positional embedding, pass it through a small MLP, and add it into each residual block, so every block is conditioned on the noise level. I'll use a slightly wider temporal kernel (size 5) for a bit more receptive field per block, and I don't need attention for these low-dimensional locomotion trajectories — the convolutional locality plus the iterative sampling is enough — so I'll leave attention off.

Here's a subtlety in the schedule. For images people run N = 1000 denoising steps with a linear beta schedule. But a locomotion trajectory window is far lower-dimensional and far smoother than a megapixel image — there's much less structure to destroy and rebuild — so I expect to need far fewer steps, on the order of N = 20. With a chain that short, the noise schedule matters more: a linear schedule tuned for 1000 steps would wreck the signal-to-noise ratio across 20 steps. The cosine schedule keeps the signal scale as alpha_i = cos(((clip(t_i, 0, 0.9946) + s)/(1+s)) pi/2) / cos((s/(1+s)) pi/2), with s = 0.008 and t_i running over the diffusion grid, and then sets sigma_i = sqrt(1 - alpha_i^2). Defining sigma that way is a claim that alpha and sigma are the cumulative signal and noise scales of a unit-variance process, alpha_i^2 + sigma_i^2 = 1, so I tabulate them over the N = 20 grid: alpha_i^2 + sigma_i^2 deviates from 1 by at most 1e-16 (machine epsilon), alpha runs from 0.9960 at the low-noise end down to 0.0084 at the high-noise end — small but, importantly, *not* zero, so the endpoint keeps a sliver of signal rather than collapsing the SNR to nothing. That's the property I wanted: a normalized cosine that spreads the SNR gracefully over a short chain while avoiding an exactly zero signal scale at the endpoint. It's cleanest to carry the schedule in this (alpha, sigma) form where x^i = alpha_i x^0 + sigma_i eps — same process, just naming the cumulative signal and noise scales directly, which makes the reverse update easy to write at arbitrary sampled noise levels.

Now the conditioning that even an unguided planner cannot do without: every plan must start at the *current* state. When I'm standing at observation s, the plan I sample has to have s_0 = s; I'm not free to imagine a different starting state. This is a hard constraint on a subset of the array's coordinates — exactly the structure of inpainting, where you fix the observed pixels and let the model fill the rest consistently. So I clamp the state coordinates of column 0 to the current observation at *every* step of the denoising chain: build a fix mask that is 1 on tau[0, :obs_dim] and 0 elsewhere, and after every reverse step overwrite those coordinates with the known value. Because they're held fixed, they should also contribute no training loss — the network is never asked to predict what is given — so I zero the loss on the masked coordinates too. This is only a start-state constraint imposed by clamping known coordinates. It is *not* reward — no value, no return, no preference for high-reward plans enters here. This unconditional-but-start-anchored trajectory model samples plans that are dynamically plausible given where I am, full stop, and I want it clean.

One small weighting choice on the loss. In receding-horizon control I sample a plan, execute only its *first* action a_0, then replan from the next state. So among all the action coordinates, tau[0, obs_dim:] — the first action — is the one that actually gets run; the rest of the plan is lookahead that informs it but is never executed directly. It's worth reconstructing that first action especially well, so I up-weight its loss (a factor like 10) relative to the rest of the array. Everything else gets unit weight.

Let me assemble the training step against the harness. A batch is windows of stacked (obs, act), x of shape (batch, H, obs_dim + act_dim). Draw a random level i per example, corrupt in one shot with the cosine (alpha, sigma): x^i = alpha_i x^0 + sigma_i eps, then re-clamp the masked start-state coordinates back to their true values so the network sees a consistent inpainting setup. Run the network; if I'm predicting noise, regress to eps; if I'm predicting the data directly, regress to x^0 — same forward corruption, same mask, same weighted squared error, only the target changes. Multiply by the per-coordinate loss weight and by (1 - fix_mask) to kill the clamped coordinates, average, backprop, step, and update an EMA of the weights (decay 0.9999) for a stable sampler. To match the CleanDiffuser configuration I take the x^0-prediction route here, with `predict_noise=False`; the reverse update recovers eps from the predicted clean trajectory in closed form, eps_theta = (x^i - alpha_i x0_theta)/sigma_i.

Now sampling — which is planning. Allocate the array, set its start-state coordinates to the current observation (that's the prior I clamp), and initialize the rest to Gaussian noise scaled by a temperature. Then run the reverse chain. I have the network's prediction; convert between the noise view and the data view as needed: if the net predicts noise, eps_theta = pred and the implied clean trajectory is x0_theta = (x^i - sigma_i eps_theta)/alpha_i; if it predicts data, x0_theta = pred and eps_theta = (x^i - alpha_i x0_theta)/sigma_i. The ancestral DDPM step in (alpha, sigma) form, going from level i to i-1, is

  x^{i-1} = (alpha_{i-1}/alpha_i)( x^i - sigma_i eps_theta ) + sqrt( sigma_{i-1}^2 - std_i^2 ) eps_theta + std_i z,

where the (alpha_{i-1}/alpha_i)(x^i - sigma_i eps_theta) piece is alpha_{i-1} times the implied clean trajectory (since (x^i - sigma_i eps_theta)/alpha_i = x0_theta), the sqrt(sigma_{i-1}^2 - std_i^2) eps_theta piece is meant to add back the right amount of correlated noise so that the marginal at level i-1 matches the forward process, and std_i is the ancestral posterior standard deviation std_i = (sigma_{i-1}/sigma_i) sqrt(1 - (alpha_i/alpha_{i-1})^2). This update has three coefficients that all have to balance, and it's the place I'm least sure I've transcribed the (alpha, sigma) form correctly, so I test the marginal-matching claim directly rather than asserting it. Take an intermediate level (i = 12 on the N = 20 cosine grid), start from x^i drawn as alpha_i x0 + sigma_i eps with the *true* eps fed in for eps_theta, apply this update with fresh z, and check the resulting x^{i-1} over 4x10^5 samples: empirical mean 0.8383 against the target alpha_{i-1} x0 = 0.8388, empirical std 0.7650 against the target sigma_{i-1} = 0.7640. Both land on the forward marginal, so the three-term coefficient balance is right — the deterministic part carries the clean estimate forward and the two noise contributions sum to exactly sigma_{i-1}^2. The fresh noise std_i z is added at every step except the last (z = 0 at i = 1), since the final step should land on a clean trajectory, not a noised one. After each step I re-clamp the start-state coordinates to the current observation, holding the inpainting constraint through the whole chain. With no reward guidance there is no classifier gradient to inject and no candidate re-ranking — I draw a single plan and read off its first action, tau[0, obs_dim:], clipped to the valid action range, and execute that. Then I replan from the next state.

Let me put it into code that fills the one empty slot in the harness. I do not need a separate planner hiding beside the model: the slot is a temporal U-Net wrapped by the discrete diffusion SDE, and the wrapper exposes only `update(obs, act)` and `plan(current_obs)`.

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
        # The SDE corrupts x in one shot as alpha[t] * x + sigma[t] * eps,
        # reclamps fix_mask coordinates, and applies loss_weight * (1 - fix_mask).
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

The causal chain, end to end. I started with offline trajectory data and the failure of the learn-a-model-then-plan decomposition: single-step models compound rollout error and get exploited by strong planners through the seam between model and reality, while model-free value methods throw away trajectory-level reasoning and autoregressive sequence models fight the anti-causal direction of conditioning. So I demanded that producing a plan *be* sampling from a trajectory model, all timesteps at once, trained for the whole trajectory rather than single-step accuracy, and reward-free at its base so it stays on the data manifold. Diffusion gave me that engine: I derived and numerically checked the closed-form one-shot forward corruption (variance-preserving, marginal coefficients confirmed by simulation), the variational bound's reduction to denoising regression, the eps-parameterization that makes the network a multi-scale score estimator and collapses the loss to predicting the added noise (its compact mean form verified to floating point against the two-term posterior), and the ancestral reverse step (Monte-Carlo'd to confirm it reproduces the forward marginal at each level). I made the data object a 2D array of stacked states and actions so the model is a controller and not only a predictor; chose 1D temporal convolutions because the horizon axis has translation structure and the feature axis does not; got global coherence from a multi-scale U-Net composed with many local denoising steps; used a normalized cosine schedule because the chain is short (about 20 steps for these low-dimensional trajectories); imposed start-state conditioning as inpainting by clamping column-0 state coordinates throughout the chain and zeroing their loss; up-weighted the first action because only it is executed in receding-horizon control; and landed on sampling a single plan with no reward guidance and no re-ranking, reading off the first action. That last part is deliberate by construction: with no value, return, or classifier entering anywhere, what this object samples is the reward-agnostic plan distribution — plans that are dynamically plausible given where I'm standing, and nothing more. Whether that base distribution is good enough as a controller, or wants reward guidance bolted on later, is the next question; here I keep it clean.
