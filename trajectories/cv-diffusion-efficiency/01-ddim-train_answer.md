I have a frozen text-to-image diffusion model, a hard ceiling of twenty denoiser calls per image, and the only thing I get to write is the update rule inside the sampling loop. There is no prior number to beat yet, so the job at this first rung is to put down the floor: the simplest correct sampler that turns the model's per-step noise prediction into an image inside the budget, so everything I try later has a baseline. The naive worry is that this is hopeless — the original ancestral chain needs hundreds to a thousand sequential network passes, and that slowness looks load-bearing: each true reverse conditional $q(z_{t-1}|z_t)$ is only near-Gaussian when the step is tiny, so a large step count is what keeps the Gaussian generative model valid. If the generative chain really had to approximate the reverse of the *Markov* forward chain, twenty steps would be a fantasy.

The crack is that "approximates the reverse of the forward chain" is doing unexamined work. Training pinned down something narrower. The objective is the unweighted $\epsilon$-MSE: feed the network $z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ and penalize $\|\epsilon_\theta(z_t,t)-\epsilon\|^2$ summed over $t$. Two facts fall out. First, it is a sum of independent per-$t$ terms, so the optimal $\epsilon_\theta$ is the same function under any reweighting across $t$ — the minimizer is reweighting-invariant. Second, the only input the network ever sees is a draw from the *marginal* $q(z_t|z_0)=\mathcal N(\sqrt{\bar\alpha_t}\,z_0,(1-\bar\alpha_t)I)$; the loss never references the joint $q(z_{1:T}|z_0)$, never asks how $z_t$ and $z_{t-1}$ correlate. So the trained network is committed only to the marginals. The Markov forward chain was one arbitrary joint with those marginals; any other joint with the same marginals is, to the loss, an equally valid story solved by the same network. The slowness came from identifying the generative chain with the reverse of that particular chain — but the network is wedded to the marginals, not to that chain.

I propose **DDIM**: a first-order deterministic ODE step on a short sub-grid of the training chain. The construction is to build a different inference process — same marginals, different joint — whose generative reverse is deterministic and short, under exactly one constraint: preserve $q(z_t|z_0)$ for every $t$. I specify the process backwards, conditioned on $z_0$, writing the reverse conditional $q_\sigma(z_{t-1}|z_t,z_0)$ directly, as a Gaussian with mean affine in $z_0$ and $z_t$ and free covariance $\sigma_t^2 I$. The natural ansatz puts a $\sqrt{\bar\alpha_{t-1}}\,z_0$ signal piece plus a term proportional to the residual $(z_t-\sqrt{\bar\alpha_t}\,z_0)/\sqrt{1-\bar\alpha_t}$ — which in distribution is the very noise $\epsilon$ that carried $z_0$ to $z_t$ — scaled by an unknown $k_t$. Imposing the marginal constraint by induction downward from $t=T$ forces $k_t^2=1-\bar\alpha_{t-1}-\sigma_t^2$, giving

$$q_\sigma(z_{t-1}|z_t,z_0)=\mathcal N\!\Big(\sqrt{\bar\alpha_{t-1}}\,z_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\tfrac{z_t-\sqrt{\bar\alpha_t}\,z_0}{\sqrt{1-\bar\alpha_t}},\,\sigma_t^2 I\Big),$$

with one free $\sigma_t$ per step subject to $0\le\sigma_t^2\le 1-\bar\alpha_{t-1}$. The induction closes — every marginal is exactly the fixed Gaussian — and a whole degree of freedom per step survives the matching: precisely the stochasticity of the reverse step. (By Bayes the implied forward conditional now depends on $z_0$, so the forward process is no longer Markovian; that is fine, I never needed Markovian, only the marginals.)

To run this generatively I have $z_t$ but not $z_0$, so the network supplies it: inverting the forward relation gives the Tweedie clean estimate $z_{0|t}=(z_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(z_t))/\sqrt{\bar\alpha_t}$, which is exactly the substrate's helper. Substituting it into the reverse conditional — and noting that the residual term collapses back to $\epsilon_\theta(z_t)$ by construction — the generative update becomes $z_{t-1}=\sqrt{\bar\alpha_{t-1}}\,z_{0|t}+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(z_t)+\sigma_t\,\epsilon$: jump the predicted clean latent to level $t-1$, re-inject deterministically the predicted noise the marginal there still wants, add fresh randomness. This is *allowed* on the frozen network because the variational objective for this generative process reduces term-by-term to a weighted noise-prediction MSE, whose minimizer — by the reweighting-invariance above — is the same network I already trained.

Now I spend the free knob, and I spend it twice. First, set $\sigma_t=0$ for all $t$: the noise term vanishes and the update becomes the deterministic DDIM step $z_{t-1}=\sqrt{\bar\alpha_{t-1}}\,z_{0|t}+\sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta(z_t)$, an implicit model where $z_0$ is a fixed deterministic pushforward of $z_T$. Determinism alone does not shorten the chain, so I use the marginal-only fact a second time, more aggressively: nothing forces the process to visit every $t$, so I define the whole construction on a sub-sequence $\tau=(\tau_1<\dots<\tau_S)$ of $[1..T]$, reusing the reverse conditional with index pairs $(\tau_i,\tau_{i-1})$ — the marginal-consistency induction only ever used the two endpoints of a step, so it does not care whether the steps are adjacent integers or jumps. Train at $T=1000$, sample on $S=20$, same network, no retraining. That is the cure for the wall clock, and it is exactly what the harness hands me: it has already chosen the 20-step grid (`self.skip` is the stride), and I walk it, one `predict_noise` call per step, twenty steps, twenty NFE.

Why the *deterministic* member is the safe few-step choice is worth seeing, because it tells me how it will fail. Change variables to $\bar z=z/\sqrt{\bar\alpha}$ and $\varsigma=\sqrt{(1-\bar\alpha)/\bar\alpha}$; the $\sigma=0$ step becomes $\bar z(t-\Delta t)=\bar z(t)+(\varsigma(t-\Delta t)-\varsigma(t))\,\epsilon_\theta$, and as $\Delta t\to 0$ this is $d\bar z=\epsilon_\theta\,d\varsigma$ — Euler integration of the probability-flow ODE of the variance-exploding diffusion, reached purely from the variational construction. The image is that ODE's solution from a fixed initial condition, and the step count is just the fineness of the Euler grid: same $z_T$, different $S$, nearly the same image, only fine detail moving as the grid coarsens. With $\sigma_t>0$, every step injects fresh noise a short chain has too few remaining steps to average back down; with $\sigma=0$ there is no injected noise to clean up, so cutting steps only coarsens an otherwise smooth map. That is the robustness argument for $\eta=0$ — and also exactly why this is the floor: twenty Euler steps each carry an $O(h)$ local error, $h$ is not small at this count, the trajectory genuinely bends between adjacent levels, and a first-order step ignores the bend by holding $z_{0|t}$ constant across the interval and moving on a straight chord. DDIM is the $k=1$ member of the solver family: no derivative estimate, no history, no intermediate evaluation, so there is nothing for a large guidance scale to corrupt — the stable fallback — and nothing to extrapolate, which is precisely the truncation error a higher-order step would later buy back.

Two substrate specifics fix the exact fill. Guidance is CFG++: the guided $\tilde\epsilon=\epsilon_{uc}+s(\epsilon_c-\epsilon_{uc})$ feeds the clean-image estimate via Tweedie, $z_{0|t}=(z_t-\sqrt{1-\bar\alpha_t}\,\tilde\epsilon)/\sqrt{\bar\alpha_t}$, but the renoising/direction term uses the bare *unconditional* $\epsilon_{uc}$, so my step is $z_{t-1}=\sqrt{\bar\alpha_{t-1}}\,z_{0|t}+\sqrt{1-\bar\alpha_{t-1}}\,\epsilon_{uc}$ — tempering the over-saturation a large $s$ would bake into the renoised latent. And I am in VAE latent space, so there is no $[-1,1]$ pixel bound to threshold against; only the numerical behaviour of the update is in play. I expect this to be the worst of anything sensible — a usable, non-diverging image on every variant, but FID well above what a higher-order step reaches at the same twenty calls, and the gap widest on SDXL, whose largest latent has the most structure to resolve in twenty coarse steps. That measured gap is exactly what should force a higher-order update next.

```python
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    """
    DDIM sampler with CFG++.
    First-order ODE solver - simple and deterministic.
    """
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="runwayml/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        super().__init__(solver_config, model_key, device, **kwargs)

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               **kwargs):

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent()
        zt = zt.requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DDIM")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # Tweedie: estimate clean image
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # DDIM update: first-order, use unconditional noise for CFG++
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # Decode final latent
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
```
