## Research question

Text-to-image diffusion models draw beautiful, prompt-faithful pictures, but only because the sampler
takes many denoiser calls to walk the noise down to an image. Here the budget is hard: exactly
**NFE = 20** denoiser evaluations per sample, the model weights are frozen, and the prompts are fixed.
The single thing being designed is the **sampler update rule** — given the current latent and the
model's noise prediction, how to step to the next noise level — so that the images come out as good as
possible (low FID, high CLIP) inside those 20 calls. Everything else about the pipeline is fixed.

## Prior art before the first rung (sampler lineage)

The first rung the ladder uses — DDIM — is itself the resolution of a line of diffusion-sampling work.
These are the methods that precede the climb; the fixed substrate below is what guided sampling
converged to.

- **DDPM (Ho et al. 2020).** Trains a noise predictor $\epsilon_\theta(x_t,t)$ by the unweighted
  $\epsilon$-MSE and samples by reversing the *Markov* forward chain one ancestral step at a time:
  $x_{t-1}=\tfrac{1}{\sqrt{\alpha_t}}\big(x_t-\tfrac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta\big)+\sigma_t z$.
  Gorgeous samples, none of a GAN's instability, but the generative chain is tied to the forward chain
  length — hundreds to a thousand sequential network passes per image. Gap: each sample is wall-clock
  unusable at the step counts the chain demands.
- **Classifier-free guidance (Ho & Salimans 2022).** Replaces the plain prediction with a steered one,
  $\tilde\epsilon=\epsilon_\theta(x_t,t,\varnothing)+s\,(\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t,\varnothing))$,
  with a *large* scale $s$ to sharpen prompt alignment — exactly what a high CLIP score rewards. Gap: a
  large $s$ amplifies the model's derivatives, which is precisely what later destabilizes fast
  high-order solvers, and pushes the converged image off the data manifold.
- **CFG++ (the substrate's renoising convention).** This codebase's solvers are written in a CFG++
  style: the *direction / renoising* term uses the **unconditional** noise $\epsilon_{uc}$ rather than
  the guided $\tilde\epsilon$, while the clean-image estimate still uses the guided $\tilde\epsilon$.
  This tempers the over-saturation a large guidance scale would otherwise bake into the renoised
  latent. Every fill below inherits this convention; it is part of the substrate, not a knob the rungs
  redesign.

## The fixed substrate

The pipeline is frozen and must not be touched. Stable Diffusion v1.5, v2.0, and SDXL with frozen
weights; a shared set of evaluation prompts; a hard budget of **NFE = 20** denoiser calls per sample.
One `predict_noise` call batches the unconditional and conditional embeddings through the UNet
together, so it returns $(\epsilon_{uc},\epsilon_c)$ and counts as **one** function evaluation. The
forward process is $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$, and the loop provides
the helpers a sampler may use:

- **SD (`latent_diffusion.py`, `BaseDDIMCFGpp.sample`):** `self.get_text_embed(null, prompt)`,
  `self.initialize_latent()`, `self.predict_noise(zt, t, uc, c) -> (noise_uc, noise_c)`,
  `self.alpha(t)` (cumulative $\bar\alpha_t$; $t<0$ returns the final cumprod), `self.skip` (timestep
  stride for the chosen step count), `self.total_alphas`, `self.timestep(sigma)`,
  `self.kdiffusion_x_to_denoised(...)`, `self.decode(z)`, `self.scheduler.timesteps`, and a
  module-level `get_sigmas_karras(n, sigma_min, sigma_max, rho)`.
- **SDXL (`latent_sdxl.py`, `BaseDDIMCFGpp.reverse_process`):** the same shape with
  `self.initialize_latent(size=...)`, `self.predict_noise(zt, t, null_embeds, prompt_embeds, add_cond_kwargs)`,
  and `self.scheduler.alphas_cumprod[t]` for $\bar\alpha_t$.

Sampling operates in the VAE **latent** space, so there is no $[-1,1]$ pixel bound to threshold
against — the data-clipping trick that helps pixel-space guided solvers is unavailable here, and only
the *numerical* behaviour of an update rule is in play.

## The editable interface

Exactly two regions are editable — the body of `BaseDDIMCFGpp.sample` in `latent_diffusion.py` (SD)
and `BaseDDIMCFGpp.reverse_process` in `latent_sdxl.py` (SDXL). Every method on the ladder is a fill
of this same contract: read the text embeddings, initialize the latent, walk the timesteps making
exactly NFE-bounded `predict_noise` calls, and at each step (i) form the guided prediction
$\tilde\epsilon=\epsilon_{uc}+s\,(\epsilon_c-\epsilon_{uc})$, (ii) estimate the clean latent by
Tweedie $z_{0|t}=(z_t-\sqrt{1-\bar\alpha_t}\,\tilde\epsilon)/\sqrt{\bar\alpha_t}$, and (iii) **update**
to the next noise level — the one line that differs across samplers — renoising with the unconditional
$\epsilon_{uc}$ in the CFG++ style. The starting point is the scaffold default: **not implemented**.
Each method below replaces exactly this body.

```python
# EDITABLE region of latent_diffusion.py — default scaffold (SD v1.5; SDXL is the analogous reverse_process)
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    # TODO: Implement your improved sampling method here.
    # Implement an improved sampling algorithm that achieves better image quality
    # (FID / CLIP) with a fixed budget of NFE = 20 steps.
    #
    # Available helpers from the parent class:
    #   self.get_text_embed(null_prompt, prompt) -> (uc, c)
    #   self.initialize_latent()                  -> zT
    #   self.predict_noise(zt, t, uc, c)          -> (noise_uc, noise_c)   # one NFE
    #   self.alpha(t)                             -> bar_alpha_t
    #   self.skip                                 -> timestep stride
    #   self.decode(z), self.scheduler.timesteps

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
        # TODO: Implement your efficient sampling method here.
        # Generate high-quality images with the fixed NFE budget. Consider the
        # update rule, the timestep spacing, multistep history, or renoising.
        raise NotImplementedError("You need to implement the sample method")
```

## Evaluation settings

Text-to-image sampling is run on three model variants — **SD v1.5**, **SD v2.0**, and **SDXL** — over
the shared evaluation prompts, at the fixed NFE = 20, on seed 42. Two metrics are reported per
variant: **FID** against a reference image set (lower is better — this is what task scoring uses) and
**CLIP score** (cosine similarity of image and prompt, higher is better). The sampler must improve
quality across variants without touching prompts, weights, the NFE budget, or the metric computation.
