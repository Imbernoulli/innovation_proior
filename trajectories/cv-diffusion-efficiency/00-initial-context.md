## Research question

Text-to-image diffusion models generate high-quality images, but standard sampling requires many denoiser calls. The budget here is fixed at exactly **NFE = 20** denoiser evaluations per sample, with frozen model weights and fixed prompts. The only design target is the **sampler update rule**: given the current latent and the model's noise prediction, choose the next noise level so that generated images have the best possible quality (low FID, high CLIP) within those 20 calls.

## Prior art / Background / Baselines

- **DDPM (Ho et al. 2020).** Trains a noise predictor and samples by reversing the Markov forward chain one ancestral step at a time, following the same discrete schedule used during training.
- **Classifier-free guidance (Ho & Salimans 2022).** At each step forms a guided prediction from unconditional and conditional outputs using a large scale to sharpen prompt alignment.
- **CFG++ renoising convention.** The sampler renoises using the unconditional noise prediction while the clean-image estimate still uses the guided prediction, tempering over-saturation.

## Fixed substrate / Code framework

The pipeline is frozen. Stable Diffusion v1.5, v2.0, and SDXL weights are fixed; the evaluation prompts and the **NFE = 20** budget are fixed. One `predict_noise` call evaluates both unconditional and conditional embeddings together and counts as **one** function evaluation. The forward process is $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$.

Available helpers:

- **SD (`latent_diffusion.py`, `BaseDDIMCFGpp.sample`):** `get_text_embed`, `initialize_latent`, `predict_noise(zt, t, uc, c) -> (noise_uc, noise_c)`, `alpha(t)`, `skip`, `total_alphas`, `timestep(sigma)`, `kdiffusion_x_to_denoised`, `decode`, `scheduler.timesteps`, and `get_sigmas_karras`.
- **SDXL (`latent_sdxl.py`, `BaseDDIMCFGpp.reverse_process`):** same shape with `initialize_latent(size=...)`, `predict_noise(zt, t, null_embeds, prompt_embeds, add_cond_kwargs)`, and `scheduler.alphas_cumprod[t]`.

Sampling runs in VAE latent space; there is no $[-1,1]$ pixel bound to clip against.

## Editable interface

Only two regions may be edited: the body of `BaseDDIMCFGpp.sample` in `latent_diffusion.py` (SD) and `BaseDDIMCFGpp.reverse_process` in `latent_sdxl.py` (SDXL). Each sampler fills the same contract: read text embeddings, initialize the latent, walk the timesteps making exactly NFE-bounded `predict_noise` calls, form the guided prediction $\tilde\epsilon=\epsilon_{uc}+s\,(\epsilon_c-\epsilon_{uc})$, estimate the clean latent by Tweedie $z_{0|t}=(z_t-\sqrt{1-\bar\alpha_t}\,\tilde\epsilon)/\sqrt{\bar\alpha_t}$, and update to the next noise level using unconditional $\epsilon_{uc}$ for renoising. The starting scaffold is **not implemented**.

```python
# EDITABLE region of latent_diffusion.py — default scaffold (SD v1.5; SDXL is the analogous reverse_process)
@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    # TODO: Implement your sampling method here.
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
        # Generate high-quality images with the fixed NFE budget.
        raise NotImplementedError("You need to implement the sample method")
```

## Evaluation settings

Text-to-image sampling runs on SD v1.5, SD v2.0, and SDXL with shared prompts, NFE = 20, seed 42. Two metrics are reported: **FID** against a reference set (lower is better and used for scoring) and **CLIP score** between generated images and prompts (higher is better). The sampler must improve quality without changing prompts, weights, the NFE budget, or metric computation.
