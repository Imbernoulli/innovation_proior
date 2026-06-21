Classifier-free guidance is the default way to turn a conditional diffusion or flow-matching model into a controllable generator. At every step the model is run twice, once with the condition and once with it dropped, producing two velocity or noise predictions. The standard rule mixes them with a single global weight: the guided prediction is a convex combination of the unconditional and conditional outputs. This works well when the learned field is accurate, but it treats both predictions as trustworthy at every timestep and every sample. In practice the learned field is imperfect, especially early in the trajectory when the latent is still close to source noise and carries almost no semantic signal. The same fixed scale that amplifies the conditional signal also amplifies conditional error, so pushing hard at the source end often hurts more than it helps.

A useful improvement has to keep the attractive parts of CFG: no retraining, no extra model calls, and the same two predictions per step. The question is how to mix those two predictions more carefully. A Gaussian linear path gives a closed-form diagnostic for the optimal velocity, and it confirms that the largest errors occur near the high-noise source end. But on real data the optimal velocity is unknown, so the mix must be chosen from the two available predictions alone, without access to the ground truth.

The method is CFG-Zero*. It changes the per-step guidance update in two independent ways. The first is an optimized scale on the unconditional prediction. Write the standard guided prediction as the conditional prediction plus a scaled difference between the conditional and unconditional predictions. The conditional prediction is fixed, and the only free scalar is the scale on the unconditional baseline. Minimizing the squared distance to the unknown optimal velocity is impossible, but a Young's-inequality upper bound on that squared error isolates a single term that depends only on the two model predictions: the squared norm of the conditional prediction minus the scalar times the unconditional prediction. Minimizing this visible surrogate gives the least-squares projection coefficient of the conditional prediction onto the unconditional prediction. Intuitively, this rescales the unconditional baseline so that it explains as much of the conditional prediction as possible, leaving a residual that is orthogonal to the unconditional direction. Guidance then amplifies only that residual, not the entire difference between the two raw predictions.

The second change is zero-init. The same Gaussian diagnostic shows that at the very beginning of the trajectory the guided update can be farther from the optimal first-step velocity than taking no step at all. When that happens, the best mix is still a bad move. CFG-Zero* therefore skips the update for the first K solver steps, leaving the latent at its initialization. Once the field becomes more reliable, the optimized-scale mix takes over. K is small, because the unreliability is localized to the source end; skipping too many steps would waste the inference budget and discard useful conditional signal.

The optimized scale is computed independently for each sample and each step by flattening the two predictions, taking their dot product, and dividing by the squared norm of the unconditional prediction plus a small floor to avoid division by zero. The result is reshaped to broadcast over the latent dimensions and cast back to the prediction dtype. The guided prediction becomes the rescaled unconditional prediction plus the guidance weight times the conditional residual. When the two predictions are collinear the residual vanishes and the rule reduces to ordinary CFG, so CFG-Zero* never underperforms the baseline where the baseline is already correct. It only acts when the two predictions genuinely disagree in direction, which is exactly where the fixed mix is most brittle.

For an eps-prediction sampler with the CFG++ renoise step, the optimized scale is applied to the two noise estimates, the denoising mix uses the rescaled unconditional baseline, and the renoise still uses the unconditional prediction to stay on the data manifold. The zero-init prefix is implemented as a simple branch that skips the first K steps. No extra network evaluations, no retraining, and no change to the model are required.

```python
import torch


def optimized_scale(positive_flat, negative_flat, eps=1e-8):
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + eps
    return dot_product / squared_norm


@torch.no_grad()
def sample_ddim_cfg_zero_star(pipeline, prompt, cfg_guidance=0.6, K=2):
    uc, c = pipeline.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
    zt = pipeline.initialize_latent()

    for step, t in enumerate(pipeline.scheduler.timesteps):
        if step < K:
            continue

        at = pipeline.alpha(t)
        at_prev = pipeline.alpha(t - pipeline.skip)

        noise_uc, noise_c = pipeline.predict_noise(zt, t, uc, c)

        bsz = noise_c.shape[0]
        c_flat = noise_c.reshape(bsz, -1)
        uc_flat = noise_uc.reshape(bsz, -1)
        dot = (c_flat * uc_flat).sum(dim=1, keepdim=True)
        sq_norm = (uc_flat ** 2).sum(dim=1, keepdim=True) + 1e-8
        alpha = (dot / sq_norm).reshape(bsz, *([1] * (noise_c.dim() - 1))).to(noise_c.dtype)

        noise_pred = noise_uc * alpha + cfg_guidance * (noise_c - noise_uc * alpha)

        z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()
        zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_uc

    return pipeline.decode(z0t)
```
