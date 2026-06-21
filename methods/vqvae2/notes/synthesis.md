# VQ-VAE-2 synthesis

Primary source: Razavi, van den Oord, and Vinyals, "Generating Diverse High-Fidelity Images with VQ-VAE-2", arXiv:1906.00446 / NeurIPS 2019.

Reference code:
- The paper explicitly cites the released DeepMind Sonnet VQ-VAE module for the vector quantizer and EMA update.
- The de facto PyTorch architecture reference used for code faithfulness is `rosinality/vq-vae-2-pytorch`, whose `vqvae.py` says it borrows the Sonnet VQ layer and ports it to PyTorch.

## Load-bearing facts

- Likelihood models optimize NLL, equivalent to forward KL up to the fixed data entropy. This gives mode-coverage pressure and a held-out generalization measure.
- Pixel-space NLL can be a poor perceptual objective and pixel autoregression is slow at high resolution.
- Lossy compression motivates modeling a compact representation instead of raw pixels.
- VQ-VAE supplies a discrete grid: `k = argmin_j ||z_e(x) - e_j||_2`; straight-through gradients route reconstruction gradients to `z_e`.
- With a deterministic one-hot posterior and uniform prior, KL is `log K` per latent position (`N log K` for `N` independent positions), so it is constant during autoencoder training.
- The VQ loss split is: reconstruction; codebook term `||sg[z_e]-e||^2`; commitment term `beta ||z_e-sg[e]||^2`.
- VQ-VAE-2 uses EMA codebook updates as a replacement for the codebook loss, with `gamma=0.99`. The differentiable latent loss is the commitment MSE and training multiplies it by `beta=0.25`.
- ImageNet-256 VQ-VAE settings from the appendix: top/bottom latent layers `32x32` and `64x64`, batch size `128`, hidden units `128`, residual units `64`, layers `2`, codebook size `512`, codebook dimension `64`, encoder filter size `3`, upsampling filter size `4`.
- ImageNet-256 prior settings from the appendix: top prior `32x32`, bottom prior `64x64`; hidden units `512` for both; residual units `2048` top and `1024` bottom; `20` layers each; top has `4` attention layers and `8` heads, bottom has no attention; dropout `0.1`; top has `20` output-stack layers; bottom has `20` conditioning-stack residual blocks.
- Stage 1 hierarchy: top code captures global structure; bottom code represents local detail while conditioned on the top. Each level separately depends on pixels, encouraging complementary information.
- Stage 2 priors: top prior uses causal attention for long-range structure on the small grid; bottom prior omits attention because the grid is larger and it is conditioned on top.
- Sampling factorization: `p(c_top, c_bottom) = p_top(c_top) p_bottom(c_bottom | c_top)`, followed by a feed-forward decoder. ImageNet priors in the method are class-conditional.
- Classifier-based rejection sampling scores generated class-conditional images by the pretrained classifier probability for the intended class and keeps a top fraction as a quality-diversity knob.

## Code notes

- `rosinality_vqvae.py` stores the codebook as `[D, K]`, flattens the last dimension, computes `||z||^2 - 2 z e + ||e||^2`, chooses nearest codes by `(-dist).max(1)`, reshapes indices to `input.shape[:-1]`, and embeds with `F.embedding(embed_id, embed.T)`.
- Its EMA update all-reduces assignment counts and sums for distributed training, applies Laplace-style count smoothing, and copies `embed_avg / cluster_size` into the codebook buffer.
- Its `Quantize.forward` returns `diff = mean((quantize.detach() - input)^2)` without multiplying by beta; `train_vqvae.py` applies `latent_loss_weight = 0.25`.
- `rosinality_train_pixelsnail.py` trains a top `PixelSNAIL([32,32], 512, ...)` and a bottom `PixelSNAIL([64,64], 512, ..., attention=False, condition=top)`. The default script loads labels but does not pass them into the prior, so class conditioning is a paper-level requirement, not present in that default script.

## Corrections made

- Removed answer leakage from `context.md`: no top/bottom hierarchy, exact stage protocol, or hyperparameter recipe is stated before the reasoning/answer.
- Corrected KL language to say `log K` per latent position and `N log K` for a field.
- Clarified minimization signs: reconstruction MSE / NLL plus penalties, not the ELBO maximization sign.
- Clarified EMA semantics: the codebook loss is replaced by EMA; the returned latent loss is commitment MSE and is weighted by `beta=0.25` in training.
- Corrected prior hyperparameter notes: top and bottom prior hidden units are both `512`; the difference is residual units (`2048` top, `1024` bottom), attention (`4` top, `0` bottom), output stack (`20` top), and conditioning stack (`20` bottom).
- Recorded that the default PyTorch prior script is not class-conditional even though the method's ImageNet prior is.
