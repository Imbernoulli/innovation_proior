OK, let me reason from the failure, not from architecture fashion. I want a generator that can reach 1024x1024, but the adversarial game gives G only whatever gradient D hands back. At low resolution the game is hard for D and manageable for G: a 4x4 face-like image has only coarse layout, so real and generated distributions have a chance to overlap. At high resolution the asymmetry flips. Real photographs have a dense set of high-frequency cues, and a random or half-trained generator gets many of them wrong. D gets millions of coordinates to find a tell, so it can become nearly perfect while G is still trying to learn global structure. That is exactly the regime where the JS-style discriminator diagnosis says the gradient becomes useless, and even WGAN-GP only fixes the distance side, not the fact that I asked for the whole latent-to-megapixel map at once.

So the useful move is to stop presenting the final problem at the start. I can train the same adversarial pair first on the smallest image where the distribution has only coarse structure, then introduce the next octave of detail only after the old scale has learned something. The old layers must stay trainable, because they may need to adjust when later details arrive, but the new work at each step is local: turn an already meaningful R x R representation into a 2R x 2R representation. That converts one brutal optimization into a sequence of smaller ones, and most of the early iterations are cheap because the networks are shallow and the images are tiny.

The first trap is insertion shock. If I append a random high-resolution block and immediately route the output through it, the old trained generator is suddenly connected to random filters, D sees garbage, and the gradient can destroy the lower-resolution solution. I need the network to behave exactly like the old network at the instant of growth, then hand control to the new block gradually. The old generator output at 2R is just the R image projected to RGB and upsampled. The new output is the new block followed by its own toRGB. During the transition I should output old_weight times the old upsampled RGB plus new_weight times the new RGB, with old_weight starting near 1 and ending at 0.

The bookkeeping convention matters here. I track a single scalar `lod` (level-of-detail) that is high at the beginning and decreases as resolution increases. For a final 1024 model, stable 4x4 is `lod=8`, stable 8x8 is `lod=7`, and a transition from 4x4 to 8x8 has `lod` between 8 and 7. The active new stage is selected by `floor(lod)`, and the fractional part of `lod` is the lower-resolution path's weight. Thus

```text
old_weight = lod - floor(lod)
new_weight = 1 - old_weight.
```

That sign is easy to get backwards. At the start of a fade, say `lod=7.999`, `old_weight` must be almost one so the random new block contributes almost nothing. At the end, `lod=7.000`, `old_weight` is zero and the new block fully owns the output. The generator blend is therefore

```text
image = new_weight * toRGB_new(block(features_R))
      + old_weight * upscale(toRGB_old(features_R)).
```

D has to use the same convention in reverse. It first maps the sharp 2R input through fromRGB_new and the new down block, producing features at R. The old path downsamples the image to R and applies fromRGB_old. Those two R-resolution feature tensors are blended with the same weights:

```text
features = new_weight * downblock(fromRGB_new(x_2R))
         + old_weight * fromRGB_old(downsample(x_2R)).
```

The real images need the same treatment before D sees them. If real images are fully sharp during the transition while fake images are a mixture of sharp and blocky paths, D can exploit high-frequency sharpness as a shortcut. So I take the current real image x, make a one-octave blurred version by average-pooling and tiling it back, and use

```text
x_faded = (1 - old_weight) * x + old_weight * upscale(downscale(x)).
```

This again has the right cases: at the start of the fade, real images are effectively lower-resolution images displayed at the higher size; at the end, they are sharp. To keep tensor shapes uniform across stages I let G and the real-image pipeline both emit at a fixed full size: I upsample by `2^floor(lod)` after the blend, and D downsamples the full tensor by the same factor before its active fromRGB path.

Now variation. A discriminator that scores each image independently can be fooled by a generator that repeats a few convincing samples, because any one sample can look real. The old minibatch discrimination idea gives D cross-sample information, but it does so with a learned tensor and extra design choices. The simplest statistic that says "this batch lacks variety" is standard deviation across the minibatch. For activations `x` with shape `[N,C,H,W]`, I split into groups, subtract each group's mean, compute the standard deviation over the group axis for each channel and location, average that to one scalar per group, tile it as a constant `[N,1,H,W]` map, and concatenate it near D's final 4x4 layers. There are no learnable parameters. If G collapses, this scalar gets too small, and D has a clean feature with which to punish it.

The magnitude problem is separate. Collapse often starts when D overshoots, gradients get exaggerated, and both networks participate in a magnitude race. BatchNorm is not the thing I want if the issue is not covariate shift but adversarial scale escalation, especially with small high-resolution minibatches. I want a parameter-free brake in G. For each pixel, I normalize the feature vector across channels to unit RMS:

```text
b_{x,y} = a_{x,y} / sqrt(mean_j (a^j_{x,y})^2 + 1e-8).
```

This preserves direction and removes only the scale degree of freedom. It belongs in G after each 3x3 convolution, and also on the latent input in the implementation; it is not used in D.

There is another scale issue that lives in the optimizer. He initialization sets each layer's effective weight scale to `gain/sqrt(fan_in)`, but Adam and RMSProp normalize updates by a running gradient scale. If the trainable weights themselves live at different dynamic ranges, then the same normalized update is a different fraction of each layer's natural range. The fix is to store all weights at unit normal scale and multiply by the He constant at runtime:

```text
w_eff = w * gain / sqrt(fan_in).
```

The forward pass has the same variance behavior as He initialization, but the learned parameters all live in the same dynamic range. Concretely I store each weight at unit normal scale and, on every forward call, multiply by `std = gain/sqrt(fan_in)` before using it; the bias is unscaled.

For the loss, I keep the WGAN-GP signs straight. D minimizes `D(fake) - D(real)` plus the penalty on interpolates, with `lambda=10` and target gradient norm 1, and G minimizes `-D(fake)`. I also add a positive, tiny drift term, `0.001 * E[D(real)^2]`, so it pulls real scores toward zero without changing the ordering objective. The default unconditional case has no label penalty; I could add AC-GAN-style label losses if labels were available, but that is not part of the ordinary unsupervised face setup.

The final recipe is now coherent. Start at 4x4 from a 512-dimensional latent normalized to the hypersphere, use mirror generator and discriminator blocks, keep old layers trainable, fade each new resolution with `frac(lod)` as the old-path weight, fade real images by the same amount, add a minibatch-stddev feature near D's end, use pixelwise normalization only in G, use runtime He scaling for every learned weight, and train with WGAN-GP plus drift using Adam `lr=0.001`, `beta1=0`, `beta2=0.99`, `epsilon=1e-8`. For the full CelebA-HQ runs I budget 800k real images for the initial 4x4 phase, then 800k to fade and 800k to stabilize each added block; these are exposed as schedule parameters so they can be retuned (a lighter preset with 600k transition/training phases works too). The method is the same; the constants have to be named as schedule choices rather than hidden in the equations.
