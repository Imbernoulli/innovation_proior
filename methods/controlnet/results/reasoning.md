I start with the failure mode rather than with a module. I have a text-to-image diffusion model whose prior is valuable precisely because it was trained at huge scale, and I want to teach it a new spatial input from a much smaller paired dataset. If I keep training all of its weights on that smaller dataset, the new control may fit, but the broad prior can be overwritten. So the first requirement is separation: the old model should continue to provide the generative prior while the new parameters learn the condition.

The simplest way to guarantee that separation is to keep the production model's weights fixed. That removes the forgetting path completely. But freezing the old model creates the next problem: a small condition module may not have enough capacity to read complicated edges, poses, depth maps, or segmentations and translate them into the internal language of the U-Net. A thin adapter or low-rank update protects the base, but the spatial input can carry high-level semantics, so I need a strong branch rather than a token gesture.

The pretrained U-Net already contains a strong encoder. If I copy the encoder-side blocks and the middle block into a trainable branch, the new branch starts with useful image features instead of random features. The small condition dataset then adapts a pretrained representation instead of teaching a deep representation from scratch. The original path stays fixed and preserves the prior; the copied path supplies capacity.

Now I have a more delicate problem. A copied branch is trainable, and the condition encoder feeding it starts untrained. If I connect that branch into the fixed U-Net with ordinary random weights, the first forward pass injects random features into a model whose internal activations already have a tuned meaning. From the very first step, the frozen prior is being corrupted by noise before the branch has learned anything useful. So the connector has to begin as an exact no-op, and only later open up. What kind of layer can start as a guaranteed zero map and yet be able to leave that state under training? A `1 x 1` convolution initialized with both weight and bias zero is the obvious candidate: for any input feature `x`, `Wx + b = 0` when `W = 0, b = 0`, independent of `x`. Whether it can subsequently learn is a separate question I have to actually check, not assume — a zero map with zero gradient would be a dead module.

For a trained block `F(x; Theta) = y`, I make a trainable copy `F(.; Theta_c)` and use two such zero convolutions, one on the condition path and one on the copied block's output:

    y_c = F(x; Theta) + Z(F(x + Z(c; Theta_z1); Theta_c); Theta_z2).

I want to see this hold concretely before trusting it, so I take a tiny example. Let `F` be a fixed `2x2` linear map with weight `[[1, 0.5], [-0.3, 0.8]]`, locked. Let the base feature be `x = [0.7, -0.2]` and the condition feature be `c = [2, 3]`. With both `Z` connectors zero-initialized, `Z(c; Theta_z1) = 0`, so the copied block receives `x + 0 = x`, not a random condition perturbation. The outer connector gives `Z(F(x); Theta_z2) = 0`. So I should get `y_c = F(x) + 0 = F(x)`. Computing `F(x) = [1*0.7 + 0.5*(-0.2), -0.3*0.7 + 0.8*(-0.2)] = [0.6, -0.37]`, and running the controlled expression numerically returns `y_c = [0.6, -0.37]` — identical to within floating point. The controlled block is exactly the original block at step zero, while the copied branch behind the dead connector is still a functional pretrained backbone waiting to be switched on.

The apparent objection is that zero-initialized layers are supposed not to learn at all — they are usually avoided for exactly that reason. I need to check the derivative directly rather than wave it away. For one output channel of a `1 x 1` convolution, write `y = W x + b`; for a scalar simplification, `y = w x + b`. The local derivatives are `dy/dw = x`, `dy/dx = w`, and `dy/db = 1`. If `g = dL/dy`, then

    dL/dw = g x,
    dL/db = g,
    dL/dx = w g.

The structure here is the whole point. The input gradient `dL/dx = w g` is proportional to `w`, so with `w = 0` it vanishes on the first backward pass — the copied branch and the inner condition-side connector behind the output connector receive no gradient yet. But the weight gradient `dL/dw = g x` is proportional not to `w` but to the connector's live input `x` and the live upstream loss gradient `g`. Neither of those is zero: `g` comes from the normal diffusion loss of the still-functioning model, and `x` comes from the copied pretrained block, which is producing real features. So the output connector's own weights and bias should pick up gradient immediately even though it passes nothing downstream.

I verify this by running one optimizer step on the same tiny example, with an MSE loss pulling `y_c` toward zero so `g = y_c = [0.6, -0.37]`. By hand, the output connector's weight gradient for its first output channel is `g_0 * F(x) = 0.6 * [0.6, -0.37] = [0.36, -0.222]`; the autograd value for that row is `[0.36, -0.222]`, matching. Meanwhile the gradients on the copied block `Fc` and on the inner connector `Z(.;Theta_z1)` both come back exactly zero at this step, as the `dL/dx = wg` term predicts. So at step zero only the output connector moves. After one SGD step its weight is no longer zero (it becomes about `[[-0.036, 0.022], [0.022, -0.014]]`). Now `w != 0`, so `dL/dx = wg` is nonzero, and at the second step the copied branch and the inner condition-side connector finally receive nonzero gradients — autograd confirms `Fc.weight.grad` and `Z(.;Theta_z1).weight.grad` are now both nonzero. The path opens from the outside in: the output connector switches on first, and only then does gradient reach back through it into the rest of the control branch.

The full convolutional case is the same argument with sums over batch and spatial positions: each connector weight accumulates a sum of upstream-gradient times input-activation products, while the input gradient is multiplication by the transpose of the current connector weights. Zero weights block the input gradient at the first step but not the connector's own weight or bias gradients — exactly the asymmetry the scalar trace and the numeric run both show. The connector starts closed and learns its way open, so the frozen prior is never hit with random features and the dead-module worry does not materialize.

I then place the branch where the U-Net can actually use it. Stable Diffusion's decoder consumes encoder features through skip connections, and the middle block sits at the bottleneck. So the trainable copy should produce one zero-convolved control tensor for each encoder-side skip and one for the middle block. Stable Diffusion's U-Net has 12 input/encoder-side blocks, so this gives `12 + 1 = 13` control tensors: 12 routed to the decoder skips and one to the bottleneck. The frozen U-Net runs its encoder and middle normally, then adds the middle control to the bottleneck and adds the skip controls to the corresponding decoder skips. This steers denoising without changing the base weights.

The condition image still has to enter at the right resolution. I need an encoder `E` that maps an image-space condition `c_i` into a feature condition `c_f = E(c_i)` compatible with the latent U-Net. Stable Diffusion's latent grid for a `512 x 512` image is `64 x 64`, so the encoder has to take the hint down by a factor of eight in each spatial dimension. Three stride-2 convolutions do that: `512 -> 256 -> 128 -> 64`, which I confirm by pushing a `512 x 512` tensor through the stack and reading off a `64 x 64` feature grid. The same zero-initialization discipline applies here — the encoder ends in a zero-initialized projection, so at step zero the hint adds zero to the copied branch and the no-op property of the whole controlled network is preserved, not just the no-op property of the per-block connectors.

The training objective does not need a new term. The base diffusion model already learns by predicting the added noise:

    L = E_{z_0, t, c_t, c_f, epsilon ~ N(0, I)} [||epsilon - epsilon_theta(z_t, t, c_t, c_f)||_2^2].

I reuse that objective and optimize the condition branch and zero connectors. Dropping the text prompt to the empty string for half of training examples has a clear purpose: it prevents the model from treating the condition as a weak hint beside a strong text label, and forces the condition branch to carry spatial semantics by itself.

Writing the core out as code makes the wiring concrete: a `zero_module` that zeroes every parameter, a copied encoder that emits one zero-convolved control per skip plus one for the middle, a small downsampling hint encoder ending in a zero projection, and a training step that reuses the diffusion loss while dropping the prompt half the time.

```python
import torch
import torch.nn as nn
import copy

def zero_module(module):
    # zero every parameter -> the module outputs 0 until training moves it
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class HintEncoder(nn.Module):
    # downsample the 512x512 condition image to the latent grid; final projection starts at zero
    def __init__(self, in_ch=3, out_ch=320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1),    nn.SiLU(),   # stride-2
            nn.Conv2d(32, 96, 3, 2, 1),    nn.SiLU(),   # stride-2
            nn.Conv2d(96, 256, 3, 2, 1),   nn.SiLU(),   # stride-2
            zero_module(nn.Conv2d(256, out_ch, 3, padding=1)),
        )
    def forward(self, c_image):
        return self.net(c_image)

class ControlNet(nn.Module):
    def __init__(self, sd_unet):
        super().__init__()
        self.locked = sd_unet                                  # frozen prior, no grad
        for p in self.locked.parameters():
            p.requires_grad = False
        self.control_encoder = copy.deepcopy(sd_unet.encoder_blocks)   # trainable copy
        self.control_middle  = copy.deepcopy(sd_unet.middle_block)
        self.hint_enc = HintEncoder(out_ch=sd_unet.model_channels)
        self.zero_convs = nn.ModuleList([
            zero_module(nn.Conv2d(ch, ch, 1)) for ch in sd_unet.block_channels
        ])                                                     # one per skip
        self.middle_zero_conv = zero_module(nn.Conv2d(sd_unet.middle_channels,
                                                      sd_unet.middle_channels, 1))

    def forward(self, z_t, t, c_t, c_image):
        h = z_t + self.hint_enc(c_image)                       # hint adds 0 at init
        controls, x = [], h
        for block, zconv in zip(self.control_encoder, self.zero_convs):
            x = block(x, t, c_t)
            controls.append(zconv(x))                          # each is 0 at init -> y_c = y
        mid_control = self.middle_zero_conv(self.control_middle(x, t, c_t))
        return self.locked.decode_with_control(z_t, t, c_t, controls, mid_control)

def diffusion_loss(eps, eps_pred):
    return ((eps - eps_pred) ** 2).mean()                      # native objective, reused

def train_step(z0, t, c_t, c_image, model, opt, alpha_bar):
    eps = torch.randn_like(z0)
    z_t = alpha_bar[t].sqrt() * z0 + (1 - alpha_bar[t]).sqrt() * eps
    if torch.rand(()) < 0.5:                                   # drop the prompt half the time
        c_t = model.empty_prompt(c_t.size(0))
    loss = diffusion_loss(eps, model(z_t, t, c_t, c_image))
    opt.zero_grad(); loss.backward(); opt.step()               # only the copy + zero convs move
    return loss
```

At inference, classifier-free guidance introduces another case distinction. The standard form is `epsilon_prd = epsilon_uc + beta_cfg (epsilon_c - epsilon_uc)`. A control condition can be present in both branches or only in the conditional branch, and the 13 control connections can be scaled to tune condition strength. That is an inference-strength mechanism, not a change to the training loss or the zero-convolution derivation.

The pieces now hang together, and each load-bearing claim has been checked rather than assumed: keep the old model fixed; copy enough pretrained encoder capacity to learn the new control; connect the copy through zero-output adapters so the initial network reproduces the old one exactly (verified numerically on a small block); let the zero connectors learn because their own weight gradients depend on live inputs and live loss while only the input gradient is gated by `w` (verified by tracing two steps, where the output connector moves first and the branch behind it opens at the next step); and train with the original diffusion objective while sometimes removing the text shortcut.
