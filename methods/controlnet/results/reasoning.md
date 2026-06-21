I start with the failure mode rather than with a module. I have a text-to-image diffusion model whose prior is valuable precisely because it was trained at huge scale, and I want to teach it a new spatial input from a much smaller paired dataset. If I keep training all of its weights on that smaller dataset, the new control may fit, but the broad prior can be overwritten. So the first requirement is separation: the old model should continue to provide the generative prior while the new parameters learn the condition.

The simplest way to guarantee that separation is to keep the production model's weights fixed. That removes the forgetting path completely. But freezing the old model creates the next problem: a small condition module may not have enough capacity to read complicated edges, poses, depth maps, or segmentations and translate them into the internal language of the U-Net. A thin adapter or low-rank update protects the base, but the spatial input can carry high-level semantics, so I need a strong branch rather than a token gesture.

The pretrained U-Net already contains a strong encoder. If I copy the encoder-side blocks and the middle block into a trainable branch, the new branch starts with useful image features instead of random features. The small condition dataset then adapts a pretrained representation instead of teaching a deep representation from scratch. The original path stays fixed and preserves the prior; the copied path supplies capacity.

Now I have a more delicate problem. A copied branch is trainable, and the condition encoder feeding it starts untrained. If I connect that branch into the fixed U-Net with ordinary random weights, the first forward pass injects random features into a model whose internal activations already have a tuned meaning. The connector has to begin as an exact no-op. A `1 x 1` convolution whose weight and bias are both zero has exactly that property: no matter what feature tensor enters it, its output is zero at initialization.

For a trained block `F(x; Theta) = y`, I make a trainable copy `F(.; Theta_c)` and use two zero convolutions, one on the condition path and one on the copied block's output:

    y_c = F(x; Theta) + Z(F(x + Z(c; Theta_z1); Theta_c); Theta_z2).

At initialization, both zero convolutions output zero. The inner one gives `Z(c; Theta_z1) = 0`, so the copied block receives `x`, not a random condition perturbation. The outer one gives `Z(F(x; Theta_c); Theta_z2) = 0`, so the combined block outputs `F(x; Theta)`. The full controlled block is therefore identical to the original block at step zero, while the copied branch remains a functional pretrained backbone.

The apparent objection is that zero-initialized layers are supposed not to learn. I check the derivative directly. For one output channel of a `1 x 1` convolution, write `y = W x + b`; for a scalar simplification, `y = w x + b`. The local derivatives are `dy/dw = x`, `dy/dx = w`, and `dy/db = 1`. If `g = dL/dy`, then

    dL/dw = g x,
    dL/db = g,
    dL/dx = w g.

With `w = 0`, the input gradient is zero on the first backward pass, so the copied branch and the inner condition-side zero convolution do not yet receive gradient through that output connector. But the weight gradient is not zero when the upstream loss gradient `g` and the connector input `x` are nonzero. Here `g` is supplied by the normal diffusion loss of the still-functioning model, and `x` is supplied by the copied pretrained block. After one optimizer step the output connector's weights can become nonzero; after that, `dL/dx = w g` also becomes nonzero and gradients can flow into the copied branch and then into the condition-side connector. The path opens from the outside in.

The convolutional case is the same argument with sums over batch and spatial positions: each connector weight gets a sum of upstream-gradient times input-activation products, while the input gradient is multiplication by the transpose of the current connector weights. Zero weights block the input gradient at the first step, but they do not block the connector's own weight or bias gradients. There is no sign flip or missing constant in the argument; the connector starts closed, then learns its way open.

I then place the branch where the U-Net can actually use it. Stable Diffusion's decoder consumes encoder features through skip connections, and the middle block sits at the bottleneck. So the trainable copy should produce one zero-convolved control tensor for each encoder-side skip and one for the middle block. The frozen U-Net runs its encoder and middle normally, then adds the middle control to the bottleneck and adds the skip controls to the corresponding decoder skips. This steers denoising without changing the base weights.

The condition image still has to enter at the right resolution. I need an encoder `E` that maps an image-space condition `c_i` into a feature condition `c_f = E(c_i)` compatible with the latent U-Net. Conceptually this can be a small convolutional encoder trained jointly with the control branch. In the concrete U-Net path I use interleaved convolutions and nonlinearities to downsample the `512 x 512` hint to the latent grid, then end with a zero-initialized projection so the hint does not perturb the copied branch at the start.

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

The design is now coherent: keep the old model fixed, copy enough pretrained encoder capacity to learn the new control, connect the copy through zero-output adapters so the initial network is exactly the old one, let the zero connectors learn because their own gradients depend on live inputs and live loss, and train with the original diffusion objective while sometimes removing the text shortcut.
