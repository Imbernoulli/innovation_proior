# ControlNet — adding spatial control to a frozen text-to-image diffusion model

## The problem it solves

Add a new spatial conditioning control (edges, depth, pose, segmentation, normals, scribbles) to a
large pretrained text-to-image diffusion model, learning it end-to-end from a *small*
condition-specific dataset (~100k) without overfitting and without catastrophically forgetting the
billions-of-images prior that makes the base model good. Text alone gives weak spatial control; direct
finetuning of the huge model on little data destroys it.

## The key idea

- **Lock the pretrained model.** Its weights never change — so it cannot forget, and the locked branch
  needs no gradient/memory at training time.
- **Trainable copy of its encoder.** Clone the pretrained encoder (and middle block) into a *trainable*
  copy and use it as the backbone for the conditioning input — a deep, already-capable feature
  extractor, so the small dataset only has to adapt it, not train it from scratch.
- **Zero convolutions.** Connect the copy back into the frozen model through `1×1` convolutions whose
  weight *and* bias are initialized to zero, so at initialization the added control is a no-op and the
  combined model is exactly the original — no random noise corrupts the protected deep features. The
  control parameters then grow from zero as training earns them.

## The ControlNet block

For a locked block `y = F(x; Θ)`, clone to a trainable copy `F(·; Θ_c)`, feed it the condition `c`, and
connect with two zero convolutions `Z(·; ·)` (`1×1`, weight = bias = 0):

    y_c = F(x; Θ) + Z( F( x + Z(c; Θ_z1); Θ_c ); Θ_z2 ).

At the first step both `Z(·) = 0`, so `y_c = F(x; Θ) + 0 = y` — identical to the original block. And
since `Z(c; Θ_z1) = 0`, the copy receives the real `x` and is fully functional (a true pretrained
backbone).

## Why a zero-initialized layer still learns

For a zero conv `y = w·x + b` with `w = b = 0`, the weight gradient is

    ∂L/∂w = (∂L/∂y) · x,

which is nonzero: the *locked* branch keeps the forward output and the loss alive (`∂L/∂y ≠ 0`), and
the cloned copy keeps the input alive (`x ≠ 0`). The gradient depends on the input and the loss, not on
`w`, so `w` moves off zero after one step. At step one `∂L/∂x = (∂L/∂y)·w = 0` (the copy gets no
gradient yet); once `w ≠ 0`, `∂L/∂x ≠ 0` and gradient flows into the copy and the inner zero conv. The
branch bootstraps off zero — producing the "sudden convergence" signature: quality is always high, and
the model abruptly starts following the condition at some step.

## Applying to Stable Diffusion

The base is a latent-diffusion U-Net (encoder, middle block, skip-connected decoder; CLIP text
encoder; positional time encoder). Make a trainable copy of the 12 encoder blocks and 1 middle block;
add their (zero-conv'd) outputs into the 12 decoder skip connections and the middle block. The locked
encoder/decoder need no gradients (~23% more memory, ~34% more time per iter than the base alone). A
small condition encoder maps the `512×512` conditioning image down to the `64×64` latent resolution
(four stride-2 conv layers, channels 16/32/64/128, ReLU, trained jointly).

## Training

Reuse the base model's native noise-prediction objective unchanged:

    L = E_{z_0, t, c_t, c_f, ε∼N(0,I)} [ ‖ ε − ε_θ(z_t, t, c_t, c_f) ‖₂² ],

with `z_t` the noisy latent at step `t`, `c_t` the text prompt, `c_f` the task condition. Randomly
replace 50% of text prompts with the empty string, forcing the condition to carry the spatial
semantics instead of leaning on text.

## Working code

```python
import torch
import torch.nn as nn
import copy

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class ConditionEncoder(nn.Module):
    # 512x512 condition image -> latent-resolution feature (Gaussian init, trained jointly)
    def __init__(self, in_ch=3, out_ch=320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1),    nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),    nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),   nn.ReLU(True),
            zero_module(nn.Conv2d(128, out_ch, 3, padding=1)),
        )
    def forward(self, c_image):
        return self.net(c_image)

class ControlNet(nn.Module):
    def __init__(self, sd_unet):
        super().__init__()
        self.locked = sd_unet
        for p in self.locked.parameters():
            p.requires_grad = False
        self.control_encoder = copy.deepcopy(sd_unet.encoder_blocks)
        self.control_middle  = copy.deepcopy(sd_unet.middle_block)
        self.cond_enc = ConditionEncoder(out_ch=sd_unet.model_channels)
        self.zero_convs = nn.ModuleList([
            zero_module(nn.Conv2d(ch, ch, 1)) for ch in sd_unet.block_channels
        ])
        self.middle_zero_conv = zero_module(nn.Conv2d(sd_unet.middle_channels, sd_unet.middle_channels, 1))

    def forward(self, z_t, t, c_t, c_image):
        h = z_t + self.cond_enc(c_image)
        controls, x = [], h
        for block, zconv in zip(self.control_encoder, self.zero_convs):
            x = block(x, t, c_t)
            controls.append(zconv(x))
        mid_control = self.middle_zero_conv(self.control_middle(x, t, c_t))
        return self.locked.decode_with_control(z_t, t, c_t, controls, mid_control)

def diffusion_loss(eps, eps_pred):
    return ((eps - eps_pred) ** 2).mean()

def train_step(z0, t, c_t, c_image, model, opt, alpha_bar):
    eps = torch.randn_like(z0)
    z_t = alpha_bar[t].sqrt() * z0 + (1 - alpha_bar[t]).sqrt() * eps
    if torch.rand(()) < 0.5:
        c_t = model.empty_prompt(c_t.size(0))         # drop the prompt 50% of the time
    loss = diffusion_loss(eps, model(z_t, t, c_t, c_image))
    opt.zero_grad(); loss.backward(); opt.step()
    return loss
```

## Why it works

Locking the base preserves the prior and makes training cheap; the trainable copy of the pretrained
encoder is a strong backbone so a small dataset suffices; and the zero convolutions make training start
as an exact no-op — yet they still learn, because their gradient depends on the (live) input and loss
rather than on the (zero) weight. So the control grows safely from zero without ever degrading the base
model's image quality. The base model's own loss is reused, and dropping the prompt half the time
forces the condition image to become a self-sufficient spatial control.
