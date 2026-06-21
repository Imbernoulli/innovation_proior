Text-to-image diffusion models can synthesize striking images from captions, yet text is a poor handle for spatial structure. A user cannot easily describe an exact pose, edge layout, depth pattern, or segmentation mask in words, so the generated result often misses the intended composition. The natural fix is to supply an additional image-valued condition such as a Canny edge map, a depth map, a human-pose skeleton, or a semantic segmentation mask. The difficulty is learning that new control from far less data than the base model saw. If we finetune all weights of the pretrained model on a small condition-specific dataset, the rich visual prior can be overwritten by overfitting and catastrophic forgetting. Lightweight adapters or low-rank updates protect the prior, but they may lack the capacity to interpret complicated spatial semantics and translate them into the internal feature language of a deep U-Net.

A better approach must keep the original model intact while adding enough trainable capacity to learn the new condition, and it must avoid disturbing the pretrained activations at initialization. ControlNet satisfies all three requirements. It locks the production U-Net, creates a trainable deep copy of the encoder-side blocks plus the middle block, and connects the copy to the base model through zero-initialized one-by-one convolutions. Because every weight and bias in those connector convolutions starts at zero, the controlled block initially outputs exactly the same features as the original block. Training then gradually opens the condition path from the outside in.

The controlled block is written as y_c = F(x; Theta) + Z(F(x + Z(c; Theta_z1); Theta_c); Theta_z2). Here F(.; Theta) is the frozen pretrained block, F(.; Theta_c) is its trainable copy, and each Z denotes a one-by-one convolution initialized to zero. At step zero the inner zero convolution maps the condition to zero, the copied block sees the original input x, and the outer zero convolution maps the copied output to zero, so y_c = y. The network therefore starts as the original generator.

The zero convolutions can still learn. For a scalar one-by-one convolution y = w x + b initialized at w = 0, the gradients are dL/dw = (dL/dy) x and dL/db = dL/dy. Both can be nonzero immediately because the upstream diffusion loss gradient dL/dy and the copied-block input x are live. Only the gradient with respect to the connector's own input, dL/dx = (dL/dy) w, is zero at the first step. After one optimizer step the outer connector weights move away from zero, and then gradients can flow into the copied branch and the inner condition-side connector. The condition path thus opens safely, one layer at a time, without ever injecting random features at initialization.

Applied to Stable Diffusion, the trainable copy mirrors the twelve encoder-side input blocks and the middle block. It emits thirteen zero-convolved control tensors, one for each decoder skip connection and one for the bottleneck. The frozen U-Net runs its encoder and middle block as usual, adds the middle control at the bottleneck, and adds the skip controls before each decoder block. A small hint encoder maps the pixel-space condition image down to the latent grid and projects it to the U-Net's model channels, ending with a zero-initialized convolution so the condition also starts as a no-op.

Training reuses the standard diffusion noise-prediction objective and optimizes only the copied blocks, zero convolutions, and hint encoder. The base model stays frozen, so its broad prior is preserved. To prevent the model from ignoring the spatial condition and relying on the text prompt, we replace the text caption with the empty string for half of the training examples. At inference, classifier-free guidance can be applied as usual, and the thirteen control tensors can be scaled to adjust condition strength.

```python
import torch
import torch.nn as nn
import copy

def zero_module(module):
    """Zero every parameter so the module outputs 0 until training moves it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class HintEncoder(nn.Module):
    """Map a 512x512 condition image to a latent-grid feature tensor."""
    def __init__(self, in_ch=3, out_ch=320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 96, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(96, 256, 3, 2, 1), nn.SiLU(),
            zero_module(nn.Conv2d(256, out_ch, 3, padding=1)),
        )

    def forward(self, c_image):
        return self.net(c_image)

class ControlNet(nn.Module):
    """Trainable control branch wrapped around a frozen pretrained diffusion U-Net."""
    def __init__(self, sd_unet):
        super().__init__()
        self.locked = sd_unet
        for p in self.locked.parameters():
            p.requires_grad = False

        self.control_encoder = copy.deepcopy(sd_unet.encoder_blocks)
        self.control_middle = copy.deepcopy(sd_unet.middle_block)
        self.hint_enc = HintEncoder(out_ch=sd_unet.model_channels)

        self.zero_convs = nn.ModuleList([
            zero_module(nn.Conv2d(ch, ch, 1)) for ch in sd_unet.block_channels
        ])
        self.middle_zero_conv = zero_module(
            nn.Conv2d(sd_unet.middle_channels, sd_unet.middle_channels, 1)
        )

    def forward(self, z_t, t, c_t, c_image):
        h = z_t + self.hint_enc(c_image)
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
    if torch.rand(()).item() < 0.5:
        c_t = model.empty_prompt(c_t.size(0))
    loss = diffusion_loss(eps, model(z_t, t, c_t, c_image))
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss
```
