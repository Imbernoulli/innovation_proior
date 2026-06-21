# ControlNet: Spatial Control for a Pretrained Diffusion U-Net

## Core Method

ControlNet adds an image-valued condition to a pretrained text-to-image latent diffusion model by keeping the production U-Net as the preserved path and adding a trainable copy of the encoder-side U-Net blocks plus the middle block. The trainable copy learns the new condition; the preserved path keeps the original image prior.

For a pretrained block

```text
y = F(x; Theta)
```

the controlled block is

```text
y_c = F(x; Theta) + Z(F(x + Z(c; Theta_z1); Theta_c); Theta_z2),
```

where `Theta` is locked, `Theta_c` is trainable, and each `Z` is a `1 x 1` convolution with both weight and bias initialized to zero. At initialization, both zero convolutions output zero, so `y_c = y`; the controlled network starts as the original network.

## Zero-Convolution Gradient

For a scalar view of a zero convolution, `y = w x + b` with `w = 0` and `b = 0`:

```text
dy/dw = x
dy/db = 1
dy/dx = w
dL/dw = (dL/dy) x
dL/db = dL/dy
dL/dx = (dL/dy) w
```

Thus the connector's weight and bias can receive nonzero gradients on the first step when its input and the upstream loss gradient are nonzero. Its input gradient is zero on the first step, so the copied branch behind the output connector opens only after the output connector moves away from zero.

## Stable Diffusion Wiring

Applied to Stable Diffusion, this copies the 12 input/encoder-side blocks and the middle block. The copied path emits 13 zero-convolved control tensors: 12 for decoder skip connections and one for the middle block. The base U-Net consumes them by adding the middle control at the bottleneck and adding skip controls before each decoder block.

The condition encoder `E(c_i) = c_f` maps a `512 x 512` condition image into a feature tensor matching the latent U-Net. Concretely this is a small convolutional hint network: `3 x 3` convolutions with SiLU activations, three stride-2 downsampling stages, and a final zero-initialized projection to `model_channels`.

## Training Objective

The training loss is the original diffusion noise-prediction loss:

```text
L = E_{z_0,t,c_t,c_f,epsilon ~ N(0,I)}
    [ ||epsilon - epsilon_theta(z_t, t, c_t, c_f)||_2^2 ].
```

During training, 50% of text prompts are replaced by the empty string so the condition image must carry spatial semantics rather than relying on the text prompt.

## Reference Code Shape

Zero modules are initialized by zeroing every parameter:

```python
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module
```

The control branch returns one zero-convolved tensor for every input block plus one for the middle block:

```python
class ControlNet(nn.Module):
    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    def forward(self, x, hint, timesteps, context, **kwargs):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels, repeat_only=False))
        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []
        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            if guided_hint is not None:
                h += guided_hint
                guided_hint = None
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        return outs
```

The locked U-Net consumes these controls in reverse order along the decoder path:

```python
class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels, repeat_only=False))
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for module in self.output_blocks:
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        return self.out(h.type(x.dtype))
```

The control tensors are computed only when an image condition is present, scaled by per-resolution `control_scales`, and passed to the U-Net. By default only `control_model.parameters()` are trained (the base stays locked); optionally unlocking the base U-Net output blocks and output head trains them too, which is a riskier option reserved for specialized or large datasets.

## Inference Controls

Classifier-free guidance starts from

```text
epsilon_prd = epsilon_uc + beta_cfg (epsilon_c - epsilon_uc).
```

A design choice is whether the image condition enters both the unconditional and conditional branches or only the conditional branch. To keep guidance stable, CFG Resolution Weighting scales the connection at block `i` by `w_i = 64 / h_i` across the 13 control resolutions. A simpler practical schedule used at inference is an exponential `guess_mode` falloff:

```python
model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)]
```

and use `[strength] * 13` outside `guess_mode`.

The essential mechanism is unchanged: a high-capacity copied encoder learns the spatial condition, and zero-initialized connectors make the controlled model start as the original model before the condition path gradually becomes active.

