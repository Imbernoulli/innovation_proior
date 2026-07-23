Text-to-image diffusion models can synthesize striking images from captions, yet text is a poor handle for spatial structure. A user cannot easily describe an exact pose, edge layout, depth pattern, or segmentation mask in words, so the generated result often misses the intended composition. The natural fix is to supply an additional image-valued condition such as a Canny edge map, a depth map, a human-pose skeleton, or a semantic segmentation mask. The difficulty is learning that new control from far less data than the base model saw. If we finetune all weights of the pretrained model on a small condition-specific dataset, the rich visual prior can be overwritten by overfitting and catastrophic forgetting. Lightweight adapters or low-rank updates protect the prior, but they may lack the capacity to interpret complicated spatial semantics and translate them into the internal feature language of a deep U-Net.

A better approach must keep the original model intact while adding enough trainable capacity to learn the new condition, and it must avoid disturbing the pretrained activations at initialization. ControlNet satisfies all three requirements. It locks the production U-Net, creates a trainable deep copy of the encoder-side blocks plus the middle block, and connects the copy to the base model through zero-initialized one-by-one convolutions. Because every weight and bias in those connector convolutions starts at zero, the controlled block initially outputs exactly the same features as the original block. Training then gradually opens the condition path from the outside in.

The controlled block is written as y_c = F(x; Theta) + Z(F(x + Z(c; Theta_z1); Theta_c); Theta_z2). Here F(.; Theta) is the frozen pretrained block, F(.; Theta_c) is its trainable copy, and each Z denotes a one-by-one convolution initialized to zero. At step zero the inner zero convolution maps the condition to zero, the copied block sees the original input x, and the outer zero convolution maps the copied output to zero, so y_c = y. The network therefore starts as the original generator.

The zero convolutions can still learn. For a scalar one-by-one convolution y = w x + b initialized at w = 0, the gradients are dL/dw = (dL/dy) x and dL/db = dL/dy. Both can be nonzero immediately because the upstream diffusion loss gradient dL/dy and the copied-block input x are live. Only the gradient with respect to the connector's own input, dL/dx = (dL/dy) w, is zero at the first step. After one optimizer step the outer connector weights move away from zero, and then gradients can flow into the copied branch and the inner condition-side connector. The condition path thus opens safely, one layer at a time, without ever injecting random features at initialization.

Applied to Stable Diffusion, the trainable copy mirrors the twelve encoder-side input blocks and the middle block. It emits thirteen zero-convolved control tensors, one for each decoder skip connection and one for the bottleneck. The frozen U-Net runs its encoder and middle block as usual, adds the middle control at the bottleneck, and adds the skip controls before each decoder block. A small hint encoder maps the pixel-space condition image down to the latent grid and projects it to the U-Net's model channels, ending with a zero-initialized convolution so the condition also starts as a no-op.

Training reuses the standard diffusion noise-prediction objective and optimizes only the copied blocks, zero convolutions, and hint encoder. The base model stays frozen, so its broad prior is preserved. To prevent the model from ignoring the spatial condition and relying on the text prompt, we replace the text caption with the empty string for half of the training examples. At inference, classifier-free guidance can be applied as usual, and the thirteen control tensors can be scaled to adjust condition strength.

```python
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

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
