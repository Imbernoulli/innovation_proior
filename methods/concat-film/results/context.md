## Research question

Many neural networks must process one input under the influence of another: answer a question about an image, render an image in a chosen style, generate a sample from a requested class, or denoise an image at a particular noise level while respecting a label. The usual architecture splits the two streams. One sub-network embeds the conditioning signal, another sub-network processes the data, and the two are fused near the top before prediction or decoding.

That late-fusion recipe leaves most of the data pipeline unchanged by the conditioning input. Early and middle convolutional features are computed the same way whether the question asks for red cubes or leftmost spheres, and the final classifier must recover the requested computation from a single fixed representation. A useful general mechanism should let the conditioning signal affect intermediate computation, should be cheap enough to apply repeatedly, should work for arbitrary embeddings rather than a small table of known ids, and should avoid hard-coding a task-specific reasoning module.

## Background

The basic substrate is a convolutional or residual network whose blocks carry feature maps `F` with shape `[B, C, H, W]`. Normalization layers in these networks already expose a small per-channel control surface. Batch Normalization normalizes each channel over batch and spatial dimensions, then restores representational freedom with one learned scale and one learned shift per channel:

```text
x_hat = (F - E[F]) / sqrt(Var[F] + eps)
BN(F | gamma_c, beta_c) = gamma_c * x_hat + beta_c
```

Instance Normalization uses per-example spatial statistics rather than batch statistics, but it keeps the same post-normalization affine form. In both cases, the scale and shift are shared across spatial positions, so their parameter count is tied to the number of feature maps rather than image resolution.

Visual question answering makes the fusion problem concrete. Standard CNN+RNN pipelines extract image features and language features separately, then combine them with concatenation, elementwise products, bilinear pooling, or attention near the top. CLEVR was designed as a diagnostic setting for multi-step compositional questions over rendered 3D scenes; generic late-fusion models and stacked-attention models struggle there compared with systems that build in stronger compositional or relational priors.

Several earlier results show that small channel-wise controls can have large behavioral effects. Conditional Instance Normalization shares all convolutional weights across many styles and changes only the normalization affine selected by style id. Conditional Batch Normalization predicts changes to frozen BatchNorm affine parameters from a language embedding and applies them throughout a pretrained ResNet. These mechanisms show that channel-wise scaling and shifting can carry conditioning information, but they are tied to particular normalization settings or fixed lookup structures.

## Baselines

**Late fusion by concatenation, product, bilinear pooling, or attention.** The image stream and conditioning stream are computed mostly independently, then combined near the classifier or decoder. This is simple and general, but the conditioning input reaches the visual pipeline after most feature extraction has already happened.

**Neural module and program-execution models.** Program Generator + Execution Engine and neural module networks assemble question-dependent modules that mirror the compositional structure of a question. They give strong reasoning priors, but they may require program supervision or hand-designed modules for particular functions, which weakens their claim as general conditioning mechanisms.

**Relation Networks.** A Relation Network compares pairs of spatial feature vectors, concatenating question features into the pairwise MLP input and summing the resulting relation vectors. This directly encodes pairwise comparison, but its pairwise stage scales quadratically in the number of spatial locations.

**Conditional Instance Normalization.** A style id `s` selects rows from learned `N x C` tables:

```text
z = gamma_s * ((x - mu) / sigma) + beta_s
```

The convolutional weights are shared across styles, and only the post-normalization affine is style-specific. The limitation is the discrete table: a new style or a continuous/compositional conditioning input is not handled by the same row lookup.

**Conditional Batch Normalization.** A language embedding `e_q` is mapped to deltas that are added to frozen pretrained BatchNorm parameters:

```text
Delta_gamma = MLP(e_q)
Delta_beta  = MLP(e_q)
gamma_hat_c = gamma_c + Delta_gamma_c
beta_hat_c  = beta_c  + Delta_beta_c
BN(F | gamma_hat_c, beta_hat_c)
```

All ResNet parameters, including the original BatchNorm affine parameters, are frozen; only the small language-conditioned predictor is trained. This reaches early visual layers with few parameters, but the mechanism is framed as a modification of BatchNorm on a pretrained backbone.

**Adaptive Instance Normalization.** A second input supplies the channel statistics:

```text
AdaIN(x, y) = sigma(y) * ((x - mu(x)) / sigma(x)) + mu(y)
```

The scale and shift come from another image's statistics rather than a learned function of an arbitrary embedding, and the operation remains attached to instance normalization.

**Concatenation or additive-bias conditioning.** Conditional DCGANs, WaveNet, and Conditional PixelCNN either concatenate constant conditioning maps to an input or add a conditioning-dependent bias. For a following linear map,

```text
W [F; z] = W_F F + W_z z
```

the `z` term is independent of `F`, so the conditioning contribution is an additive feature-wise bias on the layer output. It can shift the output features, but it does not change the coefficient on the feature computation itself.

**Feature-wise gates.** LSTM gates and Squeeze-and-Excitation-style channel gates multiply features by a bounded factor:

```text
gate = sigmoid(g(F)) in (0, 1)
F_out = gate * F
```

This can attenuate or pass a channel, but it has no additive component and cannot amplify or negate a feature map.

## Evaluation settings

CLEVR is the natural visual-reasoning diagnostic: roughly 700K image-question-answer-program tuples over rendered 3D scenes, one-word answers from a fixed answer set, and question types including counting, existence, attribute query, attribute comparison, and number comparison. Program labels exist, but a general conditioning method is expected to learn from image-question-answer triples without relying on program supervision.

CLEVR-Humans tests transfer to free-form human questions on CLEVR images. CLEVR-CoGenT changes the train/test combinations of attributes and shapes, so it probes whether a model has learned disentangled concepts or memorized attribute-shape pairs.

In class-conditional diffusion, the fixed scaffold is a CIFAR-10 `UNet2DModel` denoiser trained to predict noise from `(x_t, timestep, class_id)`. The class label has 10 values, images are 32x32, training uses the same optimizer/loss/sampler across methods, and evaluation uses clean-fid/FID on generated samples against CIFAR-10 reference statistics. Only the route by which the class embedding enters the denoiser changes.

## Code framework

The existing diffusion scaffold already embeds the timestep, embeds the class label, feeds a side-conditioning embedding to the UNet residual blocks, and exposes two neutral slots: one slot prepares the embedding consumed by the blocks, and one optional hook processes hidden states between backbone blocks. The denoising interface and training loop are fixed.

```python
import torch
import torch.nn as nn


def prepare_conditioning(time_emb, class_emb):
    """Prepare the embedding used by the residual blocks.

    time_emb:  [B, D] timestep embedding
    class_emb: [B, D] class embedding
    """
    # TODO: the conditioning rule goes here.
    pass


class ClassConditioner(nn.Module):
    """Optional hidden-state conditioning hook after a block."""

    def __init__(self, channels, cond_dim):
        super().__init__()
        # TODO: optional hook parameters go here.
        pass

    def forward(self, h, class_emb):
        # h: [B, C, H, W] ; class_emb: [B, D]
        # TODO: return the hidden state passed to the next block.
        pass


class ClassConditionalDenoiser(nn.Module):
    def __init__(self, unet, num_classes):
        super().__init__()
        self.unet = unet
        self.time_embed_dim = unet.config.block_out_channels[0] * 4
        self.class_embed = nn.Embedding(num_classes, self.time_embed_dim)

        down_channels = [
            unet.config.block_out_channels[i] for i in range(len(unet.down_blocks))
        ]
        mid_channels = unet.config.block_out_channels[-1]
        up_channels = list(reversed(unet.config.block_out_channels))

        self.down_cond = nn.ModuleList(
            [ClassConditioner(ch, self.time_embed_dim) for ch in down_channels]
        )
        self.mid_cond = ClassConditioner(mid_channels, self.time_embed_dim)
        self.up_cond = nn.ModuleList(
            [ClassConditioner(ch, self.time_embed_dim) for ch in up_channels]
        )

    def forward(self, sample, timestep, class_labels):
        t_emb = self.unet.time_proj(timestep)
        t_emb = t_emb.to(dtype=self.unet.dtype)
        time_emb = self.unet.time_embedding(t_emb)
        class_emb = self.class_embed(class_labels)
        emb = prepare_conditioning(time_emb, class_emb)

        sample = self.unet.conv_in(sample)
        down_block_res_samples = (sample,)

        for i, block in enumerate(self.unet.down_blocks):
            sample, res_samples = block(hidden_states=sample, temb=emb)
            sample = self.down_cond[i](sample, class_emb)
            down_block_res_samples += res_samples[:-1] + (sample,)

        if self.unet.mid_block is not None:
            sample = self.unet.mid_block(sample, emb)
        sample = self.mid_cond(sample, class_emb)

        for i, block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(block.resnets)]
            sample = block(sample, res_samples, emb)
            sample = self.up_cond[i](sample, class_emb)

        sample = self.unet.conv_norm_out(sample)
        sample = self.unet.conv_act(sample)
        return self.unet.conv_out(sample)
```
