# VQ-VAE-2

## Problem

Generate high-resolution, high-fidelity, diverse images while retaining the evaluation and mode-coverage pressure of likelihood-based modeling. Direct pixel likelihood wastes effort on perceptually minor local detail and samples slowly; GANs can be sharp but do not provide a test likelihood and can miss modes.

## Method

Use a two-stage likelihood model in a compressed discrete latent space.

1. Train a hierarchical vector-quantized autoencoder. For `256x256` ImageNet images, it uses a `32x32` top code for global structure and a `64x64` bottom code for local detail. The bottom code is conditioned on the top code and on image features, so the levels carry complementary information rather than making the bottom only refine the top. The decoder is feed-forward and trained with pixel MSE.

2. Freeze the autoencoder, extract top and bottom code grids for the data, and train autoregressive priors over those codes. The top prior is a class-conditional PixelCNN/PixelSNAIL-style model with causal multi-head attention; the bottom prior is a class- and top-conditioned PixelCNN without attention. Sampling is ancestral: sample top, sample bottom given top, then decode once.

3. Optionally filter class-conditional samples with an ImageNet classifier score for the intended class. Keeping only a chosen top fraction gives a quality-diversity knob without retraining the generator.

## Objective

For each latent vector, nearest-prototype quantization is

`z_q(x) = e_k`, where `k = argmin_j ||z_e(x) - e_j||_2`.

The non-EMA VQ-VAE loss for minimization is

`||x - D(z_q(x))||_2^2 + ||sg[z_e(x)] - e_k||_2^2 + beta ||z_e(x) - sg[e_k]||_2^2`.

The first term is reconstruction, the second moves the selected codebook vector toward the encoder output, and the third commits the encoder output to the selected code. With a deterministic one-hot posterior and a uniform `K`-way prior, the KL is `log K` per latent position, or `N log K` for `N` independent positions, so it has zero gradient during autoencoder training.

In the EMA variant used for the large-image model, the codebook loss is replaced by online k-means:

`N_i^(t) = gamma N_i^(t-1) + (1 - gamma) n_i^(t)`

`m_i^(t) = gamma m_i^(t-1) + (1 - gamma) sum_j z_e(x)_{i,j}^{(t)}`

`e_i^(t) = m_i^(t) / N_i^(t)`.

The differentiable latent loss returned by the reference PyTorch quantizer is the unweighted commitment MSE; the training loop multiplies it by `beta = 0.25`. The ImageNet settings are codebook size `K = 512`, code dimension `D = 64`, EMA decay `gamma = 0.99`, hidden units `128`, residual units `64`, and two residual layers in the autoencoder.

## Code

```python
import torch
from torch import nn
from torch.nn import functional as F


def maybe_all_reduce(tensor):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor)
    return tensor


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = maybe_all_reduce(embed_onehot.sum(0))
            embed_sum = maybe_all_reduce(flatten.transpose(0, 1) @ embed_onehot)
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            self.embed.data.copy_(self.embed_avg / cluster_size.unsqueeze(0))

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        return self.conv(input) + input


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]
        else:
            raise ValueError("stride must be 2 or 4")

        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks += [
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ]
        elif stride == 2:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))
        else:
            raise ValueError("stride must be 2 or 4")

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed, decay=decay)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed, decay=decay)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.dec = Decoder(embed_dim + embed_dim, in_channel, channel,
                           n_res_block, n_res_channel, stride=4)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        return self.dec(torch.cat([upsample_t, quant_b], 1))

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t).permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b).permute(0, 3, 1, 2)
        return self.decode(quant_t, quant_b)

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        return self.decode(quant_t, quant_b), diff


criterion = nn.MSELoss()
latent_loss_weight = 0.25
for img, _label in loader:
    out, latent_loss = model(img)
    recon_loss = criterion(out, img)
    loss = recon_loss + latent_loss_weight * latent_loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The reference PyTorch `PixelSNAIL` prior uses categorical cross-entropy on code indices. Its default script trains a top prior on `32x32` codes and a bottom prior on `64x64` codes conditioned on the top codes; the ImageNet prior in the method also conditions on the class label.
