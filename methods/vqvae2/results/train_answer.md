The goal I set myself is to generate high-resolution images that are sharp, varied, and faithful to the data, while keeping the one property I refuse to give up: a real likelihood objective. Minimizing negative log-likelihood is minimizing the forward KL from data to model up to the data entropy, so a likelihood model is structurally punished for putting near-zero mass where the data actually lives — that is the mode-coverage pressure that an implicit generator trained only through a discriminator does not directly enforce. The trouble is that asking pixel likelihood to do the whole job is asking the wrong object to do too much. Pixel negative log-likelihood rewards probability mass on every local bit of texture, sensor noise, and bit-plane detail, spending capacity on what matters numerically but little perceptually; and if I model pixels autoregressively I have to sample a high-resolution image one conditional at a time over an enormous grid. GANs at scale escape the speed and sharpness problem but give no held-out test likelihood and can quietly drop modes. So neither family alone is satisfactory: I want a tractable density somewhere in the system, I want to stop spending modeling effort on imperceptible pixel detail, I want to sample faster than pixel-by-pixel, and I still want enough global structure for coherent large images.

The resolution I propose is VQ-VAE-2: a two-stage likelihood model that does its density modeling not on pixels but on a compressed, discrete, hierarchical latent. Lossy compression is the guiding intuition — keep the information the decoder needs for perception and reconstruction, throw away the rest, and let the slow categorical density model operate only on that compact object while a fast feed-forward decoder fills in plausible local detail. The first stage is a vector-quantized autoencoder. The encoder maps an image to a grid of continuous vectors $z_e(x)$, a codebook holds $K$ prototypes $e_1,\dots,e_K$ in the same $D$-dimensional space, and each encoder vector is replaced by its nearest prototype,
$$z_q(x) = e_k, \qquad k = \arg\min_j \lVert z_e(x) - e_j \rVert_2,$$
so the only thing stored for the prior is a grid of integer indices — strong compression that turns prior learning into categorical sequence modeling. The first wall is that the $\arg\min$ is piecewise constant and gives zero gradient from $z_q$ back to $z_e$ almost everywhere. But $z_q$ and $z_e$ live in the same space, so the decoder's gradient with respect to its input is already a useful direction for the encoder output; I take the hard nearest code on the forward pass and pretend the quantizer is the identity on the backward pass with a straight-through estimator,
$$z_q^{\text{st}} = z_e + \operatorname{sg}\!\big[e_k - z_e\big],$$
which evaluates to $e_k$ forward and passes the gradient straight onto $z_e$ backward. It is biased but low-variance, and it lets the encoder learn. That trick creates the second wall: routing reconstruction gradient around the embeddings means the codebook vectors never see reconstruction gradient, so the selected prototype needs its own learning rule. The right rule is just vector quantization — each prototype moves toward the mean of the encoder outputs assigned to it — expressed by the codebook term $\lVert \operatorname{sg}[z_e(x)] - e_k \rVert_2^2$, whose stop-gradient freezes the encoder side so only the chosen code moves. And because the latent space has no fixed scale, the encoder could otherwise inflate its outputs and let assignments thrash while the codebook chases it, so a commitment penalty $\beta\,\lVert z_e(x) - \operatorname{sg}[e_k] \rVert_2^2$ pins the encoder to its selected code. The full minimization objective is therefore
$$\lVert x - D(z_q(x)) \rVert_2^2 \;+\; \lVert \operatorname{sg}[z_e(x)] - e_k \rVert_2^2 \;+\; \beta\,\lVert z_e(x) - \operatorname{sg}[e_k] \rVert_2^2,$$
where the stop-gradient placement is load-bearing: the reconstruction term trains the decoder and, through the straight-through path, the encoder; the codebook term trains only the embedding; the commitment term trains only the encoder. There is also a variational reason the bottleneck stays alive. The posterior over a code position is deterministic one-hot, and if the training prior is uniform over $K$ codes then for one position $\mathrm{KL}(q\Vert p) = \sum_z q(z)\log\frac{q(z)}{p(z)} = \log K$, and $N\log K$ for $N$ independent positions — constant with respect to encoder and decoder. The usual posterior-collapse route, where a powerful decoder is rewarded for making $q(z\mid x)$ match a data-independent prior, simply has no gradient here. That uniform prior is only a training device; for generation I will learn the true code distribution afterward.

For the large-image model I replace the squared codebook term with an exponential-moving-average update, which is online k-means: tracking the assignment counts and the assigned encoder-output sums for each code $i$,
$$N_i^{(t)} = \gamma N_i^{(t-1)} + (1-\gamma)\,n_i^{(t)}, \qquad m_i^{(t)} = \gamma m_i^{(t-1)} + (1-\gamma)\sum_j z_e(x)_{i,j}^{(t)}, \qquad e_i^{(t)} = m_i^{(t)} / N_i^{(t)},$$
with small count smoothing to avoid division by zero. In this variant the differentiable latent loss reduces to the unweighted commitment MSE, and the training loop multiplies it by $\beta = 0.25$. A single flat code grid, however, exposes a scale conflict on large images: a coarse grid captures global shape but loses local detail, while a fine grid preserves detail but smears global structure across many positions and forces one raster-order prior to learn both long-range geometry and local texture, correlations that live at different scales. The correction is to split the code into levels — a small top grid for global structure and a larger bottom grid for local detail, $32\times32$ and $64\times64$ for $256\times256$ images — but to wire them so they divide the work rather than letting the bottom merely refine the top. The key design choice is that each level depends on the image, while the bottom additionally sees the top: the bottom encoder downsamples the image by four; the top encoder downsamples those features by two; the top features are quantized; the top code is decoded back up to bottom resolution and concatenated with the original bottom-resolution image features before that concatenation is quantized into the bottom code. The final decoder upsamples the top code, concatenates it with the bottom code, and reconstructs pixels in one feed-forward MSE pass. I deliberately keep the pixel decoder feed-forward rather than autoregressive: an autoregressive pixel decoder would absorb too much of the modeling burden, reintroduce hierarchy-collapse, and forfeit the speed advantage, so the slow autoregressive modeling must live only in the compressed code space.

Stage one is only a compressor; generation needs the code distribution, because uniformly sampled top and bottom grids are off-manifold given their strong spatial dependencies. So I freeze the autoencoder, encode the dataset, and train autoregressive priors over the indices, factorized as
$$p(c_{\text{top}}, c_{\text{bottom}}) = p_{\text{top}}(c_{\text{top}})\, p_{\text{bottom}}(c_{\text{bottom}} \mid c_{\text{top}}).$$
The top code is a small $32\times32$ grid of long-range structure, so causal multi-head self-attention is affordable and useful — a class-conditional PixelSNAIL-style model; the bottom code is $64\times64$, where attention is too costly and less necessary because the top already supplies global context, so a class- and top-conditioned gated PixelCNN without attention is the right match. Sampling is ancestral: draw the top codes, draw the bottom codes conditioned on the top (with the class label in the class-conditional ImageNet setting), and decode once. One last lever: because likelihood training covers the whole distribution including rare and awkward modes, and a long ancestral chain accumulates mistakes, I add a post-hoc filter — score class-conditional samples with an independent ImageNet classifier by the probability it assigns to the intended class and keep a chosen top fraction. Tightening the fraction raises quality and lowers diversity; loosening it recovers the full model distribution, giving a truncation-like quality–diversity dial without retraining the generator. The ImageNet settings are codebook size $K = 512$, code dimension $D = 64$, EMA decay $\gamma = 0.99$, $128$ hidden units, $64$ residual units, and two residual layers.

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
