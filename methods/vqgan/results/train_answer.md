# VQGAN

We want to synthesize high-resolution images — into the megapixel range, unconditionally and under rich conditioning such as a class label, a semantic segmentation map, a depth map, or a low-resolution image — and we want the result to be globally coherent, not merely a quilt of locally plausible texture. The difficulty is that the two model families good at the two halves of this task pull in opposite directions. The transformer is unmatched at long-range, global reasoning because its self-attention lets every position directly compare itself to every other: it forms $\mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$ with no built-in locality assumption. But that same all-pairs scoring makes attention cost $O(n^2)$ in the sequence length $n$, and for an image the sequence is the pixel grid whose length itself grows as the square of the side, so a pixel-space transformer hits a wall around $64\times64$. Convolutional networks are the mirror image: their local, shared kernels make cost grow only linearly in the pixel count, so they scale, but their hard locality-and-translation-invariance bias makes them weak at exactly the holistic composition we need. The escape routes that were on the table do not resolve this. Restricting attention to local windows reintroduces the locality bias I wanted the transformer to avoid; sparse and axial attention keep the full receptive field but only reduce cost from $n^2$ to $n\sqrt n$, still prohibitive past 64 pixels. Pixel-space convolutional autoregressive models (PixelCNN, PixelSNAIL) are cheap and scalable but weak at long-range structure and are beaten by transformers at low resolution. And the existing two-stage VQVAE prior is convolutional too, so it inherits the same long-range weakness, while its first stage is trained with a pixel-space $L_2$ loss that forces a low compression rate to keep reconstructions acceptable — leaving the code grid long.

The resolution is to stop handing pixels to the transformer at all, and I propose VQGAN to do it. The plan is a clean division of labor across two stages: a convolutional vector-quantized autoencoder learns a short, discrete, perceptually rich vocabulary of image parts, and an autoregressive transformer models the global composition of those parts as a short sequence of code indices. The transformer never touches pixels; the decoder never has to learn the global prior. Concretely, the encoder maps an image to a spatial grid $z = E(x)\in\mathbb{R}^{h\times w\times n_z}$, each spatial vector is replaced by its nearest entry in a learned codebook $Z=\{z_k\}_{k=1}^{K}$, and the decoder reconstructs $\hat x = G(z_q)$:
$$z_q = q(z) = \Big(\arg\min_{z_k\in Z}\,\lVert z_{ij}-z_k\rVert\Big),\qquad \hat x = G(z_q).$$
The same quantized grid reads off as integer indices $s_{ij}=k$ whenever $(z_q)_{ij}=z_k$, which is the token sequence the second stage will model. Because the nearest-neighbor assignment is piecewise constant, the gradient through the $\arg\min$ is zero almost everywhere, so I use the straight-through estimator $z_q = z + \mathrm{sg}[z_q - z]$: the forward value is the selected code, but the bracket is frozen, so the gradient with respect to $z$ is the identity. That straight-through copy routes the reconstruction gradient *around* the codebook, so the codebook needs its own signal, giving the VQ objective
$$\mathcal{L}_{\mathrm{VQ}} = \mathcal{L}_{\mathrm{rec}} + \lVert \mathrm{sg}[E(x)] - z_q\rVert_2^2 + \lVert \mathrm{sg}[z_q] - E(x)\rVert_2^2,$$
where the codebook term pulls each chosen prototype toward the encoder output that selected it (online k-means dictionary learning) and the commitment term pulls the encoder output toward its chosen prototype so the encoder commits to codes rather than letting its outputs drift. The clean paper-level objective uses unit weights on both terms, dropping the original VQVAE's tuned $\beta$; the released code keeps a backward-compatible legacy quantizer that passes $\beta=0.25$ and applies it to the legacy term, and I treat the displayed objective and that code behavior as distinct facts rather than reconciling them.

The load-bearing observation is that this first stage can quietly fail even if everything downstream is perfect. The transformer demands a short sequence, so the encoder must compress hard — a $256\times256$ image might become a $16\times16$ grid, each token summarizing a large region. Under that much residual uncertainty per token, a per-pixel $L_2$ (or $L_1$) reconstruction loss is minimized by the conditional mean, and the conditional mean of plausible high-frequency textures is blur. The transformer could then learn the code distribution flawlessly and still decode to mush. So the first-stage objective must stop treating pixel averages as the main notion of fidelity. The first fix is a perceptual feature loss: I compare $x$ and $\hat x$ in a fixed deep feature space (LPIPS over VGG), keeping only a small pixel $L_1$ term for anchoring,
$$\mathcal{L}_{\mathrm{rec}} = \mathrm{mean}\big(\lvert x-\hat x\rvert + w_{\mathrm{perc}}\cdot \mathrm{LPIPS}(x,\hat x)\big),$$
which makes blur expensive in a way raw pixel error does not. Perceptual distance alone can still leave local texture slightly soft, so I add an adversarial signal — and the discriminator should be patch-based, because the missing detail under heavy compression is overwhelmingly local texture and high-frequency structure. A fully convolutional patch discriminator emits a grid of real/fake logits, one per receptive field, rather than a single scalar for the whole image; that gives dense local feedback, works at arbitrary image size, and fits the division of labor exactly — the first stage makes local parts convincing while the transformer later handles how they compose. The adversarial game is $\log D(x) + \log(1 - D(\hat x))$, implemented with the steadier hinge surrogate
$$\mathcal{L}_D = \tfrac12\big(\mathbb{E}\,\mathrm{relu}(1-D(x)) + \mathbb{E}\,\mathrm{relu}(1+D(\hat x))\big),\qquad \mathcal{L}_G = -\mathbb{E}\,D(\hat x).$$

The coefficient on that adversarial term is not a harmless detail. The reconstruction and GAN losses live on different scales, and those scales drift during training, so a fixed $\lambda$ is too weak early and too strong late. I want the adversarial gradient arriving at the decoder to be comparable in magnitude to the reconstruction gradient, so I measure both where they actually meet — the decoder's last layer $G_L$ — and set the weight to their ratio:
$$\lambda = \frac{\lVert \nabla_{G_L}\mathcal{L}_{\mathrm{rec}}\rVert}{\lVert \nabla_{G_L}\mathcal{L}_G\rVert + \delta}.$$
The paper states $\delta=10^{-6}$; the implementation uses a larger guard $\delta=10^{-4}$, clamps the ratio to $[0,10^4]$, detaches it (so it acts as a scalar, not a path for gradients), and multiplies by a base discriminator weight. Even with this adaptive ratio, I do not want a discriminator firing against a random decoder and random codebook, so I gate its weight to zero at the start, train the autoencoder as a pure perceptual reconstructor first, and only then switch the adversarial term on — a warm-up of at least one epoch. The architecture preserves the same split: convolutional ResNet-style encoder and decoder with downsampling and upsampling by a factor $f=2^m$ (so the latent grid is $H/f\times W/f$), a single self-attention block at the lowest, cheapest resolution to let the autoencoder aggregate global context before choosing or decoding codes, and a $1\times1$ convolution mapping encoder channels to the codebook dimension before quantization plus another mapping back afterward.

With the first stage trained and frozen, every image becomes an index sequence $s$, and a decoder-only GPT-style transformer models it autoregressively, $p(s) = \prod_i p(s_i\mid s_{<i})$, trained by cross-entropy on the next index. One genuine choice has no canonical answer: there is no left-to-right order for a 2D grid. I considered spiral, Morton/Z-curve, coarse-to-fine, and alternating-row scans, but row-major wins on two grounds — it gives a stable "above-and-left is already known" context, and it is what the later sliding-window sampler wants; ordering is not invariant for next-token prediction, so I treat row-major as a design decision validated by the ablation rather than a free convention. Conditioning reuses the same interface as prefix conditioning: a class label or a start-of-sequence token is a one-token prefix; a spatial condition is converted to its own token sequence and prepended, so the transformer sees $[\text{condition}, \text{previous image codes}]$ with the loss restricted to the image-code targets. The off-by-one alignment is deliberate: with a condition prefix of length $c_{\mathrm{len}}$, I feed all but the last concatenated token, so the output at position $c_{\mathrm{len}}-1$ has seen the entire condition and predicts the first image code — which is why the kept image logits start at index $c_{\mathrm{len}}-1$. Sampling for normal grids is then a plain loop: start from the prefix, predict the next token, divide logits by temperature, optionally truncate to top-$k$ (commonly temperature $1.0$ and top-$k$ around $100$, larger for big or class-conditional codebooks), sample or take argmax, append, and finally map indices back to code vectors and decode with the frozen decoder. Megapixel generation needs one further constraint, because even after compression a very large latent grid can exceed the transformer's block size and I cannot keep raising $f$ — reconstruction quality collapses past a dataset-dependent point. So I train on latent crops that fit the transformer and sample large grids with a sliding $16\times16$ latent attention window: at each raster position I take the local window around the current location together with the already-generated codes above and to the left, prepend the matching condition patch when present, predict the current local coordinate, write the sampled code into the full grid, and decode once the grid is complete (coordinate conditioning supplies position when aligned unconditional data would otherwise leave windows blind to where they sit).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer2(nn.Module):
    def __init__(self, n_e, e_dim, beta, remap=None, sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.remap = remap
        self.sane_index_shape = sane_index_shape
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.e_dim)
        d = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        indices = torch.argmin(d, dim=1)
        z_q = self.embedding(indices).view(z.shape)

        if self.legacy:
            # Canonical backward-compatible branch used by released VQGAN configs.
            diff = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        else:
            # Corrected beta placement for a VQ-VAE-style commitment weight.
            diff = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        if self.sane_index_shape:
            indices = indices.view(z_q.shape[0], z_q.shape[2], z_q.shape[3])
        return z_q, diff, (None, None, indices)

    def get_codebook_entry(self, indices, shape):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)              # shape is (B, H, W, C)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    return value if global_step < threshold else weight


def hinge_d_loss(logits_real, logits_fake):
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


def vanilla_d_loss(logits_real, logits_fake):
    return 0.5 * (F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean())


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 perceptual_weight=1.0, disc_factor=1.0, disc_weight=1.0,
                 disc_conditional=False, disc_loss="hinge"):
        super().__init__()
        self.perceptual_loss = LPIPS().eval()
        self.discriminator = NLayerDiscriminator(input_nc=3, n_layers=3, ndf=64).apply(weights_init)
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.discriminator_iter_start = disc_start
        self.disc_conditional = disc_conditional
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.discriminator_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = self.pixel_weight * torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            rec_loss = rec_loss + self.perceptual_weight * self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
        nll_loss = rec_loss.mean()

        if optimizer_idx == 0:
            fake_in = reconstructions if cond is None else torch.cat((reconstructions, cond), dim=1)
            logits_fake = self.discriminator(fake_in.contiguous())
            g_loss = -torch.mean(logits_fake)
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
            except RuntimeError:
                d_weight = torch.tensor(0.0, device=inputs.device)
            disc_factor = adopt_weight(self.disc_factor, global_step, self.discriminator_iter_start)
            return nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

        if optimizer_idx == 1:
            real_in = inputs.detach() if cond is None else torch.cat((inputs.detach(), cond), dim=1)
            fake_in = reconstructions.detach() if cond is None else torch.cat((reconstructions.detach(), cond), dim=1)
            logits_real = self.discriminator(real_in.contiguous())
            logits_fake = self.discriminator(fake_in.contiguous())
            disc_factor = adopt_weight(self.disc_factor, global_step, self.discriminator_iter_start)
            return disc_factor * self.disc_loss(logits_real, logits_fake)

        raise ValueError(f"unknown optimizer_idx {optimizer_idx}")


class VQModel(nn.Module):
    def __init__(self, ddconfig, lossconfig, n_embed, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = VQLPIPSWithDiscriminator(**lossconfig)
        self.quantize = VectorQuantizer2(n_embed, embed_dim, beta=0.25, legacy=True)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.quant_conv(self.encoder(x))
        return self.quantize(h)

    def decode(self, quant):
        return self.decoder(self.post_quant_conv(quant))

    def forward(self, x):
        quant, diff, _ = self.encode(x)
        return self.decode(quant), diff

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class Net2NetTransformer(nn.Module):
    def __init__(self, first_stage_model, cond_stage_model, transformer,
                 permuter=None, pkeep=1.0, sos_token=0, unconditional=False):
        super().__init__()
        self.first_stage_model = first_stage_model.eval()
        self.cond_stage_model = SOSProvider(sos_token) if unconditional else cond_stage_model.eval()
        self.transformer = transformer
        self.permuter = permuter if permuter is not None else Identity()
        self.pkeep = pkeep
        self.sos_token = sos_token
        self.be_unconditional = unconditional

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        return quant_z, self.permuter(indices)

    @torch.no_grad()
    def encode_to_c(self, c):
        quant_c, _, info = self.cond_stage_model.encode(c)
        indices = info[2]
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    def forward(self, x, c):
        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep * torch.ones_like(z_indices, dtype=torch.float)).long()
            random_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            z_input = mask * z_indices + (1 - mask) * random_indices
        else:
            z_input = z_indices

        cz_indices = torch.cat((c_indices, z_input), dim=1)
        logits, _ = self.transformer(cz_indices[:, :-1])
        logits = logits[:, c_indices.shape[1] - 1:]
        return logits, z_indices

    @torch.no_grad()
    def sample(self, z_start, c_indices, steps, temperature=1.0, sample=True, top_k=None):
        x = torch.cat((c_indices, z_start), dim=1)
        for _ in range(steps):
            assert x.size(1) <= self.transformer.get_block_size()
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            if sample:
                next_idx = torch.multinomial(probs, num_samples=1)
            else:
                _, next_idx = torch.topk(probs, k=1, dim=-1)
            x = torch.cat((x, next_idx), dim=1)
        return x[:, c_indices.shape[1]:]

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(index.reshape(-1), shape=bhwc)
        return self.first_stage_model.decode(quant_z)
```
