# VQGAN

## Problem

Pixel-space transformers are globally expressive but infeasible at high resolution: attention over `n = H * W` pixels costs `O(n^2)`. Convolutional models scale over pixels but are biased toward local structure. The useful split is therefore: learn a short discrete image representation with a CNN, then model the global composition of those discrete tokens with a transformer.

## Method

Train a convolutional vector-quantized autoencoder first. The encoder produces `z = E(x) in R^{h x w x n_z}`; each spatial vector is replaced by the nearest codebook entry from `Z = {z_k}_{k=1}^K`; the decoder reconstructs `x_hat = G(z_q)`.

`z_q = q(z) = (argmin_{z_k in Z} ||z_ij - z_k||)`, and `x_hat = G(z_q)`.

Use the straight-through estimator `z_q = z + sg[z_q - z]`. The paper-level VQ loss uses unit codebook and commitment terms:

`L_VQ = L_rec + ||sg[E(x)] - z_q||_2^2 + ||sg[z_q] - E(x)||_2^2`.

The final paper removed the earlier beta on the commitment term. The public code keeps a backward-compatible legacy quantizer: `VQModel` passes `beta=0.25`, and `VectorQuantizer2(legacy=True)` applies that beta to the legacy/wrong term. Treat the displayed paper objective and the released-code behavior as distinct facts.

The reconstruction term is perceptual, not pixel-only:

`L_rec = mean(|x - x_hat| + perceptual_weight * LPIPS(x, x_hat))`.

Add a patch discriminator. The paper writes the GAN game as `log D(x) + log(1 - D(x_hat))`; the canonical implementation defaults to hinge loss:

`L_D = 0.5 * (mean relu(1 - D(x)) + mean relu(1 + D(x_hat)))`

`L_G = -mean D(x_hat)`.

Balance the GAN term at the decoder's last layer `G_L`:

`lambda = ||grad_{G_L} L_rec|| / (||grad_{G_L} L_G|| + delta)`.

The paper states `delta = 10^-6`; the public code uses `1e-4`, clamps to `[0, 1e4]`, detaches, and multiplies by `disc_weight`. The discriminator factor is zero before `disc_start`; the supplement recommends at least one epoch of warm-up.

After the first stage is frozen, encode images to index sequences `s`. A decoder-only transformer models

`p(s) = prod_i p(s_i | s_<i)`, with cross-entropy next-index training.

Conditioning is prefix conditioning: a class/SOS token is a one-token prefix; spatial conditions are converted to token sequences and prepended. The loss is restricted to the image-code part. Default ordering is row-major; the paper's ordering ablation found row-major best among the tested permutations.

For large images, train on latent crops and sample with a sliding `16 x 16` latent window. At each raster position, the implementation builds a local image-token patch plus the matching condition patch, predicts the current local coordinate, samples with temperature/top-k, writes the code into the full latent grid, then decodes the completed grid.

## Code Artifact

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

## Edge Cases

- `VectorQuantizer2(legacy=True)` is code-faithful for released VQGAN configs; the paper equation is cleaner and unit-weighted.
- `lambda` uses paper `delta = 10^-6`, but code uses `1e-4`; do not mix those constants without saying which surface is meant.
- The shifted prefix slice `logits[:, c_len - 1:]` is intentional: the first kept logit predicts the first image token after seeing the full condition prefix.
- `pkeep < 1` corrupts some image-code inputs during transformer training; targets remain the true image indices.
- Unconditional sampling uses an SOS prefix; class sampling uses the class label as a one-token prefix; spatial conditions use a tokenized condition prefix.
- The default permuter is identity/row-major, but the code supports alternate permutations and reverses the permuter before decoding.
- Sliding-window high-resolution sampling is implemented outside the basic `sample` loop: it predicts one raster location from a local `16 x 16` latent patch plus condition patch.

Training uses two Adam optimizers with `betas=(0.5, 0.9)` for the first-stage autoencoder/discriminator and AdamW with weight decay grouping for the transformer.
