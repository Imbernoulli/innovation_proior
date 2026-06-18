I start with the property I do not want to give up: likelihood training gives me a real held-out objective and pressures the model to cover the data distribution. If the model puts near-zero mass on a region where the data actually lives, the data expectation of `-log p_model(x)` punishes it. That is the structural advantage over a generator trained only through a discriminator. But direct pixel likelihood asks the wrong object to do too much. It rewards probability mass on every local bit of texture, noise, and edge detail, and if I use an autoregressive pixel model I also have to sample a high-resolution image one conditional at a time.

So I need to move the density model away from pixels without losing the ability to decode images. Lossy compression gives the right mental model: keep the information needed for perception and reconstruction, throw away the rest, and model the compressed object. A feed-forward decoder can fill in plausible local detail once the compressed code has captured shape, pose, layout, and texture cues. The code should be discrete because a discrete grid is a natural target for a categorical autoregressive prior.

The obvious compressor is a vector-quantized autoencoder. The encoder maps an image to a grid of vectors `z_e(x)`. A codebook contains `K` vectors `e_1, ..., e_K`, each in the same `D`-dimensional space. For each encoder vector I choose the nearest prototype,

  `z_q(x) = e_k`, where `k = argmin_j ||z_e(x) - e_j||_2`.

The decoder reconstructs from the selected prototype grid, while the latent stored for the prior is only the grid of integer indices. This is a strong compression and it turns the prior-learning problem into categorical sequence modeling.

The argmin creates the first wall. It is piecewise constant, so ordinary backpropagation gives zero derivative from `z_q` to `z_e` almost everywhere. But `z_q` and `z_e` live in the same vector space. The decoder's gradient with respect to its input is still a useful direction for the encoder output. I can use the hard nearest code on the forward pass and pretend the quantizer is the identity on the backward pass:

  `z_q_st = z_e + sg[e_k - z_e]`.

Forward, the stop-gradient expression evaluates to `e_k`. Backward, it contributes no derivative, so the gradient lands on `z_e` unchanged. This estimator is biased, but it is low-variance and lets the encoder learn.

That trick creates a second wall: the codebook vectors do not receive reconstruction gradients. The straight-through path routes reconstruction gradient around the embeddings, so the selected prototype must get its own learning rule. The desired codebook update is just vector quantization: each prototype should move toward the mean of the encoder outputs assigned to it. In loss form, the codebook term is

  `||sg[z_e(x)] - e_k||_2^2`.

The stop-gradient freezes the encoder side, so only the selected code vector moves. But I also need to stop the encoder output from drifting arbitrarily far from the codebook while the codebook chases it. The latent space has no fixed scale, so without a commitment penalty the encoder can inflate its outputs and assignments can thrash. The encoder-side penalty is

  `beta ||z_e(x) - sg[e_k]||_2^2`.

Now the minimization loss is reconstruction plus codebook plus commitment:

  `||x - D(z_q(x))||_2^2 + ||sg[z_e(x)] - e_k||_2^2 + beta ||z_e(x) - sg[e_k]||_2^2`.

The stop-gradient placement matters. The reconstruction term trains the decoder and, through the straight-through estimator, the encoder. The codebook term trains only the embedding. The commitment term trains only the encoder. With `N` latent positions, these vector-quantization penalties are averaged over positions.

There is also a variational reason this bottleneck stays useful. The posterior over a code position is deterministic one-hot: it assigns probability one to the selected code and zero to all others. If I keep the training prior uniform over `K` codes, then

  `KL(q || p) = sum_z q(z) log(q(z)/p(z)) = log K`

for one position, and `N log K` for `N` independent positions. This is constant with respect to the encoder and decoder. The usual collapse route, where a powerful decoder is rewarded for making `q(z|x)` match a data-independent prior, has no gradient here. The uniform prior is only a training device; for generation I will still need to learn the true code distribution afterward.

The original squared codebook term can be replaced by an EMA update. Given minibatch counts `n_i` and assigned encoder-output sums for code `i`, I track

  `N_i^(t) = gamma N_i^(t-1) + (1 - gamma) n_i^(t)`

  `m_i^(t) = gamma m_i^(t-1) + (1 - gamma) sum_j z_e(x)_{i,j}^{(t)}`

and set `e_i = m_i / N_i`, with small count smoothing to avoid division by zero. This is online k-means. In that version, the differentiable latent loss is only the unweighted commitment MSE; the training loop multiplies it by `beta = 0.25`.

If I stop here, I have a flat code grid. That works as a compressor, but large images reveal a scale conflict. A single coarse grid can represent global shape but loses local detail; a single fine grid can preserve detail but spreads global structure across many positions and leaves the prior to learn both long-range geometry and local texture in one raster-order model. Those correlations live at different scales.

The natural correction is to split the code into levels. I want a small top grid for global structure and a larger bottom grid for local detail. For `256x256` images, the top grid can be `32x32` and the bottom grid `64x64`. But I have to be careful about the relationship between them. If the bottom level only refines the top, then the top has to carry nearly everything important and the hierarchy has not really divided the work. Instead, each level should depend on pixels, while the bottom also sees the top. Then the top can represent global layout and the bottom can add local information conditioned on that layout.

The encoder wiring follows that division. First I compute a bottom-resolution feature map from the image by downsampling by four. Then a second encoder downsamples those features by two to produce top-resolution features. I quantize the top features, decode that top code back up to bottom resolution, concatenate it with the original bottom-resolution image features, and quantize the result into the bottom code. The final decoder upsamples the top code, concatenates it with the bottom code, and decodes both levels to pixels in one feed-forward pass.

I deliberately keep the pixel decoder feed-forward with MSE. If I put an autoregressive decoder in pixel space, it could absorb too much of the modeling burden and reintroduce hierarchy-collapse-style problems; it would also lose the speed advantage. The slow autoregressive part should live only in the compressed code space.

The reference stage-one implementation is therefore compact:

```python
class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim, self.n_embed, self.decay, self.eps = dim, n_embed, decay, eps
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
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            self.embed.data.copy_(self.embed_avg / cluster_size.unsqueeze(0))

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
```

The implementation stores embeddings as `[D, K]`, expects the embedding dimension last for quantization, uses `(-dist).max(1)` as the nearest-code argmin, and returns the unweighted commitment loss. For distributed training, the assignment counts and sums have to be all-reduced before the EMA update; in a single-process sketch, those calls are no-ops.

The hierarchy around that quantizer is the key architectural step:

```python
class VQVAE(nn.Module):
    def __init__(self, in_channel=3, channel=128, n_res_block=2, n_res_channel=32,
                 embed_dim=64, n_embed=512, decay=0.99):
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
        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        return quant_t, quant_b, diff_t.unsqueeze(0) + diff_b.unsqueeze(0), id_t, id_b

    def decode(self, quant_t, quant_b):
        return self.dec(torch.cat([self.upsample_t(quant_t), quant_b], 1))

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t).permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b).permute(0, 3, 1, 2)
        return self.decode(quant_t, quant_b)

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        return self.decode(quant_t, quant_b), diff
```

Training this stage is just MSE plus the weighted commitment loss:

```python
out, latent_loss = model(img)
recon_loss = nn.MSELoss()(out, img)
loss = recon_loss + 0.25 * latent_loss.mean()
```

At this point generation is still missing. Uniformly sampled codes will be off-manifold, because real top and bottom code grids have strong spatial dependencies. So I freeze the autoencoder, encode the dataset, and train priors on the indices. The top code represents long-range structure on a small `32x32` grid, so causal self-attention is affordable and useful. The bottom code is `64x64`, so attention is too costly and less necessary because the top code already supplies global context; a conditional gated convolutional prior is the right match. Sampling then follows the factorization:

  `p(c_top, c_bottom) = p_top(c_top) p_bottom(c_bottom | c_top)`,

with class labels included for the class-conditional ImageNet setting. Draw top codes, draw bottom codes conditioned on the top, and decode both. The reference PyTorch prior script implements the same top-then-bottom structure with cross-entropy over code indices; its default script does not use the label it loads, so class conditioning belongs to the method specification rather than that default script.

One last issue remains. Likelihood training covers the distribution, including rare or awkward modes, and a long ancestral chain can accumulate mistakes. A post-hoc filter gives a quality-diversity dial: generate class-conditional samples, score them with an independent ImageNet classifier by the probability assigned to the intended class, and keep a chosen top fraction. Tightening the fraction raises quality and lowers diversity; loosening it recovers the full model distribution. That gives a truncation-like control while leaving the learned likelihood model intact.
