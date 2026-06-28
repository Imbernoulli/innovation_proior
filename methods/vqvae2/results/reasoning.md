I start with the property I do not want to give up: likelihood training gives me a real held-out objective and pressures the model to cover the data distribution. If the model puts near-zero mass on a region where the data actually lives, the data expectation of `-log p_model(x)` punishes it. That is the structural advantage over a generator trained only through a discriminator. But direct pixel likelihood asks the wrong object to do too much. It rewards probability mass on every local bit of texture, noise, and edge detail, and if I use an autoregressive pixel model I also have to sample a high-resolution image one conditional at a time.

So I need to move the density model away from pixels without losing the ability to decode images. Lossy compression gives the right mental model: keep the information needed for perception and reconstruction, throw away the rest, and model the compressed object. A feed-forward decoder can fill in plausible local detail once the compressed code has captured shape, pose, layout, and texture cues. The code should be discrete because a discrete grid is a natural target for a categorical autoregressive prior.

It is worth getting a feel for how aggressive that compression can be before I commit to it. A `256x256` RGB image has `256*256*3 = 196608` scalar values. If I compress to a couple of small index grids — say a `32x32` grid and a `64x64` grid of integers — that is `1024 + 4096 = 5120` symbols, a factor of `38.4` fewer numbers to model. If each symbol is one of `K = 512` codes, naively `log2(512) = 9` bits per symbol gives `5120 * 9 = 46080` bits versus `196608 * 8 = 1572864` bits at 8 bits per channel, about `34x` smaller. That is the right order of magnitude: a prior over a few thousand symbols is a tractable sequence-modeling problem, whereas a prior over two hundred thousand pixels is not. So the plan is worth pursuing if I can build a compressor whose codes are discrete and whose decoder is feed-forward.

The natural compressor is a vector-quantized autoencoder. The encoder maps an image to a grid of vectors `z_e(x)`. A codebook contains `K` vectors `e_1, ..., e_K`, each in the same `D`-dimensional space. For each encoder vector I choose the nearest prototype,

  `z_q(x) = e_k`, where `k = argmin_j ||z_e(x) - e_j||_2`.

The decoder reconstructs from the selected prototype grid, while the latent stored for the prior is only the grid of integer indices. This is a strong compression and it turns the prior-learning problem into categorical sequence modeling.

The argmin creates the first wall. It is piecewise constant, so ordinary backpropagation gives zero derivative from `z_q` to `z_e` almost everywhere. But `z_q` and `z_e` live in the same vector space. The decoder's gradient with respect to its input is still a useful direction for the encoder output. So I can try using the hard nearest code on the forward pass and pretend the quantizer is the identity on the backward pass:

  `z_q_st = z_e + sg[e_k - z_e]`.

I want to be sure this expression actually does both things at once, because it is easy to get the stop-gradient placement subtly wrong. Let me trace it concretely. Take `z_e = (0.3, -1.2, 0.7)` and suppose the selected prototype is `e_k = (1, 1, 1)`. Forward, the stop-gradient term is treated as a constant equal to `e_k - z_e`, so `z_q_st = z_e + (e_k - z_e) = e_k = (1, 1, 1)` exactly — the forward value is the hard quantized code, as required. Backward, the stop-gradient term has zero derivative, so `d z_q_st / d z_e = 1` (the identity). If some downstream loss sends back a gradient with coefficients `(2, 5, -1)` on the three components of `z_q_st`, those land on `z_e` unchanged as `(2, 5, -1)`. I checked this numerically and the gradient on `z_e` came back exactly `(2, 5, -1)`, with the forward value exactly `(1, 1, 1)`. So the single line gives the hard code forward and a clean identity backward. The estimator is biased — the true Jacobian of the argmin is zero, not the identity — but it is low-variance because there is no sampling, and it lets the encoder learn from the reconstruction signal.

That trick creates a second wall: the codebook vectors do not receive reconstruction gradients. The straight-through path routes reconstruction gradient around the embeddings, so the selected prototype must get its own learning rule. What I want from the codebook is just vector quantization: each prototype should move toward the mean of the encoder outputs assigned to it. In loss form, that is

  `||sg[z_e(x)] - e_k||_2^2`.

The stop-gradient freezes the encoder side, so only the selected code vector moves, and minimizing this term pulls `e_k` toward its assigned `z_e` — which is the k-means objective. But I also need to stop the encoder output from drifting arbitrarily far from the codebook while the codebook chases it. The latent space has no fixed scale, so without a commitment penalty the encoder could inflate its outputs and assignments could thrash. The encoder-side penalty is

  `beta ||z_e(x) - sg[e_k]||_2^2`.

Now the minimization loss is reconstruction plus codebook plus commitment:

  `||x - D(z_q(x))||_2^2 + ||sg[z_e(x)] - e_k||_2^2 + beta ||z_e(x) - sg[e_k]||_2^2`.

The stop-gradient placement is doing all the work here, so I want to read off exactly who each term trains. The reconstruction term trains the decoder and, through the straight-through estimator, the encoder. The codebook term has `sg` on the encoder side, so its only free variable is `e_k` — it trains the embedding alone. The commitment term has `sg` on the codebook side, so its only free variable is `z_e` — it trains the encoder alone. With `N` latent positions, these vector-quantization penalties are averaged over positions.

There is also a variational worry I should check before trusting this bottleneck, because VAEs are notorious for posterior collapse, where a strong decoder learns to ignore the latent and the KL term drives the posterior to the prior. Here the posterior over a code position is deterministic one-hot: it assigns probability one to the selected code and zero to all others. If I keep the training prior uniform over `K` codes, what is the KL? For one position,

  `KL(q || p) = sum_z q(z) log(q(z)/p(z))`.

Only the selected code `k*` contributes, since `q` is zero elsewhere (and `0 log 0 := 0`), so the sum is `1 * log(1 / (1/K)) = log K`. I evaluated this for `K = 512` both as `log 512` and as the explicit one-hot sum, and both give `6.238...` — a constant. For `N` independent positions it is `N log K`, still constant with respect to the encoder and decoder. So this KL term has zero gradient: the collapse route, where a powerful decoder is rewarded for making `q(z|x)` match a data-independent prior, simply has nothing to push on here. That is reassuring — the bottleneck cannot be optimized away. The uniform prior is only a training device, though; the actual code distribution is far from uniform, so for generation I will still need to learn the true code distribution afterward.

The original squared codebook term can be replaced by an EMA update. Given minibatch counts `n_i` and assigned encoder-output sums for code `i`, I track

  `N_i^(t) = gamma N_i^(t-1) + (1 - gamma) n_i^(t)`

  `m_i^(t) = gamma m_i^(t-1) + (1 - gamma) sum_j z_e(x)_{i,j}^{(t)}`

and set `e_i = m_i / N_i`, with small count smoothing to avoid division by zero. This is online k-means: a running average of the assigned encoder outputs, which is exactly the k-means centroid the codebook term was trying to reach by gradient descent. In that version, the differentiable latent loss is only the unweighted commitment MSE; the training loop multiplies it by `beta = 0.25`.

If I stop here, I have a flat code grid. That works as a compressor, but large images reveal a scale conflict. A single coarse grid can represent global shape but loses local detail; a single fine grid can preserve detail but spreads global structure across many positions and leaves the prior to learn both long-range geometry and local texture in one raster-order model. Those correlations live at different scales, and a single autoregressive model over one grid has to fit both with the same receptive field and the same conditioning order.

The natural correction is to split the code into levels. I want a small top grid for global structure and a larger bottom grid for local detail. For `256x256` images, the top grid can be `32x32` and the bottom grid `64x64`. But the relationship between the levels needs care. The tempting wiring is to make the bottom level a residual that only refines the top reconstruction. If I do that, though, the top has to carry essentially all the perceptual content and the bottom only corrects a few pixels — the top grid becomes the bottleneck again and the extra level buys little. The alternative is to let each level see the image. So I have the bottom level depend on image features directly, and additionally condition it on the top. Then the top can specialize in global layout and the bottom can add local information conditioned on that layout, without the top having to encode everything.

The encoder wiring follows that division. First I compute a bottom-resolution feature map from the image by downsampling by four. Then a second encoder downsamples those features by two to produce top-resolution features. I quantize the top features, decode that top code back up to bottom resolution, concatenate it with the original bottom-resolution image features, and quantize the result into the bottom code. The final decoder upsamples the top code, concatenates it with the bottom code, and decodes both levels to pixels in one feed-forward pass.

Before writing this up I want to confirm the resolutions and channel counts actually line up, because the two concatenations are the places where a dimension mismatch would silently break. I traced a `(1, 3, 256, 256)` tensor through the modules. The stride-4 bottom encoder takes it to `(1, 128, 64, 64)`; the stride-2 top encoder takes that to `(1, 128, 32, 32)`; the `1x1` conv to embedding dimension gives the top features `(1, 64, 32, 32)`, which is the `32x32` top grid I wanted. Decoding the top code back up (`dec_t`, stride 2) returns `(1, 64, 64, 64)`, and concatenating with the `(1, 128, 64, 64)` bottom features gives `(1, 192, 64, 64)`. That `192 = 64 + 128` is exactly what the bottom quantization conv must expect (`embed_dim + channel`), and it maps to the `(1, 64, 64, 64)` bottom grid. On the decode side, upsampling the top code gives `(1, 64, 64, 64)`, concatenating with the bottom code gives `(1, 128, 64, 64)` (`embed_dim + embed_dim`), and the stride-4 decoder returns `(1, 3, 256, 256)`. So the grids are `32x32` over `64x64`, the concat channel widths match the conv definitions, and the output is back at full resolution. That is the `5120`-symbol compression I estimated at the start.

I deliberately keep the pixel decoder feed-forward with MSE. If I put an autoregressive decoder in pixel space, it could absorb too much of the modeling burden and reintroduce hierarchy-collapse-style problems; it would also lose the speed advantage. The slow autoregressive part should live only in the compressed code space.

The reference stage-one quantizer is therefore compact:

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

The `dist` expression is the expanded `||z_e - e||^2 = ||z_e||^2 - 2 z_e . e + ||e||^2`, computed for all codes at once; `(-dist).max(1)` then returns the nearest-code index, which is the argmin I wanted. The implementation stores embeddings as `[D, K]`, expects the embedding dimension last for quantization, and returns the unweighted commitment loss `diff`. The line `quantize = input + (quantize - input).detach()` is exactly the straight-through identity I traced earlier. For distributed training, the assignment counts and sums have to be all-reduced before the EMA update; in a single-process sketch, those calls are no-ops.

The hierarchy around that quantizer is the architectural step that does the work:

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

This is the wiring I traced: `enc_b` then `enc_t` for the top code, `dec_t` plus a concat with `enc_b` for the bottom code, and `upsample_t` plus a concat for the final decode. Training this stage is just MSE plus the weighted commitment loss:

```python
out, latent_loss = model(img)
recon_loss = nn.MSELoss()(out, img)
loss = recon_loss + 0.25 * latent_loss.mean()
```

At this point generation is still missing. Uniformly sampled codes will be off-manifold, because real top and bottom code grids have strong spatial dependencies — and I already saw that the uniform prior used during autoencoder training was only there to make the KL a constant, not to describe the codes. So I freeze the autoencoder, encode the dataset, and train priors on the indices. The top code represents long-range structure on a small `32x32` grid; that is `1024` positions, so causal self-attention over the grid is affordable and useful for capturing global geometry. The bottom code is `64x64`, i.e. `4096` positions, where full self-attention is roughly sixteen times more expensive in pairwise terms and also less necessary because the top code already supplies global context — a conditional gated convolutional prior is the better-matched choice there. Sampling then follows the factorization:

  `p(c_top, c_bottom) = p_top(c_top) p_bottom(c_bottom | c_top)`,

with class labels included for the class-conditional ImageNet setting. Draw top codes, draw bottom codes conditioned on the top, and decode both. The reference PyTorch prior script implements the same top-then-bottom structure with cross-entropy over code indices; its default script does not use the label it loads, so class conditioning belongs to the method specification rather than that default script.

One last issue remains. Likelihood training covers the distribution, including rare or awkward modes, and a long ancestral chain over thousands of code positions can accumulate mistakes. A post-hoc filter gives a quality-diversity dial: generate class-conditional samples, score them with an independent ImageNet classifier by the probability assigned to the intended class, and keep a chosen top fraction. Tightening the fraction raises quality and lowers diversity; loosening it recovers the full model distribution. That gives a truncation-like control while leaving the learned likelihood model intact.
