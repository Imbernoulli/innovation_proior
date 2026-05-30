OK, let me think this through from scratch. I want to generate images that look genuinely real, at high resolution, and — this is the part I care about most — that span the *whole* variety of a dataset as messy as ImageNet, with a thousand classes and enormous within-class diversity. And I want the training objective to give me something measurable, so I can tell whether the model is generalizing rather than memorizing or quietly collapsing onto a few favorite pictures.

Let me lay out the two kinds of generative model I have and find where each one hurts. On one side, the adversarial models. A generator maps noise to an image, a discriminator scores real-vs-fake, and they play a minimax game. Scaled up they produce the sharpest, highest-resolution images anyone has — but I keep running into the same complaint: they drop modes. A generator can win the game while only ever emitting a slice of the true distribution, and I have no clean way to measure, on held-out data, how much of the distribution it has thrown away. The objective doesn't even ask it to cover everything. And evaluation is a mess: I'm reduced to staring at samples or trusting proxy numbers like Inception Score and FID. That's not a foundation I trust for "did it learn the data."

On the other side, the likelihood models — VAEs, flows, autoregressive nets. They maximize log-likelihood, i.e. they minimize the forward KL from data to model. Stare at that for a second: forward KL puts an expectation under the *data* distribution of −log p_model. If the model assigns near-zero mass to some region the data actually occupies, −log p blows up. So the objective *forces* the model to put mass on every mode; it structurally cannot ignore a chunk of the data the way a GAN can. And the number it optimizes is a real likelihood, comparable across models and measurable on a test set. That's exactly the property I wanted.

So why isn't this just the answer? Because doing maximum likelihood directly in pixel space is unsatisfying in three concrete ways. First, pixel NLL is a poor proxy for how good a sample looks — a model can have great bits-per-dim and produce mush, because most of the bits live in high-frequency local texture that barely affects perception. Second, and relatedly, nothing in the objective pushes the model to spend its capacity on *global* structure rather than that imperceptible local detail; the bits are where the noise is. Third, the strongest pixel-space likelihood models are autoregressive — they factor p(x) = Π_i p(x_i | x_{<i}) and generate one pixel at a time in raster order. For a 256×256×3 image that's around two hundred thousand sequential steps per sample. It's far too slow to push to high resolution, and most of those steps are spent predicting texture nobody would miss.

Let me hold those three complaints together, because they rhyme. They're all the same disease: the model is forced to model perceptually negligible information. What if I just... didn't make it model that?

Here's where lossy compression gives me the frame. JPEG throws away more than eighty percent of an image's data and I can't see the difference. So a huge fraction of the pixels carries no perceptual weight. If I could *first* compress the image down to a compact code that keeps only what matters perceptually, and *then* put my powerful generative model on the distribution of those codes instead of on pixels — all three complaints dissolve at once. The generative model never sees the negligible detail (a feed-forward decoder hallucinates plausible texture on the way back to pixels). The code is small, so an autoregressive model over it is fast. And by construction the code holds the structural content, so that's what the model spends its bits on.

So I need two things: a compressor that maps an image to a small code and back with little perceptual loss, and a powerful model of the distribution over those codes. Let me get the compressor first.

I'll reach for the vector-quantized autoencoder, because it's exactly built for this and it hands me a *discrete* code, which I'll want in a moment. The setup: an encoder E maps the image to a grid of continuous vectors. I keep a codebook — a table of K prototype vectors e_1…e_K, each of dimension D. To discretize, I replace each encoder vector by its nearest prototype,

  Quantize(E(x)) = e_k,  where k = argmin_j ‖E(x) − e_j‖,

and the decoder D reconstructs the image from that grid of prototypes. The code is the grid of indices k — tens of times smaller than the image.

The immediate problem is that argmin is piecewise constant: nudge the encoder output and the chosen index doesn't move, so ∂(quantized)/∂(encoder) is zero almost everywhere and no gradient reaches the encoder from reconstruction. The fix I'll borrow is the straight-through estimator: on the forward pass send the hard prototype e_k to the decoder, but on the backward pass pretend the quantizer was the identity and copy the decoder's gradient at e_k straight onto E(x). In an autodiff graph that's

  z_q = z_e + sg[e_k − z_e],

where sg is stop-gradient — forward it evaluates to e_k, backward it contributes nothing, so the gradient lands on z_e unchanged. It's biased (the gradient was computed at e_k, not at z_e) but low-variance and cheap, which is what makes it train.

That routing has a side effect: the gradient skips *around* the codebook, so the prototypes get no signal from reconstruction and would never move. They need their own objective. What I want is for each prototype to sit at the center of the encoder outputs that pick it — that's online k-means — so I add a term ‖sg[E(x)] − e‖², which drags the chosen prototype toward the encoder output while the stop-gradient keeps it from yanking the encoder. And one more failure to head off: the latent space has no fixed scale, so the encoder output can inflate and run away from whatever prototype it just chose, the prototype chases but never catches, and the assignments thrash. So I pin the encoder to its choice with a commitment term β·‖sg[e] − E(x)‖². Altogether

  L = ‖x − D(e)‖₂² + ‖sg[E(x)] − e‖₂² + β·‖sg[e] − E(x)‖₂².

The middle term I can equivalently implement as an exponential-moving-average update of each prototype toward the running mean of the encoder vectors assigned to it — literally an online k-means step with decay γ — which is more stable than pushing it by gradient. I'll use γ = 0.99. And β I'll set to 0.25; it's robust over a wide range, so it's not a knob I have to fuss over.

There's one more thing I get for free here that matters enormously given my goal. The posterior in this model is a deterministic one-hot (a vector picks exactly one code), and if I fix the prior over codes to uniform, the KL term in the variational objective is Σ q log(q/p) = 1·log(1/(1/K)) = log K — a constant. So there's no KL pressure trying to make the latent ignore the input. With a powerful decoder, an ordinary continuous VAE would happily switch its latent off (drive KL to zero by making the posterior input-independent) and let the decoder do everything — posterior collapse. Here that route is closed: the only way to lower the loss is reconstruction, which needs the code to carry information. The discrete bottleneck is load-bearing by construction. Good — that's the compressor.

Now, does this just work for big images if I drop it in as-is? Let me imagine training the single, flat version on 256×256. The encoder produces one grid of codes at one resolution, and that single grid has to carry *everything*: the global geometry of the object, its pose, the scene layout — and also the fur texture, the grass, the fine edges. There's a real conflict here. If I make the code grid coarse (say 32×32) to force it to summarize global structure, it can't hold enough to reconstruct fine detail and the images come out blurry. If I make it fine (say 64×64 or larger) so detail survives, then the global structure is now smeared across a large grid, and — here's the part that bites in stage two — the autoregressive model I'll put over this grid has to capture, in one model and one raster order, both the long-range correlations that define the object's shape and the short-range correlations of local texture. Those live at completely different scales. A single code at a single resolution forces a compromise that's bad at both ends.

So split it. Let me use a *hierarchy* of codes at different resolutions, and assign them different jobs. A **top** code at low resolution — for 256×256 I'll make it 32×32 — whose job is global structure: shape, geometry, layout. And a **bottom** code at higher resolution — 64×64 — whose job is local detail: texture, fine edges. Now each level can be modeled by a prior tuned to the correlation structure at *its* scale, instead of one model straining to do both.

How should the levels relate? My first instinct is the obvious refinement hierarchy: encode the top, and let the bottom only encode the *residual* the top couldn't capture. But think about what that does to the top. If the bottom is purely a refinement of the top, then the top is responsible for *all* the information that isn't local-residual — it has to encode every detail that the bottom merely sharpens, which overloads the small top grid and defeats the point of separating scales. I've also seen that a pure-refinement hierarchy with a strong (autoregressive) decoder tends to collapse — the levels stop dividing labor cleanly and need mitigation.

Let me back up and make a different choice: let *each* level depend on the pixels directly, not only on the level above. The bottom code is conditioned on the top code, yes — it should know the global context so it doesn't waste itself re-encoding shape — but it also looks at the image itself. Then the two levels carry *complementary* information rather than top-carries-everything / bottom-refines. Each contributes its own reduction of the reconstruction error: the top isn't forced to smuggle in detail it has no room for, and the bottom isn't reduced to a residual sharpener. This is the decision that makes the hierarchy actually divide the work.

Let me make the wiring concrete. The encoder for the bottom level, E_bottom, takes the image and downsamples it by a factor of 4: 256 → 64. Call that feature map enc_b. Then a second stack, E_top, takes enc_b and downsamples by another factor of 2: 64 → 32. Quantize the 32×32 map with the top codebook to get e_top. Now to form the bottom code with knowledge of the top: decode e_top back up to 64×64 (a small decoder, dec_t), concatenate that with enc_b — so the bottom sees both the top's global summary *and* the image features — run a 1×1 conv to mix down to the code dimension, and quantize with the bottom codebook to get e_bottom. The decoder then takes *both* levels: upsample e_top by 2 to meet e_bottom at 64×64, concatenate, and run a feed-forward decoder of residual blocks plus strided transposed convs back up to 256×256.

I'm deliberately keeping the decoder feed-forward and the reconstruction loss plain MSE in pixels — not an autoregressive pixel decoder. Two reasons. It keeps encoding and decoding fast, which is the whole appeal: one forward pass to go codes→image. And it sidesteps the hierarchy-collapse failure mode that comes with autoregressive decoders on a code hierarchy — with a feed-forward MSE decoder there's nothing for the levels to game, so I need no mitigation. The same VQ machinery applies at each level: codebook size K = 512, code dimension D = 64, β = 0.25, EMA codebook with γ = 0.99, and the per-level commitment/codebook losses just sum.

That's stage one. At this point I have, for any image, a 32×32 grid of top symbols and a 64×64 grid of bottom symbols, and a decoder that turns them back into a sharp image. But I can't *generate* yet — to sample, I need a distribution over the codes themselves. If I just sampled codes uniformly I'd get garbage; the real codes live on a thin, highly structured manifold (only certain top configurations are plausible, and the bottom is strongly conditioned on the top).

So stage two: fit a prior over the codes, after stage one is done. Why after, and why a learned prior rather than the fixed uniform one I used during quantization? Because fitting a prior to the actual distribution of codes the encoder produces closes the gap between that aggregate posterior and the prior. At sampling time I draw codes from this learned prior, and because it matches what the decoder actually saw during training, the decoded images are coherent rather than off-manifold. Information-theoretically I'm just re-encoding the codes with a distribution closer to their true one, pushing the bit rate toward their real entropy — and the smaller that gap, the more realistic the decoded samples. The fixed-uniform prior was a *training* convenience that killed posterior collapse; it was never meant to be the generative distribution.

What model do I put on the codes? The codes are a discrete grid, so the natural choice is an autoregressive model that factors p(codes) = Π_i p(code_i | code_{<i}) in raster order, with each conditional a categorical softmax over the K = 512 symbols — a PixelCNN, masked convolutions and all, but now operating on the tiny code grid instead of the full image. This is why discreteness was worth it: it makes the prior a clean categorical autoregressive model, and the grid is 30-odd times smaller than the pixels, so sampling it is correspondingly faster.

Now the hierarchy pays off a second time, because I can give each level a *different* prior matched to its job. The top code is 32×32 and carries global structure — long-range correlations between far-apart parts of the image (the two ends of an animal, symmetric features of a face). Plain masked convolutions have a bounded receptive field and are bad at long range. So for the top prior I interleave the gated residual convolutional stack with causal multi-head self-attention — attention has effectively unbounded reach, so it can tie together distant code locations. I can afford this here precisely *because* the top grid is small (32×32 = 1024 positions); attention is quadratic in sequence length but 1024 is manageable. Concretely I'll put attention in every fifth layer or so, and condition the whole top prior on the class label so it generates the right kind of object.

The bottom code is 64×64 — four times as many positions — and carries *local* detail. Two facts decide its prior. First, attention at 64×64 (4096 positions) is memory-prohibitive: the attention matrix would be enormous. Second, I don't need it: the bottom is already conditioned on the top, so the global context is handed to it for free, and local texture only needs a *local* receptive field, which convolutions give. So the bottom prior is a conditional PixelCNN with *no* attention. What it does need is a strong pathway from the top code, so I feed it through a deep residual conditioning stack that processes the top code and injects it into the bottom model. I'll also add dropout in both priors for regularization, and a small deep stack of 1×1-conv residual layers on top of the PixelCNN output, which I find lifts likelihood cheaply without much extra memory. And I train the two priors *separately* — the hierarchy lets me, since each is its own model — so each can use all the accelerator memory it needs.

Sampling is then ancestral and matches the structure: draw the top code from the top prior (conditioned on a class label), e_top ~ p_top; draw the bottom code from the bottom prior conditioned on the top, e_bottom ~ p_bottom(· | e_top); then one feed-forward pass of the decoder, x = D(e_top, e_bottom). Because the slow autoregressive sampling only happens over the small code grids and the pixel decoder is a single forward pass, the whole thing is dramatically faster than sampling pixels one at a time.

There's a tension I should confront before I call this done. I deliberately chose likelihood training because it covers all modes — but that same property has a cost at sample time. Maximum likelihood is forward KL, so the model is *forced* to put mass on every mode, including the rare and awkward ones; it can't quietly truncate the way a GAN does. So among my samples there will be some low-quality ones from the tails. And ancestral sampling from a long autoregressive chain compounds this: a slightly off code early in the raster order conditions everything after it, so errors accumulate over the sequence and occasionally derail a sample. GANs get a clean diversity-versus-quality dial (truncate the noise toward the mode for higher quality at the cost of coverage). I'd like an analogous dial that doesn't require retraining and doesn't permanently throw away the mode coverage I worked for.

The handle: a sample that lands on the true data manifold should be confidently recognizable as its intended class by an independent classifier, whereas an off-manifold or derailed sample should confuse it. So I take a classifier pretrained on ImageNet, generate many class-conditional samples, and score each by the probability the classifier assigns to the *intended* class. Keep the top-scoring fraction and discard the rest — accept/reject. Tightening the kept fraction trades diversity for quality (like truncation), but it's applied post-hoc, leaves the trained model untouched, and lets me dial back to full diversity whenever I want. It directly counters the tail-sample and error-accumulation problems by filtering on "does this look like a real member of its class."

Let me now write the stage-one model as code, because the wiring is where the hierarchy lives. The quantizer first — nearest-prototype lookup, EMA codebook, straight-through.

```python
import torch
from torch import nn
from torch.nn import functional as F

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim, self.n_embed, self.decay, self.eps = dim, n_embed, decay, eps
        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)                 # codebook [D, K]
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)                # [N, D]
        # squared distances to every prototype: ||z||^2 - 2 z.e + ||e||^2
        dist = (flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True))
        _, embed_ind = (-dist).max(1)                        # argmin distance = nearest prototype
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)                # snap to nearest code

        if self.training:                                    # EMA codebook update = online k-means
            embed_onehot_sum = embed_onehot.sum(0)           # count per prototype
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            # Laplace smoothing so empty clusters don't divide by zero
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            self.embed.data.copy_(self.embed_avg / cluster_size.unsqueeze(0))

        diff = (quantize.detach() - input).pow(2).mean()     # commitment loss (encoder -> code)
        quantize = input + (quantize - input).detach()       # straight-through: fwd=code, bwd=identity
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
```

The encoder/decoder are plain conv stacks; the only thing worth noting is the stride knob (4 to down/upsample by four, 2 by two), which is how I build the two resolutions.

```python
class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(), nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True), nn.Conv2d(channel, in_channel, 1))
    def forward(self, x): return self.conv(x) + x

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        if stride == 4:                                      # downsample by 4 (image -> bottom feats)
            blocks = [nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                      nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                      nn.Conv2d(channel, channel, 3, padding=1)]
        elif stride == 2:                                    # downsample by 2 (bottom feats -> top feats)
            blocks = [nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                      nn.Conv2d(channel // 2, channel, 3, padding=1)]
        for _ in range(n_res_block): blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x): return self.blocks(x)

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        for _ in range(n_res_block): blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4:                                      # upsample by 4 (decode back to pixels)
            blocks += [nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                       nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1)]
        elif stride == 2:                                    # upsample by 2 (decode top to bottom res)
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x): return self.blocks(x)
```

And the hierarchy — the encode path that makes the bottom see *both* the top code and the image, and the decode path that fuses both levels:

```python
class VQVAE(nn.Module):
    def __init__(self, in_channel=3, channel=128, n_res_block=2, n_res_channel=32,
                 embed_dim=64, n_embed=512, decay=0.99):
        super().__init__()
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)  # 256 -> 64
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)     # 64 -> 32
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)                                   # TOP codebook
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)                                   # BOTTOM codebook
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.dec = Decoder(embed_dim + embed_dim, in_channel, channel,
                           n_res_block, n_res_channel, stride=4)                         # both levels -> pixels

    def encode(self, input):
        enc_b = self.enc_b(input)                          # image features at 64x64
        enc_t = self.enc_t(enc_b)                          # downsample to 32x32
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)   # TOP code (global structure)
        quant_t = quant_t.permute(0, 3, 1, 2)
        dec_t = self.dec_t(quant_t)                        # decode top back to 64x64 for conditioning
        enc_b = torch.cat([dec_t, enc_b], 1)               # bottom sees top context AND image features
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)   # BOTTOM code (local detail, conditioned on top)
        quant_b = quant_b.permute(0, 3, 1, 2)
        return quant_t, quant_b, diff_t.unsqueeze(0) + diff_b.unsqueeze(0), id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)              # top 32 -> 64 to meet bottom
        quant = torch.cat([upsample_t, quant_b], 1)        # fuse both levels
        return self.dec(quant)                             # feed-forward back to 256x256 pixels

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        return self.decode(quant_t, quant_b), diff
```

Stage-one training is just reconstruction plus the bottleneck's commitment loss (the codebook itself moves via EMA inside `Quantize`), with the commitment weight β = 0.25:

```python
criterion = nn.MSELoss()
latent_loss_weight = 0.25                                  # this is beta
for img in loader:
    out, latent_loss = model(img)
    recon_loss = criterion(out, img)
    loss = recon_loss + latent_loss_weight * latent_loss.mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

Then stage two: run every training image through the frozen encoder to collect its (top, bottom) code grids, fit the top PixelCNN-with-attention on the top codes (conditioned on class), fit the conditional no-attention PixelCNN on the bottom codes (conditioned on class and top code), and at generation time sample top, then bottom-given-top, then decode once and optionally keep only the samples a pretrained classifier scores highly for their class.

```python
# stage 2 (schematic), fit after stage 1 is frozen
def extract_codes(model, loader):
    top, bottom = [], []
    for img in loader:
        _, _, _, id_t, id_b = model.encode(img)
        top.append(id_t); bottom.append(id_b)
    return top, bottom

# p_top:    autoregressive over 32x32 top codes, gated conv + causal multi-head attention, class-conditioned
# p_bottom: autoregressive over 64x64 bottom codes, gated conv, NO attention,
#           conditioned on class + the top code via a deep residual conditioning stack
def sample(p_top, p_bottom, model, label):
    code_t = p_top.sample(label)                           # global structure
    code_b = p_bottom.sample(label, condition=code_t)      # local detail given structure
    return model.decode_code(code_t, code_b)               # one feed-forward pass to pixels
```

So the causal chain, start to end: I want diverse, measurable, high-resolution generation, which points me to likelihood training for its mode coverage — but pixel-space likelihood wastes capacity on imperceptible detail and is too slow to sample. Lossy compression says throw that detail away first, so I compress the image to a small *discrete* code with a vector-quantized autoencoder (straight-through gradients, EMA codebook, commitment loss, and a deterministic one-hot posterior whose fixed-uniform prior makes the KL a constant — so no posterior collapse and a load-bearing bottleneck). A single flat code can't serve global structure and local texture at once, so I split it into a hierarchy: a small top code for global structure and a larger bottom code for local detail, with the bottom conditioned on the top *and* on the pixels so the levels carry complementary information, all reconstructed by a fast feed-forward MSE decoder that avoids hierarchy collapse. Generation needs a distribution over the codes, so after training I fit one autoregressive prior per level — self-attention on the small top grid where global long-range structure lives and it's affordable, none on the large bottom grid where it's prohibitive and unnecessary — and sample ancestrally, top then bottom, decoding in a single pass. Finally, because likelihood training forces full mode coverage at the cost of some tail samples, I add a post-hoc classifier-based rejection filter as a diversity-versus-quality dial that leaves the trained model untouched.
