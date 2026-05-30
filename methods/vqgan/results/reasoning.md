OK, let me think this through from scratch. I want to generate images — high-resolution ones, ideally up into the megapixel range — and I want them to be globally coherent, not just locally plausible. That last word is the whole problem. A model can paint convincing fur or brick texture in any local patch and still produce a face with two left eyes, or a building whose roofline doesn't agree with itself across the image. Local realism is cheap; global composition is the hard part. So whatever I build has to *understand the layout of the whole image at once*.

What's the tool for "understand the whole thing at once"? The transformer. Its defining move is that every element attends to every other element through `softmax(QKᵀ/√d_k)V` — there's no built-in assumption that interactions are local, so it's free to wire up a dependency between the top-left corner and the bottom-right corner directly. That's exactly the long-range reasoning I need for composition. For autoregressive generation I mask the upper triangle of `QKᵀ` to `−∞` so position `i` only sees `≤ i`, predict the next element's distribution, and factor `p(x) = Πᵢ p(xᵢ | x_{<i})`. Beautiful for capturing global structure.

But now the obvious thing: run a transformer over the pixels of a high-res image. The sequence length is the number of pixels. `QKᵀ` is `n×n`, so compute and memory are `O(n²)`. And `n` itself is `H·W`, which grows like the square of the side length. So the cost of a pixel transformer grows like the *fourth power* of resolution. People have tried — Image Transformer, ImageGPT — and they top out around 64 pixels, maybe 192 with heroics. There's no path from there to a megapixel by brute force. The expressivity I want and the cost I can't afford are the same property: attending to everything.

People have tried to dodge the cost. Restrict attention to a local window (Image Transformer, axial/blocked variants) — but then I've thrown away the global receptive field, which was the entire reason I reached for a transformer; I've just built a clumsy CNN. Sparse or axial attention keeps the full receptive field but only drops `n²` to `n√n` — that's a constant-factor reprieve, not a change of regime; at megapixel scale `n√n` is still hopeless. So I can't tame the transformer by making attention cheaper on pixels. The sequence is just too long.

Let me flip it. The transformer cost is set by sequence *length*, not by image resolution per se. What if the sequence the transformer sees isn't pixels at all, but something much shorter? An image has enormous local redundancy — neighboring pixels are highly correlated, textures repeat, smooth regions carry almost no information per pixel. A CNN is *exactly* the tool that exploits that: local kernels, linear cost in pixel count, and it scales to any resolution. CNNs are weak where transformers are strong (long-range, holistic structure) and strong where transformers are weak (cheap local processing). They're complementary. So split the labor: let a CNN compress the image down to a small grid that summarizes local content, and let the transformer do its global reasoning over *that* short grid. Decode back to pixels with a CNN. The image's low-level structure is handled convolutionally; the transformer concentrates on its unique strength, modeling the composition of the pieces.

For the transformer to model a distribution `Πᵢ p(sᵢ | s_{<i})` and predict a "next element," each element of that short grid should be a symbol from a finite vocabulary — a categorical token, like a word. Continuous vectors don't give me a clean categorical next-token distribution or a finite softmax head. So the compressed representation should be *discrete*: a small grid of indices into a learned vocabulary of image parts. That's a vector-quantized autoencoder. I have one off the shelf — VQVAE.

Let me lay it out. An encoder `E` maps the image to a grid of vectors `ẑ = E(x) ∈ ℝ^{h×w×n_z}` with `h,w` much smaller than `H,W`. I keep a codebook `Z = {z_k}_{k=1}^{K} ⊂ ℝ^{n_z}` of `K` prototype vectors. I quantize each spatial vector to its nearest prototype:

  `z_q = q(ẑ) := (argmin_{z_k ∈ Z} ‖ẑ_{ij} − z_k‖) ∈ ℝ^{h×w×n_z}`,

and decode `x̂ = G(z_q)`. Equivalently the grid is a sequence of `h·w` indices `s_{ij} = k` such that `(z_q)_{ij} = z_k`. The argmin is piecewise constant, so its gradient is zero almost everywhere — the gradient dies at the quantizer. The fix is the straight-through estimator: forward, use the hard quantized value; backward, pretend the quantizer was the identity and copy the decoder's gradient at `z_q` straight onto `E(x)`. Implement it as `z_q = ẑ + sg[z_q − ẑ]`, where `sg` is stop-gradient — forward this equals `z_q`, backward the bracket contributes nothing so `∂z_q/∂ẑ = I`. But that copy routes the reconstruction gradient *around* the codebook, so the prototypes get no signal from reconstruction; they need their own. So the loss carries a codebook term `‖sg[E(x)] − z_q‖²` that drags the chosen prototype toward the encoder output (online k-means), and a commitment term `‖z_q − sg[E(x)]‖²` that drags the encoder output toward its chosen prototype so the encoder commits rather than letting its output inflate and drift. Together:

  `L_VQ = ‖x − x̂‖² + ‖sg[E(x)] − z_q‖² + β‖z_q − sg[E(x)]‖²`.

So I have my two-stage plan: stage one is this discrete autoencoder, stage two is a transformer over the indices. Done? Let me imagine actually running it at the compression rate I need, and look for where it breaks.

Here's the tension. The transformer can only afford a short sequence — call the latent grid `16×16`, so `256` tokens, that's about the ceiling for a GPT-2-sized model on one GPU. For a `256×256` image that means a downsampling factor `f = 16` per side; `f = H/h`. For a megapixel image it means even more. So I'm asking the autoencoder to compress *hard* — squeeze a `16×16×3` block of pixels through a single code. And now stare at the reconstruction term `‖x − x̂‖²`. Under heavy compression there's real residual uncertainty about what's inside each block — which exact texture, which exact high-frequency detail. An `L₂` (or `L₁`) loss is minimized, in the presence of that uncertainty, by predicting the *average* — the conditional mean over all the plausible details. The average of many sharp textures is a smudge. So `L₂` reconstruction at high compression gives me blur, and the blurrier the codebook's reconstructions, the lower the fidelity ceiling on anything I sample later — the second-stage transformer can model the codes perfectly and I'll still decode to mush, because the decoder caps the achievable quality.

That's the wall. The whole plan only works if I can compress *hard* (short sequence for the transformer) *and* reconstruct *sharply* (high perceptual fidelity). VQVAE's `L₂` lets me have one or the other: low compression with okay sharpness, or high compression with blur. I need both. So the autoencoder's reconstruction objective has to change. The compression rate and the architecture are fine; it's the *loss* that's wrong.

What does `L₂` actually get wrong? It scores per-pixel error and treats every pixel independently, so it has no notion that a slightly-shifted-but-equally-sharp texture is perceptually as good as the original, while a blurred version is perceptually worse. It rewards exactly the wrong thing — pixel-mean-matching over perceptual realism. I want a loss that says "this looks like a real image of the right content," not "every pixel matches the mean."

First idea: compare in a perceptual feature space instead of pixel space. Take a fixed pretrained network — VGG — push both `x` and `x̂` through it, and compare deep features. Deep features encode texture and structure rather than raw intensities, and distances in that space track human similarity judgments far better than pixel distance. That's LPIPS. Swap the pixel `L₂` for `L_rec = LPIPS(x, x̂)` (in practice keep a small pixel `L₁` alongside it for stability, but let the perceptual term carry the fidelity). This already stops rewarding blur: a blurred reconstruction has visibly different VGG features, so it's penalized.

Is perceptual loss enough? It pulls features together, but it still doesn't have a sharp opinion about whether a *patch* of texture is realistic — VGG features can be matched by something that's close-but-still-a-bit-soft. I want a signal that actively hunts for the difference between a real image and a reconstruction and punishes it. That's adversarial. Add a discriminator `D` trained to tell real images from reconstructed ones; train the autoencoder (which is now playing the role of the generator) to fool it. As `D` gets sharper at spotting the tell-tale softness or artifacts of reconstructions, it forces `G` to produce crisper, more realistic output.

What kind of discriminator? My quality problem is *local* — it's textures and high-frequency detail inside each compressed block that go soft. Global plausibility I'm trusting the perceptual loss and the codes to handle. So I don't want a discriminator that collapses the whole image to one real/fake scalar; I want one that judges *patches*. A patch-based (fully-convolutional) discriminator outputs a grid of real/fake scores, one per `N×N` receptive field — exactly the pix2pix design. This concentrates the adversarial signal on local texture realism, gives a dense gradient (one signal per patch instead of one per image), uses fewer parameters, and applies to any image size. It also has a nice division-of-labor reading: if the patch discriminator forces the codebook to capture perceptually important *local* structure faithfully, then the transformer in stage two doesn't have to waste itself modeling low-level pixel statistics — it can spend its capacity on global composition, which is the only thing I wanted it for. The adversarial term written out (the standard GAN form for `G` vs `D`):

  `L_GAN({E,G,Z}, D) = [log D(x) + log(1 − D(x̂))]`,

with `D` maximizing and the autoencoder minimizing. (In practice the saturating `log(1−D(x̂))` is fragile, so I'll train `D` with a hinge loss — push `D(x)` above `+1` and `D(x̂)` below `−1`, i.e. `½(E[relu(1−D(x))] + E[relu(1+D(x̂))])` — and train `G` to maximize `D(x̂)`, i.e. minimize `−mean D(x̂)`. Same game, steadier gradients.)

So the complete first-stage objective is: find the autoencoder + codebook that minimizes the reconstruction-plus-codebook loss and the adversarial loss, while the discriminator maximizes its ability to discriminate,

  `Q* = argmin_{E,G,Z} max_D  E_{x∼p(x)} [ L_VQ + λ·L_GAN ]`.

Now `λ`. This is where I expect the real trouble, so let me actually think about it rather than just write a coefficient. The two terms I'm summing — the perceptual reconstruction loss and the adversarial loss — are completely different beasts with completely different gradient magnitudes, and worse, those magnitudes *change over training*. Early on the discriminator is near-random, so its gradient into `G` is tiny and noisy; later it sharpens and its gradient can swamp the reconstruction signal. If I fix `λ` to some constant, I'm guaranteed to get it wrong somewhere along the way: too small and the adversarial term does nothing and I'm back to blur; too large and the GAN gradient overwhelms reconstruction, destabilizes training, and the autoencoder stops actually reconstructing the input. A static weight can't track a moving ratio.

So what do I actually want? I want the *adversarial gradient* and the *reconstruction gradient* to arrive at the generator with comparable magnitude, automatically, at every point in training. The place they meet is the decoder — specifically its last layer `G_L`, the layer right before the output pixels, through which every gradient into the decoder must pass. So measure both gradients *there*. Let `∇_{G_L}[L_rec]` be the gradient of the reconstruction loss with respect to the last decoder layer's weights, and `∇_{G_L}[L_GAN]` the gradient of the adversarial loss there. To make the adversarial contribution match the reconstruction contribution in scale, set

  `λ = ‖∇_{G_L}[L_rec]‖ / (‖∇_{G_L}[L_GAN]‖ + δ)`,

with a small `δ = 10⁻⁴` for numerical safety when the GAN gradient is near zero. Read it: if the GAN gradient is large relative to reconstruction, `λ` shrinks and damps it; if it's small, `λ` grows and amplifies it. The product `λ·∇[L_GAN]` is normalized to the magnitude of `∇[L_rec]` at the layer where they fight. It's adaptive *by construction* — it tracks the moving ratio I was worried about, with no constant to tune. (Implementation realities: I clamp `λ` to a sane range like `[0, 10⁴]` so a vanishing reconstruction gradient can't blow it up, detach it so it's a scalar weight and not part of the graph, and I can fold in a base discriminator weight as an overall knob.)

There's still the start-of-training instability. A randomly initialized decoder and codebook produce garbage; turning the adversarial pressure on immediately means `D` trivially wins and the gradients are useless or destructive. So warm up: hold `λ = 0` for an initial phase and train the autoencoder as a pure perceptual reconstructor first, until the codebook and decoder are sane, *then* switch the adversarial term on. Empirically a longer warm-up — at least an epoch — gives better reconstructions, which makes sense: let the representation form before you start an adversarial game on top of it.

Let me also pin the architecture, because the compression rate is set by it. Encoder and decoder are a ResNet-style convolutional stack — the kind used for high-fidelity image models, residual blocks with group normalization and a Swish nonlinearity, downsampling by stride to halve resolution each of `m` blocks, so `h = H/2^m`, `w = W/2^m`, and `f = 2^m`. I adjust the compression rate purely by choosing `m`. One more thing I want even inside the autoencoder: a single self-attention (non-local) block at the *lowest* resolution of the encoder and decoder. At the bottleneck the feature map is tiny, so full self-attention there is cheap, and it lets the autoencoder aggregate context from everywhere when forming/reading the codes — global awareness where it's affordable, without paying quadratic cost on the full-resolution map. A `1×1` convolution maps the encoder's channels to the codebook dimension `n_z` before quantizing and back after. The discriminator is the patch-based convolutional classifier. This is the recipe I'll call VQGAN — a vector-quantized autoencoder whose reconstruction objective is perceptual-plus-adversarial rather than `L₂`, so it can compress hard and still reconstruct sharply.

One subtlety on `β`, the commitment weight. VQVAE used `β = 0.25`. With everything else changed I'd want to revisit it, but it's known to be robust across a wide range, and in practice I'll just keep the codebook and commitment terms at unit weight relative to the rest — the commitment term's job (keep the encoder from drifting away from its codes) doesn't need fine tuning to do its work.

Now stage two, with `E`, `G`, `Z` trained and frozen. Any image becomes its index sequence `s ∈ {0,…,K−1}^{h×w}` via `s_{ij} = k s.t. (z_q)_{ij} = z_k`. I choose an ordering to flatten the 2D grid into a 1D sequence, then learn a transformer to do next-index prediction: given `s_{<i}`, predict `p(sᵢ | s_{<i})`, and the likelihood of the whole image's code is `p(s) = Πᵢ p(sᵢ | s_{<i})`. Train by maximizing log-likelihood, i.e. minimize

  `L_Transformer = E_{x∼p(x)}[−log p(s)]`,

which is just cross-entropy of the predicted index distribution against the true next index. The model is a plain GPT-style decoder: a token embedding for each codebook index plus a learned positional embedding, a stack of blocks each with masked multi-head self-attention (lower-triangular mask) and a position-wise MLP (the usual `4×`-wide GELU sandwich), a final layer-norm, and a linear head to `K` logits. Causal masking makes the factorization autoregressive. This is exactly where the short sequence pays off: `256` tokens is a length a 300M-parameter transformer handles comfortably, where a million pixels never could.

Which ordering? The 2D grid has no canonical 1D order the way language has left-to-right. I could try a spiral (betting the subject is centered), a Z/Morton curve (preserving 2D locality), a coarse-to-fine subsample order, an alternating boustrophedon, a spiral inward. But there's a constraint coming from how I'll generate large images (below): I'll slide a window across the grid in reading order, which presupposes a row-major (raster) scan. And when I think about what the autoregressive model can actually exploit, the row-major order gives a consistent "everything above-and-left is context" structure that matches how the convolutional receptive fields were built. So I'll use raster order — top-left to bottom-right — and expect it to be at least as good as the exotic curves. (It's worth checking empirically across those permutations in a controlled run, since autoregressive code modeling is *not* permutation-invariant — the order genuinely matters — but raster is the principled default and the one the sliding window needs.)

Conditioning falls out almost for free because the transformer is just a sequence model. Suppose I'm given side information `c` — I want `p(s|c) = Πᵢ p(sᵢ | s_{<i}, c)`. If `c` is non-spatial, like a class label, it's a single extra token I prepend to the sequence; the transformer attends to it like any other context (a learned "start" token carrying the class). If `c` has spatial extent — a segmentation map, a depth map, an edge map, a low-res image to upsample — I train *another* VQGAN on that modality to get its own index sequence `r ∈ {0,…,|Z_c|−1}^{h_c×w_c}`, and because everything is autoregressive I just prepend `r` to `s` and restrict the loss to the `s` part: `p(sᵢ | s_{<i}, r)`. This decoder-only conditioning means one mechanism — prepend the condition, predict the rest — handles class labels, segmentation, depth, edges, super-resolution, pose, all without bespoke architectures. The autoencoder is reused across tasks; only the transformer's conditioning prefix changes.

Sampling: feed the condition prefix (or nothing), then autoregressively sample indices one at a time from the transformer's predicted distribution — with a temperature on the logits and a top-`k` truncation to avoid the low-probability tail (say `t = 1.0`, `k ≈ 100`, scaling `k` up for larger codebooks). Collect the `h·w` sampled indices, map each back to its codebook vector to rebuild `z_q`, and run the frozen decoder `G` to render the image. The transformer never touches a pixel; the decoder never reasons globally. Each does only its job.

Last problem: megapixels. Even with the autoencoder, a megapixel image's latent grid is still longer than the transformer's context — I can push `m` up to shrink the grid further, but past a critical `m` (dataset-dependent) reconstruction quality collapses, so I can't compress my way out arbitrarily. So I train the transformer on *crops* of the latent grid sized to a feasible sequence length, and at generation time I run it as a *sliding window*: to predict the code at a given position I center (as much as causality allows) a fixed-size attention window of already-generated context around it and slide that window across the grid in raster order. This keeps the sequence the transformer sees bounded while letting me tile out an image of arbitrary size. It rests on an assumption: that the local context inside the window is enough to predict the next code — true when the dataset's statistics are approximately spatially invariant, or when spatial conditioning information (a segmentation/depth map covering the whole canvas) is available to anchor each window. When neither holds — unconditional synthesis on aligned data, like centered faces, where position carries meaning — I can simply condition on the image coordinates so each window knows where it is. With that, the two-stage model generates coherent images well into the megapixel regime, which a pixel transformer could never reach.

Let me write the core of it as code, tying each block back to the reasoning.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# ---- Stage 1a: the discrete bottleneck (nearest-code VQ + straight-through) ----
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e, self.e_dim, self.beta = n_e, e_dim, beta
        self.embedding = nn.Embedding(n_e, e_dim)           # the codebook Z = {z_k}
        self.embedding.weight.data.uniform_(-1.0/n_e, 1.0/n_e)

    def forward(self, z):                                   # z: (B, C, h, w) = E(x) after 1x1 conv
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.e_dim)
        # squared distances to every codebook entry, then nearest -> the argmin quantization
        d = (z_flat**2).sum(1, keepdim=True) + (self.embedding.weight**2).sum(1) \
            - 2 * z_flat @ self.embedding.weight.t()
        idx = torch.argmin(d, dim=1)                        # s_{ij}: index of nearest z_k
        z_q = self.embedding(idx).view(z.shape)
        # first term pulls E(x) -> z_k (commitment); second pulls z_k -> E(x) (codebook); both via stop-grad
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        # straight-through: forward = z_q, backward = identity onto z (copy decoder gradient past argmin)
        z_q = z + (z_q - z).detach()
        return z_q.permute(0, 3, 1, 2).contiguous(), loss, idx

# ---- Stage 1b: perceptual + patch-adversarial objective with adaptive lambda ----
def hinge_d_loss(real, fake):                               # discriminator hinge objective
    return 0.5 * (F.relu(1. - real).mean() + F.relu(1. + fake).mean())

class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, perceptual_weight=1.0, disc_weight=1.0):
        super().__init__()
        self.perceptual_loss = LPIPS().eval()               # VGG-feature distance -> no reward for blur
        self.discriminator = PatchDiscriminator()           # judges N x N patches, not the whole image
        self.codebook_weight = codebook_weight
        self.perceptual_weight = perceptual_weight
        self.disc_weight = disc_weight
        self.disc_start = disc_start                        # warm-up: no GAN gradient until here

    def calculate_adaptive_weight(self, rec_loss, g_loss, last_layer):
        # measure both gradients at the decoder's last layer, the place where they meet
        rec_grad = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        g_grad   = torch.autograd.grad(g_loss,   last_layer, retain_graph=True)[0]
        # lambda = ||grad rec|| / (||grad GAN|| + delta): normalize GAN gradient to rec gradient scale
        w = torch.norm(rec_grad) / (torch.norm(g_grad) + 1e-4)
        return (self.disc_weight * torch.clamp(w, 0.0, 1e4)).detach()

    def forward(self, codebook_loss, x, x_hat, optimizer_idx, global_step, last_layer):
        # reconstruction term: small pixel L1 + perceptual loss (this replaces VQVAE's L2)
        rec = torch.abs(x - x_hat) + self.perceptual_weight * self.perceptual_loss(x, x_hat)
        rec = rec.mean()
        disc_on = 1.0 if global_step >= self.disc_start else 0.0   # warm-up gate

        if optimizer_idx == 0:                              # generator (= autoencoder) update
            g_loss = -self.discriminator(x_hat).mean()      # fool the patch discriminator
            lam = self.calculate_adaptive_weight(rec, g_loss, last_layer) if disc_on else 0.0
            return rec + lam * disc_on * g_loss + self.codebook_weight * codebook_loss.mean()

        if optimizer_idx == 1:                              # discriminator update
            d_loss = hinge_d_loss(self.discriminator(x.detach()),
                                  self.discriminator(x_hat.detach()))
            return disc_on * d_loss

# ---- Stage 1 model: encode/quantize/decode, two-optimizer min-max training ----
class VQModel(nn.Module):
    def __init__(self, ddconfig, n_embed, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)                  # conv stack, downsample x m, attn at low res
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)        # -> n_z
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.quant_conv(self.encoder(x))
        return self.quantize(h)                             # z_q, codebook_loss, indices

    def decode(self, z_q):
        return self.decoder(self.post_quant_conv(z_q))

    def forward(self, x):
        z_q, diff, _ = self.encode(x)
        return self.decode(z_q), diff

    def get_last_layer(self):
        return self.decoder.conv_out.weight                 # G_L: where adaptive lambda is measured

# ---- Stage 2: GPT-style prior over the code indices ----
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=24, n_head=16, n_embd=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)     # one embedding per codebook index
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        t = idx.size(1)
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)                               # p(s_i | s_{<i}) over the K codes
        loss = None
        if targets is not None:                             # L_Transformer = E[-log p(s)] = cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
# (Block = LayerNorm -> causal multi-head self-attention -> LayerNorm -> 4x-wide GELU MLP, both residual)

# ---- Stage 2 wrapper: image<->sequence, conditioning by prepending, sampling ----
class Net2NetTransformer(nn.Module):
    def __init__(self, first_stage, transformer):
        super().__init__()
        self.first_stage = first_stage                      # frozen VQGAN
        self.transformer = transformer

    @torch.no_grad()
    def encode_to_z(self, x):                               # image -> raster index sequence
        quant, _, idx = self.first_stage.encode(x)
        return quant, idx.view(x.shape[0], -1)              # row-major flatten

    def forward(self, x, c):                                # c: condition index sequence (label / map codes)
        _, z_idx = self.encode_to_z(x)
        cz = torch.cat((c, z_idx), dim=1)                   # prepend the condition
        logits, _ = self.transformer(cz[:, :-1])            # predict shifted-by-one
        logits = logits[:, c.shape[1]-1:]                   # keep only the predictions for s
        return logits, z_idx                                # targets = z_idx

    @torch.no_grad()
    def sample(self, c, steps, temperature=1.0, top_k=100):
        x = c
        for _ in range(steps):
            x_cond = x if x.size(1) <= self.transformer.block_size else x[:, -self.transformer.block_size:]
            logits, _ = self.transformer(x_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:                           # truncate the low-probability tail
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            x = torch.cat((x, torch.multinomial(probs, 1)), dim=1)   # autoregressive next index
        return x[:, c.shape[1]:]                             # the sampled code sequence s

    @torch.no_grad()
    def decode_to_img(self, idx, zshape):                   # codes -> codebook vectors -> pixels
        quant = self.first_stage.quantize.embedding(idx).view(zshape)
        return self.first_stage.decode(quant.permute(0, 3, 1, 2).contiguous())
```

So the causal chain: a pixel transformer can't reach high resolution because attention is quadratic in a pixel sequence that is itself quadratic in resolution; a CNN can cheaply compress an image into a short grid of discrete codes, which is the short sequence a transformer *can* afford — so let the CNN own local detail and the transformer own global composition. But compressing hard enough to make the sequence short makes an `L₂`-trained codebook reconstruct as blur, capping fidelity; replacing `L₂` with a perceptual (LPIPS) loss plus a patch-based adversarial loss — balanced by an adaptive weight `λ = ‖∇_{G_L}L_rec‖ / (‖∇_{G_L}L_GAN‖ + δ)` that equalizes the two gradients at the decoder's last layer, with a `λ=0` warm-up — yields a high-compression *and* high-fidelity discrete codebook. A GPT-style transformer then learns the autoregressive prior `p(s)=Πᵢp(sᵢ|s_{<i})` over the raster-ordered indices, with conditions prepended as a prefix for controllable synthesis; sampling means the transformer draws codes and the frozen decoder renders them, and a sliding attention window over the latent grid lets the same machinery scale to megapixel images.
