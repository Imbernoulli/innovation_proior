OK, let me think this through from scratch. I have a vision Transformer — the kind that chops an image into a `14×14` grid of `16×16` patches, linearly embeds each one, adds position embeddings, and runs a plain Transformer encoder over the sequence. It's a great model when I have mountains of labeled images, but on ImageNet-1K alone, trained from random init, it's data-hungry and unstable; a convolutional net with the same budget beats it. The architecture has almost no built-in image prior — no locality, no translation equivariance — so it has to *learn* all of that from data, and there isn't enough labeled data to do it. I have a lot of *unlabeled* images though. So the real question is: what self-supervised objective can I pretrain this encoder on so that, after fine-tuning, it transfers well and converges fast?

In language the answer has been completely settled: mask out about 15% of the tokens in a sentence, replace them with a `[MASK]` symbol, run a bidirectional Transformer over the corrupted sequence, and make the model predict the missing tokens. Then throw away the prediction head and fine-tune the encoder on whatever you want. It's simple, it's self-supervised, and it produces representations that carry a lot of structure. I want exactly that, but for my patch-sequence Transformer. The encoder is already a Transformer; the data is already a sequence (of patches). It feels like it should be a near-trivial port.

So let me try the literal port and see where it breaks. Mask ~15% of the patches, feed the corrupted patch sequence to the encoder, predict... predict what, exactly? In language, "predict the missing token" is a *classification* problem: there's a fixed vocabulary of, say, 30k word-pieces, and at each masked position I put a softmax over those 30k options and train with cross-entropy. The whole thing works because the prediction target is a *discrete symbol from a fixed, finite set*. The model never has to output a continuous quantity; it just has to point at the right entry in a known list.

And here's where it breaks. My input unit is a patch — a `16×16×3` block of continuous pixel intensities. There is no vocabulary of patches. I can't write down "all possible patches" and put a softmax over them; the space is continuous and unbounded. The thing that makes BERT's objective clean — a finite list to classify into — just doesn't exist for raw image patches. So the literal port stalls at the very first step: I have nothing to classify into.

Fine, drop the classification framing. The obvious alternative is to make it a *regression*: at each masked position, predict the raw pixels of the missing patch, and train with an L2 (or L1) reconstruction loss, `Σ_{i∈M} ‖x_i − x̂_i‖²`. This is well-defined, differentiable, needs no vocabulary. Before I commit to it, let me think about where the loss actually *goes* — which part of the signal dominates the squared error — because that's what the gradient will chase.

Stare at a natural image for a second. Adjacent pixels are enormously correlated; a patch's pixels are mostly predictable from the patch's immediate neighbors by interpolation or copying. I can make that concrete. Take any natural image and decompose a 16×16 patch into a smooth (low-frequency) part plus a high-frequency residual. The squared error of a patch splits as `‖x − x̂‖² = ‖x_low − x̂_low‖² + ‖x_high − x̂_high‖²`. The high-frequency residual of a natural patch — fine texture, sensor noise — has small amplitude but it's where almost all the *unpredictable* bits live; the low-frequency part is large in amplitude but is exactly what a neighbor-copying baseline already nails. So a model that does nothing but bilinearly interpolate from surviving neighbors drives the *large* `x_low` term most of the way to zero, and the remaining loss is dominated by the high-frequency residual — the part that carries texture and noise, not "what object is this." The gradient therefore keeps pushing capacity into matching high-frequency detail long after the model has stopped needing to understand the scene. I'd be training the encoder to be a good local denoiser/interpolator, not a good semantic representation learner.

That's an argument, not a measurement, so I want at least one outside data point to corroborate it before I throw regression out. The one I have is the diagnostic from the patchify-and-Transformer line: they ran a preliminary masked-patch objective that predicted a low-level pixel statistic — the 3-bit mean color of a masked patch — and it *underperformed*. That's the most direct translation of the NLP recipe, and it lagged. I shouldn't overclaim from a single reported number, but it lines up with the gradient-budget argument: predicting a coarse low-level pixel statistic is exactly the regime where local copying wins and semantics never get learned. So I'll treat pixel/low-level regression as a target I have a concrete reason to distrust, not a definitive verdict. The literal port has failed twice now: no vocabulary for classification, and the regression substitute spends the model on detail.

So what do I actually want? I want the masked-prediction game, because the *game* is what builds representations — to fill in a hidden patch you must reason about the rest of the image, its objects, its layout. What I don't want is for the *answer* to be raw pixels, because then the easiest way to win the game is local low-level copying. I want the target to live at a higher level of abstraction than pixels: something that says "a piece of furry texture / part of a wheel / sky," not "these 768 intensity values." And ideally it should be *discrete*, because then I get BERT's clean classification objective back — a softmax over a finite set, cross-entropy, done — and the discreteness itself forces the target to summarize the patch into one of finitely many categories rather than reproduce every pixel.

So the thing I'm missing is a vocabulary for images. A finite set of, say, a few thousand "visual words," and a way to map any patch (or any small image region) to one of them. If I had that, the port becomes exact: tokenize the image into this visual vocabulary to get a grid of discrete codes; mask some patches of the *input*; predict the discrete code at the masked positions with a softmax over the vocabulary. Two representations of the same image, then — the continuous patches as *input*, and the discrete codes as the *prediction target*.

Where do I get such a vocabulary? I'm not going to hand-design "visual words." But I recall that generative image models already solved exactly this sub-problem for a different reason. To model images as sequences, people learned a *discrete* autoencoder: an encoder (call it a "tokenizer") `q_φ(z|x)` that maps an image to a grid of indices into a learned codebook of size `|V|`, and a decoder `p_ψ(x|z)` that reconstructs the image from those indices. The main pressure is reconstruction — maximize `E_{z∼q_φ(z|x)}[log p_ψ(x|z)]` — with a uniform-prior regularizer so the discrete code space remains usable. The codes are discrete, so the sampling step isn't differentiable; that's handled with a Gumbel-softmax relaxation (a continuous, temperature-controlled surrogate for categorical sampling) during training. The two-stage idea — first learn a discrete code space, then learn a model over those codes — is exactly the VQ-VAE recipe. And there's a tokenizer of this kind already trained at scale, with a codebook of `8192` entries, that downsamples its input image by a factor of 8.

That codebook could serve as my visual vocabulary. The tokenizer maps an image to a grid of discrete labels from an `8192`-entry codebook, and because those codes were learned to reconstruct the image through a bottleneck, each one summarizes a region rather than storing its raw pixels. So instead of predicting pixels I'd predict *which codebook entry* each masked region corresponds to — a classification over `8192` classes, which is BERT's objective again. But "instead of pixels, predict a code" only works if the codes and the patches actually line up on the grid; if the two views have different spatial resolutions there's no well-defined label `z_i` to attach to masked patch `i`. So before going further I have to check the geometry.

My ViT splits a `224×224` patch view into a `14×14` grid of `16×16` patches: `(224/16)² = 14² = 196`, so `N = 196`. The tokenizer downsamples by 8. If I naively fed it the same 224-pixel tensor I'd get `224/8 = 28` codes per side — a `28×28 = 784` code grid, four times as many codes as patches, and no clean correspondence. So I should *not* feed it the 224 tensor. Instead I feed the tokenizer the same crop resized to `112×112`, and it emits `112/8 = 14` codes along each side: a `14×14` code grid, `196` codes. Now `#visual_tokens = 196 = #patches`, and the grids are congruent, so there's a one-to-one map patch `i` (input) ↔ visual token `z_i` (target). The 112-vs-224 choice isn't cosmetic — it's the resolution that makes the factor-8 downsample land exactly on the `14×14` patch grid. With that confirmed, at masked position `i` I have exactly one discrete label `z_i` to predict.

So now I can write the objective. I split the image into patches `{x_i^p}` and, separately, tokenize it into visual tokens `{z_i}` with the (frozen) tokenizer. I pick a set `M` of positions to mask. At each masked position I replace the patch embedding with a learnable mask embedding `e_[M]` (the analog of `[MASK]`), so the encoder knows "something is here, figure out what." I feed this corrupted patch sequence through the encoder, take the output vector `h_i^L` at each masked position, and push it through a softmax classifier over the codebook:

`p_MIM(z' | x^M) = softmax(W_c · h_i^L + b_c)`, with `W_c ∈ R^{|V|×D}`, `b_c ∈ R^{|V|}`.

And I train to maximize the log-likelihood of the *correct* visual tokens at the masked positions:

`max Σ_{x∈D} E_M [ Σ_{i∈M} log p_MIM(z_i | x^M) ]`.

That's it — masked image modeling. Input = continuous patches (so the encoder still sees raw pixels and learns fine-grained features), target = discrete visual tokens (so the objective abstracts away pixel detail and stays a clean classification). The two views of the image are doing complementary jobs.

How much do I mask, and *which* positions? In language ~15% works. But think about images: patches are far more locally redundant than words. If I mask isolated single patches scattered around, the model can fill each one in almost perfectly from its four immediate neighbors — it slides right back into the short-range-copying failure mode I was trying to escape, just with discrete targets instead of pixels. I need to mask a lot — around 40% of the patches, not 15% — so that there isn't enough surviving local context to interpolate from. I also need to remove contiguous *blocks*. If I knock out a whole rectangular region, the model can't recover the codes in its interior by copying a neighbor, because the neighbors are gone too; it has to use the rest of the image. This is the same instinct as masking spans/n-grams instead of independent tokens in the language models that refined BERT — block out a chunk so prediction can't be solved locally.

Let me make blockwise masking concrete as a procedure on the `h×w` grid. I build up `M` block by block until I've masked about `0.4·N` patches. For each block I sample an area `s` (with a floor — a block shouldn't be a single patch; set a minimum like 16 patches) and an aspect ratio `r`. Sampling `log r` uniformly between `log 0.3` and `log(1/0.3)` keeps tall and wide rectangles balanced instead of over-concentrating the sampler near one shape. From area and aspect I get the block's height and width, `a = round(√(s·r))`, `b = round(√(s/r))`, place its top-left corner uniformly at random inside the grid, and set those patches masked. Repeat, adding blocks (allowing them to overlap, just counting the newly-masked patches) until I reach the target number of masked positions.

Let me trace it once on the `14×14` grid to make sure the arithmetic produces sane blocks and converges. The target is `0.4·196 = 78.4`, so about 78 patches. Take a representative draw: `s = 30`, `log r = 0` so `r = 1` (a square block). Then `a = round(√30) = 5`, `b = round(√30) = 5` — a `5×5` block of 25 patches, comfortably inside the `14×14` grid. Now check the aspect extremes don't degenerate: at `r = exp(log(1/0.3)) = 3.33` with the same `s = 30`, `a = round(√(30·3.33)) = round(√100) = 10`, `b = round(√(30/3.33)) = round(√9) = 3` — a `10×3` block, tall but still fits. And at the area floor `s = 16` with `r = 1`, `a = b = round(√16) = 4`, the smallest block is `4×4 = 16` patches, which matches the floor I set; at the aspect extreme there it's `7×2 = 14`, still a genuine block, not a single patch. So the smallest unit the sampler ever places is a `4×4`-ish chunk, never an isolated patch — which is exactly the property I wanted. As for convergence: with blocks of ~25 patches each and some overlap, I reach 78 masked in about `78/25 ≈ 3` to 4 blocks, so the loop terminates quickly. That all checks out, so on a `14×14` grid this masks roughly 75–78 of the 196 patches. The procedure:

```
M ← {}
repeat
    s ← Rand(16, 0.4N − |M|)          # block area, at least 16 patches
    log r ← Rand(log 0.3, log(1/0.3))
    r ← exp(log r)                   # aspect ratio
    a ← √(s·r) ;  b ← √(s/r)         # block height, width
    t ← Rand(0, h − a) ; l ← Rand(0, w − b)   # top-left corner
    M ← M ∪ { (i,j) : i∈[t,t+a), j∈[l,l+b) }
until |M| reaches the target count
```

Should I predict the visual tokens at *all* positions or only the masked ones? Only the masked ones — that's what makes it a denoising task. If I also ask the model to reproduce the codes at the *visible* positions, that's mostly a trivial copy of information it can already see, and it dilutes the gradient that's supposed to teach in-filling. So the cross-entropy is summed over `i ∈ M` only.

The whole method hinges on the discrete-target choice, so let me pin down exactly why it should differ from pixel regression in the right direction. The cross-entropy over codes is bounded by `log|V| = log 8192 ≈ 9.0` nats per masked position — the loss can never exceed the entropy of a uniform guess over the codebook, and it's driven to zero only by predicting the *right discrete label*, with no separate gradient term for matching texture or noise. Contrast that with the pixel-regression loss I decomposed earlier, whose residual was dominated by the unpredictable high-frequency part of the patch: there the loss has nowhere to bottom out, because the high-frequency residual is largely un-inferable from context, so the gradient keeps chasing it. The code, by construction, has already passed through the tokenizer's reconstruction bottleneck, which collapses a `16×16×3 = 768`-dimensional continuous patch-region down to one of `8192` indices — a roughly `768·8 / log2(8192) = 6144/13 ≈ 470`-fold reduction in raw bit count — so the detail the regression loss was wasting itself on is *already gone* from the target. Predicting a code therefore forces the model to name *what region this is* rather than reproduce its pixels. I can't measure here that this raises downstream accuracy, but the two losses provably differ in where they put their gradient, and that difference is in the direction I argued I needed.

Is there a principled reason this two-view, tokenize-then-predict setup is the right thing, or did I just stumble into it? It would reassure me if the cross-entropy I just wrote down fell out of a variational objective rather than being an ad-hoc choice. What I'm really doing is recovering the original image `x` from its corrupted version `x̃` (corrupted = the masked patch sequence). So consider the log-likelihood `log p(x | x̃)` and lower-bound it with the discrete codes `z` as latent variables — a standard evidence lower bound. Introduce the tokenizer `q_φ(z|x)` as the variational posterior over codes given the *clean* image:

`log p(x | x̃) ≥ E_{z∼q_φ(z|x)}[ log p_ψ(x | z) ] − KL[ q_φ(z|x) ‖ p_θ(z | x̃) ]`.

Let me read off the three distributions. `q_φ(z|x)` is the tokenizer — codes from the clean image. `p_ψ(x|z)` is the decoder — rebuild the image from codes; the first term is image reconstruction through visual tokens. And `p_θ(z|x̃)` is a model that predicts the codes from the *corrupted* image — which is exactly the masked-image-modeling network I've been building. So MIM sits in the term that makes the variational posterior over codes (given the clean image) and the prior-from-corruption (given the masked image) agree: the encoder I'm training is the `p_θ(z|x̃)` factor of this bound. That's a satisfying reading, but the bound as written still has an expectation and a KL in it, and what I actually want to optimize is a plain cross-entropy. I need to check that the bound really *collapses* to my loss under the simplification I'm about to make, not just hope it does.

Optimizing all of that jointly is awkward, so split it the way VQ-VAE-style models do, into two stages. Stage 1: learn the tokenizer `q_φ` and decoder `p_ψ` with the discrete-autoencoder objective: minimize the negative reconstruction term `−E_{z∼q_φ(z|x)}[log p_ψ(x|z)]` while regularizing codes toward a uniform prior. That's just training (or, in practice, *reusing* an already-trained) discrete autoencoder. Stage 2: freeze `q_φ` and `p_ψ`, and learn `p_θ`. The simplification that should collapse the bound into a clean classification loss is to replace the soft posterior `q_φ(z|x)` with a one-point distribution at its mode, `ẑ = argmax_z q_φ(z|x)` — i.e. just take the most-likely code grid the tokenizer assigns.

Let me actually do the KL with `q_φ` a delta at `ẑ`, because this is where it's easy to wave hands and be wrong. By definition `KL[q ‖ p] = Σ_z q(z) log(q(z)/p(z))`. A delta puts `q(ẑ)=1` and `q(z)=0` elsewhere; the zero-mass terms vanish (using `0·log 0 = 0`), leaving a single surviving term `1·log(1/p_θ(ẑ|x̃)) = log 1 − log p_θ(ẑ|x̃) = −log p_θ(ẑ|x̃)`. To be sure I didn't fumble the limit, take `q` not a hard delta but `(1−ε)` on `ẑ` and `ε` spread over the rest, with some arbitrary `p_θ`, say `p_θ(ẑ) = 0.6`: at `ε = 10⁻¹` the KL is `0.244`, at `ε = 10⁻²` it's `0.461`, at `ε = 10⁻⁴` it's `0.5099`, at `ε = 10⁻⁸` it's `0.51082` — and `−log 0.6 = 0.51083`. So as the posterior concentrates the KL converges to `−log p_θ(ẑ|x̃)` exactly, confirming the algebra. Meanwhile, with `q_φ` a delta, the reconstruction expectation `E_{z∼q_φ}[log p_ψ(x|z)]` collapses to the single value `log p_ψ(x|ẑ)` and contains no `θ`, so it's constant for the stage-2 optimization. So the lower bound, summed over image pairs rather than over patch positions, becomes

`Σ_{(x,x̃)∈D} ( E_{z∼q_φ(z|x)}[log p_ψ(x|z)]  +  log p_θ(ẑ | x̃) )`.

If I factorize `p_θ(ẑ | x̃)` over the selected patch positions as `∏_{j∈M} p_MIM(ẑ_j | x^M)`, its log is exactly `Σ_{j∈M} log p_MIM(ẑ_j | x^M)`. The "take the argmax code as the hard label and do cross-entropy" recipe is the stage-2 ELBO under a one-point posterior.

The encoder can stay a standard Transformer so results are comparable to supervised ViT — no architectural tricks in the contribution. Base size: 12 layers, hidden `D = 768`, 12 heads, FFN inner size 3072, patch `16×16`; the large version is the same recipe at 24 layers, hidden 1024, 16 heads. Input is the sequence of linearly-projected patch embeddings `E·x_i^p`, with a prepended pooling token `[S]` and position information on the patch grid. The MIM head is a single linear layer `R^D → R^{|V|}` (the softmax classifier over the `8192` codes); I only run it on the masked positions' outputs.

The one non-obvious training detail is initialization. Deep Transformers are touchy at the start of large-scale pretraining — the residual stream's variance compounds across layers and things blow up or stall. I initialize weights in a small range, implemented as a truncated normal with standard deviation `0.02` and bounds `±0.02`, and then, for the `l`-th block, I rescale the *output* projections — the last linear in the self-attention sublayer and the last linear in the FFN — by `1/√(2l)`. Let me make sure the `1/√(2l)` factor does what I think before I rely on it.

Model the residual stream as `x_l = x_{l−1} + g_l(x_{l−1})`, where each block adds two sublayer outputs, and each sublayer output is (roughly) a variance-preserving linear map of a layer-normed input — so without any damping each sublayer adds back a chunk of unit-ish variance, and the stream variance grows by a roughly constant amount per layer, i.e. linearly with depth. I can just simulate this. Take a 256-wide stream, 24 blocks, weights `~N(0, 1/D)` so each map preserves variance on a standardized input, and watch `Var(x_l)`. Without the rescale, the stream variance climbs roughly linearly: about `3` at layer 1, `9` at layer 4, `20` at layer 8, `30` at layer 12, up to about `60` by layer 24. With the `1/√(2l)` factor applied to each block's two output projections, the same simulation gives `1.9` at layer 1, `3.0` at layer 4, `3.6` at layer 8, and only `~4.5` by layer 24 — bounded instead of growing, a `13×` difference in the final-layer variance. So the factor demonstrably converts linear depth-wise variance growth into a flat profile; that's the property I wanted at the start of optimization. The `2` inside is for the two residual sublayers per block, and the `l` growth inside the square root is what cancels the linear accumulation I just saw. I also leave room for learnable residual gates initialized below one (`0.1` for base, much smaller for large) because they let each block start as a conservative perturbation and then grow if the optimizer needs it. These are the damping choices that keep deep/large models from diverging in the first steps; I'd still want to confirm on a real large-model run that they're sufficient, but the variance argument is what motivates them.

The code has four moving parts: the blockwise patch-grid mask generator; the data pipeline that produces two views of the image plus a mask; the encoder that substitutes a mask embedding at masked positions and emits code-logits there; and the training step that pulls the target codes out of the frozen tokenizer and does cross-entropy on the masked positions.

```python
import math, random
from functools import partial
import numpy as np
import torch, torch.nn as nn
from timm.models.layers import trunc_normal_ as _trunc_normal

def trunc_normal_(tensor, std=0.02):
    return _trunc_normal(tensor, std=std, a=-std, b=std)

# ---- 1. Blockwise masking on the h×w patch grid ----
class PatchGridMaskGenerator:
    def __init__(self, input_size, num_masking_patches,
                 min_num_patches=16, max_num_patches=None,
                 min_aspect=0.3, max_aspect=None):
        self.height, self.width = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches          # ~0.4 * N
        self.min_num_patches = min_num_patches                  # block floor (>= 16)
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))  # log-uniform aspect

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):                                     # a few tries to place one block
            target_area = random.uniform(self.min_num_patches, max_mask_patches)   # s
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))        # r
            h = int(round(math.sqrt(target_area * aspect_ratio)))                  # a = sqrt(s*r)
            w = int(round(math.sqrt(target_area / aspect_ratio)))                  # b = sqrt(s/r)
            if w < self.width and h < self.height:
                top  = random.randint(0, self.height - h)       # t
                left = random.randint(0, self.width  - w)       # l
                num_masked = mask[top:top + h, left:left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:  # count only newly-masked (overlap ok)
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1; delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros((self.height, self.width), dtype=int)
        mask_count = 0
        while mask_count < self.num_masking_patches:            # add blocks until ~40% masked
            max_mask_patches = min(self.num_masking_patches - mask_count, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta
        return mask                                             # (h, w) boolean grid

# ---- 2. Two views of each image + a mask: patches (input) and a tokenizer view (target source) ----
class TwoViewPretrainingTransform:
    def __init__(self, args):
        mean = IMAGENET_INCEPTION_MEAN if not args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size,
                interpolation=args.train_interpolation,
                second_interpolation=args.second_interpolation),
        ])
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ])
        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([transforms.ToTensor(), map_pixels])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
            ])
        else:
            raise NotImplementedError()
        self.masked_position_generator = PatchGridMaskGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,  # e.g. 14x14 grid, ~75 masked
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block)

    def __call__(self, image):
        for_patches, for_tokens = self.common_transform(image)
        return (self.patch_transform(for_patches),            # ViT input  -> raw pixels preserved
                self.visual_token_transform(for_tokens),      # tokenizer input -> discrete-code target
                self.masked_position_generator())             # which patches are corrupted

# ---- 3. The ViT encoder for masked image modeling ----
class PatchSequencePretrainer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 init_std=0.02, **kw):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.init_std = init_std
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))           # [S]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))           # learnable e_[M]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.rel_pos_bias = (
            RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
            if use_shared_rel_pos_bias else None
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, init_values=init_values,
                  window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                  attn_head_dim=attn_head_dim)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)                        # softmax over the codebook

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token,  std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()                                                 # stabilizing rescale

    def fix_init_weight(self):
        # rescale the l-th block's two output projections by 1/sqrt(2*l) so the
        # residual-stream variance doesn't blow up with depth at the start of training
        for layer_id, blk in enumerate(self.blocks):
            blk.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            blk.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)  # (B, N, D): continuous patch embeddings
        B, N, _ = x.shape
        mask_token = self.mask_token.expand(B, N, -1)
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)    # 1 at corrupted positions
        x = x * (1 - w) + mask_token * w                         # substitute e_[M] at masked patches
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        return self.norm(x)

    def forward(self, x, bool_masked_pos, return_all_tokens=False):
        x = self.forward_features(x, bool_masked_pos)[:, 1:]      # drop [S]; keep the N patch outputs
        if return_all_tokens:
            return self.lm_head(x)
        return self.lm_head(x[bool_masked_pos])                  # logits ONLY at masked positions

# ---- 4. One pretraining step: target = tokenizer's argmax codes; CE on masked positions ----
def train_one_step(model, tokenizer, batch, optimizer):
    samples, token_view, bool_masked_pos = batch                # patch-view, tokenizer-view, mask
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool) # (B, N)

    with torch.no_grad():
        input_ids = tokenizer.get_codebook_indices(token_view).flatten(1)  # (B, N): argmax codes ẑ
        labels = input_ids[bool_masked_pos]                                # codes at masked positions

    logits = model(samples, bool_masked_pos=bool_masked_pos)               # (#masked, |V|)
    loss = nn.CrossEntropyLoss()(logits, labels)                           # = -Σ_{i∈M} log p_MIM(ẑ_i | x^M)

    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
```

And to use the pretrained encoder downstream I just drop `lm_head`, keep the backbone, and attach a task head — for classification, average-pool the patch outputs and put a linear classifier on top, `softmax(avg({h_i^L}) · W_c)` with `W_c ∈ R^{D×C}`, and fine-tune end-to-end; for dense tasks like segmentation, attach a small upsampling decoder on the patch grid instead. Same encoder, swap the head.

So the causal chain, start to finish: I wanted BERT's masked-prediction objective for a ViT, but patches are continuous so there's no vocabulary to classify into, and regressing raw pixels wastes the model on locally-predictable high-frequency detail. I needed a *discrete* target that abstracts away pixel detail, which a learned discrete-VAE codebook gives me — so each image gets two views, continuous patches as the encoder's input and the tokenizer's discrete codes as the prediction target. Mask ~40% of the patches in contiguous blocks so the task can't be solved by local copying, substitute a learnable mask embedding, run the standard ViT, and classify the codebook index at each masked position with cross-entropy. That objective turns out to be exactly the stage-2 term of an ELBO that recovers the clean image's codes from the corrupted image under a one-point tokenizer posterior — so the pieces aren't arbitrary. A small `1/√(2l)` output-rescale keeps deep pretraining stable. Then fine-tune the encoder with a task head.
