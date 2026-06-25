OK, let me think this through from scratch. The thing that keeps bothering me is the gap between how language models and vision models get their data. In NLP the recipe is settled: take a giant pile of raw text, hide part of it, train a model to predict the hidden part, and out comes a representation that transfers everywhere and scales to absurd model sizes. GPT does it autoregressively, BERT does it by masking out maybe 15% of the tokens and predicting them from both sides. No labels. The data is free. And the bigger you make the model, the better it gets.

In vision, none of this is the practical reality. The strongest backbones are still pre-trained *supervised*, on labeled data, and the appetite for labels is enormous — the Vision Transformer only really shines when it's been fed something like three hundred million labeled images, most of which nobody outside a few labs can even access. Models today can overfit a million images without breaking a sweat. So the bottleneck isn't capacity, it's the supervision. If I could pre-train on raw pixels the way BERT pre-trains on raw text, the label problem would mostly dissolve.

And the maddening part is that the idea is *right there*. Hiding part of the input and reconstructing it is a special case of the denoising autoencoder, Vincent et al. 2008: corrupt the input, train the network to recover the clean original, and the features it learns become robust to that corruption. That's older than BERT. Vision even had masked-image work before BERT: Pathak's Context Encoder in 2016 inpaints a big missing region of an image with a convnet, trained with a reconstruction plus adversarial loss, and gets features out of it. So why, years after BERT detonated in NLP, is hide-and-reconstruct pre-training in vision still trailing? People tried. The Vision Transformer line already includes a masked-patch-prediction experiment — mask some patches, predict them, exactly the BERT move — and gets about two points over training from scratch, still around four points *behind* supervised pre-training. iGPT predicts pixels from pixel sequences but at tiny resolution and ruinous compute. The recent ones, BEiT, predict discrete visual tokens from a separately trained tokenizer. Everyone's circling the same idea and it just doesn't pay off the way it does in language.

So the real question isn't "how do I hide and reconstruct image content," it's "what is *different* about vision that makes the obvious port fail?" Let me actually pull the two signals apart instead of assuming they're the same.

First difference, and the one I'd have blamed for years: architecture. Convnets run on regular grids, and it's genuinely awkward to feed a convnet an "indicator" — a mask token saying "this spot is blank, predict it" — or to add positional embeddings the way Transformers do. BERT's whole machinery assumes a sequence of tokens you can freely insert placeholders into. But this excuse is gone now. ViT turns an image into a sequence: cut it into 16×16 patches, linearly project each patch into a token, add a positional embedding, run a standard Transformer. In that world a mask token is just another token and positional embeddings are native. So architecture can't be the reason anymore. Scratch it.

Second difference. Language is a human invention — every word is a deliberate, high-entropy, semantic unit. When BERT blanks out 15% of the words and asks you to fill them in, you genuinely can't do it without understanding grammar, meaning, world knowledge. The task is hard *because the signal is information-dense*. Now look at an image. An image is just recorded light. It's a natural signal, and it's drowning in spatial redundancy — neighboring pixels, neighboring patches, are almost the same thing. If I blank out one patch and leave its eight neighbors, I can reconstruct it by smearing the neighbors in, extending the edges, copying the texture. No understanding of "this is a dog's ear" required at all. So if I port BERT's 15% masking straight over, I've built a task whose solution is *interpolation*. The model can ace it with low-level statistics and never be forced to learn anything about objects or scenes. That's why the naive port is weak — not because the idea is wrong, but because the task it creates in vision is trivial. The redundancy is doing the model's homework for it.

That reframes everything. The job isn't to copy BERT's masking, it's to *destroy the redundancy* so that filling in the holes actually requires understanding. How do you destroy spatial redundancy? You remove so much of the image that the survivors can't possibly reconstruct the rest by local extrapolation. If I keep only a sparse scatter of patches, then any missing patch is far from every visible one, there's no neighbor to copy, and the only way to guess what was there is to have some holistic sense of the whole object — to infer the gestalt. So the masking ratio shouldn't be 15%. It should be *high*. Genuinely high. Mask most of the image. The exact number I'll have to find, but the direction is forced by the redundancy argument: low ratio → interpolation → trivial; high ratio → no local shortcut → must understand.

Let me sanity-check that this is even reconstructable. If I mask, say, three quarters of the patches at random and keep a uniformly-spread quarter, is there enough left to infer the rest? I can't settle that on paper — whether a scattered quarter of the patches actually pins down the scene is an empirical question about natural-image statistics, and I'd want to look at reconstructions to be sure. But I can reason about the two failure modes. If the visible quarter is too sparse, the problem becomes ill-posed and the model just hallucinates something plausible-but-wrong; if it's too dense, the copy-from-neighbor shortcut survives. The bet I'm making is that there's a wide middle band where the evidence is enough to constrain the scene but too sparse to copy locally, and that the loss — which still compares against the *exact* held-out pixels — pushes the model toward inferring a scene-level completion rather than smearing neighbors. I won't know the exact ratio until I sweep it, but the direction is what matters here: I want the masking *random and uniform*, not clustered toward the center, or I reintroduce a center bias and an easier task.

Now the third difference, and this one is subtle. The decoder. In BERT, the thing you're predicting is a *word* — a rich, high-level semantic token. So the decoder can be trivial, basically a single linear layer / small MLP, because the heavy lifting is in producing a representation from which a word is almost readable. In vision, if I reconstruct pixels, the thing I'm predicting is *low* semantic level — it's raw RGB, far below the level of a recognition label. That asymmetry has a consequence I need to think about carefully. Somewhere in the network there has to be machinery that turns an abstract latent into concrete pixel values — that's a reconstruction-specialized computation, and it's the *opposite* of what I want my transferable representation to be specialized for. If I force that pixel-painting job into the encoder (say, by using a trivial decoder), then the encoder's output is dragged down toward pixel-level detail, and that's exactly the representation I'm going to keep for recognition. Bad. So in vision, unlike language, the decoder is not an afterthought — it's where I get to *quarantine* the reconstruction specialization, away from the encoder. I'll come back to this; first let me get the basic shape down.

So the skeleton: cut the image into patches, mask most of them at random, encode what's left, decode to reconstruct the missing pixels, take the loss. Let me think about the encoder.

There's a side effect of the high masking ratio that I should quantify, because it might change the whole cost picture. A Transformer's self-attention is quadratic in sequence length. If I only keep 25% of the patches, and if I feed the encoder *only those visible patches*, then the encoder runs on a quarter of the sequence. The attention matrix goes from L by L to (L/4) by (L/4); the number of attention entries scales as the square of the token count, so it drops by a factor of `(1/4)^2 = 1/16`. Let me put a number on it: at 224×224 with 16×16 patches there are `(224/16)^2 = 196` tokens, full attention is on the order of `196^2 ≈ 38k` entries, and on the visible 49 it's `49^2 ≈ 2.4k` — about a sixteenth, as the algebra said. I shouldn't oversell it, though: the whole layer is not 1/16 as cheap, because the QKV/output projections and the MLP scale roughly *linearly* in tokens, so those parts only fall by 4×, and the decoder still runs full-length. So the honest claim is narrower than "16× faster" — but the qualitative point holds, and it's worth pausing on: the high ratio I need to make the *task* hard is the same high ratio that makes the *encoder* cheap, instead of the usual trade where a harder task costs more.

But wait — the BERT way is to put a mask token in the encoder at every masked position, so the encoder sees a full-length sequence with placeholders. If I do that, I lose the speed entirely (full sequence again) and, worse, I'm not sure it's even good for the representation. Let me think about what feeding mask tokens to the encoder actually does. At pre-training time the encoder would see a sequence that's, say, three-quarters artificial placeholder tokens. But at deployment — when I use the encoder for recognition — the input is a clean, complete image with *no* mask tokens at all. So the encoder spends most of its pre-training capacity learning to process a kind of token it will never see again. That's a train/deploy mismatch baked right into the encoder. The fix is obvious once I see it: just don't give the encoder mask tokens. Drop the masked patches entirely; the encoder only ever sees *real* patches, exactly like at deployment. No mismatch, and as a bonus the sequence is short and cheap.

So now the architecture is *asymmetric*, and that asymmetry isn't an aesthetic choice, it falls out of two pressures pointing the same way: keep the encoder on real patches only (for the representation, no train/deploy gap) and keep it short (for compute). The encoder is a normal ViT applied to the visible quarter. But then who reconstructs the full image? I still need to predict the masked patches, and the encoder never saw them. This is where the decoder comes in, and where the mask token goes.

After the encoder, I have latent vectors for the visible patches. I now build the *full* set of tokens: the encoded visible ones, plus a mask token in every masked slot. The mask token is a single shared, learned vector — its job is just to say "something belongs here, predict it." But a bare mask token has no idea *where* it is in the image; every masked slot would be identical. So I add positional embeddings to the whole full set, which tells each mask token its location. Now a small Transformer — the decoder — attends over this full set, lets information flow from the visible encoded patches into the mask-token positions, and at each masked position produces a prediction.

How big should this decoder be? Remember the third difference: I want to quarantine the reconstruction specialization here, not in the encoder. So the decoder should be substantial enough to *absorb* the pixel-painting job. But it's only used during pre-training - I throw it away and keep only the encoder for recognition - so its design is free in a way the encoder's design is not. That freedom plus the compute argument says: make it lightweight. Narrow it, since pixel reconstruction does not need the same width as the recognition backbone; 512 dimensions is a concrete small width that still gives the decoder room to mix visible and missing locations. For depth, the minimum is a single Transformer block, because the mask-token positions need at least one round of attention to read from the visible tokens. If the decoder is too shallow, the encoder has to carry more pixel-synthesis detail in its latent; if the decoder is too deep, I spend full-sequence compute on a module I will discard. Eight narrow blocks are a reasonable middle: enough post-encoder capacity to keep reconstruction work out of the representation, but still much cheaper than the large encoder. The expensive encoder runs short; the full-length work happens only in the cheap decoder.

Now, what exactly does the decoder predict, and what's the loss? The simplest possible target is the raw pixels of each masked patch. The last layer of the decoder is a linear projection whose output dimension equals the number of pixel values in a patch, so for 16 by 16 RGB patches it predicts 16 * 16 * 3 = 768 numbers. The loss should be mean squared error between predicted and original pixels. And, following BERT's logic, I should compute that loss *only on the masked patches*, not the visible ones. Why only masked? Because the visible patches were handed to the model; asking it to reproduce them rewards an identity path and dilutes the gradient on the hard part of the task. The objective I actually want is the conditional reconstruction problem: recover the missing patches from the visible evidence. So: MSE, masked patches only, in pixel space.

Should I really predict raw pixels, though? BEiT went the other way — it predicts discrete *visual tokens* from a separately trained dVAE, on the theory that tokens are more semantic than pixels, more like predicting a word. That's tempting because it makes the vision task look more like the language task. But look at what it costs: I'd need an entire extra pre-training stage to learn the tokenizer, that tokenizer was itself trained on a couple hundred million images (so I've smuggled in extra data), and the tokenizer is a big convolutional network adding real overhead every forward pass. All of that to convert pixels into tokens. And if the high masking ratio is already making the task non-trivial — if the model already has to understand the scene to fill the holes — then maybe I don't *need* the tokenizer to inject semantics; the difficulty is coming from the masking, not the target. Predicting pixels directly is far simpler: no extra stage, no extra data, no tokenizer net. I'll bet on pixels and keep the whole thing self-contained.

One refinement on the pixel target is worth keeping as a switch. Raw pixel MSE asks the model to spend capacity on absolute color and brightness, which are often easy local statistics. If I normalize each patch by its own mean and variance before computing the loss, using `(x - mean) / sqrt(var + eps)` over the `p*p*C` values in that patch, the target puts more relative weight on local contrast and texture within the patch. A low-frequency target such as a few PCA coefficients would move in the opposite direction: it would make the output easier and more semantic-looking, but it would also throw away the fine local structure that pixel reconstruction can use as a dense supervisory signal. So the clean implementation should support per-patch normalized pixels with a `norm_pix_loss` flag while keeping the raw-pixel path simple.

Let me also reconsider the masking *pattern*, because "random" is a real choice. The alternatives are block-wise masking, where I remove a few large contiguous regions, and grid-wise masking, where I keep a regular lattice such as one of every four patches. Grid-wise keeps a downsampled version of the image; every hole has nearby anchors in a predictable arrangement, so the local-interpolation shortcut comes back. Block-wise masking removes that shortcut inside the missing block, but at a high ratio it can erase whole objects or decisive parts of them, leaving too little evidence to condition on. Plain per-sample random sampling sits between those extremes: holes are scattered everywhere, no regular low-resolution image remains, and there are still anchors across the scene. Random also lets the ratio be high, which is what gives the encoder its shortest sequence. So uniform random masking is the pattern.

Now the masking ratio number. The redundancy argument says "high," but it cannot mean "almost everything" without limit, because the decoder still needs enough visible anchors to infer the scene. With the same 196 tokens, keeping 25% leaves `196 × 0.25 = 49` visible patches, a sparse but still broad sketch of the image; keeping only 15% leaves about 29 patches, which starts to feel starved; keeping half leaves 98 patches, dense enough that many missing locations still have close neighbors. So 75% sits at the aggressive-but-reconstructable point I'd want to *test first*: far above BERT's 15%, far above the 20-50% ratios common in earlier vision attempts, and conveniently the exact ratio where the encoder sequence is one quarter of the full image and that 1/16 attention reduction I computed above applies. I'd treat 75% as the leading hypothesis and sweep around it — the redundancy argument fixes the direction, not the decimal.

A couple of consequences fall out of this. Contrastive methods build their signal from comparing two augmented views, so they need carefully chosen augmentations to avoid collapse or shortcuts. Here the fresh random mask is already a strong augmentation: the same image yields a different prediction problem each time. I should keep the image augmentations plain, with random resized crop and horizontal flip, and not make color distortion part of the core signal. Training length also changes its meaning. The encoder only sees 25% of the patches in any one pass, and each pass exposes a different subset, so a long schedule is not just optimization indulgence; it gives the model many masked views of the same image distribution. An 800-epoch default is consistent with that sampling picture, and accuracy keeps climbing out toward 1600 without saturating.

Now let me nail down the implementation, because there's a trap. "Drop 75% of the patches and run the encoder on the rest, then put mask tokens back in the right places for the decoder" sounds like it needs gather/scatter on a sparse, ragged set — the kind of thing that begs for a custom sparse kernel. I don't want that. A clean way to do it uses nothing but shuffle and gather. Tokenize *every* patch and add its positional embedding. Then generate one random number per patch and **argsort** it to get a shuffle order; the first `len_keep` entries of that order are the patches I keep, the rest are dropped — and because the order is a uniform random permutation, taking the first quarter is *exactly* sampling a quarter without replacement. Gather those kept tokens; that short sequence goes to the encoder. To put things back afterward, I need the inverse permutation. My claim is that `argsort(ids_shuffle)` gives it — but this is exactly the kind of index gymnastics where I talk myself into a sign error, so let me not trust the claim and instead run it by hand on a tiny case.

Take four patches and `mask_ratio = 0.5`, so `len_keep = 2`. Say the random noise comes out `[0.7, 0.2, 0.9, 0.4]`. Argsorting it (smallest noise first) gives `ids_shuffle = [1, 3, 0, 2]` — position 1 had the smallest noise so it leads, then 3, then 0, then 2. Now `ids_restore = argsort(ids_shuffle) = argsort([1,3,0,2]) = [2, 0, 3, 1]`. Is that actually the inverse? The test is whether gathering `ids_shuffle` by `ids_restore` returns the identity: `ids_shuffle[[2,0,3,1]] = [0, 1, 2, 3]`. It does — so `ids_restore` really is the inverse permutation, the claim survives.

Now trace the data through it. Keeping the first two of the shuffle order keeps original positions `{1, 3}`. Label each token by its original index so I can follow it: tokens are `[10, 11, 12, 13]`, the kept ones are `[11, 13]`. The encoder sees only those two. For the decoder I append two mask placeholders (write them as `-1`) to make the full shuffled sequence `[11, 13, -1, -1]`, then unshuffle by `ids_restore = [2,0,3,1]`: gathering gives `[-1, 11, -1, 13]`. Read that off — slot 0 is a mask, slot 1 holds token 11, slot 2 is a mask, slot 3 holds token 13. So the two *visible* tokens landed back in their original positions 1 and 3, and the two *masked* slots 0 and 2 got placeholders. That's precisely what I need: every token, visible or mask, aligned with its target position, with no custom sparse kernel. And the binary loss mask is built the same way — an array `[0, 0, 1, 1]` (0 on the first `len_keep`, 1 after) unshuffled by the same restore indices gives `[1, 0, 1, 0]`, i.e. positions 0 and 2 marked removed, 1 and 3 marked kept, matching the placeholder pattern exactly. Good — the bookkeeping is just shuffle, drop, unshuffle, all cheap index operations, not a single specialized sparse op. I'm glad I checked the small case rather than trusting the `argsort(argsort(.))` identity from memory; that's the one line where a silent transpose would have corrupted every target without ever throwing an error.

One more detail for using the encoder downstream. ViT carries a class token for classification. I could average-pool the patch tokens, but a class token keeps the model drop-in compatible with standard ViT recognition code, so I'll prepend that auxiliary token to the encoder input with its own positional embedding. The positional embeddings I'll use are fixed two-dimensional sine-cosine embeddings on both encoder and decoder inputs, with no learned relative-position machinery and no layer scaling; keep it plain.

The code has to make the whole argument concrete: patchify, shuffle, keep one quarter, encode only those visible tokens, reinsert learned mask tokens, unshuffle with the inverse permutation, decode the full sequence, and average MSE over the removed patches only.

```python
from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # Encoder: a standard ViT, later run only on visible patch tokens.
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  qk_scale=None, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Decoder: narrow, full-token reconstruction pathway, discarded later.
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                  qk_scale=None, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        # (N, 3, H, W) -> (N, L, p*p*3)
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p**2 * 3)

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], 3, h * p, h * p)

    def random_masking(self, x, mask_ratio):
        # Per-sample shuffle, keep the first len_keep tokens, and remember the inverse permutation.
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum()

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def mae_vit_base_patch16_dec512d8b(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def mae_vit_large_patch16_dec512d8b(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def mae_vit_huge_patch14_dec512d8b(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b
```

Images carry heavy spatial redundancy, so a low masking ratio makes reconstruction a trivial interpolation that teaches the model nothing — which forces a very high ratio (75%) to kill the redundancy and make filling the holes require holistic understanding. A high ratio means the encoder need only see a quarter of the patches, so I feed it the visible patches alone, with no mask tokens — which simultaneously slashes the quadratic encoder cost and removes the train/deploy mismatch of letting the encoder process placeholders it never sees at deployment. The masked positions are filled in only afterward, by a small, narrow decoder over the full set, where a shared mask token plus positional embeddings marks each spot to predict; the decoder is deliberately the place where pixel-reconstruction specialization gets quarantined, kept deep enough to free the encoder's latent toward abstraction yet small enough to add little compute. The target is the raw (or per-patch-normalized) pixels, scored by MSE on the masked patches only — no tokenizer, no extra data, no adversarial loss. And the sparse bookkeeping is just a shuffle, a drop, and an unshuffle, so it runs on ordinary dense ops. Hide most of it, encode the little that's left, reconstruct the rest cheaply — and the representation that falls out is what I keep.
