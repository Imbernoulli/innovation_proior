OK, let me think this through from scratch. Every image classifier I respect right now is a convolutional network. And every one of them, from the earliest small ones to the deep residual stacks that hold the state of the art, leans on the same three assumptions wired into every single convolutional layer: a unit only looks at a small local neighborhood; the input is a 2D grid where nearby pixels belong together; and the same filter is slid across all positions, so shifting the image shifts the response. Locality, 2D neighborhood structure, translation equivariance. These aren't free — they're a strong prior about what images are like, baked in before any data is seen.

And that prior is genuinely useful. It's why a CNN can learn to classify from a few hundred thousand images: most of the answer is already encoded in the architecture, the data just has to fill in the specifics. So I shouldn't sneer at it. But I keep looking over at language modeling, where the field abandoned domain-specific architecture almost entirely. The Transformer there has essentially no prior about word order beyond what you feed it explicitly — it's a generic sequence mixer — and it won anyway, not by being clever about language but by *scaling*: pretrain on a giant corpus, fine-tune on the small task, pour in parameters and data, and the curve just keeps going up with no plateau in sight. Models with a hundred billion parameters, still improving.

That juxtaposition is nagging at me. A prior helps you when you're starved for data — but does it also cap you? If the architecture already "knows" that vision is local and translation-equivariant, then in the regime where you have so much data that the model could just *learn* those regularities itself, the prior isn't buying you anything; it might even be a constraint preventing the model from discovering something better. So here's the bet I want to make: take an architecture with almost none of the convolutional prior, run it on images, and I'd expect it to *lose* in the small-data regime — the missing prior should hurt exactly there — but to catch up and overtake once the dataset is big enough. If that crossover happens, it would mean large-scale training trumps inductive bias.

Is the data even there to test this? Yes. People have already measured that convolutional accuracy grows roughly logarithmically with pre-training set size, all the way up to hundred-million-image collections, with no sign of saturating. And the strongest transfer recipe going — pretrain big residual nets on fourteen-million-image and three-hundred-million-image labeled sets, then fine-tune simply — already shows that scale plus a clean transfer protocol beats architectural tricks. So the substrate I'd need exists: huge labeled datasets, a transfer pipeline. What's missing is an architecture that scales *like a Transformer* but eats images.

So let me just try to apply a standard Transformer to an image with the fewest possible modifications. The whole point is to change as little as possible — if I start customizing the attention for images, I'm reintroducing priors and, worse, I'm building something that won't run efficiently on the matrix accelerators the standard Transformer was tuned for. Keep it vanilla.

The Transformer eats a 1D sequence of token vectors, `z ∈ ℝ^{N×D}`, and it has a fixed width `D` it carries through every layer. An image is `x ∈ ℝ^{H×W×C}`. The most literal thing: treat each pixel as a token. So `N = H·W` tokens, each projected to `D`. Let me check whether that's tractable, because self-attention's whole engine is computing pairwise interactions. The attention matrix is `A = softmax(q·kᵀ/√D_h) ∈ ℝ^{N×N}` — every token attends to every token, that's `N²` entries, cost `O(N²·D)`. For a 224×224 image, `N = H·W ≈ 50{,}000`. So `N²` is about two-and-a-half *billion*, per layer, per head. That's a complete non-starter — it's bi-quadratic in the side length of the image, `O((HW)²)`. Pixels-as-tokens dies immediately on cost.

This, I think, is exactly the wall everyone before me hit, and I can see the two escape routes they took, and why both leave me unhappy. One route: don't let attention be global — restrict each query to a local window, so it stands in for a convolution. But now I've smuggled the locality prior back in *and* I've built a custom attention pattern. The other route: keep global attention but approximate it — sparse patterns, or attend along one axis at a time. Both of those are "theoretically efficient" but they need specialized, irregular memory-access patterns that map terribly onto the dense matrix-multiply hardware I actually have. They don't scale in practice. And a third camp just keeps the CNN and bolts attention on as a side dish, which doesn't test my hypothesis at all. The one thing nobody seems to have done is just run the *plain, dense, hardware-friendly* Transformer at scale — because of the `N²` pixel wall.

So the real question isn't "how do I make attention cheaper" — it's "how do I make `N` small enough that plain dense attention is affordable, without changing the Transformer at all." `N` is the number of tokens. What if a token isn't a pixel but a *patch* — a little square block of the image? Cut the image into a grid of non-overlapping `P×P` patches. Then `N = HW/P²`. With `P = 16` on a 224×224 image, that's `14×14 = 196` tokens. A hundred and ninety-six — that's *language-sized*. `A` is then `196×196 ≈ 38{,}000` entries, which is nothing. And critically I haven't touched the attention mechanism; I've only changed what a token *is*. The image priors I'm willing to grant are exactly one: the act of cutting the image into a grid of patches. After that, it's a bag — well, a sequence — of patches, and from the Transformer's point of view there's no difference between this and a sentence.

Notice the lever here: `N ∝ 1/P²`. Bigger patches mean a shorter sequence and cheaper attention, but each token is now a coarser chunk of image — less spatial resolution to reason over. Smaller patches mean a longer, more expensive sequence but finer detail. There was prior work that took this to the extreme of `2×2` patches with full attention on top — which is so fine that it only works on tiny images, because the sequence explodes again. Sixteen feels like the sweet spot: short enough to be cheap on medium-resolution images, coarse enough that 196 tokens carry the whole picture. And the quadratic relationship means I can dial this knob later: smaller patches when I have compute, larger when I don't.

Now, a patch is a little `P×P×C` block of raw pixel values, and the Transformer wants a `D`-dimensional vector. So I flatten each patch into a vector of length `P²·C` — for `16×16×3` that's `768` numbers — and I need to map `ℝ^{P²·C} → ℝ^D`. The simplest possible map is a single learned linear projection. Call it `E ∈ ℝ^{(P²·C)×D}`. Patch `i` becomes `x_p^i · E`. That's it — one trainable matrix shared across all patches, producing what I'll call the patch embeddings. I want to resist the temptation to put a little conv stem here or anything fancy; the whole experiment is "as standard as possible," and the patch projection is the single place I'm letting any 2D structure in. (Mechanically, applying one linear map to every non-overlapping `P×P` block *is* exactly a convolution with kernel size `P` and stride `P` — so even my one concession is just the most degenerate conv imaginable, a non-overlapping patch projection, not a feature extractor.)

Now I have a sequence of `N` patch embeddings, each `D`-dimensional. Before I push them through the encoder, two things are missing, and both come from thinking carefully about what self-attention actually is.

First problem: self-attention is permutation-equivariant. It treats its input as a *set*. If I shuffle the order of the patches, the attention computation produces the same outputs in shuffled order — it has literally no idea that patch 5 was to the left of patch 6, or that the grid was 14 wide. For language people patch this with position information, and they have a choice: fixed sinusoidal signals, or a learned table of position vectors, one per slot, added to the token embeddings. Let me think about which I want. Sinusoids hard-code a particular notion of distance; a learned table just gives the model a free parameter per position and lets it figure out what the geometry is. Since my entire thesis is "let the model learn the spatial structure from data rather than hard-wiring it," a *learned* position embedding is the honest choice — it starts knowing nothing about 2D layout. So I add `E_pos`, a learned table with one vector per position.

But which geometry should the table encode — 1D or 2D? My patches really live on a 2D grid, so maybe I should give each patch a 2D-aware embedding: learn an x-table and a y-table and concatenate. Or relative-position embeddings, encoding offsets between patch pairs. Let me reason about whether that's worth it before I build it. The grid here is `14×14`. That is *tiny* — fourteen distinct positions along each axis. Learning, from a flat 1D table of 196 vectors, which positions are neighbors in 2D is a small amount of structure to recover at this coarseness. The 2D-ness would matter a lot if I were operating at pixel resolution, `224×224`, where the table would be huge and the relationships hard to learn. At patch resolution it should barely matter. I cannot omit position information entirely, because then the model is reasoning over an unordered bag of patches, but a plain 1D learned position embedding, with patches laid out in raster order, gives every slot an identity while still refusing to hard-code image geometry. I'll keep that simplest version.

Second problem, and this one's more interesting. The encoder maps a sequence of `N` vectors to a sequence of `N` vectors. But I need *one* vector to classify the whole image. How do I pool the sequence down to a single representation? The obvious move is to average all the output patch vectors — global average pooling, just like a CNN does to its final feature map — and run a linear classifier on that. That's the instinct. But let me think about what I'd be asking. If I average, I'm forcing every patch's output to do double duty: contribute to the per-patch representation *and* be a good thing to average for classification, and the averaging weights are uniform and fixed. There's a cleaner idea sitting right there in how encoder-style language models do classification. Prepend one extra token to the sequence — a token that is not any patch, just a learnable vector `x_class` — and let it ride through all the attention layers. Because attention is global from the very first layer, this extra token can attend to every patch and aggregate whatever it needs; its job is purely to be the readout slot. Its state at the output of the encoder, `z_L^0`, *is* the image representation. I prepend it, so the input becomes `[x_class; patch embeddings]` and the position table grows by one to `N+1`. Then I only ever read out position 0 and feed it to the head. This keeps me maximally close to the standard text encoder, which is the whole spirit of the exercise.

So the input to the encoder is

```
z_0 = [ x_class ;  x_p^1 E ;  x_p^2 E ;  … ;  x_p^N E ] + E_pos,
      E ∈ ℝ^{(P²·C)×D},   E_pos ∈ ℝ^{(N+1)×D}.
```

Now the encoder itself. I said "standard," so let me actually be standard, but I want to understand each piece rather than copy it. The core is self-attention. For a sequence `z`, project each element to a query, key, and value with one shared matrix: `[q, k, v] = z·U_qkv`, `U_qkv ∈ ℝ^{D×3D_h}`. The attention weights are pairwise similarities between queries and keys, normalized by softmax, and the output is the value vectors weighted by those:

```
A = softmax( q·kᵀ / √D_h ) ∈ ℝ^{N×N},      SA(z) = A·v.
```

Why the `/√D_h`? Let me actually derive it instead of taking it on faith, because if I get this wrong the whole thing won't train. Suppose the entries of `q` and `k` are roughly independent with mean 0 and unit variance — reasonable at initialization. A single logit is a dot product `q·k = Σ_{m=1}^{D_h} q_m k_m`, a sum of `D_h` independent zero-mean unit-variance terms. The variance of a sum of independent terms adds, so `Var(q·k) = D_h`, i.e. the logits have standard deviation `√D_h`. Now feed those into a softmax. If `D_h` is large, the logits are spread out over a range of order `√D_h`, which can be big — say tens. Softmax of widely-spread logits is nearly one-hot: one weight near 1, the rest near 0. And the gradient of softmax is largest when the distribution is soft and collapses toward zero as it saturates toward one-hot — a near-one-hot softmax has vanishing gradient. So large `D_h` would push me straight into a dead, saturated regime where almost no gradient flows back through attention and learning stalls. Dividing the logits by `√D_h` rescales their standard deviation back to 1 regardless of `D_h`, keeping the softmax soft and its gradients alive. That's the reason, and it's not optional — it's what makes attention trainable as I scale the head width.

One head computes one softmax — one averaging pattern over the sequence. But an image patch might need to relate to several different things at once: a head that looks broadly across the whole image for context, and a head that focuses tightly near its own location. A single softmax can't be both. So run `k` attention operations in parallel — "heads" — each in its own subspace, then concatenate their outputs and project back to `D`:

```
MSA(z) = [ SA_1(z) ; SA_2(z) ; … ; SA_k(z) ] · U_msa,    U_msa ∈ ℝ^{k·D_h × D}.
```

Different heads can specialize on different relations simultaneously. Now, how wide should each head be? If I set each head to full width `D` and use `k` of them, compute and parameters scale with `k` — expensive. The clean trick is to set `D_h = D/k`. Then the `k` heads together still produce `k·D_h = D` dimensions before the output projection, and the total work of the multi-head layer is about the same as a single full-width attention. So I get the expressive benefit of multiple specialized heads at essentially constant cost. That's why head dimension is tied to `D/k` rather than chosen freely.

Attention mixes information *across positions*, but within a position it's just a weighted average of value vectors — it doesn't transform the feature content of a single token nonlinearly. So after attention I need a per-position transformation: a small MLP applied to each token independently. Two linear layers with a nonlinearity between them. How wide should the hidden layer be? It's the only place in the block doing nonlinear feature mixing within a token, and the convention that works is to expand to about `4×` the model width and project back — wide enough to give the block real per-token capacity. For the nonlinearity I'll use the smooth gating activation that's standard in these pretrained text Transformers rather than swapping in a plain ReLU; again, the discipline is to stay vanilla, and the smooth one has been the better default in exactly this kind of large pretrained model. So: attention mixes across tokens, the MLP mixes across channels, and the block alternates the two.

Each of these two sublayers gets a residual connection — essential for training anything deep, it gives gradients a clean path to flow back — and a layer normalization. Where exactly does the normalization go? The original Transformer put it *after* the residual add. But I'm planning to train this big, at large learning rate, at scale. Post-normalization has a known failure here: at initialization the gradients near the output are large, which makes large learning rates unstable and forces a delicate warmup. If instead I put the layer norm *inside* the residual branch — normalize the input to each sublayer, then add the sublayer's output to the un-normalized input — the residual path stays a clean identity highway, gradients are well-behaved at initialization, and I can train stably at a larger learning rate. So pre-normalization it is. Concretely each block is

```
z'_ℓ = MSA( LN(z_{ℓ-1}) ) + z_{ℓ-1},     ℓ = 1 … L,
z_ℓ  = MLP( LN(z'_ℓ) )    + z'_ℓ,        ℓ = 1 … L,
```

and after the last block I take the class-token slot and normalize it once more to get the image representation:

```
y = LN( z_L^0 ).
```

For the sizes, I'll just borrow the configurations the text-encoder world has already validated rather than reinvent them: a "Base" with 12 layers, width `D = 768`, 12 heads, MLP hidden 3072; a "Large" with 24 layers, width 1024, 16 heads, MLP 4096; and since I expect scale to be the whole story, a "Huge" beyond them, 32 layers, width 1280, 16 heads, MLP 5120. Note the naming I'll use: "Large/16" means the Large config with `16×16` patches. And since sequence length goes as `1/P²`, a smaller patch size is markedly more expensive for the same model — that's the compute knob.

Now the head on top of `y`. During pre-training I want the representation to form richly, so I'll use a small MLP — one hidden layer with a tanh nonlinearity — before the class logits. When I then transfer to a downstream task, I throw that head away entirely and attach a fresh single linear layer `ℝ^{D×K}` for the `K` new classes, and I initialize it to zero. Why zero? A zero-initialized final layer starts with equal logits for every downstream class, so the first updates are driven by the downstream labels and the representation, not by arbitrary random classifier scores.

There's one more thing I have to handle, and it comes from a quirk of having a *learned* position table. It's common and beneficial to fine-tune at a higher resolution than I pre-trained at — more pixels, more accuracy. But I'm keeping the patch size `P` fixed, so a higher-resolution image produces *more patches* — a longer sequence. The Transformer itself doesn't care; attention handles any sequence length. But my learned position table `E_pos` has exactly one row per pre-training position, on the pre-training grid. At the new resolution the grid is a different shape, so those rows no longer correspond to the right locations. The fix uses the one piece of 2D knowledge I do have: I know where each pre-trained position sat in the original 2D grid, so I lay the position vectors back out on that grid and *2D-interpolate* them to the new, larger grid. This is, notably, the *only* other point besides the initial patch cut where I inject any hand-built knowledge of the image's 2D structure — everything else the model learned on its own.

The amount of image-specific prior left is tiny. In a CNN, locality and 2D neighborhood structure and translation equivariance are enforced in *every layer*, everywhere, always. Here? The self-attention layers are fully global — no locality at all. The MLP layers are applied per-token, so they're "local and translation-equivariant" only in the trivial sense of acting the same on every position, but they don't mix neighbors. The 2D neighborhood structure enters in exactly two places: cutting the image into patches at the very start, and interpolating the position embeddings when I change resolution. That's it. At initialization the position embeddings carry *no* information about where patches sit in 2D — every spatial relationship between patches has to be learned from data. Which is precisely the bet: strip the prior down to almost nothing, and lean on scale to learn the rest.

And that's also exactly why I expect the data-scale crossover. With little data — say ImageNet alone — the missing convolutional prior should hurt: the model has to spend capacity learning that vision is roughly local and translation-equivariant, structure a CNN got for free, and with only a million-ish images it'll do that imperfectly and overfit, so it should land below a comparable ResNet. The Large config should even underperform the Base config there, because more capacity with no prior and not enough data is just more overfitting. But as I grow the pre-training set toward fourteen million and then three hundred million images, the model has enough signal to *learn* the right spatial regularities directly, and now its lack of a constraining prior becomes an advantage — it can fit patterns convolution would never have allowed. So I'd predict the curves cross: ResNets ahead at small scale, the Transformer catching up as the data grows, then pulling ahead in the largest-data regime if the scale benefit is real. To check the crossover cleanly I'd train on random subsets of the big dataset (nine, thirty, ninety, three hundred million) with the same hyperparameters and no extra regularization, so I'm measuring the model's intrinsic appetite for data, not a regularization artifact. That's the validation I'd want to run; the prediction is the crossover itself.

One reassuring sanity check on the representational side. There's a known result that a multi-head self-attention layer with enough heads and a position encoding can express *any* convolution of a given kernel size. So I'm not giving up the ability to *be* convolutional — attention can imitate convolution if that's what's optimal. The difference from the prior attempts is that I'm not hard-wiring it or restricting attention to make it convolution-like; I'm leaving it free and global and betting that, with enough data, the heads can learn whatever mix of local and global relations the task needs.

Let me also make sure the cost story closes. The whole reason this works is that patchifying made `N` small. Attention is `O(N²·D)` with `N = HW/P² = 196`, so the quadratic term is tiny; the model is dominated by the linear-in-`N` projections and MLPs, just like a text Transformer. The theoretical bi-quadratic-in-resolution scaling of attention only ever bites at very large images and very large models — for the regime I care about it's a non-issue, and the model is friendly to accelerators because it is built from dense matrix operations. The escape from the pixel wall is complete and it cost me nothing but the patch cut.

```python
import torch
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    # per-token channel mixing: pre-norm, expand ~4x, GELU, project back
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    # multi-head self-attention; D_h = dim // heads keeps cost ~constant
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        assert dim % heads == 0
        dim_head = dim // heads
        inner_dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5          # the 1/sqrt(D_h) that keeps softmax unsaturated
        self.norm = nn.LayerNorm(dim)          # pre-norm
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)   # one shared U_qkv
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(b, n, self.heads, -1).transpose(1, 2) for t in qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # A logits = q k^T / sqrt(D_h)
        attn = self.dropout(self.attend(dots))                     # A = softmax(...)
        out = torch.matmul(attn, v)                                # SA = A v, per head
        out = out.transpose(1, 2).reshape(b, n, -1)                # concat heads
        return self.to_out(out)                                    # project back to D


class Transformer(nn.Module):
    # pre-norm encoder: alternate MSA and MLP, each with a residual
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x          # z'_l = MSA(LN(z)) + z
            x = ff(x) + x            # z_l  = MLP(LN(z')) + z'
        return self.norm(x)          # final LN (-> y on the class token)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, channels=3, dropout=0., emb_dropout=0.,
                 representation_size=None):
        super().__init__()
        ih, iw = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        ph, pw = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert ih % ph == 0 and iw % pw == 0
        num_patches = (ih // ph) * (iw // pw)          # N = HW / P^2
        self.grid_size = (ih // ph, iw // pw)

        # stride-P projection: the flattened-patch linear map E written as a conv
        self.patch_embedding = nn.Conv2d(
            channels, dim, kernel_size=(ph, pw), stride=(ph, pw)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))          # x_class, the readout slot
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1, dim))  # learned 1D E_pos
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        head_dim = representation_size or dim
        self.pre_logits = (
            nn.Sequential(nn.Linear(dim, representation_size), nn.Tanh())
            if representation_size else nn.Identity()
        )
        self.head = nn.Linear(head_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _position_embedding(self, h, w):
        if (h, w) == self.grid_size:
            return self.pos_embedding
        cls_pos = self.pos_embedding[:, :1]
        patch_pos = self.pos_embedding[:, 1:]
        gh, gw = self.grid_size
        patch_pos = patch_pos.reshape(1, gh, gw, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward(self, img):
        x = self.patch_embedding(img)                 # (b, D, H/P, W/P)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)              # (b, N, D)
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)                # [x_class ; patches] -> (b, N+1, D)
        x = x + self._position_embedding(h, w)        # add or interpolate E_pos
        x = self.dropout(x)
        x = self.transformer(x)                       # encoder
        x = self.pre_logits(x[:, 0])                  # y = LN(z_L^0), optional tanh pre-logits
        return self.head(x)
```

So the causal chain, start to finish: the convolutional prior is what makes CNNs sample-efficient, but I suspect it's also a ceiling, and language has shown that a prior-free sequence model wins by scaling — so I want to run a *plain* Transformer on images and bet that scale beats the prior. The plain Transformer chokes on pixels because attention is quadratic and an image has tens of thousands of pixels, so I cut the image into `16×16` patches and let each patch be a token — `N` drops to language scale and the attention mechanism is left completely untouched. Each patch is flattened and pushed through one linear projection `E` to the model width. Attention is order-blind, so I add a *learned* position table — 1D, because at `14×14` patch resolution I can give every slot an identity without hard-coding 2D geometry. The encoder maps a sequence to a sequence, but I need one vector, so I prepend a learnable class token and read out its final state, staying maximally close to the standard text encoder. Inside, scaled dot-product attention with `/√D_h` to keep the softmax unsaturated, multiple heads at `D/k` width each to capture global and local relations at constant cost, a `4×`-wide GELU MLP for per-token channel mixing, all in pre-norm residual blocks so it trains stably at scale. A rich tanh-MLP head during pre-training, swapped for a zero-init linear head at transfer, with 2D-interpolated position embeddings when fine-tuning at higher resolution. The only image-specific structure left anywhere is the patch cut and that one interpolation — everything spatial is learned. The prediction that falls out: lose to ResNets on small data, cross over and overtake as the pre-training set grows into the hundreds of millions.
