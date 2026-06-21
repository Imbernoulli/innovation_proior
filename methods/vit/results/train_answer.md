Image recognition has belonged entirely to convolutional networks, from the earliest small models through deep residual stacks, and every one of them leans on the same three assumptions wired into every convolutional layer: locality, where a unit only sees a small neighborhood; 2D neighborhood structure, where the input is a grid and nearby pixels belong together; and translation equivariance, where the same filter slides across every position so shifting the input shifts the response. These priors are what make CNNs sample-efficient — a large fraction of the answer is encoded in the architecture before any data is seen, and the data only fills in the specifics. But that very property is also what I want to interrogate. In language modeling the field abandoned domain-specific architecture almost entirely and won by scaling: the Transformer carries essentially no prior about word order beyond what is fed to it, yet pre-training on a giant corpus and fine-tuning on the small task keeps improving with parameters and data, with no plateau in sight. The nagging question is whether a prior that helps in the small-data regime also caps the ceiling — whether, once there is enough data that the model could simply learn locality and translation-equivariance itself, the hard-wired prior stops buying anything and may even constrain the model from discovering something better.

The existing options all fall short of testing this cleanly. The literal idea of running a Transformer on raw pixels dies immediately on cost: self-attention computes pairwise interactions, so its attention matrix $A = \mathrm{softmax}(qk^\top/\sqrt{D_h})$ is $N\times N$ with cost $O(N^2 D)$, and for a $224\times224$ image $N = HW \approx 50{,}000$, making $N^2$ roughly two and a half billion per layer per head — bi-quadratic in the side length, $O((HW)^2)$. The escape routes people took each reintroduce what I want to avoid: restricting attention to local windows smuggles the locality prior back in and builds a custom attention pattern; approximating global attention with sparse or axial patterns needs irregular memory access that maps terribly onto dense matrix-multiply accelerators and so has not scaled; and keeping the CNN and bolting attention on as an augmentation does not test the hypothesis at all. The one thing nobody had done was run the plain, dense, hardware-friendly Transformer at scale on images — blocked by the $N^2$ pixel wall. And the substrate to test the scaling bet already exists: convolutional accuracy is known to grow roughly logarithmically with pre-training set size up to hundred-million-image collections with no saturation, and the strongest transfer recipe of the day — pre-train large residual nets on the fourteen-million-image `ImageNet-21k` and the three-hundred-million-image `JFT-300M`, then fine-tune simply — already shows scale plus a clean protocol beating architectural tricks. What is missing is an architecture that scales like a Transformer but eats images.

I propose the Vision Transformer, ViT: take an unmodified Transformer encoder and feed it an image as a short sequence of patch tokens. The key realization is that the problem is not "make attention cheaper" but "make $N$ small enough that plain dense attention is affordable, without touching the attention mechanism at all." So a token is not a pixel but a patch — cut the image into a grid of non-overlapping $P\times P$ blocks, giving $N = HW/P^2$. With $P=16$ on a $224\times224$ image that is $14\times14 = 196$ tokens, language-sized, and $A$ becomes $196\times196 \approx 38{,}000$ entries, which is nothing. Since $N \propto 1/P^2$, the patch size is the compute knob: bigger patches mean a shorter, cheaper sequence at coarser spatial resolution, smaller patches mean a longer, more expensive sequence with finer detail — sixteen is the sweet spot, short enough to be cheap and coarse enough that the tokens still carry the whole picture. Each patch is a $P^2\!\cdot\!C$ block of raw pixel values; I flatten it and apply one shared trainable linear projection $E \in \mathbb{R}^{(P^2 C)\times D}$ to the model width $D$, so patch $i$ becomes $x_p^i E$. I resist any conv stem or feature extractor here — applying one linear map to every non-overlapping block is exactly a convolution with kernel size $P$ and stride $P$, the most degenerate conv imaginable, and that single patch cut is the one place I let any 2D structure enter the front of the model.

Two things must be supplied before the encoder, and both follow from what self-attention actually is. First, self-attention is permutation-equivariant: it sees a set, so it has no idea patch 5 was left of patch 6 or that the grid was 14 wide, and position must be injected. I add a learned position table $E_{pos}$, one vector per slot, rather than fixed sinusoids — sinusoids hard-code a notion of distance, whereas a learned table starts knowing nothing about geometry and lets the model recover it from data, which is the honest choice given the whole thesis. And I keep it 1D rather than building x/y or relative-position tables, because at $14\times14$ the grid is tiny and recovering 2D neighbor relations from a flat raster-ordered table of 196 vectors is a small amount of structure to learn; the 2D-ness would matter at pixel resolution, not here. Second, the encoder maps $N$ vectors to $N$ vectors but I need one vector to classify the image. Rather than global-average-pool the patch outputs — which forces every patch to do double duty under fixed uniform weights — I prepend one extra learnable vector $x_{class}$, not any patch, and let it ride through all the attention layers; because attention is global from the first layer it can aggregate whatever it needs, and its output state $z_L^0$ is the image representation. So the input is

$$z_0 = [\,x_{class};\ x_p^1 E;\ x_p^2 E;\ \dots;\ x_p^N E\,] + E_{pos},\qquad E \in \mathbb{R}^{(P^2 C)\times D},\quad E_{pos} \in \mathbb{R}^{(N+1)\times D}.$$

The encoder itself is deliberately standard, but each piece earns its place. Self-attention projects each element to a query, key, and value with one shared matrix, $[q,k,v] = z\,U_{qkv}$ with $U_{qkv}\in\mathbb{R}^{D\times 3D_h}$, forms the softmax-normalized pairwise similarities, and outputs the value vectors weighted by them:

$$A = \mathrm{softmax}\!\left(\frac{q k^\top}{\sqrt{D_h}}\right) \in \mathbb{R}^{N\times N},\qquad \mathrm{SA}(z) = A v.$$

The $1/\sqrt{D_h}$ is load-bearing and worth deriving rather than taking on faith. If the entries of $q$ and $k$ are roughly independent with mean 0 and unit variance at initialization, a single logit $q\cdot k = \sum_{m=1}^{D_h} q_m k_m$ is a sum of $D_h$ independent zero-mean unit-variance terms, so $\mathrm{Var}(q\cdot k) = D_h$ and the logits have standard deviation $\sqrt{D_h}$. Feeding logits spread over a range of order $\sqrt{D_h}$ into a softmax drives it nearly one-hot for large $D_h$, and the gradient of the softmax collapses toward zero as it saturates — training would stall. Dividing by $\sqrt{D_h}$ rescales the logits back to unit standard deviation regardless of head width, keeping the softmax soft and its gradients alive, which is precisely what makes attention trainable as the head width grows. One head computes a single averaging pattern, but a patch may need to relate to several things at once — a head looking broadly for context and a head focusing tightly nearby — so I run $k$ heads in parallel, each in its own subspace, then concatenate and project back:

$$\mathrm{MSA}(z) = [\,\mathrm{SA}_1(z);\ \dots;\ \mathrm{SA}_k(z)\,]\,U_{msa},\qquad U_{msa}\in\mathbb{R}^{kD_h\times D}.$$

Setting $D_h = D/k$ is the clean choice: the $k$ heads together still produce $k D_h = D$ dimensions before the output projection, so multiple specialized heads cost about the same as a single full-width attention rather than scaling with $k$. Attention only mixes across positions and within a token is just a weighted average of values, so after it I add a per-position MLP — two linear layers with a smooth GELU nonlinearity between them — expanded to about $4\times$ the model width, which is the one place doing nonlinear feature mixing within a token and needs the capacity. Attention mixes across tokens, the MLP mixes across channels, and the block alternates them. Each sublayer gets a residual connection so gradients flow cleanly through depth, and a layer normalization placed inside the residual branch — pre-normalization — rather than after the add. Post-norm has large gradients near the output at initialization, which makes large learning rates unstable and forces delicate warmup; normalizing the input to each sublayer leaves the residual path a clean identity highway and trains stably at scale. Concretely each block is

$$z'_\ell = \mathrm{MSA}(\mathrm{LN}(z_{\ell-1})) + z_{\ell-1},\qquad z_\ell = \mathrm{MLP}(\mathrm{LN}(z'_\ell)) + z'_\ell,\qquad \ell = 1\dots L,$$

and after the last block I read out and normalize the class-token slot, $y = \mathrm{LN}(z_L^0)$. The sizes borrow the validated text-encoder configurations: Base ($L{=}12$, $D{=}768$, 12 heads, MLP 3072), Large ($L{=}24$, $D{=}1024$, 16 heads, MLP 4096), and a Huge ($L{=}32$, $D{=}1280$, 16 heads, MLP 5120), with "Large/16" denoting the Large config at $16\times16$ patches. During pre-training a small MLP with a tanh nonlinearity sits before the class logits so the representation forms richly; at transfer that head is discarded and replaced by a fresh single linear layer $\mathbb{R}^{D\times K}$ for the $K$ new classes, initialized to zero so the first updates are driven by the downstream labels rather than arbitrary random classifier scores. One quirk of a learned position table must be handled: fine-tuning at higher resolution with $P$ fixed yields more patches and a longer sequence, which attention handles fine, but $E_{pos}$ has one row per pre-training grid position. The fix uses the one piece of 2D knowledge I have — I lay the position vectors back out on the original grid and 2D-interpolate them to the new grid. That interpolation and the initial patch cut are the only two places any hand-built 2D structure enters; the self-attention layers are fully global with no locality, the MLP acts per-token, and at initialization the position embeddings carry no information about where patches sit, so every spatial relationship must be learned. That is the whole bet, and it predicts a data-scale crossover: with little data the missing convolutional prior should hurt — the model overfits while spending capacity learning structure a CNN got for free, and Large should even trail Base — but as the pre-training set grows into the hundreds of millions the lack of a constraining prior becomes an advantage, the heads learn whatever mix of local and global relations the task wants, and the curves cross. Representationally nothing is given up: a multi-head self-attention layer with enough heads and a position encoding can express any convolution, so attention can imitate convolution if that is optimal, but here it is left free and global rather than hard-wired. And the cost story closes — with $N=196$ the quadratic attention term is tiny and the model is dominated by the linear-in-$N$ projections and MLPs, just like a text Transformer, built entirely from dense matrix operations.

```python
import torch
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        assert dim % heads == 0, 'model dimension must be divisible by number of heads'
        dim_head = dim // heads
        inner_dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5            # 1 / sqrt(D_h)
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        b, n, _ = x.shape
        q, k, v = (t.reshape(b, n, self.heads, -1).transpose(1, 2)
                   for t in self.to_qkv(x).chunk(3, dim=-1))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
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
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, channels=3, dropout=0., emb_dropout=0.,
                 representation_size=None):
        super().__init__()
        ih, iw = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        ph, pw = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert ih % ph == 0 and iw % pw == 0, 'image size must be divisible by patch size'
        num_patches = (ih // ph) * (iw // pw)            # N = HW / P^2
        self.grid_size = (ih // ph, iw // pw)

        self.patch_embedding = nn.Conv2d(
            channels, dim, kernel_size=(ph, pw), stride=(ph, pw)
        )                                                # E, written as stride-P projection

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1, dim))
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
        x = self.patch_embedding(img)                    # (b, D, H/P, W/P)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)                 # (b, N, D)
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)                   # (b, N+1, D)
        x = x + self._position_embedding(h, w)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.pre_logits(x[:, 0])                     # class-token readout
        return self.head(x)


# Base/16 at 224 resolution
def vit_base_patch16_224(num_classes=1000):
    return ViT(image_size=224, patch_size=16, num_classes=num_classes,
               dim=768, depth=12, heads=12, mlp_dim=3072, representation_size=768)
```
