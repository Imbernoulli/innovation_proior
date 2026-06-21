The task is to predict chemical properties from molecular structure in the regime that should reward pretraining: a few thousand labelled molecules per task, a test split by scaffold so the model must generalise to structurally novel cores, and an essentially unlimited pile of unlabelled molecules. The recipe that worked for BERT and ViT — self-supervise on the unlabelled ocean, finetune a light head on the labels — should transfer directly. What keeps biting is not the recipe but the input. Every property I care about — whether a molecule crosses the blood-brain barrier, inhibits an enzyme, is toxic in an assay — is decided by how the atoms actually sit in space: which groups are close enough to touch, the shape that does or does not fit a binding site. That has been the oldest fact in the field since Crum-Brown and Fraser. And yet the encoders everyone uses eat a SMILES string or a 2D bond graph, both of which throw the geometry away before the model ever sees it. The few "3D-aware" methods only use geometry as a side channel: GraphMVP contrasts a 2D-graph view against a 3D view to nudge a 2D encoder toward 3D-consistency, and GEM bolts bond-length and bond-angle features onto graph edges, but in both cases finetuning still ingests the 2D graph and the geometry has leaked away — the model can never take coordinates in directly, nor ever output a 3D structure. The supervised graph encoders (D-MPNN/Chemprop, AttentiveFP) are 2D-only and trained from scratch per task, so they cannot exploit the unlabelled pile at all. The genuinely 3D-correct option, the SE(3)-equivariant tensor networks, carries higher-order geometric tensors through every layer; it is accurate but far too heavy to pretrain on hundreds of millions of conformers. So the real question is how to build an encoder whose input *is* the 3D structure — atoms with coordinates — that respects rotation and translation symmetry, sees long-range spatial contacts and not just bonded neighbours, and stays cheap enough to run for a million pretraining steps.

I propose Uni-Mol, a universal 3D molecular representation framework built on an SE(3)-invariant Transformer that consumes atom types and 3D coordinates directly. The backbone has to be a Transformer rather than a message-passing GNN, because a GNN for efficiency connects each atom only to its bonded or radius-cutoff neighbours, and that locality is exactly wrong here: two atoms many bonds apart can sit right next to each other in folded space and interact strongly, and a local GNN structurally cannot see that. Self-attention gives full connectivity for free, treats the atoms as the unordered set they are, and supports an order-invariant readout. But a Transformer is position-blind — $\mathrm{softmax}(QK^\top/\sqrt d)V$ depends only on the token features — and my "position" is a continuous 3D coordinate I am not even allowed to use raw, because rotating or translating the molecule, which changes nothing chemically, would change every input vector. The way to inject position without breaking full connectivity is Graphormer's: leave the architecture alone and add a learnable per-pair bias to the attention logits. Graphormer's bias, though, indexes discrete 2D topology — shortest-path distance, bond features — and has nowhere to put a continuous Euclidean distance and no notion of rotation invariance. The slot is right; the contents must change. The resolving observation is that the one quantity about a point pair $(x_i, x_j)$ that is completely unchanged by any rigid motion of the whole molecule is the Euclidean distance $d_{ij} = \lVert x_i - x_j \rVert$. Build the bias purely out of pairwise distances and SE(3)-invariance comes for free, with none of the equivariant-tensor machinery.

A raw distance is one scalar per pair, and I will eventually backprop through it, so I featurise it against a learnable bank of $K$ Gaussians with means $\mu^k$ and widths $\sigma^k$, where a Gaussian at $\mu^k$ lights up when the distance is near it, so the vector of activations is a soft one-hot of where on the distance axis the pair lives. The Gaussian basis beats the obvious alternative of binning the distance and embedding each bin for two reasons that matter here: it is smooth and differentiable in $d$ everywhere — which I will need, because pretraining perturbs coordinates and asks the model to recover them, and a binned embedding is piecewise-constant and so flat in $d$ — and it has no bin-boundary artifacts where two nearly-equal distances on opposite sides of an edge get wildly different embeddings. One more thing bothers me: a bare distance kernel encodes a C–C pair and a C–O pair at the same 1.5 Å identically, because it only knows the number. I want the encoding aware of *what kind of pair* it is, using only inputs I already have — the two atoms' element types, not the bond, since the bond is the 2D information I am trying to live without. So I make the kernel pair-type-aware with a per-type affine on the distance *before* the basis, with $t_{ij}$ the unordered element-pair type and learnable scalars $a_{t_{ij}}, b_{t_{ij}}$:
$$A(d_{ij}, t_{ij}) = a_{t_{ij}}\, d_{ij} + b_{t_{ij}}, \qquad p_{ij}^k = G(A(d_{ij}, t_{ij}), \mu^k, \sigma^k), \quad G(d,\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\Big(\!-\frac{(d-\mu)^2}{2\sigma^2}\Big).$$
The affine rescales and shifts the distance axis per pair type, so the same physical distance lands in different regions of the shared Gaussian bank for different element pairs. Projecting the $K$ channels down to $H$ (the number of heads) with a small MLP gives the initial pair representation $q^0$, which enters the attention as a per-head bias — the pair-to-atom communication:
$$\mathrm{Attention}(Q_i, K_j, V_j) = \mathrm{softmax}\!\Big(\frac{Q_i K_j^\top}{\sqrt d} + q_{ij}\Big) V_j.$$
The only extra cost over a vanilla Transformer is the Gaussian features and one small projection — a rounding error next to attention itself.

Treating $q_{ij}$ as a frozen input bias leaves value on the table. The distance is fixed by the conformer, but the *meaning* of an interaction is something the model figures out as it processes the molecule, and right now that figuring-out lives entirely in the atom features while the pair channel stays at its initial geometric value. The signal that captures "how strongly do these two atoms interact right now" is already computed per pair per layer: the query-key product. So I let the atoms write back into the pairs — the mirror, atom-to-pair communication — by adding the current per-head $Q_i K_j^\top/\sqrt d$ into the pair representation at every layer:
$$q_{ij}^{\,l+1} = q_{ij}^{\,l} + \Big\{\tfrac{1}{\sqrt d}\, Q_i^{l,h}\, (K_j^{l,h})^\top : h = 1 \dots H\Big\}.$$
Now the two channels genuinely talk: the pair feature biases attention, and the attention affinity refines the pair feature, layer after layer. Because this is a residual accumulation rather than an overwrite, after $L$ layers the pair representation is its initial geometric value plus the sum of all per-layer corrections, so $q^L - q^0$ is exactly the *learned correction* to the raw geometry — the part of the relationship the network discovered that the bare distance never said. That delta is what I will drive the coordinate head with.

For property prediction the model need only read out a molecule-level vector, but I want the same backbone to be universal — usable for the self-supervised pretraining I am counting on and for genuinely 3D tasks that *output* a conformation. Outputting coordinates is delicate because, unlike the invariant input, the output must be equivariant: rotate the input and the predicted coordinates must rotate with it. The cleanest equivariant position update is EGNN's — move each atom along a size-normalised weighted sum of relative vectors with invariant scalar weights:
$$\hat x_i = x_i + \frac{1}{n}\sum_{j}(x_i - x_j)\, c_{ij}.$$
Under a rigid motion $x \mapsto Rx + t$, each difference $(x_i - x_j) \mapsto R(x_i - x_j)$ — the translation cancels and the rotation factors out — while the scalar $c_{ij}$, built only from invariants, does not move, so $\hat x_i \mapsto R\hat x_i + t$: the update rotates and translates exactly with the input. The scalar weight comes from the pair feature, and specifically from the *delta*, since that is the learned part and the raw geometry is already the input:
$$c_{ij} = \mathrm{ReLU}\big((q_{ij}^L - q_{ij}^0)\,U\big)\,W, \qquad U \in \mathbb{R}^{H\times H},\ W \in \mathbb{R}^{H\times 1}.$$
Unlike EGNN, which moves the coordinates at every layer, I run this update only once in the last layer, since I only need the final geometry; the $1/n$ normaliser plays EGNN's $1/(M-1)$ role, keeping the update scale independent of molecule size.

Stability decides whether a million-step run finishes, so each block is Pre-LN — normalise inside the residual branch, before the sub-layer — rather than Post-LN, which puts large gradients near the output at initialisation and forces a fragile learning-rate warmup. One further mixed-precision hazard is that the final LayerNorm can hide a very large pre-normalized state, so I add a tiny penalty when the atom and delta-pair state norms drift outside a tolerance band around $\sqrt d$, $L_{\text{norm}} = \mathrm{mean}_i\,\max(\lvert \lVert s_i\rVert - \sqrt d\rvert - \tau,\,0)$ with $\tau = 1$ and weight $0.01$. With the coordinate head in hand the pretraining task writes itself, and it has to be 3D. I keep a BERT-style masked-atom-type prediction, since atom type is discrete and a [MASK] symbol is well defined. But for coordinates masking does not transfer — any value I substitute is itself a legitimate point in space — so instead of masking I *corrupt*: perturb each coordinate by a uniform offset of about $\pm 1$ Å and train the model to recover the truth, through the equivariant coordinate head predicting clean positions and a pair-distance head regressing clean distances off $q_{ij}$. At a modest noise scale the noised-to-true correspondence is unambiguous and no re-assignment step is needed. The molecule readout prepends a [CLS] atom placed at the centroid of all atoms — so its distances to everything are meaningful — and runs its representation through a small head (dropout, dense with $\tanh$, dropout, linear to the tasks); multitask panels with missing labels use the pipeline's masked loss, and finetuning leans on cheap multi-conformer augmentation, averaging predictions over a handful of generated geometries at test time. The base model is 15 layers, embedding dim 512, FFN dim 2048, 64 heads, $K = 128$, GELU, pretrained on ~209M conformations with Adam $(\beta_1,\beta_2)=(0.9, 0.99)$, $\epsilon = 10^{-6}$, peak LR $10^{-4}$, 1M steps, 10K warmup, batch 128, weight decay $10^{-4}$, gradient clip 1.0, and pretraining loss $\mathrm{CE}(\text{atom}) + 5\cdot\mathrm{SmoothL1}(\text{coord}) + 10\cdot\mathrm{SmoothL1}(\text{dist})$, the coordinate and distance weights lifted because their raw magnitudes are small next to the cross-entropy.

The SE(3)-invariant spatial encoding (pair-type Gaussian kernel):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)        # G(d, μ, σ)


class GaussianLayer(nn.Module):
    """Pair-type-aware Gaussian distance kernel."""
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)               # μ^k
        self.stds = nn.Embedding(1, K)                # σ^k
        self.mul = nn.Embedding(edge_types, 1)        # a_t : per-pair-type scale
        self.bias = nn.Embedding(edge_types, 1)       # b_t : per-pair-type shift
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):                  # x = distances, edge_type = pair types
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias              # A(d, t) = a_t·d + b_t
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)   # [B, S, S, K]
```

Attention with pair-bias (pair → atom) and QK write-back (atom → pair):

```python
class SelfMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key_padding_mask=None, attn_bias=None, return_attn=False):
        bsz, tgt_len, _ = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        def reshape(t):
            return (t.view(bsz, -1, self.num_heads, self.head_dim)
                     .transpose(1, 2).contiguous()
                     .view(bsz * self.num_heads, -1, self.head_dim))

        q = reshape(q) * self.scaling
        k, v = reshape(k), reshape(v)
        src_len = k.size(1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))         # Q_i K_jᵀ / √d
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            attn = F.softmax(attn_weights + attn_bias, dim=-1)  # pair -> atom
            attn = F.dropout(attn, p=self.dropout, training=self.training)
        else:
            attn_weights = attn_weights + attn_bias             # updated logits = new pair repr
            attn = F.dropout(F.softmax(attn_weights, dim=-1),
                             p=self.dropout, training=self.training)

        o = torch.bmm(attn, v)
        o = (o.view(bsz, self.num_heads, tgt_len, self.head_dim)
              .transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim))
        o = self.out_proj(o)
        if not return_attn:
            return o
        return o, attn_weights, attn
```

Pre-LN block and the pair-tracking encoder:

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, attention_heads,
                 dropout=0.1, attention_dropout=0.1, activation_dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.self_attn = SelfMultiheadAttention(embed_dim, attention_heads, dropout=attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, padding_mask=None, attn_bias=None, return_attn=False):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(query=x, key_padding_mask=padding_mask,
                           attn_bias=attn_bias, return_attn=return_attn)
        if return_attn:
            x, attn_weights, _ = x
        x = residual + F.dropout(x, p=self.dropout, training=self.training)

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = residual + F.dropout(x, p=self.dropout, training=self.training)
        if not return_attn:
            return x
        return x, attn_weights, None


class TransformerEncoderWithPair(nn.Module):
    def __init__(self, encoder_layers, embed_dim, ffn_embed_dim, attention_heads,
                 emb_dropout=0.1, dropout=0.1, attention_dropout=0.1, activation_dropout=0.0):
        super().__init__()
        self.emb_dropout = emb_dropout
        self.emb_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.final_head_layer_norm = nn.LayerNorm(attention_heads)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, ffn_embed_dim, attention_heads,
                                    dropout, attention_dropout, activation_dropout)
            for _ in range(encoder_layers)
        ])

    def forward(self, emb, attn_mask, padding_mask=None):
        bsz, seq_len = emb.size(0), emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        input_attn_mask = attn_mask                            # q^0
        input_padding_mask = padding_mask

        def fill_attn_mask(am, pm, fill_val=float("-inf")):
            if am is not None and pm is not None:
                am = am.view(bsz, -1, seq_len, seq_len)
                am.masked_fill_(pm.unsqueeze(1).unsqueeze(2).to(torch.bool), fill_val)
                am = am.view(-1, seq_len, seq_len)
                pm = None
            return am, pm

        def norm_loss(t, eps=1e-10, tolerance=1.0):
            t = t.float()
            max_norm = t.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(t ** 2, dim=-1) + eps)
            return torch.relu((norm - max_norm).abs() - tolerance)

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))).mean()

        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)
        for layer in self.layers:
            x, attn_mask, _ = layer(x, padding_mask=padding_mask,
                                    attn_bias=attn_mask, return_attn=True)   # q^{l+1}

        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)

        x = self.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask          # q^L - q^0
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        delta_pair_repr = delta_pair_repr.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = masked_mean(pair_mask, norm_loss(delta_pair_repr), dim=(-1, -2))
        delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)
        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm
```

The model and [CLS] readout for property prediction:

```python
class NonLinearHead(nn.Module):
    def __init__(self, input_dim, out_dim, activation_fn=F.gelu, hidden=None):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = activation_fn

    def forward(self, x):
        return self.linear2(self.activation_fn(self.linear1(x)))


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout=0.0,
                 activation_fn=torch.tanh):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features):
        x = features[:, 0, :]                                  # [CLS] atom
        x = self.dropout(x)
        x = self.activation_fn(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class MoleculeModel(nn.Module):
    def __init__(self, num_atom_types, num_tasks, embed_dim=512, ffn_embed_dim=2048,
                 attention_heads=64, encoder_layers=15, K=128):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_atom_types, embed_dim, padding_idx=0)
        n_edge_type = num_atom_types * num_atom_types          # pair type = (type_i, type_j)
        self.gbf = GaussianLayer(K, n_edge_type)
        self.gbf_proj = NonLinearHead(K, attention_heads)
        self.encoder = TransformerEncoderWithPair(
            encoder_layers, embed_dim, ffn_embed_dim, attention_heads)
        self.cls_head = ClassificationHead(embed_dim, embed_dim, num_tasks)
        self.pair2coord_proj = NonLinearHead(attention_heads, 1)  # used for 3D-output tasks

    def forward(self, tokens, dist, edge_type, padding_mask=None):
        seq_len = tokens.size(1)
        x = self.embed_tokens(tokens)                          # atom-type embedding

        gbf_feature = self.gbf(dist, edge_type)                # pair-type Gaussian kernel
        attn_bias = self.gbf_proj(gbf_feature)                 # per-head pair bias = q^0
        attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous().view(-1, seq_len, seq_len)

        encoder_rep, pair_rep, delta_pair_repr, x_norm, delta_pair_norm = self.encoder(
            x, attn_mask=attn_bias, padding_mask=padding_mask)
        return self.cls_head(encoder_rep)                      # [B, num_tasks]

    def coord_head(self, coords, delta_pair_repr, padding_mask=None):
        atom_num = coords.shape[1] if padding_mask is None else (
            1 - padding_mask.type_as(coords)
        ).sum(dim=1).view(-1, 1, 1, 1)
        # relative vectors x_j - x_i (canonical layout; the EGNN-form x_i - x_j differs only by
        # a sign, which the learned scalar c_ij absorbs, so both are equivariant and equivalent)
        delta_pos = coords.unsqueeze(1) - coords.unsqueeze(2)
        c_ij = self.pair2coord_proj(delta_pair_repr)            # invariant scalar per pair
        coord_update = delta_pos / atom_num * c_ij              # size-normalized relative update
        if padding_mask is not None:
            pair_mask = (1 - padding_mask.float()).unsqueeze(-1) * (
                1 - padding_mask.float()
            ).unsqueeze(1)
            coord_update = coord_update * pair_mask.unsqueeze(-1)
        return coords + coord_update.sum(dim=2)
```
