# Uni-Mol

Uni-Mol is a universal 3D molecular representation framework built on an SE(3)-invariant Transformer
that takes atom types and 3D coordinates directly as input. Its core move is to inject geometry into a
fully-connected Transformer as a **continuous, rotation/translation-invariant per-pair attention bias**,
derived from pairwise Euclidean distances through a **pair-type-aware Gaussian kernel**, and to maintain
a **pair representation** that the attention both reads (pair → atom bias) and writes (atom → pair via
the query-key product). An SE(3)-equivariant coordinate head lets the same backbone *output* 3D
positions, which powers a 3D-denoising pretraining task. Pretrained on ~209M conformations, it is
finetuned for molecular property prediction by reading out a [CLS] atom through a small prediction head.

## Problem it solves

Predict chemical properties from molecular structure, generalising across scaffold-split tasks with
scarce labels, by *using 3D geometry as a first-class input* — unlike 1D/2D encoders that discard it,
and unlike prior 3D-aware methods that use geometry only as an auxiliary contrastive/transfer signal.
The encoder must be SE(3)-invariant, capture long-range (not just bonded) interactions, and be cheap
enough to pretrain at the scale of hundreds of millions of conformers.

## Key ideas

1. **Distance, not coordinates, for invariance.** The Euclidean distance `d_ij = ‖x_i − x_j‖` is the
   invariant of a point pair under any rigid motion. Building the geometric encoding purely from
   pairwise distances makes it SE(3)-invariant for free, avoiding expensive equivariant-tensor networks.

2. **Pair-type-aware Gaussian kernel.** Encode each distance with a learnable `K`-channel Gaussian
   basis, after a per-pair-type affine on the distance:
   - `A(d_ij, t_ij) = a_{t_ij}·d_ij + b_{t_ij}` (pair type `t_ij` = the *atom-type* pair, not the bond)
   - `p_ij^k = G(A(d_ij,t_ij), μ^k, σ^k)`, `G(d,μ,σ) = (1/(σ√(2π)))·exp(−(d−μ)²/(2σ²))`, `k=1..K`.
   A bare distance kernel encodes a C–C and a C–O pair at the same distance identically; the per-type
   affine shifts/scales the distance axis per pair type so they land in different regions of the basis.
   Gaussian (not binned) because it is smooth and differentiable in `d` — required to backprop into
   coordinates for the 3D head.

3. **Pair representation with two-way communication.** Project the `K` Gaussian channels to `H` heads
   to get the initial pair representation `q^0`, used as the attention bias (pair → atom):
   `Attention = softmax(Q_i K_jᵀ/√d + q_ij) V_j`. Each layer adds the per-head `Q_i K_jᵀ/√d` back into
   the pair representation (atom → pair): `q_ij^{l+1} = q_ij^l + {Q_i^{l,h}(K_j^{l,h})ᵀ/√d}`. Residual
   accumulation means `q^L − q^0` is the *learned correction* to the raw geometry.

4. **SE(3)-equivariant coordinate head** (for 3D-output tasks, including pretraining; not used for
   property prediction):
   `x̂_i = x_i + (1/n) Σ_j (x_i − x_j)·c_ij`, `c_ij = ReLU((q_ij^L − q_ij^0)U)W`, `U∈ℝ^{H×H}`, `W∈ℝ^{H×1}`.
   Equivariant because it is a sum of relative vectors `(x_i−x_j)` with invariant scalar weights; uses
   the *delta* pair representation (the learned part) and runs once in the last layer for efficiency;
   `1/n` keeps the update scale independent of molecule size.

5. **Pre-LN Transformer plus norm control** for million-step mixed-precision pretraining. Pre-LN keeps
   residual gradients well behaved, while small atom-state and delta-pair-state norm penalties prevent
   the final LayerNorm from hiding exploding pre-normalized states. **[CLS] atom** (placed at the
   centroid) is the molecule summary; **3D position recovery** corrupts coordinates with uniform noise
   (`r ≈ 1 Å`, no re-assignment) rather than masking, since continuous coordinates have no special
   [MASK] value, paired with masked-atom-type prediction.

## Architecture (base)

15 encoder layers, embedding dim 512, FFN dim 2048, 64 attention heads, `K = 128` Gaussian channels,
GELU. Pretraining: Adam `(β1,β2)=(0.9,0.99)`, `ε=1e-6`, peak LR 1e-4, 1M steps, 10K warmup, batch 128,
weight decay 1e-4, gradient clip 1.0, on ~209M conformations. Pretraining loss =
`CE(atom) + 5·SmoothL1(coord) + 10·SmoothL1(dist)`, plus small norm penalties when enabled
(`L_norm = mean max(| ||s|| − √d | − 1, 0)`, weight 0.01) to stabilize atom and delta-pair states.
Finetuning uses multiple generated conformers as augmentation and averages predictions over conformers
at test time.

## Working code

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
