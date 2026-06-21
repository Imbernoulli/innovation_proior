The D-MPNN numbers name the next move exactly. BBBP moved off chance — from GIN's **0.5099** to **0.5986** — so the diagnosis was right in direction: the global RDKit descriptor prior pulled the single-target task whose answer is global physicochemistry up off the coin flip. But $0.60$ is a weak win, not the decisive one, reading as the global prior arriving only partially with the directed-bond encoder carrying most of the load. Worse, the other two rows *fell*: BACE from $0.7261$ to **0.6881**, Tox21 from $0.7470$ to **0.7131**. The directed-bond fix and the sum readout bought nothing on the tasks where the 2D graph was already doing fine — they slightly hurt. That is the tell. Two rungs spent entirely inside the 2D bond graph, and the 2D bond graph has a ceiling: BACE and Tox21 do not move up no matter how I rearrange message passing, and BBBP creeps but does not leap. The information the 2D graph throws away is the *geometry* — and that is not a thing I can recover by tweaking aggregation, because it was never in the input.

Every one of these properties is decided by how the atoms sit in *space*: which groups are close enough to touch, the shape that fits or does not fit a binding site. A 2D bond graph discards that before the encoder ever sees it. Two atoms can be many bonds apart yet sit right next to each other in folded 3D space and interact strongly — a hydrogen bond across a ring, two aromatic groups stacking — and a message-passing GNN that only talks along bonds *structurally cannot* see that contact. That is the real ceiling under BACE and Tox21. And there is a second wall: data. These sets are a few hundred to a few thousand molecules, and an encoder learning its whole representation from scratch overfits the artifacts of a tiny training set. The fix for that is the recipe that made BERT and ViT work — pretrain a representation on an essentially unlimited pile of unlabeled molecules, then finetune a light head. The scaffold even hands me a pretrained checkpoint inside the container. Both walls — geometry and data-hunger — fall to the same move.

I propose **Uni-Mol**, an SE(3)-invariant 3D Transformer loaded from a pretrained checkpoint. The first instinct is a 3D graph net, but locality is exactly wrong: I want *every* atom able to look at *every* other atom, because the interactions that decide the property are long-range in bond-distance. That is what self-attention gives for free — a Transformer treats its input as a fully-connected set, every atom token attending to every other, and a set is the right object since atoms have no canonical order, attention is permutation-equivariant over them, and a `[CLS]` readout gives an order-invariant molecule vector. The backbone is therefore a Transformer, not a local GNN, which flips the very thing that capped the 2D rungs (bonded-only message passing) into full connectivity. But a Transformer is position-blind: $\mathrm{softmax}(QK^\top/\sqrt d)\,V$ depends only on the set of token features, not on where any atom is. In language one adds a positional encoding indexed by a *discrete* integer slot; my position is a *continuous* 3D coordinate, and one I cannot use raw — rotate or translate the whole molecule, which changes nothing about its chemistry, and every absolute coordinate changes, so the model would waste capacity learning that a molecule and its rotated copy are the same. I need to inject 3D position in a way that is continuous *and* invariant to global rotation and translation (SE(3)).

The clean way to inject structure into a Transformer without breaking full connectivity is the Graphormer move: do not touch the architecture, add a learnable per-pair bias to the attention logits, $A_{ij} = Q_iK_j^\top/\sqrt d + b_{\phi(i,j)}$. The slot is right — geometry enters as a number added to $QK^\top$ before the softmax — but Graphormer's bias is keyed by shortest-path distance and discrete bond features, all 2D topology, with nowhere to put a continuous Euclidean distance and nothing rotation-invariant in it. So the question sharpens: what continuous, SE(3)-invariant per-pair quantity goes in that slot? Anything involving an *absolute* coordinate moves under a rigid motion; what survives is the *relative* geometry, and the simplest invariant of a pair of points is the Euclidean distance $d_{ij} = \|x_i - x_j\|$. Build the per-pair bias purely out of pairwise distances and I get SE(3)-invariance *for free*, with none of the heavy equivariant-tensor machinery (tensor field networks, SE(3)-Transformers) that would be the wrong price at pretraining scale — by choosing the *invariant* of the coordinate rather than being equivariant to it, the cost stays close to a vanilla Transformer.

What function of $d_{ij}$? I want a smooth, expressive featurization, so I evaluate the distance against a bank of $K$ learnable Gaussians with means $\mu^k$ and widths $\sigma^k$: the vector $(G(d_{ij},\mu^1,\sigma^1),\dots,G(d_{ij},\mu^K,\sigma^K))$ is a soft one-hot of where on the distance axis the pair lives. Gaussians over bins because the encoding must be differentiable in $d$ (so coordinates can be a pretraining target) and free of bin-boundary artifacts. A bare distance kernel would encode a C–C pair at 1.5 Å and a C–O pair at 1.5 Å *identically* — wrong, the chemistry differs — so I make the kernel pair-type-aware: apply a per-pair-type affine $a_t\, d + b_t$ to the distance *before* the Gaussian basis, with $a_t, b_t$ learnable scalars indexed by the unordered pair of element types (from atom identities, since I am living without the 2D bond). The same number of Ångströms then lands at a different position on the shared ruler for different pair types. The $K$ Gaussian channels project down to one bias per head with a small MLP, added into the attention logits in Graphormer's slot — continuous and SE(3)-invariant now. This is the pair-to-atom communication: geometry steers which atoms each atom attends to.

The pair feature should not stay frozen. The distance is fixed by the conformer, but the *meaning* of a pair's interaction is what the model figures out as it processes the molecule, and that figuring-out lives in the atom representations while the pair channel sits at its initial geometric value. There is a per-pair, per-layer quantity that already captures "how strongly do these two atoms interact right now": the query-key product $Q_iK_j^\top$. So I let the atoms write back into the pairs — add the current per-head $Q_iK_j^\top$ into the pair representation each layer, residually. Now the two channels genuinely talk: the pair feature biases attention (pair→atom) and the attention's affinity refines the pair feature (atom→pair), layer after layer, and the residual accumulation makes $q^L - q^0$ the *learned correction* to the raw geometry, at small cost since the QK product is already computed. Optimization stability is not a detail at this scale — I am finetuning a 15-layer, 512-dim, 64-head, $\sim$86M-parameter encoder pretrained over a very long schedule — so each block is Pre-LN: normalize, self-attend with the pair bias, add; normalize, GELU feed-forward, add. Post-LN gives large gradients near the output at init and forces fragile warmup; Pre-LN puts the norm inside the residual branch and is well-behaved at init.

Made concrete in this edit surface, which exposes only the `MoleculeModel` class, I do *not* build the pretraining loop, the masked-atom / noised-coordinate recovery objective, or the equivariant coordinate head — those produced the checkpoint I am loading; for property prediction I only consume the encoder and read out a molecule vector. The fill is: a token embedding sized to the Uni-Mol dictionary (map each atom's element to its dictionary index from the 136-dim feature vector, prepend `[CLS]`, append `[SEP]`, pad with `[PAD]`); extend the distance matrix and mask for those two special tokens; form edge types as $\text{token}_i \cdot \text{dict\_size} + \text{token}_j$; run the edge-type Gaussian layer and project to per-head attention bias; embed tokens; run the Pre-LN encoder-with-pair; and read out `[CLS]` through a classification head. One faithful-but-thinner choice: the head is a simple dropout-then-linear on the `[CLS]` representation, not a dense+tanh pooler, because at finetune the heavy pretrained trunk does the work and the simplest head transfers most cleanly. The load step is the whole point — read the checkpoint, remap the fairseq/unicore key names to this flat layout, copy every key whose shape matches (`embed_tokens`, the Gaussian layer, all 15 encoder layers), skip the pretraining-only heads (LM head, distance head, coordinate projection) and any mismatch, and fall back to from-scratch only if the checkpoint is absent. The pretrained weights are how the encoder escapes the data-hunger that capped the 2D rungs.

So the delta from rung two is total, not incremental: where D-MPNN passed messages along bonds in 2D and bolted on a global descriptor prior, I feed the model atoms-in-space, attend fully-connected with an SE(3)-invariant pair-type Gaussian distance bias, let geometry and attention co-refine through a Pre-LN encoder, read out `[CLS]`, and start from weights pretrained on hundreds of millions of conformers. The falsifiable claims follow from the 2D rungs' shape: BACE and Tox21 both *fell* from GIN to D-MPNN because 2D message passing had hit its ceiling, so if geometry plus pretraining is the real lever, both should now *exceed* even GIN's marks — BACE clearly above $0.726$, Tox21 above $0.747$ — not merely recover D-MPNN's. BBBP is the harder claim: the descriptor prior got it to $0.599$, and 3D geometry plus a pretrained trunk should push it higher still, though its severe scaffold shift on a single binary target means I expect the *smallest* absolute headroom there. If instead BACE and Tox21 fail to clear GIN's marks, my story — that the 2D *input*, not the 2D *architecture*, was the ceiling — is wrong.

```python
# =====================================================================
# EDITABLE SECTION START — Uni-Mol: SE(3)-Invariant Molecular Transformer
# =====================================================================

import os as _os
import logging as _logging

_logger = _logging.getLogger(__name__)

# --------------- Uni-Mol dictionary (mirrors dict.txt) ----------------
# The pretrained model uses a token vocabulary.  We map atomic numbers
# from the featurisation to dictionary indices so that we can re-use the
# pretrained ``embed_tokens`` embedding and edge-type Gaussian layer.
# dict.txt ordering:
#   [PAD]=0, [CLS]=1, [SEP]=2, [UNK]=3, C=4, N=5, O=6, S=7, H=8,
#   Cl=9, F=10, Br=11, I=12, Si=13, P=14, B=15, Na=16, K=17, Al=18,
#   Ca=19, Sn=20, As=21, Hg=22, Fe=23, Zn=24, Cr=25, Se=26, Gd=27,
#   Au=28, Li=29
_ELEM_TO_DICT_IDX = {
    6: 4, 7: 5, 8: 6, 16: 7, 1: 8, 17: 9, 9: 10, 35: 11,
    53: 12, 14: 13, 15: 14, 5: 15, 11: 16, 19: 17, 13: 18,
    20: 19, 50: 20, 33: 21, 80: 22, 26: 23, 30: 24, 24: 25,
    34: 26, 64: 27, 79: 28, 3: 29,
}
_DICT_SIZE = 31  # 30 atoms + [MASK] token to match pretrained checkpoint
_PAD_IDX = 0
_CLS_IDX = 1
_SEP_IDX = 2
_UNK_IDX = 3


def _atomic_num_from_features(atom_feat):
    """Extract atomic number from the one-hot atom feature vector.
    The first 118 elements encode atomic_num (1..118).
    """
    idx = atom_feat[:118].argmax().item()
    if atom_feat[idx].item() < 0.5:
        return 0
    return idx + 1


def _atoms_to_tokens(atom_features_dense, mask):
    """Convert dense atom features [B, N, D] + mask [B, N] to token ids [B, N+2].
    Prepends [CLS] and appends [SEP], matching the reference data pipeline.
    Padding positions get PAD token.
    """
    B, N, D = atom_features_dense.shape
    device = atom_features_dense.device

    tokens = torch.full((B, N), _PAD_IDX, dtype=torch.long, device=device)
    for b in range(B):
        for i in range(N):
            if mask[b, i].item() > 0.5:
                anum = _atomic_num_from_features(atom_features_dense[b, i])
                tokens[b, i] = _ELEM_TO_DICT_IDX.get(anum, _UNK_IDX)

    cls_col = torch.full((B, 1), _CLS_IDX, dtype=torch.long, device=device)
    sep_col = torch.full((B, 1), _SEP_IDX, dtype=torch.long, device=device)
    tokens = torch.cat([cls_col, tokens, sep_col], dim=1)  # [B, N+2]
    return tokens


def _extend_dist_and_mask(dist_matrix, mask):
    """Extend distance matrix and mask for [CLS] and [SEP] tokens.
    Returns dist [B, N+2, N+2] and padding_mask [B, N+2] (True=pad).
    """
    B, N, _ = dist_matrix.shape
    device = dist_matrix.device
    dist = torch.zeros(B, N + 2, N + 2, device=device, dtype=dist_matrix.dtype)
    dist[:, 1:N+1, 1:N+1] = dist_matrix

    ext_mask = torch.zeros(B, N + 2, device=device, dtype=mask.dtype)
    ext_mask[:, 0] = 1.0    # CLS always valid
    ext_mask[:, 1:N+1] = mask
    ext_mask[:, N+1] = 1.0  # SEP always valid

    padding_mask = (ext_mask < 0.5)  # True = padded
    return dist, padding_mask


# -------------------- Gaussian Distance Encoding ---------------------

@torch.jit.script
def _gaussian(x, mean, std):
    pi = 3.14159
    a = (2.0 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    """Edge-type-dependent Gaussian distance encoding.
    Each atom-pair type (i,j) has its own learned scaling (mul) and bias.
    This is critical: it lets the model distinguish C-C, C-N, C-O etc.
    """
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return _gaussian(x.float(), mean, std).type_as(self.means.weight)


# -------------------- Helper Heads -----------------------------------

class NonLinearHead(nn.Module):
    """Two-layer MLP with GELU activation for projecting Gaussian features."""
    def __init__(self, input_dim, out_dim, hidden=None):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


class ClassificationHead(nn.Module):
    """Head for molecule-level prediction via [CLS] token.
    Reference Uni-Mol uses simple dropout + linear (no hidden dense/tanh).
    """
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        x = features[:, 0, :]  # [CLS] token
        x = self.dropout(x)
        return self.out_proj(x)


# -------------------- Transformer Encoder With Pair -------------------

class SelfMultiheadAttention(nn.Module):
    """Multi-head self-attention with fused QKV projection."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key_padding_mask=None, attn_bias=None,
                return_attn=False):
        bsz, tgt_len, _ = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        def reshape(t):
            return t.view(bsz, -1, self.num_heads, self.head_dim) \
                    .transpose(1, 2).contiguous() \
                    .view(bsz * self.num_heads, -1, self.head_dim)

        q = reshape(q) * self.scaling
        k = reshape(k)
        v = reshape(v)
        src_len = k.size(1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            aw = attn_weights
            if attn_bias is not None:
                aw = aw + attn_bias
            attn = F.dropout(F.softmax(aw, dim=-1),
                             p=self.dropout, training=self.training)
        else:
            attn_weights = attn_weights + attn_bias
            attn = F.dropout(F.softmax(attn_weights.clone(), dim=-1),
                             p=self.dropout, training=self.training)

        o = torch.bmm(attn, v)
        o = o.view(bsz, self.num_heads, tgt_len, self.head_dim) \
             .transpose(1, 2).contiguous() \
             .view(bsz, tgt_len, self.embed_dim)
        o = self.out_proj(o)
        if not return_attn:
            return o
        return o, attn_weights, attn


class UniMolEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer (matches reference)."""
    def __init__(self, embed_dim, ffn_embed_dim, attention_heads,
                 dropout=0.1, attention_dropout=0.1, activation_dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.self_attn = SelfMultiheadAttention(
            embed_dim, attention_heads, dropout=attention_dropout)
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
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        if not return_attn:
            return x
        return x, attn_weights, attn_probs


class TransformerEncoderWithPair(nn.Module):
    """Transformer encoder that tracks and updates pair (attention bias)
    representations through layers — a key Uni-Mol architectural feature.
    """
    def __init__(self, encoder_layers, embed_dim, ffn_embed_dim,
                 attention_heads, emb_dropout=0.1, dropout=0.1,
                 attention_dropout=0.1, activation_dropout=0.0):
        super().__init__()
        self.emb_dropout = emb_dropout
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.final_head_layer_norm = nn.LayerNorm(attention_heads)
        self.layers = nn.ModuleList([
            UniMolEncoderLayer(
                embed_dim=embed_dim, ffn_embed_dim=ffn_embed_dim,
                attention_heads=attention_heads, dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
            )
            for _ in range(encoder_layers)
        ])

    def forward(self, emb, attn_mask=None, padding_mask=None):
        bsz, seq_len = emb.size(0), emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        input_attn_mask = attn_mask
        input_padding_mask = padding_mask

        def fill_attn_mask(am, pm, fill_val=float("-inf")):
            if am is not None and pm is not None:
                am = am.view(bsz, -1, seq_len, seq_len)
                am.masked_fill_(
                    pm.unsqueeze(1).unsqueeze(2).to(torch.bool), fill_val)
                am = am.view(-1, seq_len, seq_len)
                pm = None
            return am, pm

        assert attn_mask is not None
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)

        for layer in self.layers:
            x, attn_mask, _ = layer(
                x, padding_mask=padding_mask,
                attn_bias=attn_mask, return_attn=True)

        x = self.final_layer_norm(x)

        # Compute pair representations
        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(
            delta_pair_repr, input_padding_mask, 0)
        attn_mask = attn_mask.view(
            bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        delta_pair_repr = delta_pair_repr.view(
            bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, attn_mask, delta_pair_repr


# -------------------- Main Model ------------------------------------

class MoleculeModel(nn.Module):
    """Uni-Mol: SE(3)-invariant Transformer with pretrained weights.

    Architecture: 15 layers, 512 hidden dim, 64 attention heads, 2048 FFN
    dim (~86M parameters).  Uses edge-type-dependent Gaussian distance
    encoding and pair representation tracking through the encoder layers.
    Loads pretrained weights from the checkpoint at build time.
    """

    def __init__(self, atom_dim: int, edge_dim: int, num_tasks: int,
                 task_type: str):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = task_type

        # Architecture (matches reference base_architecture)
        embed_dim = 512
        ffn_embed_dim = 2048
        attention_heads = 64
        encoder_layers = 15
        dropout = 0.1
        attention_dropout = 0.1
        K = 128
        n_edge_type = _DICT_SIZE * _DICT_SIZE  # 961

        self.embed_dim = embed_dim
        self.attention_heads = attention_heads

        # Token embedding (will be loaded from pretrained)
        self.embed_tokens = nn.Embedding(
            _DICT_SIZE, embed_dim, padding_idx=_PAD_IDX)

        # Edge-type dependent Gaussian distance encoding
        self.gbf = GaussianLayer(K, n_edge_type)
        self.gbf_proj = NonLinearHead(K, attention_heads)

        # Transformer encoder with pair tracking
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=encoder_layers,
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            attention_heads=attention_heads,
            emb_dropout=dropout,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=0.0,
        )

        # Classification / regression head.  `pooler_dropout` may be set as a
        # class attribute by the training driver to match reference per-dataset
        # settings (e.g. FreeSolv/ESOL use 0.2 per Uni-Mol README).
        pooler_dropout = getattr(type(self), "pooler_dropout", 0.0)
        self.cls_head = ClassificationHead(
            input_dim=embed_dim,
            inner_dim=embed_dim,
            num_classes=num_tasks,
            pooler_dropout=pooler_dropout,
        )

        # Initialise, then load pretrained weights
        self.apply(self._init_weights)
        self._load_pretrained()

    @staticmethod
    def _init_weights(module):
        """BERT-style weight initialisation."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _load_pretrained(self):
        """Load pretrained encoder + GBF weights from checkpoint."""
        ckpt_path = "/data/unimol_weights/mol_pre_all_h_220816.pt"
        if not _os.path.exists(ckpt_path):
            _logger.warning(
                "Pretrained weights not found at %s — training from scratch",
                ckpt_path)
            return

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)

        # Remap checkpoint keys from fairseq/unicore format to our flat format
        remapped = {}
        for key, val in state.items():
            if any(s in key for s in ["classification_head", "lm_head",
                                       "dist_head", "pair2coord_proj"]):
                continue
            new_key = key
            new_key = new_key.replace("encoder.sentence_encoder.", "encoder.")
            new_key = new_key.replace("encoder.gbf", "gbf")
            new_key = new_key.replace("encoder.embed_tokens", "embed_tokens")
            remapped[new_key] = val

        own_state = self.state_dict()
        loaded_keys, skipped_shape, skipped_missing = [], [], []

        for key, val in remapped.items():
            if key in own_state:
                if own_state[key].shape == val.shape:
                    own_state[key].copy_(val)
                    loaded_keys.append(key)
                else:
                    skipped_shape.append(f"  {key}: ckpt={list(val.shape)} model={list(own_state[key].shape)}")
            else:
                skipped_missing.append(key)

        self.load_state_dict(own_state, strict=False)
        print(f"[Checkpoint] Loaded {len(loaded_keys)} keys successfully")
        if skipped_shape:
            print(f"[Checkpoint] Shape mismatch ({len(skipped_shape)} keys):")
            for s in skipped_shape:
                print(s)
        if skipped_missing:
            print(f"[Checkpoint] Missing in model ({len(skipped_missing)} keys):")
            for k in skipped_missing[:10]:
                print(f"  {k}")
            if len(skipped_missing) > 10:
                print(f"  ... and {len(skipped_missing)-10} more")
        # Also show model keys NOT in checkpoint
        not_loaded = [k for k in own_state if k not in set(loaded_keys)]
        if not_loaded:
            print(f"[Checkpoint] Model keys NOT loaded ({len(not_loaded)}):")
            for k in not_loaded[:10]:
                print(f"  {k}")
            if len(not_loaded) > 10:
                print(f"  ... and {len(not_loaded)-10} more")

    def forward(self, batch):
        """Forward pass using dense batch format.

        Args:
            batch: MolBatch with atom_features [B, N, D],
                   dist_matrix [B, N, N], mask [B, N].
        Returns:
            predictions: [B, num_tasks]
        """
        B, N, _ = batch.atom_features.shape
        device = batch.atom_features.device

        # Map atom features to dictionary tokens
        tokens = _atoms_to_tokens(batch.atom_features, batch.mask)  # [B, N+2]

        # Extend distance matrix / mask for [CLS] and [SEP]
        dist, padding_mask = _extend_dist_and_mask(
            batch.dist_matrix, batch.mask)
        seq_len = N + 2

        # Edge types: token_i * dict_size + token_j
        edge_type = tokens.unsqueeze(1) * _DICT_SIZE + tokens.unsqueeze(2)

        # Gaussian features with edge-type-dependent scaling
        gbf_feature = self.gbf(dist, edge_type)         # [B, S, S, K]
        attn_bias = self.gbf_proj(gbf_feature)           # [B, S, S, H]
        attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous()  # [B, H, S, S]
        attn_bias = attn_bias.view(-1, seq_len, seq_len)  # [B*H, S, S]

        # Token embeddings
        x = self.embed_tokens(tokens)  # [B, S, D]

        # Run encoder (with pair representation tracking)
        encoder_rep, pair_rep, delta_pair_repr = self.encoder(
            x, attn_mask=attn_bias,
            padding_mask=padding_mask if padding_mask.any() else None,
        )

        # Predict via [CLS] token
        return self.cls_head(encoder_rep)  # [B, num_tasks]


# =====================================================================
# EDITABLE SECTION END
# =====================================================================
```
