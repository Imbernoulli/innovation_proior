Let me start from what actually fails when I try to predict a molecule's properties. I have a few thousand labelled molecules per task and an essentially unlimited pile of unlabelled ones, so the move is obvious — pretrain a representation on the unlabelled pile, finetune a light head on the labels — exactly the recipe that made BERT and ViT work. The thing that keeps nagging me is what I feed the encoder. Every property I care about — whether a molecule crosses the blood-brain barrier, inhibits an enzyme, is toxic in an assay — is decided by how the atoms actually sit in space: which groups are close enough to touch, the shape that fits or doesn't fit a binding site. That is the oldest fact in the field. And yet the encoders everyone uses eat a SMILES string or a 2D bond graph, both of which have thrown the geometry away before the model ever sees it. So the real question isn't "how do I pretrain a molecular encoder," it's "how do I build an encoder whose input *is* the 3D structure — atoms with coordinates — and learn from it at the scale of hundreds of millions of conformers."

The first instinct is to reach for a graph network, since molecules are graphs. But I keep running into the same wall: a message-passing GNN, for efficiency, connects each atom only to its bonded or radius-cutoff neighbours, and that locality is exactly wrong here. Two atoms can be many bonds apart yet sit right next to each other in folded 3D space and interact strongly — a hydrogen bond across a ring, two aromatic groups stacking. If the model can only pass messages along bonds it structurally cannot see that interaction. I want every atom able to look at every other atom. That is precisely what self-attention gives me for free: a Transformer treats its input as a fully-connected set, every token attending to every token. And a set is the right object for atoms anyway — atoms have no canonical order, self-attention is permutation-equivariant over the atom tokens, and a [CLS] readout can give me an order-invariant molecule representation. Good. The backbone should be a Transformer, not a local GNN.

But a Transformer is position-blind. `softmax(QKᵀ/√d)V` is a function of the set of token *features* and nothing else — it has no idea where any atom is. In language you fix this by adding a positional encoding indexed by a *discrete* position, the integer slot in the sentence. My "position" is a *continuous* 3D coordinate, and it is also a coordinate I'm not allowed to use naively. If I just embed the raw `(x,y,z)` of each atom and add it to the token, then rotating or translating the whole molecule — which changes nothing about its chemistry — changes every input vector, and the model has to waste capacity learning that a molecule and its rotated copy are the same thing. So I need to inject 3D position in a way that is invariant to global rotation and translation. Two constraints now: continuous, and SE(3)-invariant.

I do know how people inject *structural* position into a plain Transformer without breaking full connectivity, because Graphormer showed it. Their trick is lovely in its restraint: don't touch the architecture, just add a learnable bias to the attention logits, indexed by a structural function of the pair. For a node pair `(i,j)`,

  A_ij = (h_i W_Q)(h_j W_K)ᵀ/√d + b_{φ(i,j)} + c_ij,

where `b` is a learnable scalar keyed by the shortest-path distance between `i` and `j` in the graph, and `c_ij` aggregates discrete bond features along that path. So the geometry enters as a per-pair number added to `QKᵀ` before the softmax, and the model can learn, say, to attend more to nearby atoms by making `b` decrease with distance. This is the structural idea I want to keep: keep the fully-connected Transformer, route the prior through a per-pair attention bias. But look at what Graphormer's bias actually *is* — shortest-path distance, discrete bond features. Every quantity is a function of the 2D topology. There is nowhere in it to put a continuous Euclidean distance, and nothing in it is rotation-invariant, because there are no coordinates anywhere in the picture. The slot is right; the contents are 2D. So the question sharpens to: what continuous, SE(3)-invariant per-pair quantity do I put in that bias slot?

I stare at the coordinates. I have `x_i ∈ ℝ³` for each atom. Anything I compute that involves an atom's absolute coordinate will move when I rotate or translate the molecule. So I should never feed an absolute coordinate. What survives a global rotation and translation? The relative geometry. And the simplest invariant of a pair of points — the one quantity about `(x_i, x_j)` that is completely unchanged by any rigid motion of the whole molecule — is the Euclidean distance `d_ij = ‖x_i − x_j‖`. I want to be sure of this before I build the whole encoder on it, so let me actually check it rather than wave at it. Take a 3-atom molecule at `x_1=(0,0,0)`, `x_2=(1.2,0,0)`, `x_3=(0.4,0.9,0)`; its distance matrix has `d_12=1.2`, `d_13=0.985`, `d_23=1.204`. Now apply a genuine rigid motion — a random proper rotation `R` (I check `det R = 1` and `RRᵀ = I`) and a translation `t=(3.1,−2.0,5.5)`, so `x_i ↦ Rx_i + t`. Recompute the distance matrix on the moved coordinates: I get `1.2`, `0.985`, `1.204` again, and the entrywise gap `max|D − D_moved|` comes out at `4·10⁻¹⁶`, i.e. zero to machine precision. As a control I also try the thing I'm forbidding — features built from an atom's *absolute* coordinate, `exp(−‖x_i‖²)` — and under the same motion those change by up to `1.0`, completely scrambled. So the distance really is the invariant and the absolute coordinate really isn't; building the per-pair bias purely out of pairwise distances inherits that invariance, with no equivariant tensor machinery at all. That means I don't need the heavy SE(3)-equivariant networks — the tensor field networks, the SE(3)-Transformers — that carry higher-order geometric tensors through every layer; that machinery is the wrong price to pay when I have to pretrain on hundreds of millions of conformers. By choosing the *invariant* of the coordinate — the distance — instead of trying to be equivariant to the coordinate, I get the symmetry I need while staying close to the cost profile of a vanilla Transformer.

So the pair bias is some function of `d_ij`. What function? `d_ij` is one scalar per pair, and I'm going to feed it through layers and eventually backprop into it, so I want a smooth, expressive featurisation of a scalar distance, not a raw number. The clean way to turn one scalar into a feature vector is to evaluate it against a bank of basis functions, and for a distance the natural basis is a set of Gaussians spread along the distance axis: I pick `K` Gaussians with means `μ^k` and widths `σ^k`, and represent `d_ij` by how strongly it activates each,

  p_ij^k = G(d_ij, μ^k, σ^k) = (1/(σ^k√(2π))) · exp(−(d_ij − μ^k)² / (2(σ^k)²)),   k = 1..K.

A Gaussian at mean `μ^k` lights up when the distance is near `μ^k`, so the vector `(p_ij^1, ..., p_ij^K)` is a soft one-hot of *where on the distance axis* this pair lives. I make `μ^k` and `σ^k` learnable so the model can place and size its basis functions where the data needs them. Why a Gaussian basis rather than the obvious alternative — chop the distance into bins and embed each bin? The reason that matters here is the gradient with respect to the distance, because at pretraining I'll perturb coordinates and ask the model to recover them, which means the loss has to flow back through `d_ij` into the positions. Let me put numbers on the difference. Take eight basis centres spaced over 0–3 Å and watch one channel centred near 1.3 Å as `d` varies. For the Gaussian, the numerical derivative `dG/dd` of that channel is `+0.98` at `d=0.6`, `−1.16` at `d=1.5`, `−0.14` at `d=2.4` — a real, nonzero slope everywhere, exactly the signal a coordinate gradient needs. For the binned version I assign `d` to a bin and one-hot it: at `d=1.49` the feature is `(…,0,1,0,…)` in bin 3, at `d=1.51` it is `(…,0,0,1,…)` in bin 4 — identical (zero gradient) almost everywhere, then a discontinuous jump at the edge. So binning hands the coordinate head a gradient of zero almost everywhere and an undefined spike at boundaries, which is useless for denoising, while the Gaussian gives a smooth slope throughout. That settles it in favour of the Gaussian basis.

Now I almost stop here, but something bothers me. A bare Gaussian kernel of `d_ij` says a carbon-carbon pair at 1.5 Å and a carbon-oxygen pair at 1.5 Å are encoded *identically*, because the encoding only knows the number 1.5. That's clearly throwing away information — the chemistry of a C–C contact and a C–O contact at the same distance is different, and the model should be allowed to treat them differently from the start. I want the distance encoding to be aware of *what kind of pair* it is. The pair type here should be cheap and available from exactly the inputs I already have: the two atoms' element types. Note I deliberately use the *atom-type pair*, not the bond — the bond is 2D-graph information I'm trying to live without, whereas "this is a (C, O) pair" needs only the atom identities, which I have alongside the coordinates. So let `t_ij` be the pair type, the unordered (or ordered) pair of element types, and let me make the kernel pair-type-aware by applying a per-type affine transform to the distance *before* the Gaussian basis:

  A(d_ij, t_ij) = a_{t_ij} · d_ij + b_{t_ij},   then   p_ij^k = G(A(d_ij, t_ij), μ^k, σ^k),

with `a_{t_ij}` and `b_{t_ij}` learnable scalars indexed by the pair type. The affine rescales and shifts the distance axis per pair type, so a C–C pair and a C–O pair at the same physical distance land at *different* positions in the shared Gaussian bank and therefore get different feature vectors. The model gets, per pair type, a learnable "where does this distance fall on my ruler," which is exactly the freedom to encode that the same number of Ångströms means different things for different pairs. So the spatial encoding of a pair is: per-type affine of the distance, then a learnable Gaussian basis of `K` channels.

That gives me a `K`-dimensional feature per pair. The attention bias I need is one scalar per head per pair, so I project the `K` Gaussian channels down to `H` (number of heads) with a small two-layer MLP — call it the gbf projection — and that gives me, for every pair `(i,j)` and head `h`, a bias value. I add it into the attention logits exactly in Graphormer's slot:

  Attention(Q_i, K_j, V_j) = softmax( Q_i K_jᵀ / √d  +  q_ij )  V_j,

where now `q_ij` is the continuous, SE(3)-invariant pair feature derived from the 3D distance, not a discrete shortest-path scalar. This is the pair-to-atom communication: the geometry, encoded per pair, steers which atoms each atom attends to. So far so good, and almost free — the only extra cost over a vanilla Transformer is computing the Gaussian features and one small projection, which is a rounding error next to the attention itself.

Let me now sit with the pair feature `q_ij` for a second, because I'm treating it as a static input bias and I think that's leaving value on the table. The distance is fixed by the conformer, yes, but the *meaning* of a pair's interaction is something the model is figuring out as it processes the molecule — and right now that figuring-out lives entirely in the atom representations, while the pair channel is frozen at its initial geometric value. The atoms talk to each other through attention; the pairs never get updated by what the atoms learn. That asymmetry feels wrong. If the pair channel could absorb the model's evolving view of each interaction, the bias in later layers would reflect a *learned* relationship, not just raw distance. What signal do I already compute, per pair, per layer, that captures "how strongly do these two atoms interact right now"? It's sitting right there: the query-key product `Q_i K_jᵀ`. That is, per head, exactly the model's current estimate of pair `(i,j)`'s affinity. So let me let the atoms write back into the pairs: at each layer, add the current per-head `Q_i K_jᵀ` into the pair representation,

  q_ij^{l+1} = q_ij^l + { Q_i^{l,h} (K_j^{l,h})ᵀ / √d  :  h = 1..H }.

This is the atom-to-pair communication, the mirror of the pair-to-atom bias. Now the two channels genuinely talk: the pair feature biases attention (pair → atom), and the attention's QK affinity refines the pair feature (atom → pair), layer after layer. And because it's a *residual* accumulation — I add to `q_ij`, I don't overwrite it — after `L` layers the pair representation is its initial geometric value plus the sum of all the per-layer QK corrections. That residual structure is going to matter in a moment: the quantity `q_ij^L − q_ij^0` is precisely the *learned correction* to the raw geometry, the part of the pair relationship the network discovered that the bare distance didn't say. The cost of all this stays small because I'm reusing the QK product attention already computes.

I should pin down the layer block itself before going further, because at the scale I'm targeting — a million pretraining steps over hundreds of millions of conformers — optimisation stability is not a detail, it decides whether the run finishes at all. The standard Transformer block can place its LayerNorm in two spots. Post-LN puts the norm *between* the residual blocks, after the addition; the trouble is that at initialisation this gives large gradients near the output layer, which forces a learning-rate warmup and makes the whole thing sensitive to the warmup schedule. Pre-LN puts the norm *inside* the residual branch, before the sub-layer; the gradients are well-behaved at init, and training is stable with far less hand-tuning. When I have to run for a million steps and cannot babysit a fragile schedule, that decides it: Pre-LN. So each block is: normalise, self-attend with the pair bias, add residual; normalise, GELU feed-forward, add residual. One more stability detail falls out of that choice. The final LayerNorm can hide a very large pre-normalized state, which is dangerous in mixed precision even if the normalized output looks fine — a state with norm `10⁴` and one with norm `√d` look identical after normalization, but the first one will overflow fp16 on the way there. So I want to watch the pre-normalized atom and delta-pair states and keep their norms near `√d` (the norm a unit-variance `d`-vector has). The penalty I'll use is `L_norm = mean_i max(| ‖s_i‖ − √d | − τ, 0)`, with `τ = 1` and a small weight. Let me check it has the shape I intend, at `d=512` so `√d = 22.63`. At `‖s‖ = 22.0` it gives `max(0.63 − 1, 0) = 0`; at `22.6` and at `23.6` it is still `0` — there is a flat dead zone of width `2τ` around `√d` where a healthy state pays nothing. Outside the band it grows linearly: `‖s‖ = 10` costs `max(12.63 − 1, 0) = 11.63`, and `‖s‖ = 30` costs `max(7.37 − 1, 0) = 6.37`. So it does exactly what I want — silent while the representation is well-scaled, a gentle linear pull-back once it starts to grow or collapse, and never fighting the LayerNorm over small fluctuations.

Now, do I even need to predict 3D coordinates? For property prediction the answer is no — I read out a single molecule-level vector and regress/classify. But I want this same backbone to be the universal one, usable for the self-supervised pretraining I'm counting on and for genuinely 3D tasks like recovering or generating a conformation, and those need the model to *output* coordinates, not just consume them. So let me work out a coordinate head, because the way I build it will also tell me how to pretrain. Outputting coordinates is delicate: the *input* encoding was invariant (distances don't move under rotation), but the *output* must be equivariant — if I rotate the input molecule, the predicted coordinates must rotate with it, or the prediction is nonsense. I need an output that transforms like a position.

What kind of position update can be equivariant at all? It has to be built from things that transform like positions. The only position-like vectors I have are the relative differences `x_i − x_j`, so let me try updating each atom by a weighted sum of those, with scalar weights `c_ij` drawn from the invariant pair feature:

  x̂_i = x_i + (1/n) Σ_j (x_i − x_j) · c_ij.

Does it transform correctly? Rotate the whole molecule by `R` and translate by `t`: each `x_i ↦ R x_i + t`, so each difference `(x_i − x_j) ↦ R(x_i − x_j)` — the translation cancels in the difference and the rotation factors out — while the scalar weight `c_ij`, built only from distances, doesn't change at all (the same check I just ran on `d_ij` covers it). Substituting, `x̂_i ↦ R x_i + t + (1/n)Σ_j R(x_i − x_j)c_ij = R(x_i + (1/n)Σ_j (x_i−x_j)c_ij) + t = R x̂_i + t`. On paper the update rotates and translates exactly with the input. I don't fully trust algebra I just wrote, so I run it on the same 3-atom example: I compute `c_ij` from the distances, form `x̂`, then separately apply the rigid motion to `x̂`, and compare against recomputing the whole update on the *moved* coordinates. The two agree to `4·10⁻¹⁶` — equivariant to machine precision — and I confirm `c_ij` came out bit-identical on the original and moved inputs, which is what makes the rotation factor cleanly out front. So this construction is the equivariant counterpart to the invariant input encoding: move each atom along a combination of relative directions, weighted by invariant scalars. The scalar weight should come from the pair feature, since that's my per-pair invariant. But which pair feature — the full `q_ij^L`, or the learned correction `q_ij^L − q_ij^0`? I argued a moment ago that the *delta* is the part the network actually learned about the interaction, beyond the raw geometry it was already handed; the raw geometry it doesn't need to be told again, it's the input. So the coordinate update should be driven by the delta. The clean scalar head is:

  c_ij = ReLU( (q_ij^L − q_ij^0) U ) W,   U ∈ ℝ^{H×H},   W ∈ ℝ^{H×1},

projecting the `H`-channel delta pair representation down to a single scalar per pair through a small nonlinearity. And one efficiency choice: EGNN moves the coordinates at *every* layer, which makes sense for a deep dynamics model but is wasteful here — I only need the final geometry, so I run the coordinate update *once, in the last layer*, off the accumulated delta pair representation. The `1/n` factor plays the same role as EGNN's `1/(M−1)` normaliser: it keeps the magnitude of the update from scaling with molecule size, so a 50-atom molecule and a 10-atom molecule get comparably-scaled position corrections.

With a coordinate head in hand the pretraining task writes itself, and it has to be a 3D task or I'm not using the geometry I went to all this trouble to encode. The BERT move is to mask tokens and predict them; I'll keep a version of that — mask some atom *types* and predict them from context, a masked-atom-prediction loss, since atom type is discrete and a [MASK] symbol is well-defined for it. But the geometry is the point, and here masking doesn't transfer: a coordinate is a continuous value, there is no special "[MASK] coordinate" I can substitute the way I substitute a [MASK] token, because any value I pick is itself a legitimate point in space. So instead of masking the coordinate I *corrupt* it — add noise — and ask the model to recover the true position. Concretely, perturb each coordinate by a uniform offset in `[−r, +r]` Å and train the model to undo it. The recovery comes through two heads: the equivariant coordinate head predicts the clean positions directly, and a pair-distance head predicts the clean pairwise distances from the pair representation (a per-pair scalar regressed off `q_ij`). The noise magnitude `r` is a knob — too large and the map from a badly scrambled structure back to the truth is hopeless; small enough and it's a learnable denoising. There's a subtlety if the noise is large: which noised position corresponds to which true atom becomes ambiguous, so one could add a re-assignment step that greedily matches noised to true positions to minimise total displacement. But for a modest `r` — on the order of 1 Å — the correspondence is unambiguous and re-assignment is unnecessary; I'll keep the corruption simple and small. So pretraining = masked atom prediction + 3D position recovery (coordinate + distance), with loss weights that lift the coordinate and distance terms because their raw magnitudes are small next to the cross-entropy.

For the molecule-level readout I borrow BERT's [CLS] idea: prepend a special atom that represents the whole molecule, and give it a sensible coordinate — the centroid of all atoms, so it sits in the middle and its distances to everything are meaningful pair features. After the encoder, take the [CLS] atom's representation, run it through a small head — dropout, a dense layer with a tanh nonlinearity, dropout, then a linear projection to the number of tasks — and that's the prediction. (Mean-pooling over all atoms is a reasonable alternative; [CLS] is the cleaner BERT-style summary and what I'll use.) At finetuning I can lean on something free: generating a 3D conformer is cheap, so I can produce several conformers per molecule and use them as data augmentation, and at validation/test time average the model's predictions over a handful of conformers to smooth out the noise from any single generated geometry. Multitask datasets with missing labels — like a toxicity panel where each molecule is measured on only some assays — are handled by the fixed pipeline's masked loss, which only counts the present labels per molecule, so the readout just emits one logit per task and lets the loss ignore the rest.

The forward pass is now forced. Atoms come in as types and 3D coordinates, plus a [CLS] atom at the centroid. The atom representation `x` is initialised from the atom-type embedding. The pair representation is initialised from the spatial encoding: take pairwise distances, apply the per-pair-type affine, evaluate the learnable `K`-channel Gaussian basis, project to `H` heads — that's `q_ij^0`, the initial geometric bias. Then for each Pre-LN layer: the atoms self-attend with `q_ij` added as the per-head attention bias (pair → atom), and the per-head `Q_i K_jᵀ` is added back into `q_ij` (atom → pair), so both channels evolve together and `q_ij` accumulates its learned correction. After the stack, for property prediction I take the [CLS] representation and run the head. For 3D tasks I take the delta pair representation `q^L − q^0`, project it to per-pair scalars, and do the single equivariant coordinate update. Every component traces back to a need: full-connectivity attention for long-range interactions; distance-only encoding for SE(3)-invariance without tensor machinery; pair-type affine because the same distance means different things for different element pairs; Gaussian basis for a smooth, distance-differentiable featurisation; the pair channel and its two-way communication so geometry steers attention and attention refines geometry; the delta for the coordinate head because that's the learned part; Pre-LN plus the small norm penalty for stability at scale; [CLS] for the molecule summary; corruption-not-masking for the continuous-coordinate pretraining.

Now I can write the backbone code that fills the empty architecture slot. First the Gaussian spatial encoding:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)        # G(d, μ, σ)


class GaussianLayer(nn.Module):
    """Pair-type-aware Gaussian distance kernel: per-pair-type affine a_t·d + b_t,
    then a learnable K-channel Gaussian basis. d_ij is SE(3)-invariant, so the whole
    spatial encoding is invariant to global rotation/translation."""
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)               # μ^k, shared across pairs
        self.stds = nn.Embedding(1, K)                # σ^k
        self.mul = nn.Embedding(edge_types, 1)        # a_t : per-pair-type scale
        self.bias = nn.Embedding(edge_types, 1)       # b_t : per-pair-type shift
        nn.init.uniform_(self.means.weight, 0, 3)     # spread Gaussians over ~0-3 Å range
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)        # affine starts as identity (a=1, b=0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):                  # x = distances [B, S, S], edge_type = pair types
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias              # A(d, t) = a_t·d + b_t  -> [B, S, S, 1]
        x = x.expand(-1, -1, -1, self.K)              # broadcast to K channels
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)   # [B, S, S, K]
```

Then the attention, with both the pair-bias (pair→atom) and the QK write-back (atom→pair):

```python
class SelfMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)     # fused QKV
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key_padding_mask=None, attn_bias=None, return_attn=False):
        bsz, tgt_len, _ = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        def reshape(t):
            return (t.view(bsz, -1, self.num_heads, self.head_dim)
                     .transpose(1, 2).contiguous()
                     .view(bsz * self.num_heads, -1, self.head_dim))

        q = reshape(q) * self.scaling
        k = reshape(k)
        v = reshape(v)
        src_len = k.size(1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))         # Q_i K_jᵀ / √d  (raw pair affinity)

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            attn = F.softmax(attn_weights + attn_bias, dim=-1)  # pair -> atom: add pair repr as bias
            attn = F.dropout(attn, p=self.dropout, training=self.training)
        else:
            attn_weights = attn_weights + attn_bias             # this updated logit IS the new pair repr
            attn = F.dropout(F.softmax(attn_weights, dim=-1),
                             p=self.dropout, training=self.training)

        o = torch.bmm(attn, v)
        o = (o.view(bsz, self.num_heads, tgt_len, self.head_dim)
              .transpose(1, 2).contiguous()
              .view(bsz, tgt_len, self.embed_dim))
        o = self.out_proj(o)
        if not return_attn:
            return o
        return o, attn_weights, attn                            # attn_weights carried out as pair repr
```

The Pre-LN block, and the encoder that accumulates the pair representation across layers (so `q^{l+1} = q^l + QKᵀ/√d`, and the delta `q^L − q^0` falls out by subtraction):

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
        x = self.self_attn_layer_norm(x)                        # Pre-LN
        x = self.self_attn(query=x, key_padding_mask=padding_mask,
                           attn_bias=attn_bias, return_attn=return_attn)
        if return_attn:
            x, attn_weights, _ = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)                            # Pre-LN
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not return_attn:
            return x
        return x, attn_weights, None


class TransformerEncoderWithPair(nn.Module):
    """Carries the pair representation (the attention bias) through the stack: each
    layer adds its QKᵀ into it (atom -> pair), so it accumulates a learned correction."""
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

        input_attn_mask = attn_mask                            # q^0 : the initial geometric pair repr
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
                                    attn_bias=attn_mask, return_attn=True)   # attn_mask <- q^{l+1}

        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)

        x = self.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask          # q^L - q^0 : the learned correction
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        delta_pair_repr = delta_pair_repr.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = masked_mean(pair_mask, norm_loss(delta_pair_repr), dim=(-1, -2))
        delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)
        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm
```

And the model that wires it together — atom-type embedding, the Gaussian pair encoding into the attention bias, the encoder, and the [CLS] readout for property prediction (the equivariant coordinate head is the same delta-pair update derived above, used only for 3D-output tasks):

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
        x = features[:, 0, :]                                  # the [CLS] atom = molecule summary
        x = self.dropout(x)
        x = self.activation_fn(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class MoleculeModel(nn.Module):
    def __init__(self, num_atom_types, num_tasks,
                 embed_dim=512, ffn_embed_dim=2048, attention_heads=64,
                 encoder_layers=15, K=128):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_atom_types, embed_dim, padding_idx=0)
        n_edge_type = num_atom_types * num_atom_types          # pair type = (type_i, type_j)
        self.gbf = GaussianLayer(K, n_edge_type)               # 3D distance -> K Gaussian channels
        self.gbf_proj = NonLinearHead(K, attention_heads)      # K -> per-head attention bias
        self.encoder = TransformerEncoderWithPair(
            encoder_layers, embed_dim, ffn_embed_dim, attention_heads)
        self.cls_head = ClassificationHead(embed_dim, embed_dim, num_tasks)
        self.pair2coord_proj = NonLinearHead(attention_heads, 1)  # used for 3D-output tasks

    def forward(self, tokens, dist, edge_type, padding_mask=None):
        seq_len = tokens.size(1)
        x = self.embed_tokens(tokens)                          # atom-type embedding -> [B, S, D]

        gbf_feature = self.gbf(dist, edge_type)                # [B, S, S, K]  pair-type Gaussian kernel
        attn_bias = self.gbf_proj(gbf_feature)                 # [B, S, S, H]  per-head pair bias = q^0
        attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous().view(-1, seq_len, seq_len)

        encoder_rep, pair_rep, delta_pair_repr, x_norm, delta_pair_norm = self.encoder(
            x, attn_mask=attn_bias, padding_mask=padding_mask)
        return self.cls_head(encoder_rep)                      # [B, num_tasks]

    def coord_head(self, coords, delta_pair_repr, padding_mask=None):
        atom_num = coords.shape[1] if padding_mask is None else (
            1 - padding_mask.type_as(coords)
        ).sum(dim=1).view(-1, 1, 1, 1)
        # relative vectors x_j - x_i (this layout; the x_i - x_j form I derived above differs only
        # by a sign that the learned scalar c_ij absorbs, so both are equivariant and equivalent)
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

The chain is now complete. I needed an encoder that takes 3D structure directly and respects rotation/translation symmetry, cheaply, at pretraining scale. A local GNN can't see long-range spatial contacts, so the backbone is a fully-connected Transformer — but a Transformer is position-blind, and my position is a continuous 3D coordinate I'm not allowed to use raw. Graphormer showed how to inject structure as a per-pair attention bias without breaking full connectivity, but its bias indexes discrete 2D topology. The distance between two atoms is invariant under any rigid motion — which I checked numerically, distances unchanged to `10⁻¹⁶` while absolute-coordinate features scrambled by `~1` under the same motion — so building the bias purely from pairwise distances inherits SE(3)-invariance and dodges the slow equivariant-tensor machinery entirely. I featurise each distance with a learnable Gaussian basis (smooth and differentiable in `d`, so I can later backprop into coordinates) and make it pair-type-aware with a per-element-pair affine, because the same distance means different things for a C–C and a C–O contact. The Gaussian features, projected per head, become the pair-to-atom bias. Then, noticing the pair channel was frozen while the atoms learned, I let the per-head `QKᵀ` write back into the pair representation each layer — atom-to-pair communication — so geometry steers attention and attention refines geometry, and the residual accumulation makes `q^L − q^0` the learned correction to raw geometry. To output coordinates I built an equivariant update — a size-normalised weighted sum of relative vectors with invariant scalar weights, the same form EGNN uses — which I verified rotates with the input to `10⁻¹⁶`, driven by that delta pair representation, run once in the last layer for efficiency. Pre-LN throughout for stable gradients at million-step scale. Pretraining recovers corrupted (noised, not masked, because coordinates are continuous) 3D positions plus masked atom types; finetuning reads out the centroid [CLS] atom through a small head, with cheap multi-conformer augmentation. The design stays close to a vanilla Transformer's computation while adding the geometric state needed for direct 3D input and output.
