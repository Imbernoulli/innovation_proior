## Research question

Given a small molecule, predict chemical properties — blood-brain-barrier penetration, enzyme
inhibition, toxicity across many assays, solubility, quantum-mechanical energies — from its
structure. The supervised data is scarce (a few thousand labelled molecules per task is typical) and
the test split is by *scaffold*, so the model must generalise to structurally novel cores it never saw
in training. Meanwhile unlabelled molecules are essentially unlimited. This is the regime where
pretrain-then-finetune should pay off, exactly as it does for text and images.

The sharper problem is *what to feed the model*. From the earliest structure-activity work it has been
understood that a molecule's behaviour is governed by its three-dimensional shape — how its atoms sit
in space, which groups are close enough to interact, the geometry of a binding event (Crum-Brown &
Fraser 1865; Hansch & Fujita 1964). Yet the dominant molecular encoders consume a 1D string or a 2D
bond graph, both of which discard the actual geometry. A model that could take 3D atomic coordinates
*directly as input* — and learn from them at the scale of hundreds of millions of conformers — ought
to predict geometry-sensitive properties far better. The goal is precisely that: an encoder whose
input is the set of atoms with their 3D positions, whose representation respects the physical
symmetries of space, and which is cheap enough to pretrain on a massive conformer corpus and then
finetune on a few thousand labels per task.

Three hard constraints make this non-trivial. (1) **Symmetry.** A molecule's properties do not change
if you rotate or translate it in space; the representation of the 3D input must be invariant to global
rotation and translation, or the model will waste capacity learning that a molecule and its rotated
copy are the same thing. (2) **Long-range interactions.** Two atoms far apart in the bond graph can be
close in space and interact strongly; the encoder must not be confined to local bonded neighbourhoods.
(3) **Scale and cost.** Pretraining on a corpus of hundreds of millions of conformers rules out any
architecture whose per-step cost is many times a plain Transformer's, and demands an optimisation that
is stable enough to run for a million steps without hand-holding.

## Background

By this time the pretrain-then-finetune recipe is the default everywhere unlabelled data is abundant
and labels are scarce: BERT (Devlin et al. 2019) and the GPT line in NLP, ViT (Dosovitskiy et al.
2021) in vision. The template is fixed — learn a representation by self-supervision on a huge unlabelled
corpus, then attach a light head and finetune on the downstream task. Molecular representation learning
(MRL) imports the same template, and on property prediction these learned representations already beat
classical fingerprint-plus-tree-model pipelines.

The facts that frame the problem:

- **Properties are 3D.** The activity of a drug, its binding to a target, many of its physical
  properties, are determined by the molecule's spatial conformation, not just its atom-bond
  connectivity. This is the oldest observation in the field and the reason 3D matters.
- **Most encoders throw the geometry away.** SMILES/InChI string encoders and 2D-graph encoders never
  see coordinates. The few attempts to bring in 3D use it as an *auxiliary* signal — a second view to
  contrast against the 2D graph, or geometric features bolted onto graph edges — so that at finetuning
  time the model still consumes the 2D graph and the 3D information has leaked away.
- **The Transformer is permutation-equivariant but position-blind.** Self-attention treats its inputs as
  a set: `A = QKᵀ/√d`, `softmax(A)V`. That is useful for atoms (which have no canonical order), and an
  order-invariant molecule readout can be built on top of it, but the model cannot tell where anything
  is unless position is injected. In NLP/CV the injected position is a *discrete* index; here the
  "position" is a *continuous* 3D coordinate that must enter in a rotation/translation-invariant way.
- **Equivariant networks exist but are expensive.** A line of SE(3)-equivariant models (tensor field
  networks; SE(3)-Transformers) handles 3D correctly by carrying higher-order geometric tensors through
  every layer. They are accurate but much heavier than a plain Transformer, which is prohibitive at
  pretraining scale.
- **Locality hurts self-supervision.** Building the molecule as a radius-cutoff graph (only atoms within
  a few Å exchange messages) is the natural efficiency move, but it caps the receptive field and risks
  missing long-range spatial contacts that are far in the bond graph but close in 3D.
- **Layer-norm placement decides optimisation stability.** Placing LayerNorm between residual blocks
  (Post-LN) produces large gradients near the output layer at initialisation, so training needs a
  learning-rate warmup and is sensitive to its schedule; placing LayerNorm inside the residual branch
  (Pre-LN; Xiong et al. 2020) keeps gradients well-behaved at init and makes training stable with much
  less tuning — the deciding property when a model must run for a million pretraining steps.

## Baselines

The prior methods a new 3D encoder would be measured against and react to.

**D-MPNN / Chemprop (Yang et al. 2019).** A supervised graph network that passes messages along
*directed bonds* rather than atoms, to avoid messages bouncing straight back along an edge ("message
collision"). State-of-the-art among purely supervised graph encoders. *Limitation:* it consumes the 2D
bond graph and never sees 3D coordinates; it is trained from scratch per task, so it cannot exploit the
ocean of unlabelled molecules.

**AttentiveFP (Xiong et al. 2019).** A graph-attention encoder over the 2D molecular graph with a
learned readout. *Limitation:* again 2D-only and supervised; no geometry, no pretraining.

**Graphormer (Ying et al. 2021).** Shows a *plain* Transformer can be state-of-the-art on molecular
graphs if graph structure is injected as a **bias term added to the attention logits**. For a node pair
`(i,j)` it adds learnable scalars indexed by structural functions of the pair: a spatial term keyed by
shortest-path distance and an edge term aggregated along the path,

```
A_ij = (h_i W_Q)(h_j W_K)ᵀ/√d  +  b_{φ(v_i,v_j)}  +  c_ij ,
       c_ij = (1/N) Σ_{n=1..N} x_{e_n} (w_n^E)ᵀ           (edges along the shortest path i→j)
```

with a special [VNode] token for graph-level readout and Pre-LN layers. This is the key structural
idea: keep the full-connectivity Transformer, and let a per-pair bias carry the structural prior into
attention. *Limitation:* every structural quantity it injects — shortest-path distance, discrete bond
features — is a function of the *2D topology*. The bias is indexed by *discrete* graph structure; it has
no place to put a *continuous* Euclidean distance, and no notion of rotation/translation invariance,
because there are no coordinates in the picture at all.

**GraphMVP (Liu et al. 2022) and GEM (Fang et al. 2022).** The first wave of "3D-aware" MRL. GraphMVP
runs contrastive/predictive learning *between* a 2D-graph view and a 3D-geometry view of the same
molecule, so a 2D encoder is nudged to be 3D-consistent. GEM augments a graph encoder with bond-length
and bond-angle features as extra edge attributes. *Limitation:* in both, 3D is a *teacher* or a *side
feature*, not a first-class input. At finetuning the model still ingests the 2D graph; coordinates
cannot be fed in directly, and the model can never *output* a 3D structure. The geometry is used to
shape a 2D representation, then discarded.

**EGNN (Satorras, Hoogeboom & Welling 2021).** An equivariant graph network that updates both node
features and node *coordinates* without any higher-order tensors. Its coordinate update is a weighted
sum of relative position vectors with invariant scalar weights,

```
x_i^{l+1} = x_i^l + C · Σ_{j≠i} (x_i^l − x_j^l) · φ_x(m_ij) ,   C = 1/(M−1) ,
```

which is rotation/translation-equivariant precisely because `(x_i−x_j)` transforms with the input while
the scalar weight `φ_x(m_ij)` is invariant. It is cheap (no tensor algebra) and can predict geometry.
*Limitation:* it is a message-passing GNN aimed at supervised energy/force learning; it updates
coordinates at *every* layer (extra cost), it is not set up as a large-scale self-supervised pretraining
backbone, and as a local GNN it does not by default carry the global, all-pairs attention that helps
self-supervision.

**Smooth radial distance features (Shuaibi et al. 2021).** A way to turn a scalar interatomic
distance into a soft feature vector by evaluating it against a bank of smooth basis functions, with
parameters that can depend on the atom-pair type. *Limitation on its own:* it is a positional-encoding
primitive, not a model — it answers "how to featurise one distance," not "how a 3D encoder should be
built around distances."

## Evaluation settings

The natural yardsticks already in use:

- **MoleculeNet** (Wu et al. 2018), the standard property-prediction benchmark. Classification tasks
  with **scaffold splitting** (8:1:1 train/val/test, scaffolds disjoint across splits — a deliberately
  hard generalisation test), scored by **ROC-AUC** (higher better), averaged over valid labels per task
  and across tasks. The task at hand uses three of them: **BBBP** (blood-brain-barrier penetration,
  2,039 molecules, 1 task), **BACE** (β-secretase-1 inhibition, 1,513 molecules, 1 task), and **Tox21**
  (toxicity across 12 assays, 7,831 molecules, 12 tasks, multi-task with missing labels handled by a
  masked loss). The broader benchmark also includes regression tasks (ESOL, FreeSolv, Lipophilicity,
  QM7/8/9) scored by RMSE or MAE.
- Following prior work, results are reported as mean ± std over a few random seeds; the checkpoint with
  best validation loss is selected and reported on test.
- 3D conformers are generated cheaply (distance-geometry embedding with a molecular-mechanics
  optimisation), and multiple conformers per molecule are available as test-time augmentation.

## Code framework

The encoder plugs into a fixed pipeline that already exists: SMILES preprocessing, conformer
generation, scaffold splitting, the training loop, the optimiser schedule, target normalisation for
regression, a masked loss for missing multi-task labels, and test-time augmentation over several
conformers. What is *not* settled is the model itself — how to turn a set of atoms with 3D coordinates
into a per-molecule prediction. The substrate is a standard Transformer-style atom encoder plus the
primitives that already exist:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- existing primitives ---

class SelfMultiheadAttention(nn.Module):
    """Standard multi-head self-attention with a fused QKV projection and the usual
    optional additive mask/bias hook used by Transformer variants."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # forward computes softmax(QKᵀ·scaling + optional_bias) V

class TransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm encoder block: LN -> self-attention -> residual,
    then LN -> FFN(GELU) -> residual. Pre-LN chosen for stable gradients at init."""
    def __init__(self, embed_dim, ffn_embed_dim, attention_heads, dropout=0.1,
                 attention_dropout=0.1, activation_dropout=0.0):
        super().__init__()
        ...  # standard Pre-LN block

class NonLinearHead(nn.Module):
    """Two-layer MLP with GELU — a generic projection head."""
    def __init__(self, input_dim, out_dim, hidden=None):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


# --- the empty architecture slot ---

class GeometryBackbone(nn.Module):
    """The architecture to be designed: it must consume atom embeddings plus 3D
    coordinates and return contextual atom embeddings."""
    def __init__(self, embed_dim):
        super().__init__()
        # TODO: the architecture we'll design.

    def forward(self, atom_embeddings, coordinates, padding_mask=None):
        raise NotImplementedError

class MoleculeModel(nn.Module):
    """Consumes a batch of atoms with types and 3D coordinates; returns [B, num_tasks].
    The atom-type embedding, the Pre-LN Transformer stack, and the [CLS]/linear readout
    are standard; the backbone architecture is the open slot."""

    def __init__(self, atom_dim, edge_dim, num_tasks, task_type):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_atom_types, embed_dim)   # atom-type embedding
        self.backbone = GeometryBackbone(embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_tasks)              # [CLS] readout

    def forward(self, batch):
        # atom_features -> atom-type tokens -> embed_tokens -> x  [B, S, D]
        # coordinates available as batch.positions
        x = self.embed_tokens(...)                                   # [B, S, D]
        x = self.backbone(x, batch.positions, batch.padding_mask)
        return self.cls_head(x[:, 0, :])                            # predict from [CLS] atom
```

The atom-type embedding, Transformer blocks, GELU projection head, optimiser/loss plumbing, and
[CLS]-then-linear readout all already exist. The single empty slot is the 3D-aware backbone.
