## Research question

The problem is to predict chemical properties of a molecule from its structure: solubility,
toxicity, blood-brain-barrier penetration, enzyme inhibition, hydration free energy, and so on.
This is one of the oldest cheminformatics tasks and the central modeling step in early-stage
drug discovery, where running a real assay is slow and expensive, so a model that screens
candidates *in silico* is valuable if it is accurate on molecules the chemist has *not*
seen — new scaffolds, new chemical series, the next quarter's compounds.

Two distinct families of models exist. The first runs an off-the-shelf classifier or regressor
on a *fixed* vector encoding of the molecule — a fingerprint or a set of expert descriptors. The
second feeds the molecular graph directly into a graph convolutional network that *learns* its
own encoding end to end against the property labels. The published evidence on which family is
better is mixed: one large benchmark reports that the graph-convolution models outperform,
another reports the opposite. Part of the disagreement traces to how the data is split: under a
random train/test split, train and test can share the same scaffolds, whereas the chemistry the
model is ultimately used on is genuinely novel. The question is how to learn a single molecular
representation from the graph that can be compared against both the fixed-descriptor models and
the prior graph networks across many datasets and the diverse size/sparsity regimes
drug-discovery datasets come in, and how to evaluate it so that the number reflects performance
on new chemical space.

## Background

A molecule is naturally a graph: atoms are nodes, bonds are edges. Atom features (atomic
number, degree, formal charge, chirality, number of attached hydrogens, hybridization,
aromaticity, mass) and bond features (bond type single/double/triple/aromatic, conjugation,
ring membership, stereochemistry) are all computable from the structure with an open-source
cheminformatics toolkit (RDKit), so a molecule becomes a featured graph `(x_v for atoms,
e_vw for bonds)`.

The dominant family for graph-structured learning at this time is the *message passing neural
network* framework, which Gilmer et al. (2017) introduced as an umbrella that unifies most of
the then-current graph models. It operates on an undirected graph in two phases. A **message
passing phase** runs for `T` steps; on each step every node `v` collects a message from its
neighbors and updates its hidden state,

```
m_v^{t+1} = sum_{w in N(v)} M_t(h_v^t, h_w^t, e_vw)
h_v^{t+1} = U_t(h_v^t, m_v^{t+1})
```

where `N(v)` are the neighbors of `v`, `M_t` is a learned message function and `U_t` a learned
vertex-update function, with `h_v^0` some function of the atom features `x_v`. A **readout
phase** then maps the final node states to a prediction, `y_hat = R({h_v^T : v in G})`, where
`R` is invariant to the ordering of the nodes (a molecule has no canonical atom ordering).
The whole thing is differentiable and trained end to end. The number of steps `T` sets the
receptive field: a node hears about nodes within `T` bonds of it.

Several well-known graph models are instances of this framework, most of them placing the
main hidden states and messages **on atoms**:
- The neural-fingerprint convolution of Duvenaud et al. (2015) takes the message as a
  concatenation `M(h_v, h_w, e_vw) = (h_w, e_vw)`, updates with a degree-specific weight matrix
  through a sigmoid, and reads out by summing a softmax over all steps and atoms. The message
  sums over connected nodes and connected edges separately, `m_v = (sum_w h_w, sum_w e_vw)`.
- The gated graph network of Li et al. (2016) uses an edge-type-dependent linear message
  `M = A_{e_vw} h_w` and a GRU as the update.
- The "weave" model of Kearnes et al. (2016) is the lone prior model that maintains explicit
  edge representations and updates them during message passing.
- Interaction networks, deep tensor networks, and the spectral / Laplacian convolutions are
  further instances.

There is also a connection between message passing and *belief propagation* in probabilistic
graphical models (Pearl 1988; Koller & Friedman). On tree-structured graphical models, exact
inference is organized as local messages whose schedule and bookkeeping determine how
information moves through the graph. Dai et al. (2016), under the name structure2vec, showed
that the fixed-point updates of mean-field and loopy belief propagation on a pairwise Markov
random field can be embedded as learned graph representations. That line is a representation
learner, not a molecule-property architecture with chemical bond features, a molecular readout,
or a drug-discovery training harness.

Structural facts about the *existing* atom-centered models:
- Because `m_v^{t+1}` sums `M_t` over the full neighbor set `N(v)`, the unrolled computation
  includes walks of the form `v_1 v_2 ... v_n` with `v_i = v_{i+2}` — a step out and
  immediately back (Mahe et al. 2004).
- Datasets in this field are often tiny — hundreds to a few thousand molecules — and a model
  that learns its representation from scratch has little to learn from; in the lowest-data
  regimes the fixed-descriptor models can match or beat learned ones.
- With `T < diam(G)`, a node's receptive field does not cover the whole molecule.

## Baselines

These are the prior methods a new learned representation would be measured against.

**Fixed fingerprints / descriptors + a classifier.** Encode the molecule as a fixed vector and
run a standard learner on it. Morgan / extended-connectivity fingerprints (ECFP; Rogers &
Hahn 2010) hash circular atom environments of growing radius into a bit (or count) vector;
RDKit and Dragon descriptors are hand-designed physico-chemical quantities. On top of these
sit random forests (Breiman 2001), support vector machines (Cortes & Vapnik 1995), or
feed-forward networks; Mayr et al. (2018) concatenate several descriptor sets into one large
FFN. **Core idea:** a strong, general, *fixed* prior about chemistry, computed once.

**Atom-centered message passing networks (the MPNN instances above).** Duvenaud et al. (2015),
Li et al. (2016), Kearnes et al. (2016), and the rest learn the representation from the graph
by iterating the atom-centered update `m_v^{t+1} = sum_{w in N(v)} M_t(h_v^t, h_w^t, e_vw)`,
`h_v^{t+1} = U_t(h_v^t, m_v^{t+1})`, then read out over atoms. **Core idea:** let the network
discover task-specific atom environments rather than hand-coding them; given enough data these
are flexible and strong.

**Belief propagation as a structural reference (Pearl 1988; Dai et al. 2016).** Not a property
predictor on its own, but the principled object that many graph-message-passing stories borrow
from. It shows how local bookkeeping works in graphical models and that the same graph can
support more than one learned inference-style update.

## Evaluation settings

The yardsticks already in use, all pre-existing facts about the data and protocol:

- **MoleculeNet** (Wu et al. 2018): a suite of public benchmarks spanning quantum mechanics
  (QM7/8/9), physical chemistry (ESOL solubility, FreeSolv hydration free energy,
  Lipophilicity), biophysics (PCBA, MUV, HIV, BACE, PDBbind), and physiology (BBBP, Tox21,
  ToxCast, SIDER, ClinTox). Regression targets use RMSE or MAE; classification targets use
  ROC-AUC or PRC-AUC. Some datasets are multi-task with missing labels (e.g. Tox21 has 12
  assays, many compounds labeled for only a subset), which requires a per-task masked loss that
  ignores absent labels. Several datasets are small (a few hundred to a few thousand
  molecules); MUV is extremely class-imbalanced (~0.2% positive).
- **Three classification benchmarks** with a scaffold
  split and ROC-AUC (averaged over valid labels per task, then across tasks, higher better):
  BBBP (blood-brain-barrier penetration, 2,039 molecules, 1 task), BACE (β-secretase 1
  inhibition, 1,513 molecules, 1 task), Tox21 (toxicity across 12 assays, 7,831 molecules,
  multi-task with missing labels).
- **Splitting.** A **scaffold split** partitions molecules by their Murcko scaffold (computed
  by RDKit) so that train and test share no scaffold, which makes the test set structurally
  novel; it is used as a proxy for the **chronological / time split** that real drug-discovery
  pipelines evaluate on. Sheridan (2015) argued scaffold/time splits are harder and more
  realistic than random splits.
- **Protocol.** Identical featurization across models (RDKit-computed atom/bond features);
  cross-validation over several random-seeded splits because datasets are small; an Adam
  optimizer with a warmup-then-decay learning-rate schedule; for regression, targets normalized
  before training; for multi-task classification, a masked loss over present labels; optional
  test-time averaging over multiple input conformers.

## Code framework

The model plugs into a fixed training harness: a data pipeline that turns each SMILES into a
featured molecular graph (sparse node features `x`, COO `edge_index`, bond features
`edge_attr`, and a `batch_idx` assigning atoms to molecules), a standard optimizer and
learning-rate schedule, a masked loss for missing labels, target normalization for regression,
and a training loop — all already in place. What is *not* settled is how to turn a batch of
featured molecular graphs into a fixed-size vector per molecule and then into a prediction; that
encoder is exactly what is to be designed. The substrate below is only the generic graph-batch
plumbing and the model's outer interface; the empty slot is the graph encoder.

```python
import torch
import torch.nn as nn


def scatter_sum(src, index, dim_size):
    """Sum rows of `src` into `dim_size` buckets given by `index` (per-graph or per-atom
    aggregation). Generic primitive available before any model is designed."""
    out = torch.zeros(dim_size, src.size(-1), device=src.device)
    out.index_add_(0, index, src)
    return out


class MoleculeModel(nn.Module):
    """Maps a batch of featured molecular graphs to predictions of shape [B, num_tasks].

    The harness hands `forward` a batch with:
      batch.x          [total_atoms, atom_dim]   atom features
      batch.edge_index [2, total_edges]          graph connectivity in COO form
      batch.edge_attr  [total_edges, edge_dim]   bond features aligned with edge_index
      batch.batch_idx  [total_atoms]             atom -> molecule id (0..B-1)
    """

    def __init__(self, atom_dim: int, edge_dim: int, num_tasks: int, task_type: str):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = task_type
        # TODO: the graph encoder we will design (turns the featured graph into a
        #       per-molecule vector), plus the prediction head over it.
        raise NotImplementedError

    def forward(self, batch):
        # TODO: encode the batch of graphs -> [B, hidden], then predict -> [B, num_tasks].
        raise NotImplementedError


# existing training loop the model plugs into
def train(model, loss_fn, data_loader, optimizer, scheduler):
    for batch in data_loader:                       # batch of featured molecular graphs
        optimizer.zero_grad()
        preds = model(batch)                         # [B, num_tasks]
        loss = loss_fn(preds, batch.targets, batch.target_mask)  # masked over missing labels
        loss.backward()
        optimizer.step()
        scheduler.step()
```

The loop supplies featured graphs and consumes `[B, num_tasks]` predictions; the encoder and
head are where the model will live.
