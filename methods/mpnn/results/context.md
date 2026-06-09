# Context: learning to predict molecular properties from graphs

## Research question

We want a machine that, given a small organic molecule, predicts its quantum-mechanical properties — atomization energies, the HOMO/LUMO orbital energies and their gap, vibrational frequencies, the dipole moment, polarizability, heat capacity, and so on. These properties are physically *defined* by the laws of quantum mechanics, but Schrödinger's equation is too hard to solve exactly, so in practice they are computed by Density Functional Theory (DFT), an approximation that scales roughly as O(N_e^3) in the number of electrons. A single nine-heavy-atom molecule takes about an hour on one CPU core; a seventeen-heavy-atom molecule takes up to eight hours. Running DFT across a large chemical search space — for drug discovery or materials design — is prohibitively slow.

The goal is a learned surrogate that reproduces the DFT label to within "chemical accuracy" (a per-property error threshold set by the chemistry community) while being orders of magnitude faster at inference. Three things make this hard and define what a good solution must do:

1. **The right invariance.** A molecule is a set of atoms and bonds, not an ordered list. Relabeling the atoms, or rotating/reflecting the molecule in space, must not change the prediction. Concretely the model must be invariant to graph isomorphism (atom permutations).
2. **Learned features, not hand-engineered ones.** The model should read the molecular graph directly and discover its own features, the way convolutional nets learn image features instead of using hand-built descriptors.
3. **Variable size and connectivity.** Molecules differ in number of atoms and bond pattern; the model must accept arbitrary graphs and still produce a single fixed-size prediction per property.

## Background

**The molecule as a graph.** A molecule maps naturally onto an undirected graph: nodes are atoms, carrying features x_v (element type, charge, hybridization, hydrogen count, …); edges are bonds, carrying features e_{vw} (bond order — single/double/triple/aromatic — and, when a 3D conformation is known, the interatomic distance). The benchmark of interest provides, for each molecule, a low-energy 3D conformation, so we can study two regimes: one where full spatial geometry is available as input, and one where only the topological graph (atoms + bonds) is given and the model must implicitly recover whatever about the geometry the target depends on.

**Why hand-engineered descriptors fall short.** The prevailing approach in chemistry ML is to convert a molecule into a fixed-length vector by a hand-designed rule, then feed that vector to an off-the-shelf regressor (kernel ridge regression, random forests, a plain MLP). The descriptors build symmetries into the input by hand. This has two intrinsic limits, both observed in prior work: representations that *are* isomorphism-invariant (e.g. atom-centered symmetry functions) tend to break when the chemistry gets richer — more than three atomic species, or compositions not seen in training; and representations that are *not* invariant (e.g. a matrix indexed by atom order) force the downstream model to learn the invariance from data augmentation, wasting capacity. Either way, the features are frozen: they cannot adapt to the property being predicted.

**The diagnostic that motivates a neural graph model.** The situation parallels computer vision before convolutional nets: strong hand-engineered features plus a generic classifier, held back by a shortage of empirical evidence that a neural architecture with the *right inductive bias* can win. The symmetries of atomic systems point at neural networks that operate on graph-structured data and respect isomorphism. Several such models had already been proposed, each in its own notation, which obscures how closely related they are and makes it hard to tell which design details actually matter.

**The building blocks available.** The toolbox of operations that respect set/graph symmetry is well known: order-insensitive reductions over a collection (sums, averages, and more expressive order-invariant set encoders), recurrent cells (e.g. a GRU) for folding new information into a running state with tied parameters, and the standard differentiable layers. The open question is how to assemble such pieces into an architecture that reads a molecular graph and predicts a property while respecting isomorphism.

## Baselines

These are the prior methods a new model would be measured against or built upon.

**Hand-engineered descriptor + regressor.** The dominant family. Each turns a molecule into a fixed vector:
- *Coulomb Matrix* (Rupp et al. 2012): the matrix C with C_{ii}=½Z_i^{2.4} and C_{ij}=Z_iZ_j/‖r_i−r_j‖. Captures geometry and charge but is *not* isomorphism-invariant — its rows/columns are indexed by an (arbitrary) atom ordering, so invariance must be learned via augmentation.
- *Bag of Bonds* (BoB), *BAML* (bonds/angles ML, Huang & von Lilienfeld 2016), *Extended Connectivity Fingerprints* (ECFP4, Rogers & Hahn 2010), *projected histograms* (HDAD, Faber et al. 2017): increasingly elaborate hand-built featurizations fed to a standard regressor.
Common gap: features are fixed in advance and cannot be tuned to the target; the better-invariant ones generalize poorly across atom species/compositions (Behler & Parrinello 2007 symmetry functions), the better-expressive ones lack invariance.

**Convolutional networks on molecular fingerprints** (Duvenaud et al. 2015). A differentiable analogue of circular fingerprints. Each round, a node forms a message by concatenating each neighbor's state with the connecting bond feature, sums these, and updates through a per-degree learned matrix followed by a sigmoid: roughly h_v ← σ(H^{deg(v)} · Σ_w (h_w, e_{vw})). The graph vector is read out by summing softmax(W_t h_v^t) over all nodes and rounds. Gap: because the message sums neighbor states and bond features *separately*, it cannot model correlations between a bond and the atom it attaches to.

**Gated Graph Neural Networks** (Li et al. 2016). Assumes discrete edge types. The message into v is the sum over neighbors of A_{e_{vw}} h_w, with one learned matrix A per edge label; the node state is updated by a GRU, h_v ← GRU(h_v, m_v), weight-tied across rounds; the graph readout is a gated sum Σ_v σ(i(h_v^T, h_v^0)) ⊙ j(h_v^T) with i, j small networks. A strong, clean baseline. Gap: the discrete-edge matrices cannot ingest a continuous bond feature such as interatomic distance; the message A_{e}h_w depends only on the source node, not the destination; and with a finite number of rounds, information cannot travel between distant atoms unless the graph is augmented.

**Interaction Networks** (Battaglia et al. 2016). A physics-simulation model. The message is a network on the concatenation (h_v, h_w, e_{vw}); the update is a network on (h_v, x_v, m_v) with an optional external per-node input x_v; the graph readout is f(Σ_v h_v). Originally run for a single round (T=1). Gap: not specialized to chemistry, and the single-round form limits the receptive field.

**Molecular graph convolutions** (Kearnes et al. 2016). Maintains *edge* states in addition to node states. The node message is just the edge state e_{vw}^t; the node update is a two-layer network on (h_v, m_v); and the edge state is itself updated from its neighbors' node states. Gap: heavier bookkeeping; still uses sum/concat messages.

**Deep Tensor Neural Networks** (Schütt et al. 2017). A chemistry model whose message couples node and edge through an elementwise product, tanh(W^{fc}((W^{cf}h_w + b_1) ⊙ (W^{df}e_{vw} + b_2))); the update is residual, h_v ← h_v + m_v; the readout sums a per-node network, Σ_v NN(h_v). Gap: a fixed multiplicative interaction form.

**Spectral / Laplacian graph convolutions** (Bruna et al. 2013; Defferrard et al. 2016; Kipf & Welling 2016). Generalize convolution to graphs through the graph Laplacian L = I − D^{-1/2}WD^{-1/2}. A layer mixes node features by y_j = σ(Σ_i V F_{ij} V^⊤ x_i) with V the eigenvectors of L and F_{ij} diagonal learned filters; Kipf & Welling's first-order simplification is H^{l+1} = σ(D̃^{-1/2}ÃD̃^{-1/2} H^l W^l) with Ã = A + I. Typically applied to a single large graph (e.g. citation networks). Gap: the mixing weights come from the graph structure alone (no learned, edge-feature-conditioned interaction), and the spectral form ties the model to one fixed graph.

## Evaluation settings

The natural yardstick is a dataset of small organic molecules with quantum properties computed by DFT: about 134k molecules built from H, C, N, O, F with up to nine heavy atoms, each with a low-energy 3D conformation and thirteen computed property targets (four atomization energies U_0, U, H, G; the highest vibrational frequency ω_1 and zero-point vibrational energy; the orbital energies ε_HOMO, ε_LUMO and their gap Δε; electronic spatial extent ⟨R^2⟩, dipole-moment norm μ, static polarizability α; heat capacity C_v). Standard protocol: a random split into ~10k test, ~10k validation, and the rest for training; per-target normalization to zero mean and unit variance; train against mean-squared error but report mean absolute error; early stopping and model selection on the validation set. Each property's error is reported as the ratio of MAE to that property's chemical-accuracy threshold, so a ratio below 1 means chemical accuracy is reached. Both the topology-only and the geometry-available regimes are of interest.

## Code framework

The primitives that already exist: an automatic-differentiation framework with linear layers, GRUs, MLPs, ReLU/sigmoid; a graph data structure exposing node features, an adjacency/edge list, and edge features; mini-batching over variable-size graphs; the Adam optimizer; an MSE loss and an MAE metric. What does not yet exist is the architecture that turns a molecular graph into a prediction. We lay out the generic skeleton with one empty slot for that contribution.

```python
import torch
import torch.nn.functional as F
from torch.nn import Linear

# ---- data pipeline (exists) -------------------------------------------------
# A graph batch exposes:
#   data.x          [num_nodes, num_node_features]   atom features
#   data.edge_index [2, num_edges]                   connectivity (src, dst)
#   data.edge_attr  [num_edges, num_edge_features]   bond / distance features
#   data.batch      [num_nodes]                      which graph each node is in
#   data.y          [num_graphs]                     regression target

class GraphPropertyModel(torch.nn.Module):
    """Maps a molecular graph to a scalar property.
    Must be invariant to permutations of the atoms.
    The body is the thing we have to design."""
    def __init__(self, num_node_features, num_edge_features, dim):
        super().__init__()
        self.lin0 = Linear(num_node_features, dim)   # pad atom features to width dim

        # TODO: the architecture that maps the molecular graph (node features,
        #       edge_index, edge_attr) to a single fixed-size vector per graph,
        #       invariant to permutations of the atoms — the contribution.
        self.body = None  # TODO

        self.lin_out = Linear(dim, 1)  # final regression head

    def forward(self, data):
        h = F.relu(self.lin0(data.x))
        # TODO: process the graph (data.edge_index / data.edge_attr / data.batch)
        #       into one fixed-size, permutation-invariant vector g per graph,
        #       then return self.lin_out(g).view(-1)
        raise NotImplementedError

# ---- training loop (exists) -------------------------------------------------
def train_step(model, data, optimizer):
    optimizer.zero_grad()
    loss = F.mse_loss(model(data), data.y)   # train on MSE
    loss.backward()
    optimizer.step()
    return loss.item()

def mae(model, data):                         # evaluate on MAE
    return (model(data) - data.y).abs().sum().item()
```
