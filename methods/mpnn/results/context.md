# Context: learning to predict molecular properties from graphs

## Research question

We want a machine that, given a small organic molecule, predicts its quantum-mechanical properties — atomization energies, the HOMO/LUMO orbital energies and their gap, vibrational frequencies, the dipole moment, polarizability, heat capacity, and so on. These properties are physically *defined* by the laws of quantum mechanics, but Schrödinger's equation is too hard to solve exactly, so in practice they are computed by Density Functional Theory (DFT), an approximation that scales roughly as O(N_e^3) in the number of electrons. A single nine-heavy-atom molecule takes about an hour on one CPU core; a seventeen-heavy-atom molecule takes up to eight hours. Running DFT across a large chemical search space — for drug discovery or materials design — is prohibitively slow.

The goal is a learned surrogate that reproduces the DFT label to within "chemical accuracy" (a per-property error threshold set by the chemistry community) while being orders of magnitude faster at inference. The question is how to build a neural model that reads the molecular graph directly, learns its own features, and is invariant to relabeling of the atoms.

## Background

**The molecule as a graph.** A molecule maps naturally onto an undirected graph: nodes are atoms, carrying features x_v (element type, charge, hybridization, hydrogen count, …); edges are bonds, carrying features e_{vw} (bond order — single/double/triple/aromatic — and, when a 3D conformation is known, the interatomic distance). The benchmark of interest provides, for each molecule, a low-energy 3D conformation, so we can study two regimes: one where full spatial geometry is available as input, and one where only the topological graph (atoms + bonds) is given.

**Hand-engineered descriptors.** The prevailing approach in chemistry ML is to convert a molecule into a fixed-length vector by a hand-designed rule, then feed that vector to an off-the-shelf regressor (kernel ridge regression, random forests, a plain MLP). The descriptors build symmetries into the input by hand; example families include atom-centered symmetry functions (Behler & Parrinello 2007), Coulomb Matrix representations (Rupp et al. 2012), and fingerprint-based features (ECFP4, Rogers & Hahn 2010). These fixed-vector approaches are well-established across the quantum-chemistry ML literature.

**Neural graph models.** The situation motivating neural architectures on molecular graphs parallels computer vision before convolutional nets: strong hand-engineered features plus a generic classifier, followed by interest in neural architectures with the right inductive bias. The symmetries of atomic systems point at neural networks that operate on graph-structured data and respect isomorphism. Several such models have already been proposed, each in its own notation.

The toolbox of operations that respect set/graph symmetry includes order-insensitive reductions over a collection (sums, averages, and more expressive order-invariant set encoders), recurrent cells (e.g. a GRU) for folding new information into a running state with tied parameters, and the standard differentiable layers.

## Baselines

These are the prior methods a new model would be measured against or built upon.

**Hand-engineered descriptor + regressor.** The dominant family. Each turns a molecule into a fixed vector:
- *Coulomb Matrix* (Rupp et al. 2012): the matrix C with C_{ii}=½Z_i^{2.4} and C_{ij}=Z_iZ_j/‖r_i−r_j‖. Its rows/columns are indexed by atom ordering, so invariance is typically enforced via augmentation or eigenvalue sorting.
- *Bag of Bonds* (BoB), *BAML* (bonds/angles ML, Huang & von Lilienfeld 2016), *Extended Connectivity Fingerprints* (ECFP4, Rogers & Hahn 2010), *projected histograms* (HDAD, Faber et al. 2017): increasingly elaborate hand-built featurizations fed to a standard regressor.

**Convolutional networks on molecular fingerprints** (Duvenaud et al. 2015). A differentiable analogue of circular fingerprints. Each round, a node forms a message by concatenating each neighbor's state with the connecting bond feature, sums these, and updates through a per-degree learned matrix followed by a sigmoid: roughly h_v ← σ(H^{deg(v)} · Σ_w (h_w, e_{vw})). The graph vector is read out by summing softmax(W_t h_v^t) over all nodes and rounds.

**Gated Graph Neural Networks** (Li et al. 2016). Assumes discrete edge types. The message into v is the sum over neighbors of A_{e_{vw}} h_w, with one learned matrix A per edge label; the node state is updated by a GRU, h_v ← GRU(h_v, m_v), weight-tied across rounds; the graph readout is a gated sum Σ_v σ(i(h_v^T, h_v^0)) ⊙ j(h_v^T) with i, j small networks.

**Interaction Networks** (Battaglia et al. 2016). A physics-simulation model. The message is a network on the concatenation (h_v, h_w, e_{vw}); the update is a network on (h_v, x_v, m_v) with an optional external per-node input x_v; the graph readout is f(Σ_v h_v). Originally run for a single round (T=1).

**Molecular graph convolutions** (Kearnes et al. 2016). Maintains *edge* states in addition to node states. The node message is just the edge state e_{vw}^t; the node update is a two-layer network on (h_v, m_v); and the edge state is itself updated from its neighbors' node states.

**Deep Tensor Neural Networks** (Schütt et al. 2017). A chemistry model whose message couples node and edge through an elementwise product, tanh(W^{fc}((W^{cf}h_w + b_1) ⊙ (W^{df}e_{vw} + b_2))); the update is residual, h_v ← h_v + m_v; the readout sums a per-node network, Σ_v NN(h_v).

**Spectral / Laplacian graph convolutions** (Bruna et al. 2013; Defferrard et al. 2016; Kipf & Welling 2016). Generalize convolution to graphs through the graph Laplacian L = I − D^{-1/2}WD^{-1/2}. A layer mixes node features by y_j = σ(Σ_i V F_{ij} V^⊤ x_i) with V the eigenvectors of L and F_{ij} diagonal learned filters; Kipf & Welling's first-order simplification is H^{l+1} = σ(D̃^{-1/2}ÃD̃^{-1/2} H^l W^l) with Ã = A + I. Typically applied to a single large graph (e.g. citation networks).

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
