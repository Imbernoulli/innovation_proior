# Message Passing Neural Networks (MPNN)

## Problem

Predict the quantum-mechanical properties of small organic molecules (e.g. the 13 QM9 targets computed by DFT) with a model that reads the molecular graph directly, learns its own features, is invariant to graph isomorphism, and runs orders of magnitude faster than DFT.

## Key idea

Many neural models for graphs — differentiable molecular fingerprints, gated graph nets, interaction networks, molecular graph convolutions, deep tensor nets, and spectral/Laplacian graph convolutions — are special cases of one computation. The forward pass has a **message passing phase** of T rounds and a **readout phase**.

For each node v with hidden state h_v^t and edge features e_{vw}:

- **Message:**  m_v^{t+1} = Σ_{w ∈ N(v)} M_t(h_v^t, h_w^t, e_{vw})
- **Update:**   h_v^{t+1} = U_t(h_v^t, m_v^{t+1})
- **Readout:**  ŷ = R({ h_v^T | v ∈ G })

M_t, U_t, R are learned and differentiable. The neighbor sum is permutation-symmetric and R must be permutation-invariant; together this makes the whole model invariant to graph isomorphism.

**The framework subsumes prior models** by a choice of (M, U, R):

| Model | M | U | R |
|---|---|---|---|
| Molecular fingerprints (Duvenaud 2015) | (h_w, e_{vw}) concat | σ(H_t^{deg(v)} m) | f(Σ_{v,t} softmax(W_t h_v^t)) |
| Gated graph nets (Li 2016) | A_{e_{vw}} h_w (per discrete label) | GRU(h_v, m), tied | Σ_v σ(i(h_v^T,h_v^0)) ⊙ j(h_v^T) |
| Interaction nets (Battaglia 2016) | NN(h_v, h_w, e_{vw}) | NN(h_v, x_v, m) | f(Σ_v h_v^T) |
| Molecular graph conv (Kearnes 2016) | e_{vw}^t (edge state) | α(W_1(α(W_0 h_v), m)) | (+ edge-state updates) |
| Deep tensor nets (Schütt 2017) | tanh(W^{fc}((W^{cf}h_w+b_1)⊙(W^{df}e_{vw}+b_2))) | h_v + m (residual) | Σ_v NN(h_v^T) |
| Spectral / Laplacian (Bruna 2013, Kipf 2016) | L̃_{vw} h_w (Kipf: scalar L_{vw} h_w) | σ(m) (Kipf: σ(W^⊤ m)) | — |

**Spectral GCN as MPNN.** A spectral layer y_j = σ(Σ_i V F_{ij} V^⊤ x_i) becomes message passing by defining L̃_{v,w,i,j} = (V F_{ij} V^⊤)_{v,w} and collapsing indices: y_v = σ(Σ_w L̃_{v,w} x_w), i.e. M(h_v,h_w) = L̃_{v,w} h_w, U(h,m) = σ(m). For Kipf & Welling, H^{l+1} = σ(D̃^{-1/2}Ã D̃^{-1/2} H^l W^l) with Ã = A + I reduces per node to h_v^{t+1} = σ((W^l)^⊤ Σ_w L_{vw} h_w^t), so M_t(h_v,h_w) = L_{vw} h_w with scalar L_{vw} = Ã_{vw}(deg v · deg w)^{-1/2} and U_t = σ(W^⊤ m): a degree-normalized neighbor average followed by a linear map and nonlinearity.

## The choices for quantum chemistry (the enn-s2s model)

- **Update:** GRU, weight-tied across all T rounds. h_v^0 = atom features padded to width d. Directed treatment with separate in/out message channels (message width 2d).
- **Message — edge network:** M(h_v, h_w, e_{vw}) = A(e_{vw}) h_w, where A(·) is an MLP mapping the *continuous, vector-valued* edge feature e_{vw} to a d×d matrix. Generalizes the discrete-label matrices A_{e_{vw}} so continuous bond geometry (interatomic distance) can be used. (A *pair message* m_{wv} = f(h_w, h_v, e_{vw}) conditioned on both endpoints was tried and trained worse, so it was dropped.)
- **Readout — set2set** (Vinyals 2015): an order-invariant LSTM-attention set encoder over {(h_v^T, x_v)} producing a graph embedding, fed to an output network. More expressive than a plain sum while staying permutation-invariant.
- **Virtual graph elements** (for the no-geometry regime, to carry long-range information): a virtual edge type between non-bonded pairs, and/or a latent master node connected to every atom (own dimension d_master, own update; cost O(|E|d^2 + n·d_master^2)).
- **Towers** (efficiency): split h_v into k copies of width d/k, propagate each with its own M, U, then mix with a shared network g over the concatenation. A matrix-multiply propagation step costs O(n^2 d^2 / k) instead of O(n^2 d^2); k = 8, n = 9, d = 200 gives ≈2× speedup. The shared mixer preserves permutation invariance.
- **Input:** atom features (element one-hot H/C/N/O/F, atomic number, acceptor/donor, aromatic, hybridization, #H); raw 5-dim edge vector (Euclidean distance + bond-type one-hot) for the edge network, or distance bins / discrete bond types for the matrix-multiply message; explicit hydrogen atoms.
- **Training:** Adam; targets normalized to mean 0, variance 1; minimize MSE, evaluate MAE; early stopping on validation; one model per target; T ∈ [3, 8], set2set steps ∈ [1, 12].

## Code (enn-s2s)

```python
import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.nn import NNConv, Set2Set   # edge-network conv; set2set readout


class MPNN(torch.nn.Module):
    """Edge-network message + tied-GRU update + set2set readout (enn-s2s)."""
    def __init__(self, num_node_features, edge_dim=5, dim=64, T=3, set2set_steps=3):
        super().__init__()
        self.T = T
        self.lin0 = Linear(num_node_features, dim)          # h_v^0: pad atom features to width d

        # Message: M(h_v, h_w, e_vw) = A(e_vw) h_w, A an MLP edge_vec -> (d x d) matrix.
        edge_net = Sequential(Linear(edge_dim, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, edge_net, aggr='mean')  # m_v = sum_w A(e_vw) h_w
        self.gru = GRU(dim, dim)                              # U_t = GRU(h_v, m_v), tied over T

        self.set2set = Set2Set(dim, processing_steps=set2set_steps)  # R: invariant pooling
        self.lin1 = Linear(2 * dim, dim)
        self.lin2 = Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        for _ in range(self.T):                              # message passing phase
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)             # update each node's state
            out = out.squeeze(0)
        out = self.set2set(out, data.batch)                  # readout -> graph vector
        out = F.relu(self.lin1(out))
        return self.lin2(out).view(-1)                       # scalar property prediction


# --- training: per target, Adam, normalize targets, MSE train / MAE eval ---
model = MPNN(num_node_features=11)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_step(data):
    model.train()
    optimizer.zero_grad()
    loss = F.mse_loss(model(data), data.y)   # targets normalized to mean 0 / var 1
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def mae(loader, std):                        # report MAE in original units
    model.eval()
    err = sum((model(d) * std - d.y * std).abs().sum().item() for d in loader)
    return err / len(loader.dataset)
```
