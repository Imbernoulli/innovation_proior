We want a machine that takes a small organic molecule and returns a quantum-mechanical property — an atomization energy, an orbital energy, a dipole moment — fast enough to be useful. The honest motivation is speed: Density Functional Theory gives these numbers but costs about an hour of CPU for a nine-heavy-atom molecule and scales like the cube of the electron count, so sweeping a chemical search space with it is hopeless. If I can fit the DFT output to within chemical accuracy and evaluate in milliseconds, that is a real win. The difficulty is that a molecule is neither a vector nor an image: it is a set of atoms held together by bonds. If I flatten it into a numbered list I have lied, because the numbering is arbitrary and relabeling the atoms must not change any physical property. So whatever I build has to be invariant to permutations of the atoms — invariant to graph isomorphism — and that is precisely the constraint the standard chemistry pipeline either violates or pays dearly to satisfy.

That standard pipeline hand-designs a fixed-length descriptor of the molecule and feeds it to an off-the-shelf regressor. The Coulomb matrix is the cleanest example: build $C$ with $C_{ii} = \tfrac{1}{2}Z_i^{2.4}$ and $C_{ij} = Z_i Z_j / \lVert r_i - r_j \rVert$. It packs in nuclear charges and geometry, but its rows are indexed by atom order, so permuting the atoms permutes the matrix and the downstream model has to *learn* that all those permuted matrices mean the same molecule via data augmentation — capacity spent undoing a symmetry I could have built in for free. The other camp, atom-centered symmetry functions in the Behler–Parrinello style, is genuinely invariant but brittle: it was reported to struggle past three atomic species and to fail on compositions it had not seen. The deeper problem with all of them is that the features are frozen, designed before the target was ever seen, whereas a convolutional net learns features tuned to the task. This is exactly the pre-CNN situation in vision: strong hand features plus a generic classifier, missing the neural architecture with the right inductive bias. Several people have already built neural models that read the molecular graph directly — differentiable fingerprints, gated graph nets, interaction nets, deep tensor nets, molecular graph convolutions, spectral convolutions — but each is written in its own notation, which hides how closely related they are and makes it impossible to tell which design details actually matter.

So I lay them side by side and stare. The differentiable-fingerprint model gives each atom a hidden vector, and in each round the atom concatenates each neighbor's state with the connecting bond feature, sums those, and pushes the sum through a per-degree learned matrix and a sigmoid; it reads out by summing $\mathrm{softmax}(W_t h_v^t)$ over atoms and rounds. The gated graph net has the same skeleton: the incoming message is $\sum_w A_{e_{vw}} h_w$ with one learned matrix per discrete bond type, the node updates with a GRU, and the readout is a gated sum. The interaction net's message is a network on $(h_v, h_w, e_{vw})$, its update a network on $(h_v, x_v, m_v)$, its readout $f(\sum_v h_v)$. The deep tensor net's message is $\tanh(W^{fc}((W^{cf}h_w + b_1) \odot (W^{df}e_{vw} + b_2)))$, its update residual, its readout $\sum_v \mathrm{NN}(h_v)$. Every one of these does the same three things in the same order: each node aggregates something from its neighbors, each node updates its own state from that aggregate, repeat for a few rounds, then collapse all the node states into one graph vector. They are not different algorithms; they are different functions filling the same three slots.

I propose to write that one algorithm down and treat it as the object of study. The method is the Message Passing Neural Network. A node $v$ carries a hidden state $h_v^t$ at round $t$, and the forward pass has a message passing phase of $T$ rounds followed by a readout phase, governed by three learned, differentiable functions $M_t$, $U_t$, $R$:

$$m_v^{t+1} = \sum_{w \in N(v)} M_t\big(h_v^t,\, h_w^t,\, e_{vw}\big),\qquad h_v^{t+1} = U_t\big(h_v^t,\, m_v^{t+1}\big),\qquad \hat y = R\big(\{\, h_v^T \mid v \in G \,\}\big).$$

The single load-bearing choice is the *sum* over neighbors. I use a sum rather than a concatenation in neighbor order because it is the cheapest reduction that is permutation-symmetric: reorder the neighbors and the sum is unchanged. That, together with the requirement that the readout $R$ be a function of the *set* of final node states, is what makes the whole model invariant to graph isomorphism by construction — relabeling the atoms permutes the $h_v$'s but leaves $\hat y$ untouched. Any symmetric reduction would do, but sum is the natural first choice and it is what all the prior models effectively use. The message phase builds local features; the readout phase pools them; I get the symmetry for free instead of paying for it with augmentation.

This abstraction is not a vague analogy — it genuinely subsumes the prior models, each as a choice of $(M, U, R)$, and reducing them this way exposes flaws their own notation hides. The fingerprint model is $M = (h_w, e_{vw})$, the concatenation; expanding the sum gives $m_v^{t+1} = (\sum_w h_w,\, \sum_w e_{vw})$, which sums neighbor *states* and bond *features* in separate blocks, so it can never see that *this* bond attaches to *that* atom — it cannot model node–edge correlations, and that is a hint about what a good $M$ should do. The gated graph net is $M_t = A_{e_{vw}} h_w$, $U_t = \mathrm{GRU}(h_v, m)$ tied across rounds. The interaction net is $M = \mathrm{NN}(h_v, h_w, e_{vw})$, $U = \mathrm{NN}(h_v, x_v, m)$, run at $T=1$. The deep tensor net's residual update $U = h + m$ is the same trick that lets rounds stack without washing the signal out. The strongest evidence the abstraction is right comes from the spectral lineage, which is derived from the graph Fourier transform and looks nothing like neighbor aggregation. A spectral layer $y_j = \sigma(\sum_i V F_{ij} V^\top x_i)$ can be rewritten by absorbing the products into a tensor $\tilde L_{v,w,i,j} = (V F_{ij} V^\top)_{v,w}$ and grouping the channel indices into vectors, which collapses the double sum to $y_v = \sigma(\sum_w \tilde L_{v,w} x_w)$ — exactly $M(h_v, h_w) = \tilde L_{v,w} h_w$ and $U(h,m) = \sigma(m)$, a message passing round whose only oddity is that the neighborhood is every node because a dense spectral filter is global. Pushed to the first-order Kipf–Welling form $H^{l+1} = \sigma(\tilde D^{-1/2} \tilde A \tilde D^{-1/2} H^l W^l)$ with $\tilde A = A + I$, one node's row reduces to $h_v^{t+1} = \sigma((W^l)^\top \sum_w L_{vw} h_w^t)$, so $M_t = L_{vw} h_w$ with scalar $L_{vw} = \tilde A_{vw}(\deg v \cdot \deg w)^{-1/2}$ and $U_t = \sigma(W^\top m)$ — a degree-normalized neighbor average followed by a linear map. The whole zoo collapses onto three slots.

With the framework in hand the design space *is* $(M, U, R)$, so I search it deliberately, starting from the strongest baseline — the gated graph net — and probing each slot; the resulting configuration is the enn-s2s model. For the update I keep the GRU and weight-tie it across all $T$ rounds: untying it into a fresh $U_t$ per round does not help, and the better use of parameters is to tie and instead widen the hidden state $d$. Tying turns the $T$ rounds into a recurrent propagation I can run for more steps without paying in parameters, and the GRU's gating keeps the state from blowing up or washing out as I iterate. I initialize $h_v^0$ to the atom feature vector padded up to width $d$, and I treat the undirected bonds as directed with separate in/out message channels (message width $2d$) purely as a handle for parameter tying.

The slot that matters most is the message. The baseline $M = A_{e_{vw}} h_w$ uses one learned matrix per discrete bond label, but the single best signal QM9 offers is geometry — the interatomic distance — and distance is *continuous*. A per-label matrix scheme cannot ingest a real number without bucketing it into bins and pretending the bins are labels, throwing away resolution. So I make the matrix itself depend on the edge: let a small network $A(\cdot)$ read the edge vector $e_{vw}$ and *output* a $d \times d$ matrix, giving the edge-network message

$$M(h_v, h_w, e_{vw}) = A(e_{vw})\, h_w.$$

The discrete-label matrices are just the special case where $A$ is a lookup table over labels; making $A$ a network lets a continuous bond geometry continuously deform the linear map that mixes the neighbor in, and it is exactly the fix for the fingerprint model's failure to couple edge and node — here the edge multiplicatively gates the node's contribution instead of being added in a separate bin. I also tried a *pair message* $m_{wv} = f(h_w, h_v, e_{vw})$ conditioned on both endpoints, even cooking up a toy pathfinding task designed to reward knowing the receiver; it trained worse even on that task, because conditioning on both endpoints makes the message harder to optimize, so I dropped it and kept the simpler, better $A(e_{vw}) h_w$.

For the readout a plain sum of per-node terms throws away structure in the set of node states, so I want something more expressive that stays permutation-invariant. The set2set encoder is built for exactly this: it runs an LSTM that, at each step, computes content-based attention weights over the set elements, forms an order-invariant weighted sum, and refines a query, so it can attend to global structure while remaining invariant because the attention is over a set, not a sequence. I project each final node state paired with its input features into the set, run set2set, and push the resulting graph embedding through an output network. Two further ingredients address range and cost. Because $T$ rounds (empirically anything past $3$ suffices, up to $8$) give only a few-hop receptive field, distant atoms never inform each other when geometry is withheld; I fix this within the framework by adding a virtual edge type between non-bonded pairs and/or a single latent master node wired to every atom with its own width $d_{\text{master}}$ and update, a global scratchpad through which any two atoms communicate in two hops at cost $O(|E|d^2 + n\, d_{\text{master}}^2)$ rather than the $n^2 d^2$ of a fully connected width-$d$ graph. And because a matrix-multiply propagation step costs $O(n^2 d^2)$, I optionally split each node's state into $k$ copies of width $d/k$, propagate each with its own $M, U$, and mix them back with a shared network $g$ over the concatenation; $k$ copies of $O(n^2 (d/k)^2)$ cost $O(n^2 d^2 / k)$, a factor of $k$ cheaper, and the shared mixer (acting identically at every node) preserves invariance — these are the towers. The input is atom features (element one-hot over H/C/N/O/F, atomic number, acceptor/donor, aromaticity, hybridization, hydrogen count) and a raw 5-dimensional edge vector (Euclidean distance plus bond-type one-hot) for the edge network, with hydrogens made explicit nodes. Training is the obvious recipe — Adam, targets normalized to mean $0$ and variance $1$, minimize MSE but report MAE, early-stop on validation, one model per target since a single shared model across all 13 targets is consistently worse by up to 40%.

The best configuration that falls out of this search is the edge-network message with a tied-GRU update and a set2set readout on graphs with explicit hydrogens, and the core of it fills the three slots of the generic graph model as follows.

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
