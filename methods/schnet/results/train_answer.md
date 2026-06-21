I want one model that takes a molecule — atoms with nuclear charges $Z_1,\dots,Z_n$ at positions $r_1,\dots,r_n \in \mathbb{R}^3$ — and predicts both its total potential energy $E(r_1,\dots,r_n)$ and the forces on every atom, and I want it to generalize across both chemical variation (different compositions and sizes) and conformational variation (the same molecule far from equilibrium). The forces are not a target I get to fit freely: they are $F_i = -\partial E/\partial r_i$, the negative gradient of the energy. That one fact disciplines the whole design. If I let a network emit a force vector per atom decoupled from any scalar, nothing stops the field from having nonzero curl, and a curl-bearing force field is not conservative — I could march the atoms around a closed loop in configuration space and return with more energy than I left with, which is nonsense. So I insist on a scalar energy and let the forces be its gradient; then curl-freeness and energy conservation come for free, since the gradient of a scalar always has zero curl. The energy is a scalar observable, so it must be invariant to rotation, translation, and atom relabeling, and the force, being the gradient of an invariant scalar, is automatically rotationally equivariant. The whole target therefore collapses to building a permutation- and rotation-invariant scalar function of the geometry and letting autodiff hand me the forces. The constraint that quietly kills most obvious designs is smoothness: I will put the force, already $\partial E/\partial r$, inside a loss and descend it, which differentiates the energy a second time, so $E$ must be twice differentiable everywhere. Any kink or jump in $E(R)$ becomes a spike or a hole in the force, and force training falls apart there.

Measured against these three constraints, every existing camp leaves a gap. The physically careful descriptor methods (Behler–Parrinello high-dimensional NN potentials, and Gaussian approximation potentials) write $E = \sum_i E_i$ with each atomic contribution depending only on neighbors within a cutoff $r_{\text{cut}}$, and feed each atom's network a fixed vector of hand-crafted atom-centered symmetry functions, smoothed by a cosine cutoff. These are invariant, smooth, energy-conserving, and linear-scaling — but the descriptors are hand-built and frozen; a practitioner must choose the radial widths, angular resolutions, and cutoff, and a good set for one chemistry is not automatically right for another. The representation is prescribed, not learned. The deep-tensor line learns it instead, starting each atom from a learnable type embedding and refining it as $c_i^{t+1} = c_i^t + \sum_{j\neq i} v_{ij}$, but distance enters through an opaque factorized bilinear tensor coupling with no spatial-filter reading, it shares interaction parameters across passes, and its accuracy trails the best message-passing models. The message-passing framework is the cleanest statement of the learned idea — $m_v = \sum_{w} M(h_v,h_w,e_{vw})$ then $h_v \leftarrow U(h_v,m_v)$ — and its strongest molecular instance generates a full $F\times F$ matrix $A(e_{vw})$ per edge and sends $A(e_{vw})\,h_w$; it takes the QM9 benchmark. But its edge features include one-hot bond types, and the instant a bond stretches and its character flips, that one-hot input jumps and the energy jumps with it: the surface is discontinuous, so the model cannot give a smooth PES, cannot be trained on forces, and cannot do molecular dynamics. The gradient-domain kernel method is energy-conserving by construction and stunningly data-efficient, but its Gram matrix grows quadratically in both atoms and examples, so it does not scale to large data or span compositions. The tension is now sharp: the descriptor methods are smooth, physical, and scalable but not learned; the learned graph nets are scalable and learned but tie geometry to discrete, jumpy inputs that wreck the surface.

I propose SchNet, built around what I call the continuous-filter convolution (cfconv). The root cause of the discreteness is that an ordinary convolutional filter is a lookup table $W[\Delta]$ indexed by integer grid offsets — pixels, audio samples, evenly spaced — and atoms live at arbitrary real positions, so any attempt to put them on a grid makes an atom's contribution snap from one filter tap to the next as it crosses a voxel boundary; the bond-type one-hot is the same disease in another coordinate. So I replace the filter tensor by a filter-generating network $W^l$ that maps a continuous offset to a vector of filter values, and convolve at atom $i$ as a sum over the others,
$$x_i^{l+1} = \big(X^l * W^l\big)_i = \sum_j x_j^l \circ W^l(r_i - r_j),$$
where $\circ$ is an elementwise product. There is no grid: an atom at any position contributes through $W$ evaluated at its exact offset, and as it moves $W$ moves with it continuously, with no binning discontinuity. I make $W$ act feature-wise — a vector in $\mathbb{R}^F$ multiplied channel by channel into $x_j$ — rather than as a full $F\times F$ matrix, which would cost $F^2$ per edge and recreate the heavy edge-network message; this is the depthwise-separable split, doing the spatial part cheaply per channel and leaving cross-feature mixing to ordinary atom-wise dense layers, so the cfconv carries the geometry and the dense layers recombine channels, at linear cost in $F$. The raw displacement is not rotation-invariant, so I feed the filter network only the distance $d_{ij} = \lVert r_i - r_j\rVert$; then $E$ is invariant and $F = -\nabla E$ is equivariant and curl-free by construction. Read this way, the cfconv message $x_j \circ W(d_{ij})$ is exactly the diagonal, continuous-distance special case of the edge-network message $A(e)\,h_w$: I have replaced the full matrix by an elementwise filter and the discrete edge feature by a continuous distance, and that single move buys both smoothness and cheapness.

A bare scalar fed to the filter network is a trap. At initialization a network is nearly linear, so each of the $F$ output channels is approximately the same linear ramp in the single input $d_{ij}$, just scaled — the filters are almost identical, carrying one degree of freedom instead of $F$, and training stalls on a plateau with no filter diversity to exploit. I decorrelate the channels by lifting the distance into a bank of localized radial basis functions before the dense layers,
$$e_k(d_{ij}) = \exp\!\big(-\gamma\,(d_{ij}-\mu_k)^2\big),$$
with centers $\mu_k$ on a uniform grid from $0$ out to the cutoff. A given distance now lights up only the few Gaussians whose centers are near it, so different output channels of the filter net latch onto different distance ranges and the filters start diverse even from a near-linear net; the plateau disappears. This gives two interpretable knobs for free — the number of centers is the filter resolution, the span of the centers is the filter size — and if the centers sit at spacing $\Delta$ I match the width with $\gamma = 1/(2\Delta^2)$ so neighboring Gaussians overlap smoothly. The expansion is itself $C^\infty$ in $d$, so no discontinuity is reintroduced.

The nonlinearity matters because I differentiate this network twice. ReLU has a corner — its second derivative is a delta at zero — so it would put a discontinuity in the force exactly at the kinks. I use the shifted softplus,
$$\mathrm{ssp}(x) = \mathrm{softplus}(x) - \ln 2 = \ln\!\big(0.5\,e^x + 0.5\big),$$
which is $C^\infty$ everywhere so the twice-differentiability holds throughout the energy network, and which satisfies $\mathrm{ssp}(0)=0$ (since plain softplus gives $\ln 2$ at zero, biasing the activations) so the activations stay centered. I use it in the filter net and in every atom-wise layer, because each sits inside the energy I differentiate twice. Each atom begins from a learnable per-type embedding $x_i^0 = a_{Z_i}$, and the network then interleaves atom-wise dense layers — shared across atoms, hence permutation-equivariant and size-agnostic, doing the cross-feature recombination the feature-wise cfconv deliberately omits — with cfconv layers that do the geometry. I wrap each geometric update in a residual connection, $x_i^{l+1} = x_i^l + v_i^l$, so each block learns only a correction and a deep stack stays trainable, and I do not share weights across blocks: distinct parameters let earlier blocks build short-range structure and later blocks build on it. This is what manufactures many-body terms from strictly pairwise radial filters — after one block $x_j$ already encodes $j$'s own neighbors, so when $i$ pulls in $x_j$ in the next block it implicitly feels the $(i,j,k)$ triple, and a few blocks reach several hops out while every individual filter only ever sees one invariant distance, so the invariance is never spent.

A hard neighbor cutoff would undo all of this: as an atom drifts across $r_{\text{cut}}$ during dynamics its contribution would snap to zero, a jump in $E$ and a delta in $F$. So I fold a Behler-style cosine cutoff into the filter,
$$f_{\text{cut}}(d) = 0.5\,\big[1 + \cos(\pi d / r_{\text{cut}})\big]\quad (d < r_{\text{cut}}),\qquad 0\ \ (d \ge r_{\text{cut}}),$$
whose value $\cos(\pi)=-1$ and slope $-(\pi/2r_{\text{cut}})\sin(\pi d/r_{\text{cut}})$ are both zero at $r_{\text{cut}}$, so the effective filter $W_{\text{eff}}(d) = \text{filter\_net}(e(d))\cdot f_{\text{cut}}(d)$ makes a neighbor fade out smoothly; the energy stays $C^1$ across the boundary and the bounded neighborhood keeps cost linear. The forces are then the analytic gradient $\hat F_i = -\partial \hat E/\partial r_i$ from autodiff, which is curl-free, energy-conserving, and equivariant for free. To train on energies and forces together I use
$$\ell(\hat E,(E,F_1,\dots,F_n)) = \rho\,\lVert E - \hat E\rVert^2 + \frac{1}{n}\sum_i \big\lVert F_i - (-\partial\hat E/\partial r_i)\big\rVert^2,$$
with $\rho \approx 0.01$ down-weighting the energy term, since one energy per molecule and $3n$ force components live on very different scales; the force model is effectively twice the depth of the energy model. The energy is extensive, so an atom-wise output head maps each final representation to a scalar contribution and I sum over atoms (an intensive property would average), and I optimize with Adam at an initial learning rate of $10^{-3}$ with exponential decay, small molecule minibatches, and an exponential moving average over weights for the test-time model. The identical geometric core transfers directly to encoding protein structure from alpha-carbon coordinates: distances stay rotation-invariant, the residual stack still builds multi-scale structure from local radial filters, and only two things change, both forced by the data — the type embedding becomes a lazy linear map over per-node feature vectors, and distances are computed on a supplied spatial neighbor graph over residues rather than on a radius graph the energy model builds — producing invariant per-node and per-graph embeddings.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi as PI
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class GaussianSmearing(nn.Module):
    """e_k(d) = exp(-(d - mu_k)^2 / (2 Delta^2)); centers on a uniform grid 0..cutoff."""
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * dist.pow(2))


class ShiftedSoftplus(nn.Module):
    """ssp(x) = softplus(x) - ln 2; C-infinity, ssp(0) = 0."""
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    """Continuous-filter convolution. Message j->i: x_j (projected) (*) W(d_ij),
    W(d) = filter_net(e(d)) * cosine_cutoff(d), summed over neighbors."""
    def __init__(self, in_channels, out_channels, num_filters, filter_net, cutoff):
        super().__init__(aggr="add")
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.filter_net = filter_net
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)   # smooth cutoff
        W = self.filter_net(edge_attr) * C.view(-1, 1)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.filter_net, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class ProteinEncoder(nn.Module):
    """SchNet continuous-filter encoder over alpha-carbon coordinates."""
    def __init__(self, hidden_channels=128, out_dim=1, num_layers=6,
                 num_filters=128, num_gaussians=50, cutoff=10.0, readout="add"):
        super().__init__()
        self.cutoff = cutoff
        self.readout = readout
        self.embedding = nn.LazyLinear(hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            for _ in range(num_layers)
        ])
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.LazyLinear(out_dim)

    def forward(self, batch):
        edge_index = batch.edge_index
        u, v = edge_index
        edge_weight = (batch.pos[u] - batch.pos[v]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        h = self.embedding(batch.x)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        h = self.lin1(h)
        h = self.act(h)
        node_emb = self.lin2(h)
        graph_emb = scatter(node_emb, batch.batch, dim=0, reduce=self.readout)
        return {"node_embedding": node_emb, "graph_embedding": graph_emb}
```

For the molecular energy/force model, replace the lazy node-feature embedding with a per-type `nn.Embedding`, build the graph with a radius graph at `cutoff`, keep the scalar output head `hidden_channels -> hidden_channels // 2 -> 1`, pool with sum/mean to obtain `E`, and obtain forces as `-torch.autograd.grad(E, pos)`.
