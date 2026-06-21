The problem is learning on sets of points embedded in Euclidean space, such as molecules, N-body particles, or latent graph geometries, where the target must respect the symmetries of the Euclidean group E(n): translations, rotations, and reflections, together with arbitrary permutations of the points. Scalar targets like energy must be invariant, while vector targets like future positions or velocities must be equivariant, rotating and translating exactly as the input does. The straightforward baselines all fall short in different ways. Plain message-passing GNNs are permutation equivariant but completely blind to spatial geometry; feeding raw coordinates as features breaks rotation invariance. SchNet fixes this by making messages depend only on invariant interatomic distances, but that collapses the geometry to scalars and can never produce an equivariant vector output. The steerable approaches, Tensor Field Networks and the SE(3)-Transformer, do achieve true equivariance through spherical harmonics and Clebsch-Gordan coefficients, but they are computationally heavy, tied to three dimensions, and difficult to implement. The radial-field update is cheap and equivariant, yet it operates only on coordinates and discards the learned node-feature channel that gives GNNs their flexibility. What is missing is a single layer that is cheap, dimension-agnostic, carries a full feature channel, and can emit both invariant scalars and equivariant vectors.

The method I propose is EGNN, short for E(n) Equivariant Graph Neural Networks. It is a message-passing layer that keeps two quantities per node: invariant scalar features h_i and equivariant coordinates x_i. Both are updated in every layer, and they exchange information through the edge operation. The key insight is to separate where the geometry enters the computation. The edge message is built from quantities that the group leaves alone: the source and target features, any edge attributes, and the squared relative distance ||x_i - x_j||^2. Because distances are unchanged by translation, rotation, or reflection, the message m_ij is E(n)-invariant. The coordinate update, by contrast, must be equivariant, so it is formed as a weighted sum of relative-difference vectors (x_i - x_j). Each weight is an invariant scalar phi_x(m_ij) produced from the message by a small MLP. Under a transformation x -> Qx + g, the shared translation g cancels in every difference, and the orthogonal matrix Q factors cleanly out of the linear combination because the weights are scalars. Adding this displacement to x_i preserves the type-1, equivariant nature of the coordinates. By induction across a stack of layers, the whole network is invariant on h and equivariant on x.

This split is what makes EGNN both cheap and general. All nonlinearities live on invariant channels, inside phi_e, phi_x, and phi_h, so they never need to commute with the group. The coordinate itself is only ever touched by linear combinations with invariant scalar weights, which is the safe equivariant operation. The coordinate weight phi_x is constrained to output a single scalar; a matrix or vector weight would itself transform under Q and would require the expensive steerable machinery to handle correctly. Equivariance lives in the difference vectors, learning lives in the invariant scalars, and they meet only through a scalar-vector product. The coordinate displacement is mean-aggregated over the M-1 contributing neighbors, so its magnitude stays order-one regardless of graph size; the feature messages remain summed as in a standard GNN. For stability, the last layer of phi_x is initialized with a tiny Xavier gain so coordinates barely move at initialization, and the difference vector can optionally be normalized by its length before scaling, which preserves equivariance because the normalization factor is an invariant scalar.

Several natural extensions fit into the same framework without changing the symmetry argument. When an initial velocity is available, it can be gated by an invariant scalar phi_v(h_i) and added to the coordinate velocity before moving the position; when the initial velocity is zero this collapses back to the plain update. Soft edge gates can be learned from the invariant messages to let the network prune or weight its own connectivity. For purely invariant tasks such as molecular energy prediction, the coordinate update can be skipped entirely; in that case the pairwise distance matrix is a complete identifier of the geometry up to E(n), so nothing is lost by using only scalar distances in the messages.

```python
from torch import nn
import torch


class E_GCL(nn.Module):
    """E(n)-equivariant graph-convolutional layer.
    Invariant features h, equivariant coordinates x; updates both."""

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False):
        super().__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg          # 'mean' => C = 1/(M-1)
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1                     # the invariant scalar: ||x_i - x_j||^2

        # phi_e: invariant edge message
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf), act_fn)
        # phi_h: invariant node update (residual applied in node_model)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf), act_fn,
            nn.Linear(hidden_nf, output_nf))
        # phi_x: message -> scalar; last layer init tiny for stability
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)
        if self.attention:                     # phi_inf: soft invariant edge gate
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        out = torch.cat([source, target, radial], dim=1) if edge_attr is None \
            else torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)               # m_ij = phi_e(h_i, h_j, ||x_i-x_j||^2, a_ij)
        if self.attention:
            out = out * self.att_mlp(out)
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))   # m_i = sum_j m_ij
        agg = torch.cat([x, agg], dim=1) if node_attr is None \
            else torch.cat([x, agg, node_attr], dim=1)
        out = self.node_mlp(agg)               # phi_h(h_i, m_i)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)         # (x_i - x_j) * phi_x(m_ij)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))   # C = 1/(M-1)
        else:
            raise Exception('Wrong coords_agg parameter: %s' % self.coords_agg)
        return coord + agg                     # x^{l+1} = x^l + C sum (x_i-x_j) phi_x(m_ij)

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]               # x_i - x_j  (equivariant, type-1)
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)  # ||x_i - x_j||^2  (invariant)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


def unsorted_segment_sum(data, segment_ids, num_segments):
    result = data.new_full((num_segments, data.size(1)), 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result = data.new_full((num_segments, data.size(1)), 0)
    count = data.new_full((num_segments, data.size(1)), 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class EGNN(nn.Module):
    """Embed invariant features, run n_layers EGCLs, read out.
    Returns h (E(n)-invariant) and x (E(n)-equivariant)."""

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0,
                 act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False,
                 normalize=False, tanh=False):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)
        for i in range(n_layers):
            self.add_module("gcl_%d" % i,
                            E_GCL(hidden_nf, hidden_nf, hidden_nf, edges_in_d=in_edge_nf,
                                  act_fn=act_fn, residual=residual, attention=attention,
                                  normalize=normalize, tanh=tanh))

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x
```
