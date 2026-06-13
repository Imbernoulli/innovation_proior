# EGNN (E(n) Equivariant Graph Neural Networks), distilled

EGNN is a message-passing layer for point sets in `R^n` that is permutation equivariant (like a
GNN), `E(n)`-invariant on its scalar node features, and `E(n)`-equivariant on per-node
coordinates — all by construction, with no spherical harmonics and no restriction to three
dimensions. Each node carries an invariant feature `h_i` and an `n`-dimensional coordinate `x_i`;
a layer updates both. Invariance enters the message through the squared relative distance;
equivariance enters the coordinate update as a sum of relative-difference vectors weighted by
invariant scalars.

## Problem it solves

Learning on geometric point sets (3D molecules, N-body systems, point clouds, latent graph
geometries) where targets must respect the Euclidean group `E(n)` = translations + orthogonal
transformations (rotation and reflection), plus permutation of the set. Scalar targets (energy,
labels) must be **invariant**; vector targets (positions, velocities, displacements) must be
**equivariant** (rotate the input by `Q`, the output rotates by the same `Q`). Prior methods each
miss something: invariant message passing (SchNet) cannot emit vectors; steerable methods (Tensor
Field Networks, SE(3)-Transformer) need spherical harmonics and are 3D-only; the coordinate-only
equivariant update (Radial Field) carries no learnable feature channel.

## Key idea

Carry two quantities per node — invariant features `h_i` and equivariant coordinates `x_i` — and
let them exchange information in the edge operation:

- The **message** is fed the squared relative distance `||x_i - x_j||^2`, which is the only
  low-order quantity invariant under translation, rotation, and reflection. So the message
  `m_ij` is `E(n)`-invariant.
- The **coordinate update** is a weighted sum of relative-difference vectors `(x_i - x_j)`, with
  each weight an *invariant scalar* `phi_x(m_ij)`. This is equivariant: under `x -> Q x + g`, the
  shared `g` cancels in the difference and `Q` factors out of the linear combination, while the
  invariant weights are unchanged.

Equivariance lives in the difference vectors; learning lives in the invariant scalars; they meet
only through the product (vector)·(scalar). `phi_x` must output a scalar — a non-scalar weight
would itself transform under `Q` and break the factoring. Nonlinearities act only on invariant
channels, never on `x` directly (a pointwise nonlinearity does not commute with `Q`).

## Final layer (EGCL)

```
m_ij      = phi_e( h_i^l, h_j^l, ||x_i^l - x_j^l||^2, a_ij )          # invariant message
x_i^{l+1} = x_i^l + C * sum_{j != i} (x_i^l - x_j^l) * phi_x(m_ij)    # equivariant coord update
m_i       = sum_{j != i} m_ij                                          # aggregate
h_i^{l+1} = phi_h( h_i^l, m_i )                                        # invariant node update
```

- `phi_e`: 2-layer MLP, two Swish nonlinearities.
- `phi_x`: `R^nf -> R^1`, 2-layer MLP with one nonlinearity; last linear initialized tiny (Xavier
  gain ~0.001) so coordinates barely move at initialization.
- `phi_h`: 2-layer MLP with one nonlinearity and a residual connection `h^{l+1} = h^l + (...)`.
- `C = 1/(M-1)`: averages the coordinate displacement over the `M-1` summed differences, keeping
  the update magnitude `O(1)` independent of graph size (i.e. mean-aggregate the coordinate
  messages; feature messages are summed).
- Optional difference normalization `(x_i - x_j) / (||x_i - x_j|| + eps)` for stability; dividing
  by an invariant scalar preserves equivariance.

## Equivariance (proof)

Assume `h^l` is `E(n)`-invariant (encode no absolute pose into `h^0`). Then:
- `||Q x_i + g - (Q x_j + g)||^2 = ||Q(x_i - x_j)||^2 = (x_i-x_j)^T Q^T Q (x_i-x_j) = ||x_i-x_j||^2`,
  so `m_ij` is invariant.
- `Q x_i + g + C sum (Q x_i + g - [Q x_j + g]) phi_x(m_ij)`
  `= Q x_i + g + C sum Q(x_i-x_j) phi_x(m_ij)`
  `= Q(x_i + C sum (x_i-x_j) phi_x(m_ij)) + g = Q x_i^{l+1} + g`, so the coordinate update is
  equivariant.
- `m_i` and `h_i^{l+1}` depend only on invariants, so `h^{l+1}` is invariant — re-establishing the
  hypothesis. By induction the whole stack is `E(n)`-equivariant on `x`, invariant on `h`.

## Variants

**Velocity / momentum.** Replace the coordinate update with
```
v_i^{l+1} = phi_v(h_i^l) * v_i^init + C * sum_{j != i} (x_i^l - x_j^l) * phi_x(m_ij)
x_i^{l+1} = x_i^l + v_i^{l+1}
```
`phi_v: R^nf -> R^1` is an invariant scalar gating the (type-1) initial velocity. Equivariant by
the same argument; reduces exactly to the plain update when `v^init = 0`.

**Edge inference.** Rewrite aggregation as `m_i = sum_{j != i} e_ij m_ij` and approximate the gate
`e_ij ~= phi_inf(m_ij)`, with `phi_inf` = Linear -> sigmoid (`R^nf -> [0,1]`). A soft, invariant
attention over messages; preserves `E(n)` properties since it only touches invariant messages.
Used to infer connectivity on variable-size graphs.

**Invariant-only mode.** For an invariant target (e.g. molecular energy), skip the coordinate
update entirely — the model becomes plain `E(n)`-invariant message passing on distances.

## Invariant features are all you need (why the simple model suffices for invariant tasks)

When only positions are given, the pairwise distance matrix for a fixed node indexing is a
*unique* identifier of the geometry up to `E(n)`; the ordinary permutation-equivariant graph
machinery handles relabelling. No information is lost by using only the scalar distances.

- *Invariance*: `l2(Q x_i + t, Q x_j + t) = l2(x_i, x_j)` (orthogonal preserves norm; `t` cancels).
- *Uniqueness*: if `l2(x_i, x_j) = l2(y_i, y_j)` for all `i, j`, then `x_i = A y_i + t` for some
  orthogonal `A` and translation `t`. Proof: center both configurations, so `x_0 = y_0 = 0`.
  Then `||x_i|| = ||y_i||`. Expanding
  `||x_i - x_j||^2 = ||x_i||^2 - 2<x_i,x_j> + ||x_j||^2` and equating to the `y` version forces
  `<x_i, x_j> = <y_i, y_j>` for all `i, j`, so the centered configurations have the same Gram
  matrix. Therefore, for any coefficients `c,d`,
  `<sum c_i x_i, sum d_j x_j> = <sum c_i y_i, sum d_j y_j>` (*). Pick a basis `{y_{i_j}}` of
  `span{y_i}`; the matching `{x_{i_j}}` is independent by (*). Define `A y_{i_j} = x_{i_j}`.
  For `y_i = sum_j c_j y_{i_j}`, expanding `||x_i - sum_j c_j x_{i_j}||^2` and converting every
  inner product by (*) gives `<y_i,y_i> - 2<y_i,y_i> + <y_i,y_i> = 0`, so `A y_i = x_i`.
  For arbitrary `u,v` in `span{y_i}`, (*) gives `<A u, A v> = <u, v>`, so `A` is an isometry on
  the span; extend it orthogonally to the complement. Undo centering with
  `t = x_0 - A y_0` using the original anchors. QED.

So higher-order (type-`l`) representations and even the difference vectors carry no geometric
information beyond the scalar distances for an `E(n)`-invariant target — the distance-only model
is justified, not a compromise.

## Working code

The clean PyTorch implementation uses the squared distance in the edge message, a scalar
coordinate MLP with tiny final-layer initialization, mean aggregation for coordinate
displacements, summed feature messages, and a residual feature update. A PyTorch-Geometric port
used in ProteinWorkshop keeps the same invariant/equivariant split but feeds `||pos_i-pos_j||`
instead of the squared distance, scales the displacement as
`(pos_i-pos_j)/(||pos_i-pos_j||+1) * mlp_pos(msg)`, mean-aggregates the coordinate displacement,
and applies residual additions to both node features and positions in the outer model.

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

        # phi_e
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf), act_fn)
        # phi_h (with residual applied in node_model)
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

Typical settings: N-body / graph-autoencoder — 4 layers, 64 hidden, Swish, Adam; QM9 — 7 layers,
128 hidden, edge inference (`attention=True`), coordinate update skipped (invariant mode),
sum-pool readout, Adam with cosine learning-rate decay.
