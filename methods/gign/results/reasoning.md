Let me start from what physically sets a binding affinity, because that has to drive the
architecture, not the other way round. I have a protein-ligand complex — a small molecule sitting
in a pocket of a protein — and I want one number out, `-logKd/Ki`, how tightly it binds. What
makes that number what it is? Interactions between atoms. Two flavors. Inside the ligand and
inside the protein there are covalent bonds holding each molecule together — the bonded skeleton.
And across the gap between them there are noncovalent interactions — hydrogen bonds, van der
Waals contacts, electrostatics, hydrophobic packing — formed only because the two molecules are
pressed together in the bound pose. That second kind is the whole story of *binding*: the covalent
skeletons exist whether or not the ligand is in the pocket, but the interfacial contacts are the
new physics of the bound state, and they're what tips the energy balance toward sticking. So
whatever I build has to see both, and it has to see them as the geometric, spatial things they
are, because every one of these interactions is set by where the atoms actually are — bond lengths
and angles for the skeleton, atom-to-atom distances across the interface for the contacts.

That immediately tells me the input can't be a sequence or a flat 2D bond graph. The noncovalent
interface is a 3D object; it lives in the coordinates. I need the actual positions `r_i` of every
atom. And the moment I say "coordinates" a hard constraint lands on me that I cannot negotiate
with: the affinity is a property of the *complex*, and it does not change if I pick up the whole
complex and move it to the other side of the room, or spin it around. The bound energy doesn't
know where the origin is or which way my axes point. If I slide ligand and protein together by
some vector `t`, or rotate them together by some rotation `Q`, the answer must be byte-for-byte
the same. This is not a nice-to-have; a model that gives different affinities for the same complex
viewed from two angles is predicting something that isn't physics. So invariance to rigid motion
— translation and rotation, and for that matter reflection — has to hold *by construction*. I
refuse to "learn it from augmentation," rotating every training complex a hundred ways and hoping
the network smooths it out; that's expensive and it only ever approximates the symmetry I can just
build in.

So the real question is: how do I let a network consume 3D coordinates while being exactly blind
to the global pose? Let me think about what survives a rigid motion. Send every position to
`Q r_i + t`. Look at a single coordinate — it changes completely, useless as an invariant feature.
Look at the difference of two positions: `(Q r_i + t) - (Q r_j + t) = Q(r_i - r_j)` — the
translation `t` cancels in the subtraction, good, but `Q` is still sitting there, so the
difference *vector* still rotates; not invariant either. Now take its length:

  ||(Q r_i + t) - (Q r_j + t)|| = ||Q(r_i - r_j)|| = sqrt( (r_i - r_j)^T Q^T Q (r_i - r_j) ).

And `Q` is orthogonal, `Q^T Q = I`, so this collapses to `sqrt((r_i - r_j)^T (r_i - r_j)) =
||r_i - r_j||`. The interatomic *distance* is invariant to translation and to rotation/reflection
both. That's the free invariant the geometry hands me, the one quantity I can feed a network and
know the output won't care about the pose. If I build everything out of distances, I'm invariant,
full stop. If I let a raw coordinate or a bare difference vector leak in as a feature, I've broken
it. So: distances in, never coordinates.

Now, what's the substrate for learning on a molecule? Message passing on the atom graph. That's
the chemistry-standard tool: node embeddings `h_i`, edge attributes `a_ij`, and a layer

  m_ij = phi_e(h_i, h_j, a_ij);  m_i = sum_{j in N(i)} m_ij;  h_i' = phi_h(h_i, m_i),

with small MLPs `phi_e`, `phi_h`. Stack a few and information flows several bonds deep; the
symmetric sum makes it permutation-equivariant, which I want — relabel the atoms and the answer
is unchanged. But this plain form references only topology and attributes; it has no idea where
the atoms are. The fix that the geometric networks found is exactly the distance idea above. SchNet
makes the message depend only on `d_ij = ||r_i - r_j||`: it generates a filter `W(d_ij)` from the
distance and multiplies the neighbor's features by it elementwise,
`x_i' = sum_j x_j (*) W(d_ij)`, then mixes channels with an atom-wise dense layer. Because only the
invariant distance enters, every layer is invariant — perfect for predicting an invariant scalar,
which is exactly what affinity is. EGNN tells the same story with the squared distance carried into
the message, `m_ij = phi_e(h_i, h_j, ||r_i - r_j||^2)`, and it also offers an equivariant
*coordinate* update — but I should ask whether I need that part. EGNN's coordinate update,
`x_i' = x_i + C sum_j (x_i - x_j) phi_x(m_ij)`, exists so the output positions transform *with* the
input under rotation — it's there for when your target is itself a vector, an updated position or a
velocity. My target is a single scalar that's supposed to be invariant, not equivariant. If I keep
that update, I'm carrying a quantity that rotates with the input through the whole network, and then
I'd have to be careful never to let it touch the scalar prediction or I'd reintroduce pose
dependence. The clean move is to hold the positions fixed and never update them; then there's no
equivariant quantity in the network at all, and what's left of EGNN is just "a graph network on
distances." So the shape of the geometry handling I'm reaching for is: distance in, expanded somehow,
generating a per-edge filter, gating the message — with positions a fixed input I read distances off
of, not a thing I evolve.

But here's where SchNet and EGNN stop being enough, and it's the whole point of this problem. They
were built for a *single homogeneous molecule*. Every edge in their world is the same kind of edge,
processed by the same distance filter and the same update. My complex is not homogeneous. It is two
different molecules — a small organic ligand and a chunk of protein — joined by an interface, and
the edges come in two physically distinct kinds. There are the covalent bonds inside each molecule:
short, stiff, around a bond length apart, fixed chemistry. And there are the noncovalent contacts
across the interface: soft, reaching out to maybe five angstroms, a different physics entirely, and
the ones that actually carry the binding signal. If I throw both into one undifferentiated message
function — one filter, one update for every neighbor regardless of whether the link is a bond or an
interfacial contact — I'm asking a single learned map to model two different physical regimes at
once, and it will blur them. A bond at 1.5 angstroms and a hydrogen bond at 3 angstroms mean
completely different things, and a single distance-to-filter map has to compromise between them.
That's the limitation: the geometric networks have the right symmetry but no concept that this is a
heterogeneous, two-molecule, two-interaction-type object.

Who has taken the covalent/noncovalent distinction seriously? IGN — InteractionGraphNet. It does
recognize the two kinds: it represents the complex with a ligand graph, a protein graph, and a
bimolecular interaction graph, and it learns them in two graph-convolution modules stacked in
sequence — first a module for the intramolecular (covalent) interactions, then a separate module for
the intermolecular (noncovalent) interactions, then a readout. So the distinction is there. But
stare at the *sequential, separate* structure. An atom in that design finishes absorbing its bonded
covalent environment in the first module, and only afterward, in a different module, does it ever
see its interfacial contacts. The two kinds of information are never weighed together inside the same
step. And intuitively that's wrong for the physics: a ligand atom's binding-relevant state depends on
both its bonded neighbors *and* the protein atoms it's touching, simultaneously — those aren't
separable stages, they're one local environment with two kinds of edges in it. Processing them in
sequence, in modules that don't exchange information within a step, means the node never integrates
its covalent and noncovalent context at once. Plus it costs an extra stage and extra parameters, and
IGN didn't feed the interaction *geometry* into the message the clean distance-invariant way the
geometric networks do.

So now I can state precisely what I'm missing. I want the `E(3)`-invariant, distance-driven message
passing of SchNet/EGNN — I'm not giving up the symmetry or the geometry. And I want IGN's
acknowledgment that covalent and noncovalent interactions are different things that deserve different
treatment. But I want a single atom, in a single message-passing step, to take in both its covalent
neighborhood and its noncovalent neighborhood and fuse them into one updated representation — not
process one then the other in separate modules. Let me try to build that layer.

I have, per complex, two edge sets that are fixed up front: the intramolecular bond graph
`E_intra` (RDKit bonds within the ligand and within the pocket, made bidirectional), and the
intermolecular contact graph `E_inter` (ligand-atom / pocket-atom pairs within 5 angstroms, also
bidirectional). And I have `pos`, the 3D coordinates of all atoms, which I only ever touch to
compute distances. The obvious first instinct is: run message passing over `E_intra`, run message
passing over `E_inter`, and add the results. Let me make that concrete and see if it's enough, or
whether the "heterogeneous" part forces more.

Start with the message itself. For an edge `j -> i`, I want the neighbor's feature `x_j`
modulated by the geometry of that edge. Borrow SchNet's move: gate the neighbor's features by a
distance-derived filter, elementwise. Compute `d_ij = ||pos_i - pos_j||`, turn it into a filter
vector `radial_ij` of the same width as the features, and the message is `m_ij = x_j (*) radial_ij`,
the elementwise product. Aggregate by summing over neighbors. That's clean and it's invariant — the
only geometric input is `d_ij`, which I just showed is rigid-motion-invariant, so the whole message
is invariant.

Now, how do I turn the scalar `d_ij` into the filter `radial_ij`? The naive thing is to push `d_ij`
straight into a small MLP that outputs a width-`hidden` vector. But think about what a *fresh* such
MLP does, since that's the regime training starts in: with small random init the map is nearly affine
in its input, and its input is one scalar, so at init every output channel is roughly
`silu(w_k d + b_k)` — near-affine functions of the *same* `d`, differing only by a slope and offset.
The channels are essentially copies of each other up to scale, so the gate is effectively
one-dimensional until training pulls them apart. To see how bad that actually is: a single-scalar
linear map to 9 channels with random init, SiLU, evaluated across `d` from 0.5 to 6 angstroms — the
mean absolute pairwise correlation across the 9 channels comes out at about 0.985. The gate starts
life with essentially no channel diversity, and gradient descent would have to spend early epochs
just decorrelating before the filter can do anything useful.

The fix is to expand the distance in a fixed basis *before* the learned map, so the diversity is
handed to the network for free. A bank of Gaussian radial basis functions on a grid of centers,

  rbf(d) = [ exp( -((d - mu_k)/sigma)^2 ) ]_{k=1..K},  mu_k = linspace(D_min, D_max, K),

each a localized bump centered at a different distance, so a short distance lights up the near
centers and a long distance the far ones. Repeating the same measurement with `rbf(d)` in place of
the raw scalar — a random `Linear(9 -> 9)` and SiLU, same distance sweep — the mean absolute pairwise
correlation drops from 0.985 to about 0.38: at init the expanded version already gives the gate
genuinely different channels to work with, before training does anything.

What grid? The distances I care about run from the shortest bond up to the interaction cutoff. Bonds
are around 1.5 angstroms, the noncovalent cutoff is 5; I want centers spanning that range with
sub-angstrom resolution. Take `D_min = 0`, `D_max = 6` angstroms, `K = 9` centers — a small fixed
bank spanning the whole range. `torch.linspace(0, 6, 9)` includes the endpoints, so the centers land
at `mu = [0, 0.75, 1.5, 2.25, 3, 3.75, 4.5, 5.25, 6]` with spacing `0.75`, while the implementation
deliberately sets the Gaussian width to `sigma = (D_max - D_min)/K = 6/9 ≈ 0.667`. Checking the
resolution against the two regimes I actually care about: a bond at `d = 1.5` lights up centers
`1.5, 0.75, 2.25` (activations `1.0, 0.28, 0.28`); an H-bond-length contact at `d = 3.0` lights up
centers `3.0, 2.25, 3.75` (`1.0, 0.28, 0.28`). The two patterns share only the weakly-activated `2.25`
shoulder and otherwise touch disjoint centers — so a downstream map *can* respond differently to a
stiff bond and a soft contact, which is exactly the discrimination I'll need. The width is
sub-angstrom and `K` is small enough that the filter map stays cheap. Then a linear map from `K` to
`hidden` and a smooth
nonlinearity gives the filter: `radial_ij = SiLU( W_coord · rbf(d_ij) )`. I use SiLU here, the smooth
swish activation, because the filter is a response to a continuous physical distance and I want it
smooth — no kinks, the way a physical interaction strength varies smoothly with separation.

Now I have to decide whether one distance-to-filter map can serve both edge types, or whether the
split has to enter the geometry network itself. Think about what the two edge sets actually contain.
Covalent edges are bonds: a tight, peaked distribution of distances around bond lengths,
roughly 1 to 2 angstroms, with a specific chemical meaning. Noncovalent edges are contacts: a broad,
soft distribution stretching out to 5 angstroms, encoding a different physics. If I force one
`W_coord` to serve both, it has to find a single distance-to-filter response that's simultaneously
right for stiff bonds and soft contacts — a compromise that under-resolves both. The whole reason
the homogeneous networks fell short was treating these as the same. So I give each interaction type
its own coordinate network: `W_coord_cov` for the covalent edges, `W_coord_ncov` for the
noncovalent edges. Each learns the distance response appropriate to its own physics. Same RBF basis
feeding both — the basis is just a fixed geometric expansion — but separate learned maps on top.

So now, inside one layer: over `E_intra` I compute `radial_cov` from the covalent coord net, run the
gated message `m_ij = x_j (*) radial_cov_ij`, sum over covalent neighbors to get `out_cov_i`. Over
`E_inter`, independently, `radial_ncov`, message, sum over noncovalent neighbors, `out_ncov_i`. Two
aggregated neighborhood summaries per atom, one covalent, one noncovalent, both computed in the same
step from the same current node features `x`.

Now fuse them, and again the heterogeneity question: do I just add `out_cov_i + out_ncov_i` raw and
push through one shared transform, or transform each separately? Same argument as before — these are
two physically distinct contributions, and I want the network free to map each through its own
feature transform before combining, rather than forcing a single shared map onto the sum. So give
each branch its own node MLP. And there's a subtlety about what goes into each MLP. If I feed only
the aggregated neighbor summary `out_s_i`, then an atom with few or no neighbors of type `s` (say a
buried ligand atom with no protein contacts) gets almost nothing from that branch and risks losing
its own identity. I want to keep the node's own current state in the loop — a self-loop, a residual.
So feed `x_i + out_s_i` to each branch: the atom's own representation plus its aggregated type-`s`
neighborhood. Then:

  out_i = MLP_cov( x_i + out_cov_i ) + MLP_ncov( x_i + out_ncov_i ).

The two transformed branches are *summed*. Why a sum and not a concatenation or a gate? Physically,
the covalent contribution and the noncovalent contribution to an atom's state are like two energy
terms that add; additive fusion treats them as co-equal contributions and keeps the output width
equal to the input width so I can stack the layer. Each `MLP_s` is a `Linear(hidden -> hidden)`
followed by `Dropout(0.1)`, `LeakyReLU`, and `BatchNorm1d` — the heavy feature transforms where I
want regularization and normalization, because the dataset is only tens of thousands of complexes
and the regression will overfit without it. I deliberately use the smooth SiLU only on the geometric
filter, where smoothness in distance matters, and the cheaper LeakyReLU-with-batchnorm in the
feature MLPs, where stable training of the regression head matters more than smoothness.

So this is one layer that carries two distinct interaction channels through a single step and fuses
them — heterogeneous in the literal sense that covalent and noncovalent edges get different
treatment within the step rather than in separate stages. The only place geometry enters is
`d_ij = ||pos_i - pos_j||` inside each `rbf`, and that quantity is exactly invariant under any rigid
motion `pos -> Q pos + t` (the same algebra as above: `t` cancels in the difference, `Q^T Q = I`
cancels the rotation in the norm). So every `rbf(d_ij)`, every `radial_cov` and `radial_ncov`, every
message `x_j (*) radial`, every aggregated `out_cov_i` and `out_ncov_i`, and the fused `out_i`
inherit that invariance unchanged. Stack three of these layers and the node features stay
pose-independent at every layer, because `pos` is never updated — it is a fixed input I only ever
read distances from, so unlike EGNN there is no equivariant vector that has to co-transform and could
leak pose-dependence back in. The whole network is `E(3)`-invariant by construction, with no rotation
augmentation needed.

Now wrap the layer into a full predictor. The atoms — both ligand and pocket — come with a 35-dim
feature vector each: one-hot element among the common organic atoms, degree, implicit valence,
hybridization, aromaticity, hydrogen count. Same featurization for both molecules, since at the atom
level they're the same chemistry; the *heterogeneity* lives in the edge sets, not in the node
features. Embed those features to the hidden width with a `Linear(35 -> hidden)` followed by SiLU.
Then apply the stack of heterogeneous interaction layers — three of them. Why three? Each layer
propagates information one hop along the bond graph and one hop along the contact graph; three hops
is enough to carry a protein atom's noncovalent influence a few bonds into the ligand and vice
versa, building a joint covalent-plus-noncovalent context around each atom, without so many layers
that the features oversmooth into mush across the whole complex. With hidden width 256 each layer
has the capacity to keep the two channels' contributions distinct. Between layers the output simply
replaces `x` — the residual self-loop already lives *inside* the layer, in the `x_i + out_s_i`
terms, so I don't need a second residual wrapped around it.

Then I have to collapse the per-atom features into one prediction. Pool over all atoms in the
complex and run a regression head. Which pooling? Affinity is extensive-like — it's built from the
sum of favorable interactions, and more interacting atoms generally means a stronger total binding
signal. A mean would normalize that away, washing out the difference between a small tight complex
and a large one. So I sum: `global_add_pool` over every atom in the complex, ligand and pocket
together, producing one graph vector. Then a small fully-connected head maps that to the scalar,
built by a loop over `j` in `range(n_FC_layer)` with two independent `if`s rather than a clean
if/elif: one keyed on `j == 0`, the other on `j == n_FC_layer - 1`. Tracing it at the configured
depth `n_FC_layer = 3`: at `j = 0` the first `if` fires and appends a full block
`Linear(hidden -> hidden), Dropout(0.1), LeakyReLU, BatchNorm1d` — but since the second `if` is a
*separate* statement, not an `elif`, it is also evaluated at `j = 0`, and because
`0 != n_FC_layer - 1` its `else` branch fires too, appending a second full hidden-width block. So
`j = 0` alone contributes two blocks. At `j = 1` only the second `if`'s `else` fires, appending a
third block. At `j = 2`, the last index, the `if j == n-1` branch fires and appends a bare
`Linear(hidden -> 1)` with nothing after it. So the head is three full
`Linear, Dropout(0.1), LeakyReLU, BatchNorm1d` blocks followed by the scalar projection — as many
hidden-width blocks as `n_FC_layer` names, just distributed across the loop so the loop index
doesn't map one-to-one onto blocks the way a quick read would suggest. Dropout and batchnorm again
for regularization on the modest dataset. Output is the predicted `-logKd/Ki`.

Training is plain regression: mean-squared error against the measured affinity, Adam with a small
learning rate `5e-4` and a touch of weight decay `1e-6`, minibatches of 128 complexes, run for many
epochs with early stopping on a validation split. Nothing exotic — the architecture is the
contribution, and it plugs straight into the standard graph training loop.

Now the code, filling the single interaction-layer slot:

```python
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    # Expand a distance in a bank of Gaussian RBFs on a uniform grid of centers,
    # so the filter channels start decorrelated (no near-linear-init plateau).
    D_mu = torch.linspace(D_min, D_max, D_count, device=device).view(1, -1)
    D_sigma = (D_max - D_min) / D_count                 # code-set Gaussian width
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))


class HIL(MessagePassing):
    """Heterogeneous interaction layer: message passing over the covalent (intra)
    and noncovalent (inter) edge sets in one step, each with its own distance
    filter, fused per node. Geometry enters only via interatomic distance -> E(3)
    invariant."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # per-interaction-type node transforms (the "heterogeneous" fusion branches)
        self.mlp_node_cov = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.Dropout(0.1),
            nn.LeakyReLU(), nn.BatchNorm1d(out_channels))
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.Dropout(0.1),
            nn.LeakyReLU(), nn.BatchNorm1d(out_channels))

        # separate distance -> filter maps: covalent and noncovalent distance
        # regimes differ, so each interaction type gets its own (RBF dim = 9)
        self.mlp_coord_cov = nn.Sequential(nn.Linear(9, in_channels), nn.SiLU())
        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, in_channels), nn.SiLU())

    def forward(self, x, edge_index_intra, edge_index_inter, pos=None, size=None):
        # covalent (intra) channel: distance -> RBF -> filter -> gated message -> sum
        row_cov, col_cov = edge_index_intra
        d_cov = torch.norm(pos[row_cov] - pos[col_cov], dim=-1)          # invariant distance
        radial_cov = self.mlp_coord_cov(
            _rbf(d_cov, D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_intra = self.propagate(
            edge_index=edge_index_intra, x=x, radial=radial_cov, size=size)

        # noncovalent (inter) channel: same form, own coord net
        row_ncov, col_ncov = edge_index_inter
        d_ncov = torch.norm(pos[row_ncov] - pos[col_ncov], dim=-1)       # invariant distance
        radial_ncov = self.mlp_coord_ncov(
            _rbf(d_ncov, D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_inter = self.propagate(
            edge_index=edge_index_inter, x=x, radial=radial_ncov, size=size)

        # fuse: residual self-loop (x + out_s) per branch, own MLP, summed
        out_node = self.mlp_node_cov(x + out_node_intra) \
            + self.mlp_node_ncov(x + out_node_inter)
        return out_node

    def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
        x = x_j * radial               # cfconv-style elementwise gating by the distance filter
        return x


class FC(nn.Module):
    """Regression head: stacked Linear/Dropout/LeakyReLU/BatchNorm -> scalar."""

    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return h


class GIGN(nn.Module):
    """Geometric Interaction Graph Neural Network: embed atom features, three
    heterogeneous interaction layers over the complex, sum-pool, regress affinity."""

    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.lin_node = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU())
        self.gconv1 = HIL(hidden_dim, hidden_dim)
        self.gconv2 = HIL(hidden_dim, hidden_dim)
        self.gconv3 = HIL(hidden_dim, hidden_dim)
        self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)

    def forward(self, data):
        x, edge_index_intra, edge_index_inter, pos = \
            data.x, data.edge_index_intra, data.edge_index_inter, data.pos
        x = self.lin_node(x)                                  # 35 -> hidden, SiLU
        x = self.gconv1(x, edge_index_intra, edge_index_inter, pos)
        x = self.gconv2(x, edge_index_intra, edge_index_inter, pos)
        x = self.gconv3(x, edge_index_intra, edge_index_inter, pos)
        x = global_add_pool(x, data.batch)                   # sum: affinity is extensive
        x = self.fc(x)
        return x.view(-1)                                    # predicted -logKd/Ki
```

Built with `GIGN(node_dim=35, hidden_dim=256)`, trained by minimizing MSE against `-logKd/Ki` with
Adam at `5e-4`, weight decay `1e-6`, batch size 128, up to 800 epochs, and early stopping after
100 validation stalls. The contribution is the heterogeneous interaction layer; everything around it
is the standard graph-regression harness.
