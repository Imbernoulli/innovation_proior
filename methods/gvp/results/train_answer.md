Protein design from backbone structure asks us to read the three-dimensional arrangement of a protein chain and predict, for every position, which of the twenty standard amino acids belongs there. This inverse-folding problem has two inseparable faces. One is geometric: residues point in particular directions, backbone segments bend and twist, and neighboring atoms occupy specific locations in space. The other is relational: which residues contact one another, how interactions are wired across the chain, and the sequential order of residues all matter. Any useful model must honor both. Worse, the correct answer must not change if the entire molecule is rotated or reflected, so the final per-residue predictions must be invariant to rigid motions.

Existing architectures each capture only one face and pay a price for it. Voxel-based three-dimensional convolutions reason directly about spatial shape, but they tie the representation to a fixed grid, discard the residue-residue graph, and are not naturally rotation-invariant without expensive augmentation. Graph neural networks with invariant scalar geometry represent the residue graph naturally, but to stay invariant they project every directional quantity into local coordinate frames at the input, freezing all geometry into scalars before the network can manipulate it as geometric objects. The SO(3)-equivariant irreducible-representation networks keep geometry live at every layer, but their spherical-harmonic and tensor-product machinery is too costly for proteins with hundreds of residues. What is needed is a graph method whose directional features remain live and manipulable through depth, while its scalar readouts stay invariant and its cost scales linearly with the number of residues.

The method I propose is the Geometric Vector Perceptron, or GVP. It is a drop-in replacement for a dense layer that operates on a tuple of scalar and vector features. Scalars are ordinary channels that must be invariant under rotation; vectors are channels in R^3 that must rotate equivariantly. The key observation is that only three operations on vector channels commute with rotation and reflection: bias-free linear mixing over the channel dimension, taking the L2 norm, and rescaling a vector by a scalar function of its own norm. A vector bias is forbidden because no nonzero constant vector is rotation-invariant, and a coordinate-wise nonlinearity is forbidden because coordinate axes are arbitrary. The GVP applies a bias-free map W_h to the input vectors, feeds the row-wise norms into the scalar path alongside the input scalars, and applies a standard biased linear layer plus scalar nonlinearity to produce new scalars. For the new vectors it applies a second bias-free map W_mu and rescales each output vector by a sigmoid of that vector's own row-wise norm — the one nonlinearity that survives on a vector, since scaling by any function built purely from an invariant scalar stays equivariant — so the scalar output is invariant while the vector output is equivariant. The split between W_h and W_mu decouples the number of geometric invariants extracted for the scalars from the number of directional vectors propagated forward.

Stacking GVPs into a message-passing graph neural network yields the GVP-GNN. Each node and edge carries a scalar-vector tuple. Messages are formed by concatenating source-node, edge, and target-node tuples and running them through a short GVP stack, with the final GVP using identity activations so the summed message stays expressive. Incoming messages are aggregated by averaging over the k nearest neighbors, and a residual node update plus an equivariant LayerNorm refines the representation. The equivariant LayerNorm rescales all vector channels by their root-mean-square norm with no per-coordinate parameters, and vector dropout zeros out entire vector channels rather than individual coordinates to preserve equivariance. Because every layer updates both scalars and vectors, the directional state stays alive and gets refined through depth, unlike scalar-only methods that freeze geometry at the input.

For the protein backbone we build a k-nearest-neighbor graph over Cα atoms with k=30. Each node is initialized with six scalar dihedral features from sin and cos of phi, psi, and omega plus three vector directions: the forward unit vector toward the next residue's Cα, the reverse unit vector toward the previous residue's Cα, and the imputed Cβ direction built from the two backbone bond vectors under a tetrahedral-geometry assumption. Each edge carries Gaussian radial basis functions of the Cα-Cα distance, a sinusoidal encoding of the residue offset, and a single direction vector from the center Cα to the neighbor Cα. Input GVPs with identity activations lift nodes to 100 scalars and 16 vectors and edges to 32 scalars and 1 vector, each followed by the equivariant LayerNorm, and three GVPConv layers propagate structure alone to form the encoder embeddings. For sequence design each amino acid is embedded, and its embedding is appended to the scalar features of every edge pointing away from it — but zeroed whenever that neighbor's index is not earlier in the chain, so the graph is causally masked. Three further autoregressive decoder layers propagate this sequence-augmented graph: edges running forward in the chain draw their source features from the live decoder embeddings, while edges running backward draw instead on the frozen encoder embeddings so no information about an undecoded residue leaks in, and the two contributions are summed and divided by the true neighbor count. A final GVP with a twenty-scalar, zero-vector output and identity activations produces per-residue logits — softmax or log-softmax turns them into a distribution or a loss. Training minimizes masked per-residue cross-entropy against the native sequence, teacher-forced during training and sampled left-to-right at inference.

```python
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


class GVP(nn.Module):
    """(s, V) -> (s', V'); scalar out invariant, vector out equivariant."""
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid)):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.scalar_act, self.vector_act = activations
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
        else:
            self.ws = nn.Linear(self.si, self.so)
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)                                  # V_h = W_h V
            vn = _norm_no_nan(vh, axis=-2)                   # ||V_h||
            s = self.ws(torch.cat([s, vn], -1))             # scalars see vector norms
            if self.vo:
                v = self.wv(vh)                              # V_mu = W_mu V_h
                v = torch.transpose(v, -1, -2)
                if self.vector_act:                          # V' = sigma+(||V_mu||) (.) V_mu
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        return (s, v) if self.vo else s


def tuple_sum(*args):
    return tuple(map(sum, zip(*args)))

def tuple_cat(*args, dim=-1):
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

def tuple_index(x, idx):
    return x[0][idx], x[1][idx]

def _merge(s, v):
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)

def _split(x, nv):
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    return x[..., :-3*nv], v


class _VDropout(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if not self.training:
            return x
        mask = torch.bernoulli((1 - self.drop_rate) *
                               torch.ones(x.shape[:-1], device=self.dummy_param.device)).unsqueeze(-1)
        return mask * x / (1 - self.drop_rate)


class Dropout(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))      # RMS norm over channels
        return self.scalar_norm(s), v / vn


class GVPConv(MessagePassing):
    def __init__(self, in_dims, out_dims, edge_dims, n_layers=3, aggr="mean"):
        super().__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        GVP_ = functools.partial(GVP)
        module_list = [GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)]
        for _ in range(n_layers - 2):
            module_list.append(GVP_(out_dims, out_dims))
        module_list.append(GVP_(out_dims, out_dims, activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        x_s, x_v = x
        message = self.propagate(edge_index, s=x_s,
                                 v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1]),
                                 edge_attr=edge_attr)
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        m = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        return _merge(*self.message_func(m))


class GVPConvLayer(nn.Module):
    def __init__(self, node_dims, edge_dims, n_message=3, n_feedforward=2,
                 drop_rate=.1, autoregressive=False):
        super().__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                            aggr="add" if autoregressive else "mean")
        GVP_ = functools.partial(GVP)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])
        hid = (4 * node_dims[0], 2 * node_dims[1])
        self.ff_func = nn.Sequential(GVP_(node_dims, hid),
                                     GVP_(hid, node_dims, activations=(None, None)))

    def forward(self, x, edge_index, edge_attr, autoregressive_x=None, node_mask=None):
        if autoregressive_x is not None:
            src, dst = edge_index
            fwd = src < dst
            bwd = ~fwd
            dh = tuple_sum(
                self.conv(x, edge_index[:, fwd], tuple_index(edge_attr, fwd)),
                self.conv(autoregressive_x, edge_index[:, bwd], tuple_index(edge_attr, bwd)))
            count = scatter_add(torch.ones_like(dst), dst,
                                dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)
        else:
            dh = self.conv(x, edge_index, edge_attr)
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x


class CPDModel(nn.Module):
    """GVP-GNN for structure-conditioned autoregressive protein design.
    node_in_dim=(6,3), node_h_dim=(100,16), edge_in_dim=(32,1), edge_h_dim=(32,1)."""
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim,
                 num_layers=3, drop_rate=0.1):
        super().__init__()
        self.W_v = nn.Sequential(GVP(node_in_dim, node_h_dim, activations=(None, None)),
                                 LayerNorm(node_h_dim))
        self.W_e = nn.Sequential(GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
                                 LayerNorm(edge_h_dim))
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) for _ in range(num_layers))
        self.W_s = nn.Embedding(20, 20)
        edge_h_dim = (edge_h_dim[0] + 20, edge_h_dim[1])       # sequence appended to edge scalars
        self.decoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate, autoregressive=True)
            for _ in range(num_layers))
        self.W_out = GVP(node_h_dim, (20, 0), activations=(None, None))

    def forward(self, h_V, edge_index, h_E, seq):
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        encoder_embeddings = h_V
        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0               # causal: i sees only seq of j<i
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)
        return self.W_out(h_V)                                 # (n_nodes, 20) logits
```
