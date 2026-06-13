Let me start from what I actually have and what I actually want. I have a protein as a chain of residues that has folded up into a 3D shape, and the only thing I'm handed is, for each residue, a 3D coordinate — I'll use the alpha-carbon, one point per residue — and the residue identity. I want a function that turns that into vectors: one vector per residue and one vector for the whole protein, good enough that a simple head on top can predict function and fold. The reason structure is worth the trouble at all is that function follows from the fold, not from the bare sequence, so whatever I build has to actually use the 3D arrangement — the distances and the angles between residues — and not just who is bonded to whom along the chain. And there's a constraint I can't negotiate away: if I take the same protein and translate it, rotate it, or reflect it, it is the same protein, so my vectors must not change. If they changed, I'd be learning the coordinate frame I happened to write the protein down in, which is meaningless. So everything I do has to be invariant to the full set of rigid motions, E(3). One more pressure on top: experimentally solved structures are scarce — order 10^5 in the PDB against 10^7 sequences — so I can't afford a heavy, slow, parameter-hungry encoder that only works with a giant labeled corpus. It has to be simple enough to learn from few structures.

The natural substrate is a graph. Put a node at every residue, draw edges between residues that are related, and run message passing: each layer, every node gathers a message from each neighbor, sums them, transforms, and that becomes its new state; stack a few layers and information from L hops away reaches each node. The general shape is h_i^{(l+1)} = σ(Σ_{j∈N(i)} g(h_i, h_j)). Fine. Two things about this shape are going to drive every decision.

The first is how I get invariance for free. If the only geometric quantity that ever enters a message is itself invariant — a distance, an angle — then the entire network is E(3)-invariant by construction, no special equivariant layers, no carrying coordinate frames. Euclidean distance between two residues doesn't change under translation, rotation, or reflection; neither does the angle at a residue between two of its edges. So if I'm careful to feed the message function only distances and angles, invariance is handled and I never think about it again. That's the cheap route and I'll take it. I notice there's a more elaborate route — keep the coordinates inside the network and update them equivariantly, the way an E(n)-equivariant net does, where x_i ← x_i + Σ_j (x_i − x_j) φ(m_ij) so the coordinates rotate with the molecule — and it's elegant, but it's heavier than I need. My target is an *invariant* embedding, not an equivariant coordinate field. Why pay to carry and update coordinates through every layer when I only want a number at the end that's invariant? I'll get the invariance by feeding invariant features in and let the messages be plain vectors. So: invariant scalar inputs, ordinary message passing.

Now what message passing exactly? The plainest option is a graph convolution where every edge is treated identically — one shared kernel W transforms every neighbor before the normalized sum, H ← σ(D^{-1/2} A D^{-1/2} H W). It's cheap and it's a real baseline. But stare at what my edges actually mean. Some edges are "these two residues are next to each other in the chain." Some are "these two residues are far apart in the chain but close in space" — a tertiary contact, the whole point of folding. Those two kinds of edge carry totally different geometric information, and a single shared W transforms them the same way. That's structurally wrong, the same way one global learning rate is wrong when different parameters live at different scales: I'm forcing one operator onto edges that mean different things. So I want edges to carry a *type*, and I want the transform to depend on the type.

That's exactly what a relational graph convolution gives me. It was built for graphs whose edges carry a discrete type r from a set R, and the update is h_i^{(l+1)} = σ(Σ_{r∈R} Σ_{j∈N_r(i)} (1/c_{i,r}) W_r h_j + W_0 h_i): one learnable kernel W_r per edge type, shared across all edges of that type, a self-connection W_0, and a per-relation normalizer c_{i,r}. The key accounting fact is that the number of kernels is |R|, the number of distinct edge *types* — not the number of edges. That's the lever I need. Let me hold that thought, because there's a competitor that fits proteins even better and shows me exactly which trap to avoid.

The protein-specific encoder of the moment forms a message by passing each edge's geometry through a small MLP that outputs *an entire kernel matrix for that edge*: (κ * F)_k(x) = Σ_{i∈N(x)} Σ_j F_{i,j} · κ_k(τ_e, τ_{i1}, τ_{i2}), where τ_e is the Euclidean distance and τ_{i1}, τ_{i2} are two intrinsic graph-geodesic distances capturing primary and secondary structure. This is genuinely expressive — every edge gets a kernel continuously tailored to its own geometry — and it's invariant, and it respects locality. But count the cost: a distinct kernel matrix *per edge*. The memory grows with the number of edges, and the intrinsic-distance computation is expensive on top. On a residue graph with thousands of edges, that's a lot of kernel matrices, and it makes the thing cumbersome to scale and miserable to pretrain on many structures. So I have a clean trade-off in front of me. Per-edge kernels: maximal capacity, blows up in memory. One shared kernel: minimal memory, can't tell edge types apart. The relational convolution sits exactly in between — share a kernel *within each type*, so capacity scales with |R| (a handful) and memory does not scale with the number of edges at all. That's the sweet spot, and it tells me my backbone: relational message passing with one kernel per edge type. The question becomes: what are the types?

Let me design the graph and its edge types from the geometry, deliberately. I need edges that capture sequence locality and edges that capture spatial proximity, because those are the two distinct things. Start with the sequence. Connect residue i to residue j when they're within a small sequence window, |j − i| < d_seq. But here's a choice I should not make lazily: do I call all of these "sequential edges," one type? No — direction and exact offset along the backbone are geometrically meaningful. The relationship i to i+1 is not the relationship i to i−2; the backbone is directional, and a residue's neighbor two steps ahead sits differently in space than its immediate predecessor. So I type each sequential edge by its *relative position* d = j − i. With d_seq = 3 the offsets are d ∈ {−2, −1, 0, 1, 2}, which is 2·d_seq − 1 = 5 sequential edge types. The d = 0 case is a self relation — a residue to itself in sequence space — which gives me a clean place for self-information inside the relational machinery. Typing by offset lets the model learn order-aware, direction-aware local backbone patterns instead of smearing them into one bucket.

Now the spatial edges, the contacts that sequence misses. The obvious move is a radius rule: connect i and j whenever ||x_i − x_j|| < d_radius. And I should also consider a k-nearest-neighbor rule: connect each residue to its k closest residues by Euclidean distance. Which one? Let me think about what each does to the graph, because this matters more than it looks. If I use radius only, the trouble is that the typical spacing between residues isn't the same from protein to protein — some structures are loosely packed, some tightly — so a single fixed radius is too small for many of them, and on a large database tens of thousands of proteins come out with average degree below two. A graph that sparse is almost edgeless; there are too few edge pairs and triplets for any angle- or dihedral-based learning to get any signal, and tuning the radius can't fix it because the right radius differs per protein. If instead I use kNN only, every node gets exactly k neighbors, so the degree is nearly constant across all proteins — connectivity guaranteed — but that's also the problem: it flattens away the genuinely dense regions where many residues pack together, because it caps everyone at k regardless of how crowded they are. So the two failure modes are complementary. Radius alone leaves holes; kNN alone erases density variation. The fix isn't to pick the better one — it's to use *both*, as two separate edge types. The radius edges restore the density information in crowded regions; the kNN edges put a floor under the degree so no protein collapses to an edgeless graph. Two edge types, not a compromise between them.

One more refinement on the spatial edges. A radius or kNN edge between two residues that are *also* close in the chain is redundant — I already have sequential edges covering local neighbors, and a spatial edge there just re-says "these are adjacent." What I actually want from spatial edges is the long-range tertiary contacts, the residues far apart in sequence that fold back to touch. So I add a filter: drop a radius or kNN edge if |i − j| < d_long. With d_long = 5 that keeps spatial edges focused on genuine long-range structure and stops them from duplicating the sequential ones. So my edge-type set is: 5 sequential offsets, 1 radius type, 1 kNN type — total |R| = 2·d_seq + 1 = 7 relation types. Seven kernels, that's all, independent of how many edges any protein has.

Features. I want to keep this simple — that data-scarcity pressure again, and the prior encoders that loaded up on hand-crafted chemical features are exactly the ones that got cumbersome. For the node, just the residue identity: a one-hot over the 20 amino acids plus one slot for "unknown," 21 dimensions, nothing else. The geometry will come in through the graph structure and the edge features, not through elaborate node descriptors. For an edge (i, j, r), I concatenate what's relevant and invariant: the two endpoint residue features, a one-hot of the edge type r, the absolute sequence separation |i − j|, and the spatial distance ||x_i − x_j||_2. In code the absolute sequence separation is clamped and one-hot expanded, so the edge feature is Cat(f_i, f_j, onehot(r), onehot(clamp(|i − j|)), ||x_i − x_j||_2). Every geometric quantity in there is a distance, so it's all E(3)-invariant, and the encoder stays invariant.

Now the layer itself. The relational template is h_i^{(l+1)} = σ(Σ_r Σ_{j∈N_r(i)} (1/c_{i,r}) W_r h_j + W_0 h_i). I want to adapt three things. First, that normalization c_{i,r} = |N_r(i)|: on protein graphs the degree varies wildly between residues and between proteins, and a fixed degree-divide is a blunt instrument for controlling the scale of the aggregated message. I'd rather normalize the *scale* of the result directly — drop the 1/c and put a batch-normalization on the aggregated, transformed message, which adapts to whatever scale the sums come out at across the batch. Second, the activation: I'll put σ = ReLU after the batch-norm. Third, the self-connection W_0 h_i: instead of a separate learned self kernel inside the sum, I'll make the layer compute an update u and add it residually to the previous state. So define u_i^{(l)} = σ(BN(Σ_{r∈R} W_r Σ_{j∈N_r(i)} h_j^{(l−1)})) and then h_i^{(l)} = h_i^{(l−1)} + u_i^{(l)}, starting from h_i^{(0)} = f_i. The residual does double duty: it's the self-information path (replacing W_0), and it's the thing that lets me stack several layers without the optimization getting hard, so early local features survive to the output instead of being overwritten. Six layers, hidden width 512 — deep enough to propagate information across the fold, wide enough to hold it.

Let me make sure I can compute this aggregation efficiently, because Σ_r W_r Σ_{j∈N_r(i)} h_j looks like it wants a separate pass per relation, and seven passes over the graph per layer would be wasteful. There's a tidy trick. For each directed edge with source j, destination i, type r, the message is just h_j (the neighbor's features). I want, for each destination node i and each relation r, the sum of h_j over neighbors of that type. So I scatter every message into a bucket keyed by (i, r): give edge → destination i and type r the bucket index i·|R| + r, and scatter-add all messages into an array of size num_nodes·|R|. Reshape that to (num_nodes, |R|·input_dim): now each row holds, concatenated across relations, the per-relation neighbor sums for one node. Then a single linear map Linear(|R|·input_dim → output_dim) applied to that row is exactly Σ_r W_r (sum over N_r(i)), because that one big weight matrix *is* the seven W_r stacked side by side. One scatter-add, one reshape, one matmul — all relations in a single pass. BatchNorm, ReLU, residual add, and the layer is done.

I now have a perfectly serviceable encoder: a relational message-passing network on a 7-type residue graph, invariant, cheap, simple. But I have a nagging feeling it's leaving something on the table, and I want to name it precisely. Every message in this layer is h_j — one neighbor's features, transformed by the kernel for its type, summed over neighbors *independently*. The neighbors don't talk to each other. And I know from the general theory of these networks that messages built from pairwise relations alone have a real blind spot: a network that sees only the distances to a node's neighbors cannot distinguish geometries that present the same multiset of neighbor distances. The clean illustration is that at a short cutoff a hexagonal ring and two separate triangular rings with the same bond lengths look identical from every atom, because each atom's distance-neighborhood is the same in both. What separates them is *directional* — the angles between the edges incident to a node. My relational layer types the edges and shares kernels nicely, but it never looks at the angle between two edges meeting at a residue. It can't tell whether residue j's two contacts go off in nearly the same direction or in opposite directions, and in a folded protein that's exactly the kind of geometry that distinguishes one local arrangement from another.

So I want to model interactions *between edges*, not just between nodes — how one edge of a residue relates to another edge of the same residue. And I have strong evidence this is the right thing to chase for proteins specifically: the most accurate structure models of the time spend most of their effort updating a representation of each residue *pair* using triplets — for the pair (i, j) they bring in every third residue k, which is precisely a statement that the interaction between two contacts that share a residue is what carries the structural signal. The triangle operations doing this are powerful. But they're dense over all triplets, costing on the order of n^3 in the number of residues. I can't afford that, and I shouldn't have to — most pairs of edges in a sparse residue graph don't share a residue at all and have no business interacting. I want the *sparse* version: only let two edges interact when they actually meet at a residue, and condition the interaction on the angle between them.

How do I let edges interact through message passing? The clean construction is to build a graph *whose nodes are the edges* of the original graph — a line graph. Each directed edge (i, j, r_1) of the protein graph becomes a node in this new graph G′. When do two of these edge-nodes connect? I want them to connect when they physically share a residue so that one edge's information can flow into another edge of the same residue. Concretely, connect edge (i -> j, r_1) to edge (j -> k, r_2), sharing the middle residue j, and require i != k so I don't make a degenerate two-step walk that leaves and returns to the same residue. Now message passing on G′ updates each edge's representation by aggregating from the edges adjacent to it, which is exactly edge-to-edge interaction.

And what should the *type* of a G′-edge be? Two adjacent edges meeting at a residue have a well-defined angle between them, and that angle is the directional information my node-only layer was blind to — edges pointing in nearly the same direction interact differently than edges pointing apart. So I let the angle between the two edges set the relation type in G′, and I'm right back in relational-convolution territory, which I already know how to do cheaply. But the angle is continuous, and a relation type has to be discrete. I could embed the angle in a continuous basis the way the directional small-molecule networks do — a spherical Bessel/harmonic expansion — but that's an extra basis and an extra MLP and it's heavy, which is the cost I'm specifically trying to avoid. The cheaper move that reuses everything I've already built: discretize. Chop the range [0, π] into a small number of bins — 8 bins — and use the bin index as the edge type in G′. Now G′ is a relational graph with 8 relation types, and I run the *same* relational convolution on it that I run on the residue graph. No new machinery. The angle binning is what turns "model edge-edge interactions conditioned on geometry" into "run a cheap 8-relation conv on the line graph."

Let me pin down the angle computation exactly, because the formula matters. The two adjacent edges are (i → j) — its head j is the shared residue — and (j → k), whose tail is that same j; they meet at j, and the angle I want is the one at j between the two *non-shared* endpoints i and k. Form the two vectors from j out to those endpoints: v_1 = x_i − x_j and v_2 = x_k − x_j. The angle between v_1 and v_2 is most robustly gotten from both the dot product and the cross product: let x = v_1 · v_2 and y = ||v_1 × v_2||, then angle = atan2(y, x). Using atan2 of the cross-product norm against the dot product is numerically stable across the whole range and gives the unsigned angle in [0, π], which is what I want — the geometry doesn't care about a sign of the angle here, just how open it is. Then the bin is floor(angle / π · 8), clamped to 7 so the boundary angle = π lands in the last bin. That index is the relation type. Eight kernels on the line graph, indexed by how sharp the angle between the two edges is.

Now stitch the edge messages back into the node update — the whole point is to make the residue features better, not just to have edge features. The edge layer produces, for each edge (i, j, r_1) at layer l, a message representation m_{(i,j,r_1)}^{(l)}, updated by the same relational rule on G′: m_{(i,j,r_1)}^{(0)} = f_{(i,j,r_1)} (the edge's input features), and m_{(i,j,r_1)}^{(l)} = σ(BN(Σ_{r∈R′} W′_r Σ_{(w,k,r_2)∈N′_r((i,j,r_1))} m_{(w,k,r_2)}^{(l−1)})), where R′ are the 8 angle bins and N′ are the incoming line-graph neighbors. Then, when I do the node update, instead of each neighbor j contributing just h_j, it contributes h_j *plus* the message that lives on the very edge (j, i, r) carrying it. Algebraically I can write a learned FC on m_{(j,i,r)}^{(l)}, but in the implementation I can absorb that FC into the edge convolution itself by making the edge layer at depth l output exactly the input dimension of the node layer at depth l. Then the node aggregation is u_i^{(l)} = σ(BN(Σ_{r∈R} W_r Σ_{j∈N_r(i)} (h_j^{(l−1)} + m_{(j,i,r)}^{(l)})))) with the dimensions already matched. So each contact now carries not only the neighbor's state but a summary of how that contact relates, angularly, to the other contacts at the same residue. That's the directional information the node-only layer couldn't see, delivered sparsely — only along edges that actually share a residue — and conditioned on the angle through the 8 line-graph relations. It's the triangle idea, made sparse and discrete and therefore cheap.

I should make sure I haven't broken invariance by introducing the line graph. The line graph is built from which edges share residues, which is purely combinatorial and frame-free; the only geometric quantity that enters is the angle between edges, which is invariant to translation, rotation, and reflection; and the residue-graph construction and edge features were already all distances. So everything the network ever sees is a distance or an angle. The whole encoder, with or without the edge layer, is E(3)-invariant by construction. Good — that was the non-negotiable constraint and it survives.

There's a clean optional add-on I can see now that the architecture is flexible. The protein-specific kernel-MLP convolution I rejected for memory has real expressivity in its geometry-to-kernel mapping; I can keep a stripped-down version as a parallel branch when I want extra continuous geometric capacity. The expensive parts — dynamically changing receptive fields, pooling, smoothing, and costly intrinsic graph-geodesic distances — do not have to be part of the core encoder. A small kernel MLP can map cheap relative-positional edge features to a per-edge kernel on the already-built residue graph, and its output can be added in parallel with the relational update. The core construction does not depend on it.

Last, how do I read out the embeddings. Two outputs: per-residue and per-protein. For the per-residue embedding I want all scales, not just the last layer — the early layers hold local backbone geometry, the later layers hold the propagated global fold, and the downstream head benefits from seeing both. So I concatenate the hidden states from every layer, h^{(1)} through h^{(L)}, into the node representation; with L = 6 layers of width 512 that's a 6·512-dimensional per-node vector. For the per-protein embedding I pool that concatenated node representation over each protein. Sum pooling rather than mean: summing keeps a notion of the total signal and the protein's size, which mean would normalize away, and for these classification tasks that size/total information is worth keeping. So graph_feature = Σ_{i∈protein} node_feature_i.

Let me write the encoder, filling the slot in the harness. The graph-construction module supplies a residue graph with 7 relation types: five sequential offsets from d_seq = 3, one radius relation, and one kNN relation; its edge feature is the concatenation of endpoint residue types, edge-type one-hot, clamped sequence-distance one-hot, and Euclidean distance, which is 59 dimensions in the usual residue vocabulary. The relational convolution layer does the one-pass scatter-add-reshape-linear trick, with BN and ReLU after the aggregation. The spatial line graph supplies 8 angle-bin relations for edge messages. The model stacks six layers, updates the edge states before the corresponding node layer, passes the node update through the configured dropout module, applies the residual short-cut, concatenates all hidden layers, and sum-pools.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class GeometricRelationalGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None,
                 batch_norm=True, activation="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        self.edge_linear = nn.Linear(edge_input_dim, input_dim) if edge_input_dim else None
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

    def message(self, graph, input, edge_input=None):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear is not None:
            message = message + self.edge_linear(graph.edge_feature.float())
        if edge_input is not None:
            assert edge_input.shape == message.shape
            message = message + edge_input
        return message

    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation
        dst = graph.edge_list[:, 1]
        relation = graph.edge_list[:, 2]
        bucket = dst * self.num_relation + relation
        weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * weight, bucket, dim=0,
                             dim_size=graph.num_node * self.num_relation)
        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def combine(self, update):
        update = self.linear(update)                # Sum_r W_r Sum_{j in N_r(i)}
        if self.batch_norm is not None:
            update = self.batch_norm(update)
        if self.activation is not None:
            update = self.activation(update)
        return update

    def forward(self, graph, input, edge_input=None):
        message = self.message(graph, input, edge_input)
        update = self.aggregate(graph, message)
        return self.combine(update)


class SpatialLineGraph(nn.Module):
    def __init__(self, num_angle_bin=8):
        super().__init__()
        self.num_angle_bin = num_angle_bin

    def forward(self, graph):
        line_graph = graph.line_graph()
        node_in, node_out = graph.edge_list[:, :2].t()
        prev_edge, next_edge = line_graph.edge_list.t()
        # line_graph enumerates consecutive directed edges src -> mid -> dst.
        src = node_in[prev_edge]
        mid = node_out[prev_edge]
        dst = node_out[next_edge]
        v1 = graph.node_position[src] - graph.node_position[mid]
        v2 = graph.node_position[dst] - graph.node_position[mid]
        angle = torch.atan2(torch.cross(v1, v2).norm(dim=-1), (v1 * v2).sum(dim=-1))
        relation = (angle / math.pi * self.num_angle_bin).long()
        relation = relation.clamp(max=self.num_angle_bin - 1)
        edge_list = torch.cat([line_graph.edge_list, relation.unsqueeze(-1)], dim=-1)
        return type(line_graph)(edge_list, num_nodes=line_graph.num_nodes,
                                offsets=line_graph._offsets, num_edges=line_graph.num_edges,
                                num_relation=self.num_angle_bin, meta_dict=line_graph.meta_dict,
                                **line_graph.data_dict)


class SumReadout(nn.Module):
    def forward(self, graph, node_feature):
        return scatter_add(node_feature, graph.node2graph, dim=0, dim_size=graph.batch_size)


class GearNet(nn.Module):
    def __init__(self, input_dim=21, hidden_dims=(512, 512, 512, 512, 512, 512),
                 num_relation=7, edge_input_dim=59, batch_norm=True, concat_hidden=True,
                 short_cut=True, readout="sum", dropout=0, num_angle_bin=8):
        super().__init__()
        self.num_relation = num_relation
        self.concat_hidden = concat_hidden
        self.short_cut = short_cut
        self.num_angle_bin = num_angle_bin
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]

        self.layers = nn.ModuleList([
            GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                         None, batch_norm, "relu")
            for i in range(len(self.dims) - 1)
        ])
        self.dropout = nn.Dropout(dropout)

        if num_angle_bin:
            self.spatial_line_graph = SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList([
                GeometricRelationalGraphConv(self.edge_dims[i], self.edge_dims[i + 1],
                                             num_angle_bin, None, batch_norm, "relu")
                for i in range(len(self.edge_dims) - 1)
            ])

        if readout != "sum":
            raise ValueError("This configuration uses sum readout")
        self.readout = SumReadout()

    def forward(self, graph, input):
        hiddens = []
        layer_input = input
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_hidden = line_graph.node_feature.float()
        else:
            edge_hidden = None

        for i, layer in enumerate(self.layers):
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_hidden)
            hidden = layer(graph, layer_input, edge_hidden)
            hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        node_feature = torch.cat(hiddens, dim=-1) if self.concat_hidden else hiddens[-1]
        graph_feature = self.readout(graph, node_feature)
        return {"graph_feature": graph_feature, "node_feature": node_feature}
```

Let me retrace the causal chain. I needed an encoder over residue 3D coordinates that is invariant to rigid motion, uses real geometry, captures both local and global structure, and is cheap enough for scarce data. Invariance I bought for free by feeding the network only distances and angles and refusing to carry coordinate frames, which is why I passed on the heavier equivariant-coordinate route. Plain graph convolution treats all edges alike, which is wrong because sequence neighbors and spatial contacts mean different things, so I went relational — one shared kernel per edge *type*. The protein-specific per-edge-kernel convolution had the capacity but blew up in memory at one kernel per edge; relational convolution sits exactly between that and a single shared kernel, with capacity scaling in the number of types, so it became my backbone. I designed the seven types from the geometry: five sequential offsets, because backbone direction and exact offset matter; a radius type and a kNN type together, because radius-only leaves tens of thousands of proteins near-edgeless while kNN-only flattens away dense regions, and the two are complementary; with a long-range filter so spatial edges carry genuine tertiary contacts instead of duplicating sequence neighbors. The layer dropped R-GCN's degree normalizer for a batch-norm on the aggregated message and turned the self kernel into a residual add, and I made the seven-relation aggregation a single scatter-reshape-matmul. Then I noticed the node-only layer is blind to the angles between a node's edges — the directional information that distinguishes geometries with identical distance neighborhoods — and that the strongest structure models spend their effort on exactly these edge-edge interactions via dense triplet updates. So I built the sparse version: a line graph over directed two-step walks (i -> j) then (j -> k), typed by the binned angle at j, with the same cheap relational convolution running on it; the resulting per-edge messages fold back into each node update along the edge that carries them. The angle binning is what reuses the relational machinery instead of a heavy continuous basis, and the sparsity is what avoids the cubic cost. Everything seen is a distance or an angle, so the whole thing stays E(3)-invariant. Concatenating all layer outputs gives the head every scale; sum pooling gives the per-protein vector while keeping size and total signal. And because the layers are additive, a stripped-down geometry-to-kernel convolution can be added in parallel as an optional extra for the tasks that want it.
