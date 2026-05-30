OK, let me think this through from scratch. The Transformer has eaten language, then speech, then vision — every domain where it shows up, it eventually wins, because one self-attention layer gives every element a global receptive field and the whole thing scales with data and compute in a way nothing else does. And yet on graph-level prediction leaderboards it just doesn't show up at the top. The winners there are still message-passing GNNs. Every time someone says "Transformer for graphs," I look closely and it turns out they took a GNN and swapped the neighborhood aggregator for a softmax — it's a GNN wearing a Transformer's clothes. Nobody has taken a plain Transformer encoder, fed it a graph, and won. So either the architecture is genuinely unsuited to graphs, or we're feeding it the wrong thing. I want to know which.

Let me write down exactly what a self-attention layer does, because I suspect the failure is right there in the equations. Stack the node representations as rows, $H=[h_1^\top,\dots,h_n^\top]^\top$. Project to queries, keys, values: $Q=HW_Q,\ K=HW_K,\ V=HW_V$. Form scores $A=QK^\top/\sqrt{d_K}$, and output $\mathrm{softmax}(A)\,V$. Stare at $A_{ij}=\langle h_iW_Q,\,h_jW_K\rangle/\sqrt{d_K}$. That number depends on the *features* of node $i$ and node $j$ and nothing else. It does not know whether $i$ and $j$ are neighbors. It does not know whether they're three hops apart or fifty. It does not know what edge — if any — connects them. It does not even know that node $i$ has twenty neighbors and node $j$ has one. The score is pure feature similarity. And the output for node $i$ is a function of the *multiset* of input vectors: permute the nodes and the outputs permute identically. So an attention layer fed a graph sees an unordered bag of node feature vectors. It is, structurally, blind.

That's actually the same disease attention has on sequences — a sequence is also just a set to attention, which is why we bolt on positional encodings. For words we either add an absolute position embedding to each token, or we add a learned bias $A_{ij}\mathrel{+}=$ (something depending on $i-j$) so the score knows the relative offset. Fine. But a graph has no canonical ordering of its nodes — relabel them however you like, it's the same graph — so an absolute "node 1, node 2, node 3" embedding is meaningless, it would just memorize an arbitrary labeling. And there's no scalar offset $i-j$ between two nodes either; "how far apart" on a graph isn't a one-dimensional thing you read off the index. So the sequence fixes don't port over directly. The whole problem reduces to: what *is* the structure of a graph that a bag-of-vectors throws away, and how do I hand exactly that to attention?

Let me just enumerate what's true about a graph that's invisible in the bag of node features. First: some nodes are more important than others — in a social graph a celebrity with a million followers is not interchangeable with a random user. The standard handle on that is *degree centrality*: how many edges touch a node. That's a per-node integer, knowable from the graph alone. Second: there's a notion of distance — the shortest-path distance between two nodes, $\phi(v_i,v_j)$, which I can compute for all pairs with Floyd–Warshall. That's a per-pair integer. Third: edges carry features — in a molecule the bond between two atoms has a type — and more than that, the *sequence of edges* lying between two nodes describes how they're connected. So I have one per-node structural signal (degree), and two per-pair structural signals (distance, and the edges along the connecting path). Three things to inject. Let me take them one at a time and figure out *where* in the attention computation each one belongs, and let the form drop out of that.

Start with degree, since it's the simplest — it's per-node. Where does a per-node quantity belong? The attention score is built from $Q$ and $K$, which are linear images of the node inputs $h_i^{(0)}$. If I want the attention to be *able* to condition on a node's importance — to learn "celebrities get attended to more," or whatever the task wants — then the importance has to be present in $Q$ and $K$, which means it has to be present in the input $h_i^{(0)}$. So the place to put degree is the input embedding itself, added to the node feature before the first layer. I don't want to hand the network the raw integer degree — that's a single scalar, and I want it to interact with all $d$ dimensions of the representation and be learnable. So give each possible degree value its own learnable $d$-dimensional embedding vector and add it. For a directed graph I have two numbers, in-degree and out-degree, so two embedding tables:
$$h_i^{(0)} = x_i + z^-_{\deg^-(v_i)} + z^+_{\deg^+(v_i)},$$
with $z^-,z^+\in\mathbb{R}^d$ learnable, indexed by the node's in- and out-degree. For an undirected graph the two collapse to one degree term. That's it — call it a centrality encoding. Now $Q$ and $K$ carry degree information, so the score $A_{ij}$ can reflect both the semantic similarity *and* the relative importance of $i$ and $j$. It's almost embarrassingly cheap, but it's the right cheap thing: a per-node signal goes into the per-node input.

Now distance. This one is per-pair, $\phi(v_i,v_j)$, so it does *not* belong in the per-node input — there's no way to write a function of two nodes by adding something to each node's vector separately (that would only ever produce a sum of a function of $i$ and a function of $j$, and a generic pairwise function isn't separable like that). Where does a per-pair quantity live? Right in the score matrix $A_{ij}$, which is exactly the per-pair object. And the sequence world already showed me the mechanism: a relative-position bias adds a learned scalar to the score depending on the relation between $i$ and $j$. So mirror that, but index the bias by graph distance instead of by sequence offset. Assign each shortest-path-distance value a single learnable scalar $b_{\phi}$, and add it:
$$A_{ij}=\frac{(h_iW_Q)(h_jW_K)^\top}{\sqrt{d}} + b_{\phi(v_i,v_j)}.$$
The bias depends only on the distance, not on the layer or the content, so I share one table of scalars across all layers — that's a tiny number of parameters, one per distinct distance (per head), and it's content-free, which is the point: it's a structural prior, not a learned feature interaction. If two nodes are disconnected I just give $\phi$ a special value (say $-1$, or a reserved bucket) and let it learn its own scalar. What does this buy me? Two things, and they're the things the GNN couldn't have. First, the receptive field is still global — node $i$ attends to *every* node, near or far, in one layer, unlike a GNN that has to stack $k$ layers to reach $k$ hops. Second, the bias lets the model *modulate* that global reach by structure: if $b_\phi$ is learned to decrease in $\phi$, then each node down-weights far-away nodes and concentrates on near ones — it can recover locality when locality is what the task wants — but it can also learn to attend far when that helps, adaptively, per head. I get the GNN's locality as a special case and the Transformer's globality for free.

There's a subtlety I should pin down: why a *learned scalar per distance* and not, say, just plugging the raw distance $\phi$ into the score, or a one-hot of the distance through a linear layer? Plugging in the raw integer forces a monotone, linearly-scaled relationship between distance and attention — but I have no reason to believe attention-vs-distance is monotone; maybe two-hop neighbors matter most for some chemical property, and the model should be free to put the biggest bias there. A free learnable scalar per bucket imposes no shape at all on the distance–attention curve; the model picks it. And a scalar per bucket (per head) is the minimal thing that does that. Good.

Now edges. Edges have features — bond types — and they're clearly structural and clearly matter for, say, molecules. How do GNNs use them today? Two ways. One: add the edge's feature to the features of its two endpoint nodes. Two: when a node aggregates its neighbors, fold the connecting edge's feature into that aggregation. Both have the same flaw, and once I say it out loud it's obvious: an edge's information only ever reaches the *two nodes it touches*. But in an attention model I'm computing a correlation between *every* pair $(i,j)$, including pairs that are far apart, and the natural question for that pair is "how are these two connected?" — and the answer is the *chain of edges* between them. The edge information that's relevant to the pair $(i,j)$ is the whole path from $i$ to $j$, not just the edges hanging off $i$. So edges, for a pair, should also go into the per-pair object, the score $A_{ij}$, and they should summarize the connecting path.

So take a shortest path from $i$ to $j$, $\mathrm{SP}_{ij}=(e_1,\dots,e_N)$, and I need to turn that sequence of edge feature vectors into one scalar to add to $A_{ij}$. I want each edge to contribute, I want a learnable interaction (not just a raw sum of features), and I want position along the path to matter (the first bond out of $i$ might mean something different from the last bond into $j$). The clean form: give each path-position $n$ its own learnable weight vector $w^E_n$, dot it with that edge's feature, and average over the path:
$$c_{ij}=\frac{1}{N}\sum_{n=1}^{N} x_{e_n}\,(w^E_n)^\top,\qquad A_{ij}=\frac{(h_iW_Q)(h_jW_K)^\top}{\sqrt{d}} + b_{\phi(v_i,v_j)} + c_{ij}.$$
Why average rather than sum over the path? Because path lengths differ across pairs, and a raw sum would make the edge bias systematically larger for distant pairs purely because the path is longer — I'd be smuggling distance into the edge term, and I already have a dedicated, properly-learnable distance term $b_\phi$ for that. Averaging normalizes out the length so $c_{ij}$ is about *what kind* of edges connect $i$ and $j$, leaving *how far* to $b_\phi$. The two pairwise terms factor cleanly. So now $A_{ij}$ carries three things: semantic similarity (the original $QK^\top$), how far apart (the spatial bias), and what connects them (the edge bias). The per-node input carries the fourth, importance (centrality). That's the whole structural story, and every piece lands in the place that matches its arity — per-node into the input, per-pair into the score.

For the block itself I'll keep the standard Transformer encoder essentially untouched — self-attention plus a position-wise FFN, residuals around each. One choice: put the layer norm *before* each sublayer (pre-LN) rather than after. The reason is purely optimization: with post-LN, deep Transformers need careful warmup and are finicky to train because the residual path gets renormalized every layer; pre-LN keeps a clean identity residual all the way down and trains stably at depth without that fragility. So
$$h'^{(l)}=\mathrm{MHA}(\mathrm{LN}(h^{(l-1)}))+h^{(l-1)},\qquad h^{(l)}=\mathrm{FFN}(\mathrm{LN}(h'^{(l)}))+h'^{(l)}.$$
One more knob: the FFN inner width. The sequence Transformer blows it up to $4d$. But these graphs are tiny — a molecule has on the order of fifteen atoms — and the parameter budget is tight on the smaller graph benchmarks. If the structural information is doing real work in attention, I should not need to spend most of the parameters in the position-wise MLP. So I set the inner width to $d$ instead of $4d$: it keeps the block shape standard while putting capacity pressure on the graph-aware attention rather than on a wide FFN.

Now the question that actually worries me. I've been hand-waving that these structural biases make the model "good for graphs." But the bar is the GNNs, and GNNs are a well-understood family with a known expressive ceiling. If I can't even *match* a GNN, this is a dead end. So let me ask precisely: with these encodings, can a single one of my attention layers *reproduce* what a GNN layer does — the AGGREGATE and COMBINE steps of GCN, GraphSAGE, GIN? If yes, then my model contains the GNNs as special cases and can only be more expressive. Let me try to actually construct the weights, not just assert it.

Take the GCN-style step first: MEAN aggregation over the one-hop neighbors. I want my attention layer, for node $i$, to output the *average* of its neighbors' value vectors. Look at $A_{ij}=QK^\top/\sqrt d + b_\phi$. The shortest-path distance to a one-hop neighbor is exactly $\phi=1$, and to a non-neighbor it's $\geq 2$ (or the disconnected value). So set the spatial bias to $b_\phi=0$ when $\phi=1$ and $b_\phi=-\infty$ otherwise. That alone masks attention down to exactly the one-hop neighborhood — softmax sends $-\infty$ scores to zero weight. If I want the self-inclusive mean used by some GCN variants, I keep $\phi=0$ in the allowed set too; I will keep the self term separate here because COMBINE needs it anyway. Now I want the *within-neighborhood* weights to be uniform, i.e. a plain mean, not a content-weighted average. Kill the content term: set $W_Q=W_K=0$, so the $QK^\top$ part is zero everywhere and the only thing left in $A_{ij}$ among the neighbors is the constant $b_1=0$. Equal scores → softmax gives a uniform distribution over the neighbors → the output is the unweighted average of the value vectors. Set $W_V=I$ so the values are the neighbor representations themselves. Then $\mathrm{softmax}(A)V$ is exactly $\frac{1}{|\mathcal N(i)|}\sum_{j\in\mathcal N(i)}h_j$ — MEAN aggregation. So GCN's aggregator falls out of one head with three weight settings. The spatial bias was the load-bearing piece: it's what lets attention *carve out the neighbor set at all*, which raw attention could never do because it has no idea who the neighbors are.

Now SUM aggregation — that's GIN's. SUM is just MEAN times the number of neighbors: $\sum_{j\in\mathcal N(i)}h_j = |\mathcal N(i)|\cdot\frac{1}{|\mathcal N(i)|}\sum_{j}h_j$. I already have MEAN from the construction above. And $|\mathcal N(i)|$ is the node's degree — which I can read off the centrality encoding. So use one head to produce the MEAN, and a second head (or the centrality term directly) to surface the degree; concatenate the degree onto the mean-aggregated vector, and let a sufficiently wide FFN multiply them. An MLP can realize the map "(mean vector, degree) ↦ degree·mean = sum." So SUM is recoverable too — it just needs the degree, which is exactly why centrality encoding had to be there. GIN's sum aggregator falls out of MEAN-plus-degree-times.

MAX aggregation — GraphSAGE's pooling variant — is harder, because max isn't linear and softmax is a *soft* average, not a hard selection. But softmax becomes a hard argmax in the low-temperature limit. So do it per dimension. For output dimension $t$, use one head: again $b_\phi=0$ on $\phi=1$ and $-\infty$ otherwise to restrict to neighbors; set $W_K=e_t$ (the $t$-th standard basis vector), so the key for neighbor $j$ is its $t$-th coordinate $h_{j,t}$; set $W_Q=0$ but give $Q$ a constant bias $T\mathbf{1}$ with $T$ a large temperature, so the score for neighbor $j$ becomes $T\cdot h_{j,t}$ — proportional to the value I want to maximize, scaled up by $T$. As $T\to\infty$, $\mathrm{softmax}(T\cdot h_{\cdot,t})$ concentrates all weight on the neighbor with the largest $t$-th coordinate — hard max. Set $W_V=e_t$ so the value read off is that same $t$-th coordinate. The head's output is then $\max_{j\in\mathcal N(i)}h_{j,t}$. Use one head per dimension and you've got element-wise MAX over the neighborhood. So all three aggregators — mean, sum, max — are special cases of one attention layer.

COMBINE is the last piece: it fuses the aggregate with the node's *own* previous representation. That just needs an extra head that returns node $i$ itself. Set $b_\phi=0$ when $\phi=0$ (the self-distance) and $-\infty$ otherwise — so attention collapses onto the single node $i$ — with $W_Q=W_K=0$ and $W_V=I$, giving back $h_i$. Now I have, in parallel, the aggregate of the neighbors (from the AGGREGATE head) and the self-feature (from this head); a sufficiently wide FFN can approximate the desired COMBINE function of the two. So one Graphormer layer, with the right weights and the SPD as the distance function, represents the full AGGREGATE–COMBINE step of GIN, GCN, and GraphSAGE. The useful claim is containment: those GNN updates sit inside the parameter space, and the all-pairs attention layer still has degrees of freedom those local updates never use.

And it *is* strictly more powerful, and I can see why from the same machinery. The whole GNN family is capped at the 1-WL test — there are non-isomorphic graphs whose 1-WL color refinement is identical, and no message-passing GNN can ever separate them. But my spatial encoding gives every node its *set of shortest-path distances to all other nodes*, which is information 1-WL never computes. Picture two graphs that 1-WL declares identical — same color multisets at every round. If their multisets of SPD-sets differ — one graph has a node whose distances to the rest are $\{0,1,1,2,2,3\}$ while no node in the other graph has that profile — then the spatial bias feeds the attention a signal that distinguishes them, and a model that's *given* those distances can tell the graphs apart even though 1-WL, and therefore every plain GNN, cannot. So the spatial encoding lifts me above the 1-WL ceiling, not just up to it.

Now the readout — the whole-graph vector. The textbook move is a permutation-invariant pool over the final node states. But I keep circling a different, slicker idea that the architecture seems to *want*. People get a global pathway in GNNs by adding a supernode wired to every node — it gathers the whole graph and rebroadcasts it, which empirically helps a lot. Conceptually that's exactly a READOUT followed by a re-injection. But here's the thing: a self-attention layer, where every node already attends to every node, can *do that READOUT by itself*, without adding any node. Let me check it can simulate a MEAN READOUT. If a head has no pairwise structural bias active, set $W_Q=W_K=0$ and set the query/key biases to the same constant vector. Then every score in a row is exactly the same; softmax is uniform over all $n$ nodes, and with $W_V=I$ each node's output is $\frac1n\sum_k h_k$ — the global mean over the entire graph. I have to be careful here: a huge constant score cannot "wash out" a nonconstant spatial bias, because softmax subtracts row-wise constants implicitly. So the exact mean-readout construction needs a bias-free head, or a head whose pairwise bias is zeroed for this purpose. With that caveat, vanilla self-attention represents MEAN READOUT for free, because each node can already see the whole graph. That's the deep reason the supernode trick works and also why I don't strictly need it: globality is built in.

But I'll still add a dedicated readout node, for a clean reason. Rather than averaging the final node states (which fixes the pooling to be a mean), I'd rather *learn* the pooling. So I add one special virtual node — call it [VNode] — connected to every real node, exactly like the [CLS] token in a sentence encoder. It participates as a normal node through all the layers, attending to everyone and being attended to; and its representation in the final layer becomes $h_G$, the graph vector. Because it's a real participant, the "pooling" is whatever attention learns, not a hard-wired mean — strictly more flexible than a fixed READOUT, and (from the fact above) the network can still fall back to plain mean-pool if that's optimal, so I lose nothing.

One catch with the [VNode]. It's connected to all nodes, so its shortest-path distance to every node is $1$, and likewise every node is at distance $1$ from it. But that connection is *virtual* — it isn't a real edge in the molecule, and I don't want the spatial bias to treat "one hop to the readout node" the same as "one real bond away." If I let $b_{\phi(\text{VNode},\cdot)}=b_1$ share the scalar with genuine one-hop neighbors, the model would conflate the artificial readout link with real chemical adjacency. So I reset the spatial bias on every VNode-to-node and node-to-VNode pair to its *own distinct learnable scalar*, separate from the $b_\phi$ table for physical distances. That keeps the virtual connections distinguishable from physical ones — the readout pathway gets its own learned strength.

Let me now sanity-check the data side, because all of this presupposes I can precompute the structural quantities per graph. The raw categorical columns need disjoint id ranges before a shared embedding table sees them, so I offset feature column $r$ by a fixed multiple of $r$ and keep zero reserved for padding. From the edge index and edge attributes I build the adjacency, run Floyd-Warshall to get all-pairs shortest-path distances (that's my $\phi$) and the predecessor table, recover the actual edge sequence along each shortest path (for the edge bias), and read off in/out degrees. Unreachable pairs get a sentinel distance from the graph algorithm. All of this is graph-native, computed once per graph before any learning. Then the collator pads a batch of graphs to dense tensors, shifts real node, distance, and edge-path ids by one so zero remains padding, masks too-long structural distances with $-\infty$, and leaves one extra row and column for the [VNode]. I also need the attention module to receive a key padding mask, because padding nodes must never become attended-to keys.

Let me write the preprocessing first, because if this part is wrong then the embeddings and shortest-path tables are indexed incorrectly:

```python
def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset,
                                      dtype=torch.long, device=x.device)
    return x + feature_offset

def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True
    if edge_attr.dim() == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )

    shortest_path_result, path = floyd_warshall(adj.numpy())
    max_dist = shortest_path_result.max()
    edge_input = gen_edge_input(max_dist, path, attn_edge_type.numpy())
    item.x = x
    item.attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = torch.from_numpy(shortest_path_result).long()
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree
    item.edge_input = torch.from_numpy(edge_input).long()
    return item
```

Then I need the batcher to preserve the padding convention and apply the practical distance cutoff used by the dense implementation:

```python
def pad_1d_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze(x, padlen):
    x = x + 1
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos,
         item.in_degree, item.out_degree, item.x,
         item.edge_input[:, :, :multi_hop_max_dist, :], item.y)
        for item in items
    ]
    (idxs, attn_biases, attn_edge_types, spatial_poses,
     in_degrees, out_degrees, xs, edge_inputs, ys) = zip(*items)
    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    return {
        "idx": torch.LongTensor(idxs),
        "attn_bias": torch.cat([
            pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases
        ]),
        "attn_edge_type": torch.cat([
            pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types
        ]),
        "spatial_pos": torch.cat([
            pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses
        ]),
        "in_degree": torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]),
        "out_degree": torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees]),
        "x": torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs]),
        "edge_input": torch.cat([
            pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist)
            for i in edge_inputs
        ]),
        "y": torch.cat(ys),
    }
```

Now I can make the node-input module add the degree embeddings and prepend the graph token:

```python
class GraphNodeFeature(nn.Module):
    def __init__(self, num_atoms, num_in_degree, num_out_degree, hidden_dim):
        super().__init__()
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        # centrality encoding: a learnable vector per (in/out) degree value
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        # the [VNode] / graph token, used for readout
        self.graph_token = nn.Embedding(1, hidden_dim)

    def forward(self, batched_data):
        x, in_degree, out_degree = (batched_data["x"],
                                    batched_data["in_degree"],
                                    batched_data["out_degree"])
        n_graph = x.size(0)
        node_feature = self.atom_encoder(x).sum(dim=-2)          # raw node features
        node_feature = (node_feature                            # h_i^(0) = x_i
                        + self.in_degree_encoder(in_degree)     #   + z^-_{deg^-}
                        + self.out_degree_encoder(out_degree))  #   + z^+_{deg^+}
        graph_token = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        return torch.cat([graph_token, node_feature], dim=1)    # prepend [VNode]
```

I build the per-pair additive term as the spatial bias from SPD, the VNode-distance reset, and the edge bias averaged along shortest paths. Each term gets its own learnable table, one scalar per head:

```python
class GraphAttnBias(nn.Module):
    def __init__(self, num_heads, num_edges, num_spatial, num_edge_dis,
                 edge_type="multi_hop", multi_hop_max_dist=20):
        super().__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        # spatial encoding: one learnable scalar (per head) per SPD value -> b_phi
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        # edge encoding: a learnable embedding per edge feature, plus a per-path-
        # position weight (edge_dis_encoder) -> c_ij
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        # distinct learnable scalar for the virtual [VNode] connections
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

    def forward(self, batched_data):
        attn_bias, spatial_pos, x = (batched_data["attn_bias"],
                                     batched_data["spatial_pos"],
                                     batched_data["x"])
        edge_input, attn_edge_type = (batched_data["edge_input"],
                                      batched_data["attn_edge_type"])
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone().unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # --- spatial encoding: A_ij += b_{phi(i,j)} (offset by 1 for the VNode row/col)
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # --- reset VNode distances to their own distinct learnable scalar
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # --- edge encoding: average edge-feature . weight along the shortest path
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
            edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            edge_input = self.edge_encoder(edge_input).mean(-2)  # embed each edge
            max_dist = edge_input.size(-2)
            eib = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            eib = torch.bmm(eib, self.edge_dis_encoder.weight.reshape(
                -1, self.num_heads, self.num_heads)[:max_dist, :, :])  # per-position w
            edge_input = eib.reshape(max_dist, n_graph, n_node, n_node,
                                     self.num_heads).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2)                     # sum over the path,
                          / spatial_pos_.float().unsqueeze(-1)   # divide by length:
                          ).permute(0, 3, 1, 2)                  # -> the average c_ij
        else:
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        return graph_attn_bias + attn_bias.unsqueeze(1)
```

The attention itself can stay standard once I add the structural bias to the raw scores before softmax — that one line is the whole structural intervention:

```python
attn_weights = torch.bmm(q, k.transpose(1, 2))     # QK^T / sqrt(d) (scaling folded into q)
attn_weights = attn_weights + attn_bias.reshape(bsz * heads, length, length)
if key_padding_mask is not None:
    attn_weights = attn_weights.view(bsz, heads, length, length)
    attn_weights = attn_weights.masked_fill(
        key_padding_mask[:, None, None, :], float("-inf")
    )
    attn_weights = attn_weights.view(bsz * heads, length, length)
attn_probs = softmax(attn_weights, dim=-1)
out = torch.bmm(attn_probs, v)
```

Then the block is a plain pre-LN Transformer layer:

```python
class GraphormerLayer(nn.Module):
    def __init__(self, d, ffn_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d, num_heads, dropout=dropout)
        self.attn_ln = nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, ffn_dim), nn.Linear(ffn_dim, d)
        self.ffn_ln = nn.LayerNorm(d)
        self.act = nn.GELU()

    def forward(self, x, attn_bias, padding_mask=None):
        # pre-LN: MHA(LN(x)) + x, then FFN(LN(.)) + .
        h, _ = self.self_attn(
            self.attn_ln(x),
            attn_bias=attn_bias,
            key_padding_mask=padding_mask,
        )
        x = x + h
        h = self.fc2(self.act(self.fc1(self.ffn_ln(x))))
        return x + h
```

The whole model now embeds nodes with centrality, builds the structural bias once, runs the pre-LN stack, and reads the graph vector off the [VNode] at position 0:

```python
class Graphormer(nn.Module):
    def __init__(self, num_atoms, num_in_degree, num_out_degree, num_edges,
                 num_spatial, num_edge_dis, d, ffn_dim, num_heads, n_layers):
        super().__init__()
        self.node_feature = GraphNodeFeature(num_atoms, num_in_degree,
                                             num_out_degree, d)
        self.attn_bias = GraphAttnBias(num_heads, num_edges, num_spatial,
                                       num_edge_dis)
        self.layers = nn.ModuleList(
            [GraphormerLayer(d, ffn_dim, num_heads) for _ in range(n_layers)])
        self.out = nn.Linear(d, 1)

    def forward(self, batched_data):
        padding_mask = batched_data["x"][:, :, 0].eq(0)
        padding_mask = torch.cat(
            [padding_mask.new_zeros(padding_mask.size(0), 1), padding_mask],
            dim=1,
        )
        x = self.node_feature(batched_data)        # node feats + centrality + [VNode]
        bias = self.attn_bias(batched_data)        # b_phi + c_ij + VNode reset
        x = x.transpose(0, 1)                       # [N+1, B, d]
        for layer in self.layers:
            x = layer(x, bias, padding_mask=padding_mask)
        graph_repr = x[0]                           # [VNode] is the readout
        return self.out(graph_repr)
```

So the causal chain: raw self-attention is a permutation-equivariant function of a bag of node vectors and sees no graph structure — no degree, no distance, no edges. A graph has no canonical order, so sequence positional encodings don't port; but it *does* expose degree (per-node), shortest-path distance (per-pair), and the edges along connecting paths (per-pair). Match each signal to the place with its arity: degree goes into the per-node input as a learnable centrality embedding; distance goes into the per-pair score as a shared learnable bias $b_\phi$, which simultaneously keeps the global receptive field and lets the model learn locality; the connecting edges go into the per-pair score as a length-normalized average of learnable edge–weight products $c_{ij}$. Add a [VNode] wired to all nodes for a learned readout, giving it a distinct virtual-distance scalar so its artificial links aren't confused with real bonds. With those biases the spatial encoding lets one attention layer carve out the one-hop neighborhood and reproduce MEAN/SUM/MAX aggregation and COMBINE — so GCN, GraphSAGE, and GIN are special cases — while there exist 1-WL-indistinguishable graph pairs whose all-pairs distance profiles differ, and the spatial encoding exposes exactly that missing signal. The block stays a standard pre-LN Transformer; the only structural intervention is one additive bias on the attention scores.
