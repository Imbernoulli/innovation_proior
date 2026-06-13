The equivariant rung did almost exactly what I predicted, and the one place it did not is the seam I want
to pull on. Fold jumped from SchNet's 0.184 to 0.331 — that is the directional-discrimination gain I
argued for, equivariance helping most precisely where two folds share distance statistics and differ in
direction, and it is the strongest confirmation that the radial ceiling was the real problem. EC rose
from 0.589 to 0.747, a large gain too, because active-site geometry benefits from the directional
structure the coordinate channel injects into the distances. But GO-BP went the *wrong way*: 0.238, down
from SchNet's 0.245. I flagged that risk — GO-BP is a coarse multilabel metric and its signal is largely
sequence-composition rather than fine geometry — and the regression confirms it: piling batch-norm-heavy
equivariant machinery onto a task that wanted the cheap encoder hurt it slightly. So EGNN is the stronger
baseline overall (it wins EC and Fold by wide margins), but it is not uniformly better, and the GO-BP
regression plus the *shape* of the EGNN layer tells me what it is still missing. The equivariant
coordinate update is a single scalar weight per edge: `φ_x(m_ij)` reads one edge's geometry and moves the
coordinate along that one difference vector. The neighbors never talk to each other, and the encoder never
asks what *kind* of edge each one is. Two things follow that I want the next rung to fix: edges that mean
different things are all transformed by the same machinery, and the relational structure between a
residue's contacts is invisible.

Let me take the edge-type problem first, because it is the more basic structural error and it is the
one most likely to recover GO-BP while holding the gains. Stare at what a residue's edges actually mean
in a folded protein. Some edges are "these two residues are adjacent along the chain" — sequential,
backbone, local. Some are "these two residues are far apart in the chain but close in space" — a tertiary
contact, the whole point of folding. Those two kinds of edge carry completely different geometric
information, yet both SchNet and EGNN build a single kNN graph and run one shared transform over every
edge. That is the same error as using one global learning rate when parameters live at different scales: I
am forcing one operator onto edges that mean different things, and a kNN graph in particular *mixes* the
two — a residue's `k` nearest neighbors are a blend of its sequential neighbors and its spatial contacts,
indistinguishable to the layer. So the next move is to type the edges and let the transform depend on the
type.

That is a relational graph convolution: one learnable kernel `W_r` per edge type `r`, shared across all
edges of that type, with the per-type neighbor sums combined as `Σ_r W_r Σ_{j∈N_r(i)} (·)`. The key
accounting fact is that the number of kernels is the number of edge *types*, not the number of edges — a
handful, independent of how many edges any protein has. That matters here because the database is
structurally diverse and proteins vary wildly in size; a per-edge tailored kernel would blow up in memory,
and a single shared kernel cannot tell types apart, and the relational convolution sits exactly between
those two — capacity scaling in the number of types, memory not scaling with the number of edges at all.
So the backbone of the rung is relational message passing, and the design question becomes: what are the
types?

I design them from the geometry deliberately. For the sequential structure, I do not lump all backbone
neighbors into one "sequential" type — direction and exact offset along the chain are geometrically
meaningful, the relationship `i → i+1` is not the relationship `i → i-2`. So I type each sequential edge
by its relative position. With offsets `{-2, -1, 0, 1, 2}` that is five sequential relation types,
including the `0` self-relation that gives a clean place for self-information inside the relational
machinery. For the spatial structure I use *two* complementary types, and the reason is exactly the
failure modes of each rule alone. A radius rule — connect `i, j` when `‖pos_i - pos_j‖ < cutoff` — gives
density information in crowded regions but leaves loosely-packed proteins near-edgeless, because no single
fixed radius fits every structure. A kNN rule gives every residue a guaranteed degree but flattens away
the density variation in packed regions, capping everyone at `k` regardless of how crowded. The two are
complementary: radius restores density, kNN puts a floor under the degree so no protein collapses. So I
add both as separate edge types — one radius relation and one kNN relation — for a total of seven relation
types. Seven kernels, that is all.

Now I make this concrete in this task's edit surface, and here the honest accounting matters more than
anywhere on the ladder, because this task's relational encoder is a *stripped-down* version of the
full relational-with-edge-messages design, and I must derive the version the harness actually builds, not
the richer one. The biggest omission: there is **no line graph, no edge-to-edge message passing, and no
angle binning** here. The richest version of this idea would build a graph whose nodes are the edges of
the residue graph, connect two edges that share a residue, type that connection by the binned angle
between them, and run a second relational convolution on it — that is what would let a residue's contacts
*relate* to each other and would deliver the directional, between-edge information that the node-only layer
is blind to. The harness does not expose that branch at all. So this rung gets the relational *node*
convolution and the seven-type graph, but **not** the edge-enhanced angle machinery. I am explicit about
this because it bounds what I can expect: the rung fixes the edge-type error but it does *not* recover the
between-edge directional structure, and that is a real ceiling I will name at the close.

The concrete construction, then. The encoder builds the seven-relation graph itself from `pos`,
`node_feat`, and `batch`: five sequential offset relations enumerated per protein within each batch
element (with the `0` offset as a self-loop sequential relation, and positive/negative offsets as distinct
relation types so direction is preserved), one spatial radius relation (`radius_graph` at `cutoff = 10.0`,
a generous max-neighbor budget so dense regions are not truncated), and one kNN relation (`knn_graph` at
`k = max_neighbors = 16`). The per-relation aggregation is done in a single pass with the scatter trick:
scatter every edge message into a bucket keyed by `dst * num_relation + r`, into an array of size
`num_nodes * num_relation`, reshape to `(num_nodes, num_relation * input_dim)`, and apply one
`Linear(num_relation * input_dim, output_dim)` — that single weight matrix *is* the seven `W_r` stacked
side by side, so all relations are handled in one matmul. The layer also reads an edge feature: for each
edge it concatenates the two endpoint node features, a one-hot of the relation type, the absolute sequence
separation `|i - j|`, and the spatial distance `‖pos_i - pos_j‖` — every geometric quantity in there a
distance, so the encoder stays E(3)-invariant. The harness modulates each message by this edge feature
multiplicatively through a sigmoid gate, `message = h[src] * sigmoid(edge_linear(edge_feat))`, which is a
detail to get right because the canonical relational layer *adds* a projected edge feature rather than
gating by a sigmoid — this task gates. The layer applies the relation-combining linear, then (as a
separate module after the conv, not inside it) a batch norm, a residual short-cut when the dimensions
match, dropout, and ReLU. Six layers, hidden width 512.

Two readout choices follow the relational design and both differ from EGNN. First, `concat_hidden`: the
per-node representation is the concatenation of *all six* layers' hidden states, not just the last, because
the early layers hold local backbone geometry and the later layers hold the propagated global fold, and the
downstream head benefits from seeing every scale — this is a multi-scale readout EGNN did not have. Second,
the graph embedding is a **sum** pool (`global_add_pool`) over those concatenated node embeddings, not
EGNN's mean: summing keeps a notion of the total signal and the protein's size, which the classification
heads can use and which mean would normalize away. The whole scaffold module is in the answer.

So the delta from EGNN is precise. EGNN ran one shared transform over a single kNN graph that blended
sequential and spatial edges, with a single scalar weight per edge and no notion of edge type. GearNet
types the edges into five sequential offsets plus a radius and a kNN relation, gives each type its own
kernel, reads an invariant edge feature, and reads out multi-scale (concat all layers, sum pool). It does
*not* add the line-graph angle machinery the harness omits, so it remains, like its predecessors, a
node-level encoder.

Here is what I expect against EGNN's numbers, stated to be falsified. The clearest prediction is
**GO-BP**: typing the edges and the multi-scale concat readout should recover and exceed the regression —
I expect GO-BP to climb back above SchNet's 0.245 and clear EGNN's 0.238, because the relational structure
lets the cheap, sequence-correlated signal through cleanly rather than drowning it in equivariant
coordinate machinery; if GO-BP does not recover, the edge-typing is not the lever I think it is. **EC**
should rise above EGNN's 0.747 — distinguishing sequential from spatial contacts is exactly what
active-site reasoning wants, and the multi-scale readout helps — so I expect the strongest EC of the
ladder here. **Fold** is the prediction I am most cautious about: EGNN's equivariant coordinate channel
gave it a genuine directional edge on Fold (0.331), and this rung *removed* the explicit coordinate
update and did *not* replace it with the line-graph angle structure that would supply between-edge
direction, so Fold could plausibly come in slightly *below* EGNN even as EC and GO-BP rise — the relational
graph captures which contacts exist and what type they are, but not the angle between two contacts of a
residue, which is precisely the directional cue Fold rewards. If Fold lands near or just under EGNN while
EC and GO-BP clearly lead, that is the signature of this exact trade: the node-relational encoder wins on
contact-type and sequence-correlated structure and pays a little on the pure directional fold cue it
cannot see without the omitted edge branch — and that omitted branch is the obvious thing a further rung
would restore.
