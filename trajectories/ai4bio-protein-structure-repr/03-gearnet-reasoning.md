The equivariant rung did almost exactly what I predicted, and the one place it did not is the seam I want
to pull on. Let me read the three deltas against SchNet before I decide anything, because the pattern in
them is the whole diagnosis. Fold jumped from 0.184 to 0.331 — an absolute gain of `0.147` and a ratio of
`1.80`, very nearly a doubling — and the loss column agrees: EGNN's Fold test loss is `5.71` against
SchNet's `6.60`, `0.89` nats better, so the coordinate channel did not just shuffle the top-1 guess, it
concentrated probability mass. That is the directional-discrimination gain I argued for, equivariance
helping most precisely where two folds share distance statistics and differ in direction, and it is the
strongest confirmation that the radial ceiling was the real problem. EC rose from 0.589 to 0.747, `+0.158`
absolute, `1.27×`, with the loss falling from `2.43` to `1.73`, `0.70` nats better — a large gain too,
because active-site geometry benefits from the directional structure the coordinate channel injects into the
distances. But GO-BP went the *wrong way*: 0.238, down `0.007` from SchNet's 0.245, and this is not a
metric-only wobble, because the GO test loss also moved the wrong way, up from `0.1665` to `0.1758`. Every
one of the three accuracy moves is corroborated by a loss move in the same direction, including the
regression, so I trust that the GO-BP drop is a real effect and not seed noise. I flagged that risk — GO-BP
is a coarse multilabel metric and its signal is largely sequence-composition rather than fine geometry — and
the regression confirms it: piling batch-norm-heavy equivariant machinery onto a task that wanted the cheap
encoder hurt it slightly. So EGNN is the stronger baseline overall, winning EC and Fold by wide margins, but
it is not uniformly better, and the GO-BP regression plus the *shape* of the EGNN layer tells me what it is
still missing. The equivariant coordinate update is a single scalar weight per edge: `φ_x(m_ij)` reads one
edge's geometry and moves the coordinate along that one difference vector. The neighbors never talk to each
other, and the encoder never asks what *kind* of edge each one is. Two things follow that I want the next
rung to fix: edges that mean different things are all transformed by the same machinery, and the relational
structure between a residue's contacts is invisible.

Let me take the edge-type problem first, because it is the more basic structural error and it is the one most
likely to recover GO-BP while holding the gains. Stare at what a residue's edges actually mean in a folded
protein. Some edges are "these two residues are adjacent along the chain" — sequential, backbone, local.
Some edges are "these two residues are far apart in the chain but close in space" — a tertiary contact, the
whole point of folding. Those two kinds of edge carry completely different geometric information, yet both
SchNet and EGNN build a single kNN graph and run one shared transform over every edge. That is the same
error as using one global learning rate when parameters live at different scales: I am forcing one operator
onto edges that mean different things, and a kNN graph in particular *mixes* the two — a residue's `k`
nearest neighbors are a blend of its sequential neighbors and its spatial contacts, indistinguishable to the
layer. And this is exactly the kind of error most likely to be behind the GO-BP regression: if the useful
GO signal is carried by which residues are sequence-neighbors versus which are genuine spatial contacts, a
transform that cannot separate the two, wrapped in equivariant machinery that further dilutes the cheap
compositional cue, would lose ground. So the next move is to type the edges and let the transform depend on
the type.

Before I build that, I make sure it beats the two cheaper things I could do instead. The first temptation is
to simply scale EGNN — more layers, more width — and hope GO-BP recovers. But the GO-BP regression is not a
capacity failure: EGNN already runs about `11M` parameters and is `3.5`-plus nats below uniform on EC, so it
is not starved of capacity, it is applying the *wrong inductive bias* to GO — spending equivariant
coordinate machinery and batch-norm coupling on a signal that is largely cheap and compositional. More
capacity of the same wrong kind will not turn a `−0.007` into a gain; it would more likely deepen the
regression. The second temptation is to jump straight to the richest relational design with the line-graph
angle branch, which would target the between-edge directional structure EGNN's single scalar weight could
not represent. But that branch is not in this edit surface, and reaching for it now would mean inventing
machinery the harness does not build; I would rather fix the more basic, clearly-diagnosed edge-type error
first and name the angle branch as the thing left for later. So the relational node convolution is the right
size of step: it addresses the exact structural error the delta pattern points at, without over-reaching.

That is a relational graph convolution: one learnable kernel `W_r` per edge type `r`, shared across all
edges of that type, with the per-type neighbor sums combined as `Σ_r W_r Σ_{j∈N_r(i)} (·)`. The key
accounting fact is that the number of kernels is the number of edge *types*, not the number of edges — a
handful, independent of how many edges any protein has. I can put the number on it directly: with seven
types and a hidden width of 512, the relation-combining transform is a single `Linear(7·512, 512)`, about
`7 · 512 · 512 ≈ 1.84M` parameters, and that one weight matrix *is* the seven `W_r` stacked side by side, so
all relations are handled in one matmul whose cost does not scale with the edge count at all. That matters
here because the database is structurally diverse and proteins vary wildly in size; a per-edge tailored
kernel would blow up in memory and could not generalize across proteins with different edge sets, while a
single shared kernel cannot tell types apart, and the relational convolution sits exactly between those two
— capacity scaling in the number of types, memory not scaling with the number of edges. So the backbone of
the rung is relational message passing, and the design question becomes: what are the types?

I design them from the geometry deliberately. For the sequential structure, I do not lump all backbone
neighbors into one "sequential" type — direction and exact offset along the chain are geometrically
meaningful, the relationship `i → i+1` is not the relationship `i → i-2`. So I type each sequential edge by
its relative position. With offsets `{-2, -1, 0, 1, 2}` that is five sequential relation types, including
the `0` self-relation that gives a clean place for self-information inside the relational machinery and
distinguishing positive from negative offsets so the direction along the chain is preserved rather than
folded together. For the spatial structure I use *two* complementary types, and the reason is exactly the
failure modes of each rule alone. A radius rule — connect `i, j` when `‖pos_i - pos_j‖ < cutoff` — gives
density information in crowded regions but leaves loosely-packed proteins near-edgeless, because no single
fixed radius fits every structure. A kNN rule gives every residue a guaranteed degree but flattens away the
density variation in packed regions, capping everyone at `k` regardless of how crowded. The two are
complementary: radius restores density where kNN would flatten it, kNN puts a floor under the degree so no
protein collapses to no edges. So I add both as separate edge types — one radius relation and one kNN
relation — for a total of seven relation types. Seven kernels, that is all.

Now I make this concrete in this task's edit surface, and here the honest accounting matters more than
anywhere on the ladder, because this task's relational encoder is a *stripped-down* version of the full
relational-with-edge-messages design, and I must derive the version the harness actually builds, not the
richer one. The biggest omission: there is **no line graph, no edge-to-edge message passing, and no angle
binning** here. The richest version of this idea would build a graph whose nodes are the edges of the
residue graph, connect two edges that share a residue, type that connection by the binned angle between
them, and run a second relational convolution on it — that is what would let a residue's contacts *relate*
to each other and would deliver the directional, between-edge information that the node-only layer is blind
to, and it is exactly the between-edge angle structure I noticed EGNN could not represent with its single
scalar weight per edge. The harness does not expose that branch at all. So this rung gets the relational
*node* convolution and the seven-type graph, but **not** the edge-enhanced angle machinery. I am explicit
about this because it bounds what I can expect: the rung fixes the edge-type error but it does *not* recover
the between-edge directional structure, and that is a real ceiling I will name at the close — the same
directional cue EGNN's coordinate channel supplied and that this rung is about to remove.

The concrete construction, then. The encoder builds the seven-relation graph itself from `pos`, `node_feat`,
and `batch`: five sequential offset relations enumerated per protein within each batch element (with the `0`
offset as a self-loop sequential relation, and positive/negative offsets as distinct relation types so
direction is preserved), one spatial radius relation (`radius_graph` at `cutoff = 10.0`, a generous
max-neighbor budget so dense regions are not truncated), and one kNN relation (`knn_graph` at `k =
max_neighbors = 16`). The per-relation aggregation is done in a single pass with a scatter trick I want to
trace concretely so I trust the bookkeeping. I scatter every edge message into a bucket keyed by `dst ·
num_relation + r`, into an array of size `num_nodes · num_relation`. Take node `2` receiving a message on
relation `5` with seven relations: its bucket index is `2·7 + 5 = 19`, and node `2` owns the contiguous
block of buckets `14` through `20`, one per relation, inside the length-`7N` array. After the scatter I
reshape to `(num_nodes, num_relation · input_dim)` — each node's row is its seven per-relation aggregates
laid end to end — and apply one `Linear(num_relation · input_dim, output_dim)`, which reads all seven blocks
of every node at once; that single matmul *is* the seven `W_r` applied to their respective per-relation
sums and summed. The layer also reads an edge feature, and here I check the width: for each edge it
concatenates the two endpoint node features (`28 + 28`), a one-hot of the relation type (`7`), the absolute
sequence separation `|i - j|` (`1`), and the spatial distance `‖pos_i - pos_j‖` (`1`), which is `65`
numbers, and every geometric quantity in there is a distance or a sequence index, so the encoder stays
`E(3)`-invariant. The harness modulates each message by this edge feature multiplicatively through a sigmoid
gate, `message = h[src] · sigmoid(edge_linear(edge_feat))`, where `edge_linear` maps the `65`-dim edge
feature to the layer's input width so the gate multiplies `h[src]` elementwise — and this is a detail to get
right because the canonical relational layer *adds* a projected edge feature rather than gating by a sigmoid;
this task gates. The layer applies the relation-combining linear, adds a self-loop transform of the node's
own features, then (as a separate module after the conv, not inside it) a batch norm, a residual short-cut
when the dimensions match, dropout, and ReLU. Six layers, hidden width 512.

Two of those details reward a second look. The sigmoid gate is weaker than the canonical additive edge
feature, and I want to be honest about what that costs. A sigmoid lives in `(0,1)`, so
`h[src] · sigmoid(edge_linear(edge_feat))` can only *attenuate* the source message channel by channel — it
can down-weight an irrelevant contact or pass a relevant one through, but it can never add edge-specific
content or flip a sign the way an additive projected feature could. So the gate is a per-channel relevance
filter, not a content injector. That said, it is load-bearing precisely because the pre-gate message is just
`h[src]` with nothing concatenated: the gate is the *only* place the `65`-dim edge feature — the relation
one-hot, the sequence separation, the spatial distance — enters the message content at all. So relation
information reaches a node by two distinct routes, a hard one and a soft one: the hard route is the scatter
bucket that sends each edge into its type's `W_r`, and the soft route is the gate that lets the edge feature
modulate which channels of the source survive. The self-loop is a third, cleaner route: `out = W_r-combined
+ self_loop(h)` adds a direct learned transform of the node's own features, independent of any aggregation,
so a residue always has an unmediated path from its own features to its update. And the short-cut has a
subtlety I have to get right: it adds the incoming `h` back only when `hidden.shape == h.shape`, which holds
for layers one through five where the width is `512 → 512`, but *not* for layer zero, where `h` enters at
the raw `28`-dim node feature and the output is `512` — so the first layer runs without a residual and the
later five run with one. That is the correct behavior for a dimension-changing first layer, and worth
noticing so I do not mistake the missing residual for a bug.

The graph-construction details carry their own correctness checks. The two spatial relations are given
deliberately asymmetric neighbor budgets that make their complementarity concrete: the radius relation is
built with a generous `max_num_neighbors = 512` so a densely packed core residue can keep all its
within-cutoff contacts and the *density* is preserved, while the kNN relation caps at `k = 16` so every
residue, however loosely packed, gets a floor of sixteen edges — the radius relation reports how crowded a
neighborhood is, the kNN relation guarantees nobody is starved, exactly the two failure modes I wanted to
cover with two types. The sequential relations are enumerated per protein inside each batch element rather
than globally, and this is not incidental: a global offset would wrongly wire the last residue of one
protein to the first residue of the next, inventing a backbone bond across a batch boundary, so the
per-protein masking that restricts offset edges to residues sharing a batch index is what keeps the
sequential types meaning what they say.

Tracing the widths through a layer once fixes the wiring in my head and catches the place the dimensions
change. At layer zero `h` is the raw `28`-dim node feature: the message `h[src]` is `28`-wide, the gate
`sigmoid(edge_linear(edge_feat))` maps the `65`-dim edge feature to `28` and multiplies elementwise, the
per-relation scatter fills an array of shape `(7N, 28)` which reshapes to `(N, 196)`, the relation linear
`Linear(196, 512)` lifts it to `512`, and the self-loop `Linear(28, 512)` adds a direct transform of the
node's own `28` features — so layer zero legitimately has no residual because it is a `28 → 512` map. At
layers one through five `h` is `512`-wide: the scatter array is `(7N, 512)` reshaping to `(N, 3584)`, the
relation linear is `Linear(3584, 512)`, the self-loop is `Linear(512, 512)`, and now the short-cut fires
because output and input widths agree. Everything closes, and the only place any coordinate touched the
computation was through the invariant distances inside the `65`-dim edge feature — so, as with every rung on
this ladder, the encoder is `E(3)`-invariant by the construction of what it reads, now checked at the level
of shapes.

Two readout choices follow the relational design and both differ from EGNN. First, `concat_hidden`: the
per-node representation is the concatenation of *all six* layers' hidden states, `6 · 512 = 3072` dimensions
projected by a single `Linear(3072, out_dim)`, not just the last layer, because the early layers hold local
backbone geometry and the later layers hold the propagated global fold, and the downstream head benefits
from seeing every scale — this is a multi-scale readout EGNN did not have, and it is a second, cheaper route
to letting the cheap local signal survive to the head instead of being overwritten by the deepest layer.
That mechanism is worth spelling out, because it is a second lever on the same GO-BP problem the edge-typing
attacks. The first layer's hidden state is still close to the raw compositional features, only lightly mixed
by one round of relational aggregation, so it carries the cheap sequence-correlated signal almost intact;
the sixth layer's hidden state has propagated the global fold and may well have written over that cheap
signal in favor of higher-order structure. A last-layer-only readout, as in EGNN, forces the head to read
whatever survived to the top. The concat readout instead hands the head all six layers at once, so the
shallow, composition-rich representation reaches the classifier directly regardless of what the deep layers
did to it — which is exactly the kind of signal GO-BP rewards and exactly the signal EGNN's deep equivariant
stack was drowning. So edge-typing and the concat readout are two independent reasons to expect GO-BP to
recover, and if it does not despite both, my read of what GO-BP wants is wrong.
Second, the graph embedding is a **sum** pool (`global_add_pool`) over those concatenated node embeddings,
not EGNN's mean: summing keeps a notion of the total signal and the protein's size, which the classification
heads can use and which mean would normalize away. The whole scaffold module is in the answer.

So the delta from EGNN is precise. EGNN ran one shared transform over a single kNN graph that blended
sequential and spatial edges, with a single scalar weight per edge and no notion of edge type. GearNet types
the edges into five sequential offsets plus a radius and a kNN relation, gives each type its own kernel,
reads an invariant edge feature through a sigmoid gate, and reads out multi-scale (concat all layers, sum
pool). It does *not* add the line-graph angle machinery the harness omits, so it remains, like its
predecessors, a node-level encoder — and it removes EGNN's explicit equivariant coordinate update, which is
the one thing I have to weigh honestly when I predict Fold.

Here is what I expect against EGNN's numbers, stated to be falsified. The clearest prediction is **GO-BP**:
typing the edges and the multi-scale concat readout should recover and exceed the regression — I expect
GO-BP to climb back above SchNet's 0.245 and clear EGNN's 0.238, because the relational structure lets the
cheap, sequence-correlated signal through cleanly rather than drowning it in equivariant coordinate
machinery; if GO-BP does not recover, the edge-typing is not the lever I think it is. **EC** should rise
above EGNN's 0.747 — distinguishing sequential from spatial contacts is exactly what active-site reasoning
wants, and the multi-scale readout helps — so I expect the strongest EC of the ladder here. **Fold** is the
prediction I am most cautious about, and the caution is quantitative: EGNN's equivariant coordinate channel
bought Fold `0.147` of accuracy and `0.89` nats of loss, a genuine directional edge, and this rung *removed*
that explicit coordinate update and did *not* replace it with the line-graph angle structure that would
supply between-edge direction. So Fold could plausibly come in slightly *below* EGNN's 0.331 even as EC and
GO-BP rise — the relational graph captures which contacts exist and what type they are, but not the angle
between two contacts of a residue, which is precisely the directional cue Fold rewards. If Fold lands near or
just under EGNN while EC and GO-BP clearly lead, that is the signature of this exact trade: the
node-relational encoder wins on contact-type and sequence-correlated structure and pays a little on the pure
directional fold cue it cannot see without the omitted edge branch — and that omitted branch is the obvious
thing a further rung would restore.
