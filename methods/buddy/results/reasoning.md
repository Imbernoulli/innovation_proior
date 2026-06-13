I have a sharp, uncomfortable diagnosis to start from: the structural signal that decides links on these
graphs — neighborhood overlap, the common-neighbor / Adamic–Adar / resource-allocation family — is
exactly what a plain message-passing GNN cannot represent, because a GNN gives one embedding per node
and two automorphic nodes get identical embeddings, so for a fixed source u it cannot distinguish a
true neighbor v from a structurally-identical non-neighbor v′. And the tool that *can* represent it —
extract the k-hop enclosing subgraph around each pair, label every node by its distances to the two
endpoints, run a GNN with pooling — does so only by recomputing massively overlapping subgraph
structure once per candidate edge, which is O(|E|) per edge on power-law graphs and overflows memory if
I try to precompute the subgraphs. So I want the expressiveness of the subgraph method at the cost of a
plain GNN. The thing I have to find is whether the per-edge subgraph computation can be replaced by
something computed *per node, once*.

Let me look hard at *what* the subgraph method actually uses, because if most of its power comes from a
small, structured quantity I might be able to compute that quantity directly. In the labeled enclosing
subgraph, every node w carries a label (d(w,u), d(w,v)) — its distance to each endpoint. The
downstream GNN reads these labels and pools. What does pooling over those labels amount to? It is, in
effect, *counting*: how many nodes are at distance d_u from u and d_v from v, for each small (d_u, d_v).
Call that count A_{uv}[d_u, d_v] = number of nodes at distance exactly d_u from u and exactly d_v from
v. This single family of counts is the load-bearing content of the labeling. Notice A_{uv}[1,1] is the
number of nodes adjacent to both u and v — that is *common neighbors exactly*, the triangle count, the
backbone of every heuristic. A_{uv}[1,2] and A_{uv}[2,1] count the one-hop/two-hop overlap, A_{uv}[2,2]
the two-hop/two-hop overlap, and so on. So the distance-label counts *subsume* the overlap heuristics:
CN is A[1,1], and a learned function of the whole count table can express AA, RA, Katz-like mixtures,
whatever the data wants. If the predictive power of the subgraph method really sits in these counts —
and intuition says most of it sits in the *low*-distance counts, because A[1,1] is the dominant LP
signal — then I do not need the subgraph at all. I need the counts.

So restate the goal precisely: for each candidate pair (u, v), compute the count table
{A_{uv}[d_u, d_v] : d_u, d_v ≤ k} for a small receptive field k, plus the "boundary" counts
B_{uv}[d] = (nodes at distance d from u but beyond distance k from v) that catch the mass that falls
outside the joint window so the features don't depend on graph size — and do it *without* building the
subgraph, ideally from quantities I can attach to each node once. Let me write the counts in terms of
neighborhoods. Let N_d(u) be the set of nodes within distance ≤ d of u. The number of nodes at distance
exactly d_u from u and ≤ d_v from v is essentially an *intersection cardinality*:
|N_{d_u}(u) ∩ N_{d_v}(v)|, and the exact-distance counts A_{uv}[d_u, d_v] follow by inclusion-exclusion
across the smaller windows — subtract off the counts already attributed to nearer shells. The boundary
B_{uv}[d] = |N_d(u)| − B_{uv}[d−1] − Σ_{i,j≤d} A_{uv}[i,j], peeling the within-window joint counts off
the d-hop neighborhood of u. So everything reduces to two primitives per node pair: *the cardinality of
each node's d-hop neighborhood*, |N_d(u)|, and *the cardinality of the intersection of two nodes'
neighborhoods*, |N_{d_u}(u) ∩ N_{d_v}(v)|. If I can get those two cheaply, I get the whole count table
by arithmetic.

Now the obstacle: computing |N_{d_u}(u) ∩ N_{d_v}(v)| exactly for every candidate pair means
intersecting potentially huge sets, and storing N_d(u) explicitly for every node is back to the memory
blowup. This is precisely the situation streaming/database systems face — estimate set cardinalities and
set intersections over enormous sets without materializing them — and they solve it with *sketches*:
small fixed-size summaries of a set from which cardinality and similarity can be estimated. Two
classical sketches fit exactly the two primitives I need. HyperLogLog summarizes a set in O(1) space
(a small register array) and estimates its cardinality |S| via a cardinality function card(·) on the
sketch; the relevant property is that the HyperLogLog sketch of a *union* is the elementwise *max* of
the two sketches, so I can build neighborhood sketches by max-aggregating. MinHash summarizes a set so
that the Hamming similarity between two MinHash sketches estimates their *Jaccard* similarity
J(S, T) = |S ∩ T| / |S ∪ T|; its sketch of a union is the elementwise *min*. Put them together: a set
intersection cardinality is |S ∩ T| = J(S, T) · |S ∪ T| ≈ H(MinHash(S), MinHash(T)) · card(max(HLL(S),
HLL(T))), where H is the Hamming similarity. That is the whole estimator — MinHash gives the Jaccard,
HyperLogLog gives the union size, the product gives the intersection size — and crucially the sketch
sizes are fixed constants *independent of graph size*, with a precision/cost knob.

This makes the per-node primitive concrete. Give each node u an initial sketch pair (m_u^{(0)},
h_u^{(0)}) — the MinHash and HyperLogLog of the singleton {u}. The d-hop neighborhood sketch follows a
beautiful recurrence: N_d(u) = ∪_{v ∈ N(u)} N_{d−1}(v), and because MinHash-of-union is elementwise min
and HLL-of-union is elementwise max, the d-hop sketches are m_u^{(d)} = min_{v ∈ N(u)} m_v^{(d−1)} and
h_u^{(d)} = max_{v ∈ N(u)} h_v^{(d−1)}. That is *message passing* — but the messages are the sketches,
and the aggregators are elementwise min and max. So with k rounds of min/max sketch propagation I get,
for every node, the sketches of all its d-hop neighborhoods up to d = k, in O(k|E|·h) time with h the
sketch size — node-level, once, no subgraph. Then for any candidate pair (u, v) I read off
|N_{d_u}(u)| ≈ card(h_u^{(d_u)}), |N_{d_u}(u) ∩ N_{d_v}(v)| ≈ H(m_u^{(d_u)}, m_v^{(d_v)}) · card(max(...)),
and turn those into the count table A and boundary B by the inclusion-exclusion arithmetic above. The
expensive per-edge subgraph construction has been replaced by cheap per-pair sketch comparisons on
fixed-size summaries.

That is the structural side. The other half of a link predictor is node *features*, and here I should
not overthink it. The subgraph method propagates node features over the subgraph; I can propagate them
over the *whole graph* with an ordinary GNN, or — and this is the scalability lever — I can fix the
feature propagation, since fixed and learned propagation give nearly equivalent link-prediction
performance. A fixed sparse propagation x_u^{(l)} = mean_{v ∈ N(u)} x_v^{(l−1)} can be precomputed once
with scatter operations, and concatenating the features diffused at each hop, Z = [X^{(0)} ‖ X^{(1)} ‖
… ‖ X^{(k)}], gives a multi-scale node feature with no message passing at training time at all. This is
the decoupling-propagation-from-learning idea (SGC/SIGN, Wu et al. 2019; Rossi et al. 2020) applied to
the feature stream.

Now assemble the predictor. For a pair (u, v): take the multi-scale node features z_u, z_v (propagated
or GNN-encoded), combine them by the Hadamard product z_u ⊙ z_v (the standard edge-pooling
combination, which is subgraph-free and empirically beats mean/sum pooling for links), and *concatenate*
the structural count features {B_{uv}[d], A_{uv}[d_u, d_v] : d, d_u, d_v ≤ k}. Pass the concatenation
through an MLP ψ to a single logit: p(u, v) = ψ(z_u ⊙ z_v, {B_{uv}[d], A_{uv}[d_u, d_v]}). The MLP
learns *how* to weight the structural counts — recovering CN, AA, RA, or any mixture the data prefers —
and *how* to fuse them with the feature interaction. Two free design points worth settling: the
readout combines the two endpoints by an *edge*-level pooling (Hadamard), not a graph-level pool over
all subgraph nodes, because edge pooling is both subgraph-free and the better-performing choice for
links; and the structural counts enter as *direct inputs to the MLP*, not as labels propagated through
a GNN, because most of the predictive content is in the low-distance counts and an MLP directly on them
matches or beats propagating them. The receptive field k is small — k = 2 or 3 — both because the
low-distance counts dominate and because keeping k below the size scale guarantees a fixed-length,
size-independent feature vector that resists overfitting; the sketch sizes (HyperLogLog precision p,
number of MinHash permutations) are the accuracy/speed knobs, again independent of N.

Two distinct realizations fall out of *when* I compute the sketches. The full-graph version (call it
ELPH) folds the sketch propagation (min for MinHash, max for HyperLogLog) and the feature propagation
into the message-passing loop, computing the edge features on the fly each forward pass — clean and
expressive, provably stronger than a plain message-passing GNN because the structural features
distinguish automorphic-node links a GNN cannot, but it needs the whole graph (and its sketches) in GPU
memory. The scalable version (BUDDY) observes that *both* the sketches and the fixed feature
propagation can be precomputed once with sparse scatter operations — scatter_min for MinHash,
scatter_max for HyperLogLog, scatter_mean for features — so the per-edge structural and node features
{B_{uv}[d], A_{uv}[d_u, d_v]} and z_u, z_v are materialized ahead of time and cached, and training/
inference is then *just an MLP on precomputed edge features*. That converts a GNN into an MLP: no
subgraph sampling, no message passing at train time, memory independent of the graph fitting on the
GPU. The cost is O(kEd) for feature propagation plus O(kEh) for sketch propagation in preprocessing,
then O(k²h + kd²) per link prediction — both independent of the size of the graph, which is the entire
point.

Let me sanity-check the expressiveness claim, because it is the justification for the structural counts
existing at all. A plain message-passing GNN assigns automorphic nodes identical embeddings, so it
cannot separate certain non-automorphic *links* whose endpoints are automorphic. The structural counts
A_{uv}[d_u, d_v] are computed *relative to the pair* — they depend on the joint neighborhood geometry of
u and v, not on either node alone — so two links that a GNN cannot separate can have different count
tables and thus different scores. With exact estimates the model provably does not suffer the
automorphic-node problem, and it is strictly more powerful than message-passing GNNs: every pair of
links a GNN can separate, this can too, and there exist pairs it separates that a GNN cannot. The
counts are the mechanism, and A[1,1] = common neighbors is the concrete witness — a quantity a GNN
cannot count and this model reads off directly.

So the method is: per node, propagate MinHash and HyperLogLog sketches by elementwise min/max for k
hops to summarize every d-hop neighborhood; per candidate pair, estimate the neighborhood intersection
and cardinality from those sketches and convert them by inclusion-exclusion into the distance-label
count table A and boundary counts B; combine those structural counts, as direct MLP inputs, with the
Hadamard product of multi-scale (precomputable) node features; and read out a single link logit through
an MLP. ELPH does this in-loop on the full graph; BUDDY precomputes the sketches and features so the
learned model is an MLP on cached edge features — subgraph-class expressiveness at GCN-class cost.
