I have a sharp, uncomfortable diagnosis to start from: the structural signal that decides links on these
graphs — neighborhood overlap, the common-neighbor / Adamic–Adar / resource-allocation family — is
exactly what a plain message-passing GNN cannot represent, because a GNN gives one embedding per node
and two automorphic nodes get identical embeddings, so for a fixed source u it cannot distinguish a
true neighbor v from a structurally-identical non-neighbor v′. Take the smallest graph that shows this:
a star where p1 and p2 both attach to a hub s, and separately q1 attaches to t1 and q2 to t2. p1 and q1
are both degree-1 leaves — to a message-passing GNN their rooted neighborhoods are identical, so they
receive identical embeddings. But the *link* (p1, p2) has one common neighbor (s) while the link
(q1, q2) has zero. So two links whose endpoints a GNN cannot tell apart have different overlap
structure: a scorer that reads only node embeddings is forced to give them the same score; the overlap
signal is invisible to it. That is the wall, confirmed on four nodes. And the tool that *can* see it —
extract the k-hop enclosing subgraph around each pair, label every node by its distances to the two
endpoints, run a GNN with pooling — does so only by recomputing massively overlapping subgraph
structure once per candidate edge, which is O(|E|) per edge on power-law graphs and overflows memory if
I try to precompute the subgraphs. So I want the expressiveness of the subgraph method at the cost of a
plain GNN. The thing I have to find is whether the per-edge subgraph computation can be replaced by
something computed *per node, once*.

Let me look hard at *what* the subgraph method actually uses, because if most of its power comes from a
small, structured quantity I might be able to compute that quantity directly. In the labeled enclosing
subgraph, every node w carries a label (d(w,u), d(w,v)) — its distance to each endpoint. The
downstream GNN reads these labels and pools. What does pooling over those labels amount to? At bottom
it is *counting*: how many nodes are at distance d_u from u and d_v from v, for each small (d_u, d_v).
Call that count A_{uv}[d_u, d_v] = number of nodes at distance exactly d_u from u and exactly d_v from
v. Does the lowest cell equal the classical overlap signal? I take a small graph with u = 0, v = 1,
shared neighbors 2 and 3 (both adjacent to 0 and 1), plus some peripheral nodes hanging off at distance
2, and compute exact shortest-path distances. The exact-distance table comes out A[1,1] = 2, A[1,2] = 0,
A[2,1] = 0, A[2,2] = 1. And counting common neighbors directly, |N(0) ∩ N(1)| = |{2, 3}| = 2 — which is
exactly A[1,1]. So on this graph A[1,1] *is* the common-neighbor count, not approximately but on the
nose. A[1,2] and A[2,1] count the one-hop/two-hop overlap, A[2,2] the two-hop/two-hop overlap, and so
on. So the distance-label counts subsume the overlap heuristics: CN is A[1,1], and a learned function of
the whole count table can express AA, RA, Katz-like mixtures, whatever the data wants. Most of the LP
signal sits in the *low*-distance counts, since A[1,1] is the dominant heuristic across these
benchmarks, so if the predictive power of the subgraph method really sits in these counts I may not need
the subgraph at all. I need the counts.

So restate the goal precisely: for each candidate pair (u, v), compute the count table
{A_{uv}[d_u, d_v] : d_u, d_v ≤ k} for a small receptive field k, plus the "boundary" counts
B_{uv}[d] = (nodes at distance d from u but beyond distance k from v) that catch the mass that falls
outside the joint window so the features don't depend on graph size — and do it *without* building the
subgraph, ideally from quantities I can attach to each node once. Let me write the counts in terms of
neighborhoods. Let N_d(u) be the set of nodes within distance ≤ d of u. The number of nodes within d_u
of u and within d_v of v is just an *intersection cardinality* C_{uv}[d_u, d_v] = |N_{d_u}(u) ∩
N_{d_v}(v)|, a cumulative quantity. To get the *exact*-distance count A from these cumulative C's I have
to peel off the nearer shells. The natural guess is two-sided inclusion–exclusion: A[d_u, d_v] =
C[d_u, d_v] − C[d_u−1, d_v] − C[d_u, d_v−1] + C[d_u−1, d_v−1] (with C[0, ·] and C[·, 0] the
endpoint-only counts). On the same small graph I compute every cumulative C[a, b] = |N_{≤a}(u) ∩
N_{≤b}(v)| by brute force and then run the four-term formula for each (a, b) with a, b from 1 to 3 —
and every reconstructed A[a, b] matches the directly-counted exact-distance table. So the cumulative
intersection cardinalities, run through this four-term subtraction, recover the exact counts; the
boundary then follows the same way, B_{uv}[d] = |N_d(u)| − (within-window joint mass already
attributed), peeling the within-window joint counts off the d-hop neighborhood of u. So everything
reduces to two primitives per node pair: the cardinality of each node's d-hop neighborhood, |N_d(u)|,
and the cardinality of the intersection of two nodes' neighborhoods, |N_{d_u}(u) ∩ N_{d_v}(v)|. If I can
get those two cheaply, I get the whole count table by arithmetic.

Now the obstacle: computing |N_{d_u}(u) ∩ N_{d_v}(v)| exactly for every candidate pair means
intersecting potentially huge sets, and storing N_d(u) explicitly for every node is back to the memory
blowup. This is precisely the situation streaming/database systems face — estimate set cardinalities and
set intersections over enormous sets without materializing them — and they solve it with *sketches*:
small fixed-size summaries of a set from which cardinality and similarity can be estimated. Two
classical sketches line up with the two primitives I need. HyperLogLog summarizes a set in O(1) space
(a small register array) and estimates its cardinality |S| via a cardinality function card(·) on the
sketch; the property I care about is that the HyperLogLog sketch of a *union* is the elementwise *max*
of the two sketches, so I can build neighborhood sketches by max-aggregating. MinHash summarizes a set
so that the Hamming similarity between two MinHash sketches estimates their *Jaccard* similarity
J(S, T) = |S ∩ T| / |S ∪ T|; its sketch of a union is the elementwise *min*. If I want an intersection
size, neither sketch gives it alone, but together they might: |S ∩ T| = J(S, T) · |S ∪ T|, so if MinHash
delivers J and HyperLogLog delivers |S ∪ T| via the max-merged sketch, the product is the intersection.
On two concrete sets S = {1..6}, T = {4..8}: |S ∩ T| = 3, and J · |S ∪ T| = (3/8) · 8 = 3 — exact, as it
must be since it is just the definition of Jaccard rearranged. The union properties need checking too: I
MinHash two random 400-element subsets of a 2000-element universe with 256 permutations and check that
the elementwise minimum of their two sketches equals the MinHash computed directly on the union — it
does, exactly, every register. And the Hamming similarity of the two sketches estimates Jaccard 0.07
against a true 0.091 — close, with the gap the expected sampling noise at 256 permutations, tightening
with more permutations. So the estimator is |S ∩ T| ≈ H(MinHash(S), MinHash(T)) · card(max(HLL(S),
HLL(T))): MinHash gives the Jaccard, HyperLogLog gives the union size, the product gives the
intersection size — and the sketch sizes are fixed constants independent of graph size, with a
precision/cost knob.

This makes the per-node primitive concrete. Give each node u an initial sketch pair (m_u^{(0)},
h_u^{(0)}) — the MinHash and HyperLogLog of the singleton {u}. The d-hop neighborhood decomposes as
N_d(u) = ∪_{v ∈ N(u)} N_{d−1}(v), a union over neighbors. Since I just verified that MinHash-of-union is
elementwise min and HLL-of-union is elementwise max, applying that to this recurrence gives
m_u^{(d)} = min_{v ∈ N(u)} m_v^{(d−1)} and h_u^{(d)} = max_{v ∈ N(u)} h_v^{(d−1)}. That is *message
passing* — but the messages are the sketches, and the aggregators are elementwise min and max. So with k
rounds of min/max sketch propagation I get, for every node, the sketches of all its d-hop neighborhoods
up to d = k, in O(k|E|·h) time with h the sketch size — node-level, once, no subgraph. Then for any
candidate pair (u, v) I read off |N_{d_u}(u)| ≈ card(h_u^{(d_u)}), |N_{d_u}(u) ∩ N_{d_v}(v)| ≈
H(m_u^{(d_u)}, m_v^{(d_v)}) · card(max(...)), and turn those into the count table A and boundary B by the
inclusion-exclusion arithmetic I checked above. The expensive per-edge subgraph construction has been
replaced by cheap per-pair sketch comparisons on fixed-size summaries. One caveat worth keeping honest:
the inclusion-exclusion above was verified on *exact* counts; with sketch estimates the four-term
subtraction can produce small negatives from estimation noise, so I clamp the counts at zero rather than
pretend the estimates are exact. The expressiveness argument is about the exact counts; the
implementation is about good estimates of them.

That is the structural side. The other half of a link predictor is node *features*. The subgraph method
propagates node features over the subgraph; I can propagate them over the *whole graph* with an
ordinary GNN, or — and this is the scalability lever — I can fix the feature propagation, on the bet
that fixed and learned propagation give nearly equivalent link-prediction performance: the
decoupled-propagation line found exactly this for node tasks, and I expect it to carry over to links. A
fixed sparse propagation x_u^{(l)} = mean_{v ∈ N(u)} x_v^{(l−1)} can be precomputed once with scatter
operations, and concatenating the features diffused at each hop, Z = [X^{(0)} ‖ X^{(1)} ‖ … ‖ X^{(k)}],
gives a multi-scale node feature with no message passing at training time at all. This is the
decoupling-propagation-from-learning idea (SGC/SIGN, Wu et al. 2019; Rossi et al. 2020) applied to the
feature stream.

Now assemble the predictor. For a pair (u, v): take the multi-scale node features z_u, z_v (propagated
or GNN-encoded), combine them by the Hadamard product z_u ⊙ z_v, and *concatenate* the structural count
features {B_{uv}[d], A_{uv}[d_u, d_v] : d, d_u, d_v ≤ k}. Pass the concatenation through an MLP ψ to a
single logit: p(u, v) = ψ(z_u ⊙ z_v, {B_{uv}[d], A_{uv}[d_u, d_v]}). The MLP learns *how* to weight the
structural counts — and since A[1,1] just turned out to be the common-neighbor count, a linear readout
on the count vector already contains CN, and a nonlinear one can reach AA, RA, or any mixture the data
prefers — and *how* to fuse them with the feature interaction. Two design points need settling. The
endpoints: should I pool the two node representations at the *edge* level (Hadamard of z_u and z_v) or
pool over all subgraph nodes (graph-level)? Graph-level pooling drags the subgraph back in, which is the
whole cost I am trying to escape, so edge pooling wins on cost alone — and it is the standard,
well-performing choice for links. The structural counts: feed them as *direct inputs to the MLP*, or
propagate them as labels through a GNN? Since the predictive content concentrates in the low-distance
counts — A[1,1] and its immediate neighbors — and those are already a short fixed vector, an MLP
directly on them has nothing to gain from further propagation and keeps the model an MLP; so direct
inputs. The receptive field k is small — k = 2 or 3 — both because the low-distance counts dominate (I
would not expect A[3,3] to add much over A[1,1]) and because keeping k below the size scale gives a
fixed-length, size-independent feature vector that resists overfitting; the sketch sizes (HyperLogLog
precision p, number of MinHash permutations) are the accuracy/speed knobs, again independent of N.

Two distinct realizations fall out of *when* I compute the sketches. The full-graph version folds the
sketch propagation (min for MinHash, max for HyperLogLog) and the feature propagation into the
message-passing loop, computing the edge features on the fly each forward pass — clean and expressive,
stronger than a plain message-passing GNN because (by the four-node witness above) the structural
features distinguish automorphic-node links a GNN cannot, but it needs the whole graph and its sketches
in GPU memory. Call it ELPH. The scalable version observes that *both* the sketches and the fixed
feature propagation can be precomputed once with sparse scatter operations — scatter_min for MinHash,
scatter_max for HyperLogLog, scatter_mean for features — so the per-edge structural and node features
{B_{uv}[d], A_{uv}[d_u, d_v]} and z_u, z_v are materialized ahead of time and cached, and training/
inference is then *just an MLP on precomputed edge features*. Call it BUDDY. That converts a GNN into an
MLP: no subgraph sampling, no message passing at train time, memory independent of the graph fitting on
the GPU. The cost is O(kEd) for feature propagation plus O(kEh) for sketch propagation in preprocessing,
then O(k²h + kd²) per link prediction — both independent of the size of the graph, which is the entire
point.

The four-node witness now closes the expressiveness argument: leaves p1 and q1 are indistinguishable to
a plain GNN, yet A[1,1] = 1 for the link (p1, p2) and A[1,1] = 0 for (q1, q2). The structural counts
A_{uv}[d_u, d_v] are computed *relative to the pair* — they depend on the joint neighborhood geometry of
u and v, not on either node alone — so exactly those links a GNN collapses can carry different count
tables and thus different scores. With exact counts the model separates every pair a GNN separates (it
can ignore the counts and use the embeddings) and additionally separates pairs like (p1, p2) vs (q1, q2)
that the GNN cannot, so it is strictly more expressive on this family. With *sketch* estimates rather
than exact counts the separation holds whenever the estimation error is smaller than the gap between the
two count tables, which is why the sketch sizes are a knob and not free.

BUDDY is that answer: sketches and multi-scale features precomputed once per node, per-pair structural
counts read off in constant time by the inclusion-exclusion arithmetic above, and the entire learned
model reduced to an MLP on cached edge features — subgraph-class expressiveness at GCN-class cost, work
that grows with the graph once at preprocessing time rather than per candidate edge.
