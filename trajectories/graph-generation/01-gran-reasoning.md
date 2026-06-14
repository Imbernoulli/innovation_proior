The model is the whole point, but it has to live inside a fixed harness, and the cheapest thing that
fills the contract is the floor — so the pain to start from is just *learning a distribution over
these adjacency matrices at all*, under the constraints that actually bind here. The loop hands me
padded binary adjacency matrices `[B, max_nodes, max_nodes]`, calls `train_step` on each batch for 500
epochs, then calls `sample` once and scores degree, clustering, and orbit MMD against a held-out
reference set. The default fill is a plain-MLP VAE: flatten the adjacency, encode to a Gaussian, decode
the whole matrix back, train on per-entry reconstruction. That floor is order-blind in the worst way —
it compares the decoded adjacency to the target *entrywise*, so a correct graph emitted under a
different node permutation is punished hard, and the latent has to memorize orderings instead of
structure. I want something that at least *reasons about the graph as a graph* before it predicts
edges: a model where each edge decision is informed by the topology around it, not just by a flat MLP
readout of a latent. That points away from the latent-variable family and toward a model that passes
messages over the nodes and predicts edges from the resulting node states.

Let me write down what I am really after so I don't fool myself. I want a generator that (1) predicts
each edge `(i,j)` as a function of *node representations* that have absorbed the surrounding structure,
so edge decisions are topology-aware rather than independent readouts of a single bottleneck vector;
(2) produces a whole symmetric, self-loop-free adjacency at fixed `max_nodes`, since that is exactly
what `sample` must return and what the loop thresholds and scores; and (3) trains and samples cheaply
enough to fit 500 epochs on three datasets, including `enzymes` at `max_nodes` up to 125. The
autoregressive lineage is the natural place to look for topology-aware edge decisions — DeepGMG runs a
full GNN before every atomic decision so each edge sees the real graph, and that is the right
instinct — but it pays a full message-passing pass per decision, costing on the order of
`m·n²·diam(G)`, which is hopeless here, and worse it bakes in a *generation order* the loop never asked
for. GraphRNN scales by dropping the GNN and emitting rows under a BFS ordering, but then every edge
decision is filtered through a recurrent state rather than conditioned directly on the topology, and it
spends `O(N²)` strictly-sequential steps that I would have to unroll inside `train_step`. Both
autoregressive routes drag in an ordering and a long sequential dependency I would rather not carry in
this harness.

So let me ask the heretical question for an autoregressive family: do I need the *sequence* at all, or
do I only need the *topology-aware edge predictor*? The thing that made DeepGMG good was that each edge
saw the graph through message passing; the thing that made it slow was running that message passing
once per atomic decision and serializing the decisions. What if I keep the message passing — node
representations that absorb structure, attention so a node listens to the neighbors that matter — but
*drop the sequential construction entirely*, and instead predict the entire adjacency in one shot from
the refined node states, then refine that prediction a handful of times? That gives me the
topology-aware edge predictor without the `O(N²)` sequential unroll and without an imposed ordering. It
is a one-shot model with a message-passing core: attention-and-edge-MLP refinement over the full node
set, an edge predictor reading pairs of node states, and a node-existence predictor so a fixed-size
adjacency can represent graphs on fewer than `max_nodes` nodes. This is the version of attentive graph
generation that fits the harness's `sample` contract directly — produce a full adjacency, no node
ordering, no recurrent rollout.

Let me build the refinement core, because that is where the topology-awareness lives. The state is a
node-feature tensor `[B, N, D]` and an edge-feature tensor `[B, N, N, 1]` initialized from the (current)
adjacency. One refinement block does three things. First, **attention over nodes**: multi-head
self-attention so each node's representation is updated by a learned weighting of all other nodes —
`attn = softmax(QKᵀ/√d)`, `out = attn·V`, with a residual and a LayerNorm — which is the message-
passing step, letting a node absorb the structure it sits in. Why attention rather than a fixed
degree-normalized GCN aggregation? Because the edges I am conditioning on are themselves uncertain
(during sampling they start as guesses), so I want the node update to *learn* which neighbors to trust
rather than average them uniformly. Second, an **edge update**: for every pair `(i,j)` form
`[n_i, n_j, e_ij]`, push it through an MLP, and add the result back to the edge feature — so the edge
representation accumulates evidence from both endpoints. Third, an **edge-to-node aggregation**: sum
each node's incident edge features (masked to real nodes when a mask is available), concatenate with
the node state, run an MLP, residual-and-norm — so the node state also hears from its edges, not only
from attention. Stacking a few of these blocks (the fill uses three) and treating the stack as the
refinement is what gives me, in one forward pass, node states that have integrated several rounds of
structure.

From the refined node states I emit two predictions. The **edge logits**: for each pair form
`[n_i, n_j, e_ij]` again, an MLP to a scalar, then symmetrize `(L + Lᵀ)/2` and zero the diagonal — the
two constraints the loop requires of `sample`'s output, baked into the predictor rather than hoped for.
The **node-existence logits**: an MLP per node to a scalar, which lets the fixed-`N` adjacency say
"this slot is empty" so a graph on `n < N` nodes can come out. Training is then almost embarrassingly
direct compared to the autoregressive likelihoods: there is no ordering to marginalize and no sequence
to unroll, so I just teacher-force the *true* adjacency through the refinement and ask the edge logits
to reconstruct it. The edge loss is per-entry binary cross-entropy of the edge logits against the
target adjacency; the node loss is BCE of the node-existence logits against the true node mask
(`adj.sum(-1) > 0`); the total is `edge_loss + 0.5·node_loss`, with the node term down-weighted because
there are `O(N)` node terms against `O(N²)` edge terms and I do not want node existence to dominate. A
global grad-norm clip at 1.0 keeps the attention stack from taking a destructive step early. That is
the entire `train_step`: one forward pass on the real adjacency, one reconstruction loss, one Adam
step — no sequential inner loop, which is exactly the cost I was trying to buy back from the
autoregressive route.

Now sampling, and this is where I have to be most careful, because a one-shot refinement model has no
ground-truth prefix to lean on — it has to hallucinate a graph from nothing. If I start the refinement
from an all-zeros adjacency, watch what happens: the edge features are all zero, attention has nothing
structural to weight, the edge-to-node aggregation sums to zero, so every node looks identical, the
node-existence predictor sees uniform empty states and predicts every node *absent*, and the output is
the empty graph. The model collapses to nothing. So I cannot start from zeros. The fix is to start the
refinement from a **random sparse adjacency** — draw each upper-triangular entry Bernoulli with a small
`p_init` (the fill uses 0.3), symmetrize — so there is real edge signal for the attention and the edge
MLPs to chew on. Then I refine: at each of `n_refine_steps` (the fill uses 5), run the forward pass,
take `edge_probs = sigmoid(edge_logits)`, and *resample* the adjacency as `Bernoulli(edge_probs)`,
symmetrizing each time. The refinement is a fixed-point-style sweep that pulls the random initial graph
toward something the edge predictor is confident about. After the last sweep I read node counts off the
realized connectivity (`(adj.sum(-1) > 0).sum(-1)`, clamped to ≥ 2) rather than off the node-existence
predictor, because that predictor was trained on *real* adjacency rows and is unreliable when fed the
random-then-refined adjacency at inference time — so I trust the structure the refinement actually
produced over the auxiliary head's guess.

I want to be honest about what this construction is and is not. It is *not* the block-wise
autoregressive generator with a per-step GNN, mixture-of-Bernoulli edge outputs, and a family of
canonical orderings that the autoregressive line points toward — that machinery is the principled way
to get *correlated* edges (a shared latent mixture index couples the edges in a block) and a tractable
permutation-aware likelihood (a logsumexp over canonical orderings). This harness fill keeps none of
it: the edges are predicted as **independent** Bernoullis given the node states (no mixture latent, so
no modeled edge correlation), there is **no ordering and no autoregression** (the whole adjacency is
predicted at once and refined), the training signal is a plain **reconstruction BCE** rather than a
sequence likelihood, and "iterative refinement" here means resampling the full adjacency a fixed five
times at sampling, not block-by-block construction. I am keeping only the *attention-based message-
passing edge predictor* and re-expressing it as a one-shot refiner that fits the `sample` contract.
That is a deliberate simplification forced by the contract — `sample` wants a whole adjacency, not a
sequence — and I should expect it to cost me exactly the things I dropped: independent edges cannot
represent "both these edges fire or neither," which is precisely the kind of correlation that makes a
two-community graph or a clustered ego-graph look right, and a model that reconstructs the true
adjacency entrywise still has the order-sensitivity problem the default VAE had, only now with a
fancier predictor in front of it.

So at step 1 this is the rung I run, and I should reason carefully about where it will land, because it
is the start of the climb. The thing I genuinely improved over the plain-MLP floor is the *edge
predictor*: edges now come from node states that have passed messages over the graph, with attention
choosing whom to listen to, rather than from a flat decode of one latent vector. That should help most
where local structure is regular and learnable — the dense two-community blocks of `community_small`,
where a node's edges are strongly predictable from its community, and the small `enzymes` motifs.
Where I expect trouble is everywhere the *sampling* dynamics bite. The refinement starts from a random
`Bernoulli(0.3)` adjacency, which on a dataset of sparse graphs is far too dense, and five resampling
sweeps from that starting point are a short, high-variance walk: whether they land on a clean sparse
graph or stall on a dense blob depends on the random init and on how confident the edge predictor
became in 500 epochs. That is a recipe for *seed sensitivity* — some seeds should converge to sensible
sparse graphs and score well, while a seed whose refinement stalls on an over-dense or degenerate
adjacency will score terribly on every MMD at once. The node-count-from-connectivity rule adds a second
failure mode: if the refinement leaves spurious scattered edges, node counts inflate and the degree and
orbit statistics drift away from the reference.

My falsifiable expectations for this floor, then. On `community_small`, where the dense local block
structure is the easiest thing for an attention edge predictor to learn, I expect the *good* seeds to
do genuinely well — this should be the rung's best dataset. On `ego_small` and `enzymes` I expect the
refinement's random-init-plus-five-sweeps sampler to be the dominant story: a wide seed-to-seed spread,
with at least one seed collapsing to a high-MMD blob because the sweep stalled on an over-dense graph,
dragging the *mean* up even if the other seeds are fine. The independent-edge assumption means even the
good seeds will miss the higher-order correlations the orbit statistic measures. So the prediction is:
strong on the regular-structure dataset, but a high-variance, seed-fragile generator overall whose mean
is pulled up by collapse on the larger/sparser graphs — a model that *can* produce good graphs but does
not do so reliably, which is exactly the weakness the next rung will have to fix. The distilled
scaffold module is in the answer.
