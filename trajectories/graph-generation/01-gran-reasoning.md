The model is the whole point, but it lives inside a fixed harness, so the pain to start from is just
*learning a distribution over these adjacency matrices at all*, under the constraints that bind. The loop
hands me padded binary adjacency `[B, max_nodes, max_nodes]`, calls `train_step` per batch for 500 epochs,
then calls `sample` once and scores degree, clustering, and orbit MMD against a held-out reference. Three
facts bind hard before I write any model. The graphs are **unordered**: the same graph on `n` nodes is up
to `n!` adjacency matrices, so any exact likelihood is a sum over `n!` permutations I cannot pay. They are
**sparse and discrete**: `O(n)` real edges in `O(n²)` binary slots, and what gets scored is topology —
degree histograms, per-node clustering, 4-orbit counts — not the entry values. And the **size range is
wide**: `community_small` and `ego_small` top out near 20 nodes but `enzymes` runs to 125, so
`max_nodes = 125` sets the padded shape and every `O(N²)` tensor is a 125×125 object even when the real
graph inside has thirty nodes. Those three facts are the whole difficulty; the rest is scaffolding.

The default fill is a plain-MLP VAE: flatten the adjacency, encode to a Gaussian, decode the whole matrix
back, reconstruct per-entry. That floor is order-blind in the worst way — it compares the decoded adjacency
to the target *entrywise*, so a correct graph emitted under a different node permutation is punished hard
and the latent memorizes orderings instead of structure. It is also enormous exactly where it hurts: the
flatten encoder is a `Linear(max_nodes², hidden)` and the decoder ends in a `Linear(hidden, max_nodes²)`,
so on `enzymes` each is 15625 wide and the two together are already `2·15625·256 ≈ 8M` parameters, most of
it spent shuttling a flattened adjacency in and out — precisely the ordering-memorization I want to escape.
I want a model that *reasons about the graph as a graph* before it predicts edges: each edge decision
informed by the topology around it, not by a flat MLP readout of one bottleneck vector. That points away
from the latent family and toward a model that passes messages over the nodes and predicts edges from the
resulting node states.

So I want a generator that (1) predicts each edge `(i,j)` from node representations that have absorbed the
surrounding structure; (2) produces a whole symmetric, self-loop-free adjacency at fixed `max_nodes`, which
is exactly what `sample` must return and what the loop thresholds and scores; and (3) trains and samples
cheaply enough for 500 epochs on three datasets, including `enzymes` at `max_nodes` 125. Constraint (3) is
not decorative — it is what kills most of the tempting options before I can fall in love with them.

Walk the alternatives with the cost arithmetic in front. DeepGMG runs a full GNN before every atomic
decision — add a node, add an edge, choose an endpoint — so each choice sees the real current graph, which
is exactly the topology-awareness I want. But price it: a graph has `O(n²)` potential edges, so `O(n²)`
add-edge decisions, each triggering a message pass on the order of `n²`, giving `O(n⁴)` per graph. On
`enzymes`, `n⁴ ≈ 2.4×10⁸` per graph, times a batch of 32, times ~15 batches an epoch, times 500 epochs —
hopeless, and it bakes in a *generation order* the loop never asked for. GraphRNN drops the per-decision
GNN and emits rows under a BFS ordering at `O(M·n)`, cheaper, but spends `O(n²)` *strictly sequential*
steps I would unroll inside a single `train_step` — up to `125² ≈ 15000` recurrent steps per graph, with a
backprop-through-time chain that long, and again an imposed ordering. The latent family is cheap per step
but either keeps the order-blind entrywise reconstruction (the floor's disease) or cures it with graph
matching at `O(k⁴)` per forward — `125⁴ ≈ 2.4×10⁸` similarity entries plus an inner solver, the same wall
as DeepGMG in different clothes. So every principled route drags in an ordering and a long sequential
dependency, pays a quartic per-step cost, or is not topology-aware at all.

The heretical question for the autoregressive family: do I need the *sequence*, or only the *topology-aware
edge predictor*? What made DeepGMG good was that each edge saw the graph through message passing; what made
it slow was running that pass once per decision and serializing the decisions. So keep the message passing —
node representations that absorb structure, attention so a node listens to the neighbors that matter — but
*drop the sequential construction entirely*, and predict the whole adjacency in one shot from the refined
node states, then refine that prediction a handful of times. That gives the topology-aware edge predictor
without the `O(n²)` unroll and without an imposed ordering, and it fits the `sample` contract directly: a
full adjacency, no node ordering, no recurrent rollout, with a per-forward cost that is a fixed number of
passes over the graph, not something that grows with the number of edges I place.

Now the refinement core, where the topology-awareness lives, and I want the shapes right before I trust
them. State is a node tensor `[B, N, D]` and an edge tensor `[B, N, N, 1]` initialized from the current
adjacency. I seed the node features from an identity input `eye(N)` through a `Linear(N, D)`, so node `i`
starts from a learned embedding of its *slot index* — a positional encoding. That is not
permutation-invariant featurization: the model can and will lean on slot position, a small down payment on
the same order-sensitivity the floor had, which I accept because the harness always presents each graph in
one fixed padded order. One block does three things. **Attention over nodes**: multi-head self-attention,
`softmax(QKᵀ/√d)·V` with a residual and a LayerNorm, so each node absorbs the structure it sits in — I use
attention rather than a fixed degree-normalized aggregation because the edges I am conditioning on are
themselves uncertain (during sampling they start as guesses), so I want the node update to *learn* which
neighbors to trust rather than average them uniformly under a normalization that assumes the adjacency is
real. An **edge update**: for every pair `(i,j)` form `[n_i, n_j, e_ij]`, push through an MLP, add back to
the edge feature, so the edge representation accumulates evidence from both endpoints. And an
**edge-to-node aggregation**: sum each node's incident edge features (masked to real nodes), concatenate
with the node state, MLP, residual-and-norm — so the node state hears from its edges too, not only from
attention. Stacking three of these blocks gives, in one forward pass, node states that have integrated
several rounds of structure.

The parameter count is small — three blocks at roughly 133k each plus a node embedding, edge predictor, and
node head comes to about 0.46M, an order of magnitude *under* the 8M floor, because only the node embedding
scales with `max_nodes` and everything else is `O(D²)`, so the `1.05×` param cap is nowhere near binding.
What *is* nearly binding is activation memory: the edge MLP's input on `enzymes` is `[32, 125, 125, 257]`,
about `1.3×10⁸` floats or 514 MB in float32, for one block, held by autograd across three blocks plus the
edge predictor. Affordable but not free — the tax for predicting a full adjacency in one shot instead of one
edge at a time.

From the refined node states I emit two predictions. The **edge logits**: form `[n_i, n_j, e_ij]` again, an
MLP to a scalar, then symmetrize `(L+Lᵀ)/2` and zero the diagonal — the two constraints the loop requires
of `sample`'s output baked into the predictor. The **node-existence logits**: an MLP per node so the
fixed-`N` adjacency can say a slot is empty and a graph on `n < N` nodes can come out. Training is direct:
no ordering to marginalize, no sequence to unroll, so I teacher-force the *true* adjacency through the
refinement and ask the edge logits to reconstruct it. Edge loss is per-entry BCE against the target; node
loss is BCE of the existence logits against the true node mask `adj.sum(-1) > 0`; total
`edge_loss + 0.5·node_loss`, the node term down-weighted because there are `O(N)` node terms against
`O(N²)` edge terms and I do not want node existence to dominate. A grad-norm clip at 1.0 keeps the attention
stack from an early destructive step. The whole `train_step` is one forward pass, one loss, one Adam step.

One thing about that reconstruction loss quietly shapes what the predictor learns. The edge BCE is a *mean
over all `N²` entries*, and the padded target is overwhelmingly zero. A 30-node `enzymes` graph padded to
125 has a real block of `30² = 900` entries out of `15625` — under 6% carries any structure, and even inside
the real block the graph is sparse. So the mean BCE is dominated by "predict zero," and after 500 epochs the
predictor is good at mapping *sparse real* inputs to *sparse* outputs. That sets a trap: the predictor is
only ever *trained* on inputs that are already sparse real graphs (teacher-forced), so its behavior on a
*dense* input is entirely unconstrained by the loss.

Sampling is where I have to be most careful, because a one-shot refiner has no ground-truth prefix — it must
hallucinate a graph from nothing. Start from an all-zeros adjacency and watch it collapse: the edge features
are all zero, attention has nothing structural to weight, the edge-to-node aggregation sums to zero, every
node looks identical, the node-existence predictor sees uniform states and predicts every node *absent*, and
the output is the empty graph. That is the deterministic fixed point of the zeros start, not a risk. So I
start the refinement from a **random sparse adjacency** — draw each upper-triangular entry Bernoulli with a
small `p_init`, symmetrize — so there is real edge signal for the attention and edge MLPs to chew on. The
fill uses `p_init = 0.3`. At `p_init = 0.3` on `N = 125` each node starts with expected degree
`0.3·124 ≈ 37` and the graph has ~2300 edges — a *dense* random graph, nothing like the single-digit
degrees of the sparse `enzymes` targets. On `community_small` at `N = 20` the same `p_init` gives expected
degree ~5.7, far closer to the dense two-community graphs there. So `p_init = 0.3` is roughly matched to the
small dense datasets and badly over-dense for the large sparse ones, and it lands the refinement's *input*
squarely in the dense region the training loss never constrained — the trap made concrete: on `enzymes` the
first pass feeds the predictor a `Bernoulli(0.3)` blob it was never trained to denoise.

Then at each of `n_refine_steps` (the fill uses 5) I run the forward pass, take
`edge_probs = sigmoid(edge_logits)`, and resample `adj = Bernoulli(edge_probs)`, symmetrizing. This is a
fixed-point-style sweep meant to pull the random init toward something the predictor is confident about, but
with no schedule and no defined corruption process behind it there is no contraction guarantee — whether
five sweeps walk down to a clean sparse graph or stall on a dense blob depends entirely on how the predictor
behaves in the off-distribution dense region the loss never pinned down. After the last sweep I read node
counts off the realized connectivity (`(adj.sum(-1) > 0).sum(-1)`, clamped ≥ 2) rather than off the
node-existence predictor, which was trained on *real* adjacency rows and is unreliable on the
random-then-refined input — I trust the structure the refinement produced over the head's guess. That has
its own failure mode: spurious scattered edges inflate node counts and drift the degree and orbit statistics
with them.

What this construction is and is not: it is *not* the block-wise autoregressive generator with a per-step
GNN, mixture-of-Bernoulli edge outputs, and a family of canonical orderings — the machinery that gets
*correlated* edges (a shared latent mixture couples the edges in a block) and a tractable permutation-aware
likelihood (a logsumexp over orderings). This fill keeps none of it: edges are predicted as **independent**
Bernoullis given the node states (no mixture latent, so no modeled edge correlation), there is **no ordering
and no autoregression**, the signal is plain reconstruction BCE, and "refinement" means resampling the full
adjacency five times at sampling. I keep only the *attention-based message-passing edge predictor*,
re-expressed as a one-shot refiner that fits the `sample` contract. I should expect it to cost exactly what
I dropped: independent edges cannot represent "both these edges fire or neither," which is the correlation
that makes a two-community graph or a clustered ego-graph look right, and entrywise reconstruction keeps the
order-sensitivity of the floor with a fancier predictor in front.

So where does this land? The thing I genuinely improved is the edge predictor — edges now come from node
states that have passed messages, with attention choosing whom to listen to. That should help most where
local structure is regular and the init density is matched: the dense two-community blocks of
`community_small`, where a node's edges are strongly predictable from its community and `p_init = 0.3`
starts near the right neighborhood. Trouble lives wherever the *sampling* dynamics bite: on the sparse
`ego_small` and `enzymes` the `Bernoulli(0.3)` init is far too dense and off-distribution, and five
unscheduled sweeps are a short, high-variance walk with no convergence guarantee — a recipe for seed
sensitivity, some seeds converging to sensible sparse graphs and a seed whose sweep stalls on a dense blob
scoring terribly. When a seed does stall on a blob I can say *which* statistics blow up: the blob has huge
degrees, so the degree histogram shifts wholesale and `mmd_degree` explodes, and it has combinatorially many
4-node substructures, so the raw orbit counts under the σ=30 Gaussian kernel explode too, while per-node
clustering — a ratio bounded in `[0,1]` — is far less sensitive and should stay comparatively flat. So I
expect this floor to be strong on the regular-structure dataset but a high-variance, seed-fragile generator
overall, its mean pulled up by degree/orbit blowups on the larger sparse graphs — a model that *can* produce
good graphs but does not do so reliably, which is exactly the weakness the next step has to fix.
