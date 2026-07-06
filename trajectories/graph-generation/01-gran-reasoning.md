The model is the whole point, but it has to live inside a fixed harness, and the cheapest thing that
fills the contract is the floor — so the pain to start from is just *learning a distribution over
these adjacency matrices at all*, under the constraints that actually bind here. The loop hands me
padded binary adjacency matrices `[B, max_nodes, max_nodes]`, calls `train_step` on each batch for 500
epochs, then calls `sample` once and scores degree, clustering, and orbit MMD against a held-out
reference set. Before I write a line of model, three facts about that setup bind hard. The graphs are
**unordered**: the same graph on `n` nodes is up to `n!` adjacency matrices, so any exact likelihood is
a sum over `n!` permutations I cannot pay. They are **sparse and discrete**: `O(n)` real edges out of
`O(n²)` binary slots, and what gets scored is topology — degree histograms, per-node clustering, 4-orbit
counts — not the entry values themselves. And the **size range is wide**: `community_small` tops out
near 20 nodes, `ego_small` near 18, but `enzymes` runs to 125, so `max_nodes = 125` sets the padded
shape and every `O(N²)` tensor I touch is a 125×125 object even when the real graph living inside it has
thirty nodes. Those three facts are the whole difficulty; the rest of the harness is scaffolding.

The default fill is a plain-MLP VAE: flatten the adjacency, encode to a Gaussian, decode the whole
matrix back, train on per-entry reconstruction. That floor is order-blind in the worst way — it
compares the decoded adjacency to the target *entrywise*, so a correct graph emitted under a different
node permutation is punished hard, and the latent has to memorize orderings instead of structure. It is
also enormous exactly where it hurts: the flatten encoder is a `Linear(max_nodes², hidden)` and the
decoder ends in a `Linear(hidden, max_nodes²)`, so on `enzymes` each of those is a 15625-wide layer and
the model carries on the order of eight million parameters — I can total it: `15625·256` in and
`256·15625` out is already 8.0M, everything else is rounding — most of it spent to shuttle a flattened
adjacency in and out, which is precisely the ordering-memorization I want to escape. I want something
that at least *reasons about the graph as a graph* before it predicts edges: a model where each edge
decision is informed by the topology around it, not just by a flat MLP readout of a bottleneck vector.
That points away from the latent-variable family and toward a model that passes messages over the nodes
and predicts edges from the resulting node states.

Let me write down what I am really after so I don't fool myself. I want a generator that (1) predicts
each edge `(i,j)` as a function of *node representations that have absorbed the surrounding structure*,
so edge decisions are topology-aware rather than independent readouts of a single bottleneck vector;
(2) produces a whole symmetric, self-loop-free adjacency at fixed `max_nodes`, since that is exactly
what `sample` must return and what the loop thresholds and scores; and (3) trains and samples cheaply
enough to fit 500 epochs on three datasets, including `enzymes` at `max_nodes` up to 125. Those are the
three the design has to satisfy simultaneously, and (3) is not decorative — it is what kills most of the
tempting options before I can fall in love with them.

Let me walk the alternatives with the cost arithmetic in front of me rather than by taste. The
autoregressive lineage is the natural place to look for topology-aware edge decisions. DeepGMG runs a
full GNN before every atomic decision — add a node, add an edge, choose an endpoint — so each choice
sees the real current graph; that is exactly the topology-awareness I want. But price it: a graph has
`O(n²)` potential edges, so `O(n²)` add-edge decisions, and each triggers a message-passing pass on the
order of `n²`, giving something like `O(n⁴)` message-passing work per graph before I even count the
diameter factor. On `enzymes`, `n=125` makes `n⁴ ≈ 2.4×10⁸` per graph, times a batch of 32, times ~15
batches an epoch, times 500 epochs — hopeless, and it bakes in a *generation order* the loop never asked
for. GraphRNN drops the per-decision GNN and emits rows under a BFS ordering, scaling as `O(M·n)` with
the BFS bandwidth `M` bounding row width; cheaper, but it spends `O(n²)` *strictly sequential* steps
that I would have to unroll inside a single `train_step` — on `enzymes` that is up to `125² ≈ 15000`
sequential recurrent steps per graph, with a backprop-through-time chain that long, and again an imposed
ordering. The other pole, staying in the latent family, is cheap per step but does not advance the one
thing I care about: a one-shot latent decode either keeps the order-blind entrywise reconstruction (the
floor's disease) or cures it with graph matching at `O(k⁴)` per forward pass, which on `enzymes` is
`125⁴ ≈ 2.4×10⁸` similarity entries plus an inner solver — the same wall as DeepGMG in different clothes.
So every principled route either drags in an ordering and a long sequential dependency, or pays a
quartic per-step cost, or fails to make edge decisions topology-aware at all.

So let me ask the heretical question for the autoregressive family: do I need the *sequence* at all, or
do I only need the *topology-aware edge predictor*? The thing that made DeepGMG good was that each edge
saw the graph through message passing; the thing that made it slow was running that message passing once
per atomic decision and serializing the decisions. What if I keep the message passing — node
representations that absorb structure, attention so a node listens to the neighbors that matter — but
*drop the sequential construction entirely*, and instead predict the entire adjacency in one shot from
the refined node states, then refine that prediction a handful of times? That gives me the
topology-aware edge predictor without the `O(n²)` sequential unroll and without an imposed ordering. It
is a one-shot model with a message-passing core: attention-and-edge-MLP refinement over the full node
set, an edge predictor reading pairs of node states, and a node-existence predictor so a fixed-size
adjacency can represent graphs on fewer than `max_nodes` nodes. This is the version of attentive graph
generation that fits the harness's `sample` contract directly — produce a full adjacency, no node
ordering, no recurrent rollout — and its per-forward cost is a fixed constant of passes over the graph,
not something that grows with the number of edges I place.

Let me build the refinement core, because that is where the topology-awareness lives, and I want the
shapes to be exactly right before I trust them. The state is a node-feature tensor `[B, N, D]` and an
edge-feature tensor `[B, N, N, 1]` initialized from the (current) adjacency. The node features
themselves I seed from an identity input `eye(N)` pushed through a `Linear(N, D)` — so node `i` starts
from a learned embedding of its *slot index*, a positional encoding. I note in passing that this is not
permutation-invariant node featurization: the model can and will lean on slot position, which is a small
down payment on the same order-sensitivity the floor had, and I accept it because the harness always
presents each graph in one fixed padded order. One refinement block does three things. First,
**attention over nodes**: multi-head self-attention so each node's representation is updated by a
learned weighting of all other nodes — `attn = softmax(QKᵀ/√d)`, `out = attn·V`, with a residual and a
LayerNorm — the shapes going `[B,N,D] → qkv [B,N,3D] → [B, heads, N, head_dim]`, scores `[B,heads,N,N]`,
back to `[B,N,D]`. This is the message-passing step, letting a node absorb the structure it sits in. Why
attention rather than a fixed degree-normalized GCN aggregation? Because the edges I am conditioning on
are themselves uncertain — during sampling they start as guesses — so I want the node update to *learn*
which neighbors to trust rather than average them uniformly under a normalization that assumes the
adjacency is real. Second, an **edge update**: for every pair `(i,j)` form `[n_i, n_j, e_ij]` — a
`[B, N, N, 2D+1]` tensor — push it through an MLP, and add the result back to the edge feature, so the
edge representation accumulates evidence from both endpoints. Third, an **edge-to-node aggregation**:
sum each node's incident edge features (masked to real nodes when a mask is available), concatenate with
the node state, run an MLP, residual-and-norm — so the node state also hears from its edges, not only
from attention. Stacking three of these blocks and treating the stack as the refinement is what gives
me, in one forward pass, node states that have integrated several rounds of structure.

That `[B, N, N, 2D+1]` edge tensor is where I should sanity-check the budget, because it is the real
constraint, not the parameter count. With `hidden_dim = 128` the whole model is small: I can add it up —
three refinement blocks at roughly 133k parameters each, plus a node embedding, an edge predictor, and a
node head, comes to about 0.46M parameters, an order of magnitude *under* the 8M flatten-VAE floor it
replaces, because only the node embedding scales with `max_nodes` and everything else is `O(D²)`. So the
`1.05×`-largest-baseline parameter cap is nowhere near binding. What *is* nearly binding is activation
memory: the edge MLP's input on `enzymes` is `[32, 125, 125, 257]`, which is `32·125·125·257 ≈ 1.3×10⁸`
floats, about 514 MB in float32 — for one block, and autograd holds them across three blocks plus the
edge predictor. That is the cost I am actually spending, and it is affordable but not free; it is the
tax for predicting a full adjacency in one shot instead of one edge at a time.

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

One thing about that reconstruction loss deserves a hard look, because it quietly shapes what the
predictor learns. The edge BCE is a *mean over all `N²` entries*, and the padded target is
overwhelmingly zero. Take a 30-node `enzymes` graph padded to 125: the real block is `30·30 = 900`
entries out of `125·125 = 15625`, so about 5.8% of the target carries any structure and over 94% is
structural-zero padding — and even inside the real block the graph is sparse. So the mean BCE is
dominated by "predict zero," and after 500 epochs the edge predictor is very good at mapping *sparse,
real* adjacency inputs to *sparse* outputs. I flag this now because it sets a trap for sampling: the
predictor is only ever *trained* on inputs that are already sparse real graphs (teacher-forced), so its
behavior on a *dense* input is entirely unconstrained by the loss. Whatever I feed it at sampling time
had better not look wildly out of that training distribution, or the predictor is extrapolating.

Now sampling, and this is where I have to be most careful, because a one-shot refinement model has no
ground-truth prefix to lean on — it has to hallucinate a graph from nothing. If I start the refinement
from an all-zeros adjacency, watch what happens step by step: the edge features are all zero, attention
has nothing structural to weight (every node's key and value derive from the same identity-seeded
embedding modulated by zero edges), the edge-to-node aggregation sums to zero, so every node looks
identical, the node-existence predictor sees uniform states and predicts every node *absent*, and the
output is the empty graph. The model collapses to nothing — this is not a risk, it is the deterministic
fixed point of the all-zeros start. So I cannot start from zeros. The fix is to start the refinement
from a **random sparse adjacency** — draw each upper-triangular entry Bernoulli with a small `p_init`
and symmetrize — so there is real edge signal for the attention and the edge MLPs to chew on. The fill
uses `p_init = 0.3`, and I should check what that actually is: at `p_init = 0.3` on `N = 125`, each node
starts with expected degree `0.3·124 ≈ 37`, and the whole thing has `~2300` edges — a *dense* random
graph, nothing like the sparse `enzymes` targets whose real degrees are single digits. On `community_small`
at `N = 20` the same `p_init` gives expected degree `0.3·19 ≈ 5.7`, which is far closer to the dense
two-community graphs that dataset actually contains. So `p_init = 0.3` is roughly matched to the small
dense datasets and badly over-dense for the large sparse ones — and, worse, it lands the refinement's
*input* squarely in the dense region the training loss never constrained. This is the trap I flagged
above made concrete: on `enzymes` the first refinement pass feeds the edge predictor a `Bernoulli(0.3)`
blob it was never trained to denoise.

Then I refine: at each of `n_refine_steps` (the fill uses 5), run the forward pass, take
`edge_probs = sigmoid(edge_logits)`, and *resample* the adjacency as `Bernoulli(edge_probs)`,
symmetrizing each time. The refinement is a fixed-point-style sweep meant to pull the random initial
graph toward something the edge predictor is confident about. But there is no schedule and no defined
corruption process behind it, so I have no contraction guarantee: whether five sweeps from a dense init
walk down to a clean sparse graph or stall on a dense blob depends entirely on how the predictor
behaves in that off-distribution dense region, which the loss never pinned down. After the last sweep I
read node counts off the realized connectivity (`(adj.sum(-1) > 0).sum(-1)`, clamped to ≥ 2) rather
than off the node-existence predictor, because that predictor was trained on *real* adjacency rows and
is unreliable when fed the random-then-refined adjacency at inference time — so I trust the structure
the refinement actually produced over the auxiliary head's guess. That choice has its own failure mode:
if the refinement leaves spurious scattered edges, node counts inflate and the degree and orbit
statistics drift with them.

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
where a node's edges are strongly predictable from its community, and where the `p_init = 0.3` init is
also best matched to the real density, so the refinement starts near the right neighborhood. Where I
expect trouble is everywhere the *sampling* dynamics bite. The refinement starts from a random
`Bernoulli(0.3)` adjacency, which on the sparse `ego_small` and `enzymes` graphs is far too dense and
lands off the training distribution, and five resampling sweeps from that starting point are a short,
high-variance walk with no schedule to guarantee convergence: whether they land on a clean sparse graph
or stall on a dense blob depends on the random init and on how confident the edge predictor became in
500 epochs. That is a recipe for *seed sensitivity* — some seeds should converge to sensible sparse
graphs and score well, while a seed whose refinement stalls on an over-dense or degenerate adjacency
will score terribly. And when a seed does stall on a dense blob, I can predict *which* statistics blow
up: a blob has huge node degrees, so the degree histogram shifts wholesale and `mmd_degree` explodes,
and it has combinatorially many 4-node substructures, so the raw orbit counts under the σ=30 Gaussian
kernel explode too — while per-node clustering, being a ratio bounded in `[0,1]`, is far less sensitive
to a blob and should stay comparatively flat across seeds.

My falsifiable expectations for this floor, then. On `community_small`, where the dense local block
structure is the easiest thing for an attention edge predictor to learn and the init density is
well-matched, I expect the *good* seeds to do genuinely well — this should be the rung's best dataset by
mean. On `ego_small` and `enzymes` I expect the refinement's random-init-plus-five-sweeps sampler to be
the dominant story: a wide seed-to-seed spread, with at least one seed collapsing to a high-MMD blob
because the sweep stalled on an over-dense graph, and that collapse showing up specifically as blown-up
`mmd_degree` and `mmd_orbit` with `mmd_clustering` comparatively spared — one bad seed dragging the
*mean* up even if the other two are fine. The independent-edge assumption means even the good seeds will
leave the higher-order correlations the orbit statistic measures on the table. So the prediction is:
strong on the regular-structure dataset, but a high-variance, seed-fragile generator overall whose mean
is pulled up by degree/orbit collapse on the larger/sparser graphs — a model that *can* produce good
graphs but does not do so reliably, which is exactly the weakness the next rung will have to fix. The
distilled scaffold module is in the answer.
