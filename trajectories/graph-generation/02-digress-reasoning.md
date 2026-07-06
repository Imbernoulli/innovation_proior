GRAN's numbers told me exactly what its one-shot refinement bought and what it cost, and they point
straight at the next move. Read the seed-42 column across all three datasets: `community_small`
mmd_avg 0.219, `ego_small` 0.819, `enzymes` 0.814 — every one wrecked. Now read the other two seeds:
`community_small` drops to 0.045 and 0.043, `ego_small` to 0.149 and 0.152, `enzymes` to 0.230 and
0.237. So on every dataset the bad seed is roughly 3.5–5.5× worse than the good ones — `0.819/0.150 ≈
5.5` on ego, `0.814/0.234 ≈ 3.5` on enzymes — and the overall mean of 0.301 is one catastrophic seed
averaged with two perfectly respectable ones. GRAN is *not* a bad edge predictor; it is a *seed-fragile
sampler*.

I want to be precise about *how* the bad seed fails, because the shape of the failure names the fix.
Look at what blows up on seed-42 versus what stays put. The degree MMD explodes — 1.22 on ego against
0.25 on the good seeds, 1.44 on enzymes against 0.29, a clean ~5× jump — and the orbit MMD explodes
from essentially zero (0.0002–0.001) to 1.12 on ego and 0.97 on enzymes. But the clustering MMD barely
moves with the seed: on ego it is 0.116 on the bad seed and 0.19–0.20 on the good ones (the bad seed is
actually *lower*), on enzymes 0.037 on the bad seed against 0.39–0.42 on the good ones. That
decomposition is not noise; it is mechanistic, and it matches how the harness scores. Degree MMD is a
Gaussian-EMD kernel over the degree histogram, and orbit MMD is a plain Gaussian on *raw* 4-orbit count
vectors — both are unbounded and count-sensitive, so a dense blob (huge degrees, combinatorially many
4-node substructures) sends them to the moon. Clustering MMD, by contrast, compares histograms of a
per-node coefficient that lives in `[0,1]`; a blob just parks every node near clustering 1, bounded, so
the statistic cannot explode the same way. So the seed-42 collapse is specifically a *density
catastrophe* read out through the two count-sensitive statistics: the refinement starts from a random
`Bernoulli(0.3)` adjacency — expected degree `0.3·124 ≈ 37` on enzymes, far too dense for those sparse
graphs — and five resampling sweeps from that init *stall on a dense blob* for that seed. The diagnosis
is sharp: the failure is in the *generation dynamics*, not the model's capacity — an uncontrolled
five-step resampling from an arbitrary dense init has no schedule and no defined corruption process, so
it has no reason to converge, and whether a seed lands well is a coin flip. And the independent-edge
assumption means even the good seeds leave higher-order correlations on the table. I want a generator
whose *sampling process itself* is principled — a controlled trajectory from noise to a clean graph, not
a five-shot gamble.

So I want the strongest paradigm I know for "iteratively turn noise into a sample": diffusion. The
shape of it is a fixed forward process that slowly corrupts a clean datapoint into noise through a
Markov chain, and a learned network that inverts one step at a time; to sample I start from the noise
prior and denoise repeatedly. The reason this is the right answer to GRAN's failure is structural:
GRAN's refinement was an *unanchored* fixed-point sweep with no schedule and no defined corruption
process, so it had no reason to converge; diffusion replaces that with a *defined* corruption process
whose exact reverse the network is trained to follow, step by calibrated step, which is exactly the
"controlled trajectory" I am after. And there is a subtlety I want to keep because it is why diffusion
trains well: I do not train the network to predict the previous noisy state — that target is high-
variance, it depends on whatever noise I happened to draw. Instead I train it to predict the *clean*
graph and reconstruct the reverse step from that, which is the same x₀-prediction that DDPM uses to
strip out the label noise.

The obvious port is the one GDSS took: embed the adjacency in a continuous space and add Gaussian
noise. Let me run that forward process in my head on one of these graphs and watch it break, because
the break is the whole reason I cannot just reuse the image recipe. At t=0 I have a sparse 0/1
adjacency, almost all zeros — an `enzymes` node has degree in the single digits out of 125 possible
neighbors, so a given row is maybe 5 ones and 120 zeros. Add a little Gaussian noise; a bit more; by
the middle of the chain every entry is some real number around the noise scale and the matrix is
*dense* — those 120 structural zeros are now 120 small nonzeros. Now ask: what is the degree of node i
in this object? Summing the row gives a real number, not a count. How many triangles, how many 4-orbits?
Undefined — there are no edges, just a fog of continuous values. The sparsity that *was* the data is
gone, destroyed by the very Gaussian noise I added, and these are exactly the degree/clustering/orbit
statistics I am scored on. The denoiser would be reconstructing a sparse graph from something with no
graph structure at all. That is the wall, and it is the same wall in different clothes as GRAN's
over-dense init: a process that passes through dense, structureless intermediate states has no good way
to land on a sparse structured graph. So Gaussian noise is the wrong noise for a graph. The problem is
not diffusion; it is that I would be continuizing a thing that is discrete.

So I will not continuize it. Keep the graph discrete the whole way down: add noise that *edits the
graph* — flips an edge in or out — but leaves it, at every step, an honest discrete graph. Then the
intermediate states are always sparse and the structural notions stay defined, and denoising becomes a
clean question: given a corrupted graph, say which edges should really be there. For the binary
adjacency this is the simplest possible discrete corruption: at step t, *flip* each edge entry with
some probability, symmetrically. Concretely I tie the flip probability to a noise schedule. Take a
cosine schedule for the cumulative survival `alpha_bar_t` — `alpha_bar_t = cos²((t/T + s)/(1+s)·π/2)`
normalized to `alpha_bar_0 = 1`, with a small `s ≈ 0.008` so it corrupts gently near both ends — and
set the per-step flip probability `flip_prob = 0.5·(1 - alpha_bar_t)`. Let me tabulate it so I know the
process I am committing to rather than trusting the formula blindly. At `t/T = 0` the survival is 1 and
`flip_prob = 0`, the graph is clean. At `t/T = 0.1`, `alpha_bar ≈ 0.972` and `flip_prob ≈ 0.014` — a
gentle 1.4% edge-flip, so early steps barely perturb, which is the point of the cosine `s`-offset. At
the midpoint `t/T = 0.5`, `alpha_bar ≈ 0.494` and `flip_prob ≈ 0.253` — a quarter of the entries flip.
At `t/T = 0.9`, `alpha_bar ≈ 0.024` and `flip_prob ≈ 0.488`, and at `t/T = 1` exactly `0.5`. So the
schedule ramps the flip rate from 0 toward 0.5, spending most of its gentleness at the clean end where a
few wrong edges matter most.

The `flip_prob → 0.5` limit is worth verifying rather than asserting, because it is the whole basis for
the sampling prior. At `flip_prob = 0.5` each entry flips with probability one-half independent of its
clean value, so the probability the noisy entry is a 1 is `P(clean=1)·0.5 + P(clean=0)·0.5 = 0.5`
regardless of the clean input — the corrupted graph is a `Bernoulli(0.5)` random symmetric graph,
statistically independent of the datapoint I started from. That is exactly the property I need: a fixed
prior I can sample from without knowing the clean graph, which is what lets sampling start from noise. To
apply the corruption I draw a flip mask, symmetrize it (keep the upper triangle, mirror it) so the noisy
adjacency stays symmetric, and XOR it into the clean adjacency: `adj_noisy = |adj - flip_mask|`. I
sample a random timestep per graph, jump straight to the corrupted graph, and the whole chain trains in
parallel — no unrolling, so `train_step` stays a single forward pass exactly as GRAN's did, just now
conditioned on a timestep.

Let me trace one corruption on a concrete small graph so I trust the mechanics rather than the formula.
Take a 4-node ego star from `ego_small`: a center connected to three leaves, three real edges out of
six upper-triangular slots — adjacency rows `[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]`. Corrupt it at
`t/T = 0.5`, where `flip_prob ≈ 0.25`. In expectation `0.25·6 = 1.5` of the six independent
upper-triangular entries flip, so I might lose one of the three real edges (a leaf detaches) and gain
one of the three absent leaf-leaf edges (a spurious triangle side appears). The result is still an
honest 0/1 graph on four nodes — maybe a path plus a stray edge — and I can still read a degree
sequence and a triangle count off it, which is the whole point: the corrupted state never leaves the
space of graphs, so the denoiser's job stays "which of these present edges are real, which absent edges
should be there," a well-posed per-edge question at every noise level. Contrast the Gaussian port, where
the same midpoint would have turned all six entries into real numbers and the degree of the center would
be a sum of continuous noise, not a count.

The denoiser takes the noisy adjacency and the timestep and predicts the *clean* edge logits. Because
the target of each edge is a binary label — edge or no-edge — the entire generative problem dissolves
into a pile of independent per-edge classifications: "is this edge really present?" No graph matching
(that `O(k⁴)` alignment the latent family needs to make its reconstruction well-defined), no decoding a
continuous adjacency, no alignment — just binary cross-entropy of the predicted edge logits against the
clean adjacency. That is the satisfying payoff of keeping things discrete: where GRAN had to gamble on a
refinement sweep, diffusion-with-an-x₀-target turns generation into supervised classification, because
the forward process already told me, for each edge, exactly what the clean answer was. I keep the
node-existence head from GRAN — a per-node MLP trained with BCE against the true node mask
`adj.sum(-1) > 0`, weighted at 0.5 against the edge loss — so the fixed-`N` adjacency can still drop
empty slots and represent graphs on fewer than `max_nodes` nodes. The total loss is
`edge_loss + 0.5·node_loss`, with a grad-norm clip at 1.0, unchanged from GRAN because the loss
geometry is the same per-entry BCE.

The denoiser itself I build as a graph transformer, because attention is a natural fit for edge
prediction: every pair of nodes already has an attention score, which is exactly the object I want to
turn into an edge decision. Each layer does multi-head self-attention over node features, but the
*edges modulate the attention* rather than being passive — I take the current (noisy) adjacency, embed
each entry into a per-head bias through a `Linear(1, n_heads)`, and add that bias into the attention
scores, so an edge between two nodes raises or lowers how much they attend to each other. Trace one
entry: `adj_noisy[i,j] = 1` maps to a learned per-head bias `b_h` that is added to the pre-softmax
score between `i` and `j`, so a present edge tips the softmax toward (or away from) that neighbor before
any value is read — the graph is literally steering the attention. Node features come from a linear
embedding of an identity input plus a time embedding (a small SiLU MLP of the normalized timestep,
broadcast over nodes), so the network knows *how noisy* the graph it is looking at is and can denoise
more aggressively early and gently late — the time signal is what lets a single network cover the whole
schedule from `flip_prob = 0.014` to `0.5`. After the transformer stack, the edge predictor reads
`[n_i, n_j, adj_noisy_ij]` per pair into a logit, symmetrizes `(L+Lᵀ)/2`, and zeroes the diagonal —
the same symmetric, self-loop-free output the loop requires. Residual connections and LayerNorm
throughout. This is the attentive edge predictor I already trusted from GRAN, now placed inside a
*defined* denoising process instead of an unanchored refinement.

Let me check the budget stays affordable, since I have grown the core from three refinement blocks to
four transformer layers with a 4×-wide feed-forward. Adding it up at `hidden_dim = 128`: each
transformer layer is roughly `3·(128²)` for q/k/v, `128²` for the projection, and the feed-forward's
`128·512 + 512·128 ≈ 131k` dominates, giving about 198k per layer; four layers plus the node and time
embeddings, the edge predictor, and the node head comes to about 0.87M parameters — still comfortably
under the flatten-VAE floor and roughly twice GRAN's 0.46M, well inside the `1.05×` cap. The one
hyperparameter I do move is the learning rate: GRAN ran at `1e-3`, but the denoiser now has to fit a
*multi-task* target — reconstruct the clean graph from every noise level from 1.4% to 50% flips at
once — which is a harder and more conflicting objective than GRAN's single teacher-forced reconstruction,
and a graph transformer with softmax attention is more prone to an early destructive step. So I drop to
`2e-4`, a 5× smaller step, trading a slower ramp for stability across the schedule; the grad clip at 1.0
is the same safety net.

Sampling is where the diffusion structure earns its keep over GRAN's gamble. I start from the
corruption limit — a `Bernoulli(0.5)` random symmetric adjacency, the prior the forward process
converges to, which I just verified is input-independent — and walk t from T-1 down to 0. At each step
I run the denoiser on the current adjacency and the current timestep to get
`edge_probs = sigmoid(edge_logits)`; for every step but the last I *resample* `adj = Bernoulli(edge_probs)`,
symmetrizing; at the final step I threshold at 0.5 for a clean discrete output. Then I mask the
adjacency by the node-existence prediction and read node counts off it (clamped to ≥ 2). Notice the
contrast with GRAN, and it is quantitative, not vibes: GRAN started from an *arbitrary* `Bernoulli(0.3)`
init that matches no corruption limit and took a fixed five sweeps with no schedule; here I start from
the *exact* `Bernoulli(0.5)` prior the forward process defines and take 50 scheduled denoising steps —
a tenfold finer trajectory — each one a calibrated partial cleanup whose allowed flip budget is set by
the schedule. That tenfold refinement is not free at sampling — 50 denoiser forward passes instead of
GRAN's 5, each the same `O(B·N²·D)` pass over the full adjacency — but sampling happens once at the end
of training, so the extra cost is a fixed one-time tax I gladly pay to trade GRAN's five-shot gamble for
a fifty-step controlled descent. The trajectory is anchored at both ends — known noise prior, known
clean target — and the schedule controls how much structure each step is allowed to commit. That is the
mechanism GRAN's collapse-prone seed-42 was missing.

I want to be honest about what this fill is and is not, because it is a deliberately reduced form of
the discrete-diffusion idea, forced by the harness contract. The fuller version of discrete graph
diffusion does several things this fill drops. It diffuses *node types* as well as edges with their own
transition matrix; here only the binary edges diffuse and node existence is a separate auxiliary BCE
head, not part of the diffusion. It uses *marginal* transition matrices `Q = αI + β·1mᵀ` whose limit is
the data's empirical edge frequency — so the noisy graphs stay as sparse as the data all the way down;
here the corruption is *uniform* edge-flipping whose limit is `Bernoulli(0.5)`, a dense random graph.
That difference is not cosmetic, and it is worth the edge-count arithmetic. My `Bernoulli(0.5)` prior on
`enzymes` at `N = 125` has expected degree `0.5·124 = 62` and about `0.5·125·124/2 ≈ 3875` edges;
a real `enzymes` graph has degree in the single digits, so a few hundred edges at most. So the reverse
chain does not start near the data manifold at all — it starts as a nearly-complete graph and must
*delete on the order of 3500 edges* before the surviving structure could even resemble a protein graph,
and it has to do that deletion in the first handful of high-`flip_prob` steps. A marginal prior that
started near the data's few-percent density would skip that entire re-sparsification phase; my uniform
prior spends its early budget there instead of on placing the right edges, exactly the inefficiency the
marginal prior was designed to remove.
It samples by marginalizing an *analytic Bayes posterior* `q(z_{t-1} | z_t, x)` over the network's
clean-graph belief; here I skip the analytic posterior and just resample from the predicted edge
probabilities at each step, which is a coarser reverse step and keeps the *same independent-edge
assumption GRAN had* — the 50 steps spread the independence over time but each step still treats the
edges as conditionally independent Bernoullis, so within-block correlations are still not modeled. And
the fuller version feeds the denoiser *structural and spectral features* (cycle counts, Laplacian
eigenvalues) computed from the noisy graph at each step to beat the 1-Weisfeiler-Leman expressivity
ceiling — the one big advantage discreteness unlocks — whereas this fill's denoiser is a plain
edge-bias graph transformer with none of those extras, so it cannot perceive a triangle or a 4-cycle
directly; it only sees pairwise attention. That matters precisely for the two statistics I am scored on
that *are* higher-order — clustering counts triangles, the orbit metric counts 4-node substructures —
so the one class of features I dropped is the one class of features those metrics reward, and I should
not be surprised if they are where the residual lands. So I am keeping the load-bearing idea — *discrete
edge-flipping diffusion with an x₀-predicting graph-transformer denoiser and a scheduled reverse
process* — and dropping the marginal prior, the analytic posterior, the node diffusion, and the
structural features. I should expect to pay for the uniform-prior and missing-features choices in
exactly the orbit/clustering statistics those pieces were meant to protect.

Reading GRAN's shape, here are my falsifiable expectations. The single biggest claim is about
*variance*: the scheduled 50-step reverse from the true noise prior should stop the seed-42 collapse
that GRAN suffered, because there is no longer an arbitrary dense init and an unanchored sweep — the
process is anchored and the schedule budgets each step — so I expect DiGress's worst-seed mmd_avg to be
*far* below GRAN's seed-42 (0.819 on ego, 0.814 on enzymes), even if its best seed is no better than
GRAN's good seeds. Concretely I expect the seed-42 ego/enzymes numbers to come down from ~0.8 toward the
0.2–0.4 range, and since the collapse was carried by degree and orbit specifically, I expect the orbit
MMD in particular to fall back toward the good-seed near-zero it had on GRAN's stable seeds, lifting the
overall mean below GRAN's 0.301. Where I am unsure, and where the uniform prior should bite: because the
corruption limit is a dense `Bernoulli(0.5)` graph rather than the sparse data marginal, the reverse
process spends effort re-sparsifying, and without the structural/spectral features the denoiser cannot
perceive cycles directly — so I expect `mmd_clustering` and `mmd_orbit` on the harder, sparser graphs
(`ego_small` and `enzymes`) to be the residual weakness, possibly *worse* than a model that nails
sparsity directly, and if any seed's re-sparsification stalls it should show up as a clustering blow-up
rather than the degree/orbit blow-up GRAN had, because the failure mode has moved from "dense init the
loss never saw" to "dense prior the reverse chain must undo." So the prediction is: DiGress trades
GRAN's catastrophic-seed fragility for a more *uniform* but not uniformly-better profile — lower overall
mean, much tamer worst seed, but a clustering/orbit residual on the sparse datasets that a model with a
better-matched prior could still beat. The distilled scaffold module is in the answer.
