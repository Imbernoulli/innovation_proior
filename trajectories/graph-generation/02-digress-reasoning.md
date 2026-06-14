GRAN's numbers told me exactly what its one-shot refinement bought and what it cost, and they point
straight at the next move. Read the seed-42 column across all three datasets: `community_small`
mmd_avg 0.219, `ego_small` 0.819, `enzymes` 0.814 — every one wrecked, and wrecked the same way, by a
blown-up `mmd_degree` (1.22 on ego, 1.44 on enzymes) and a blown-up `mmd_orbit` (1.12, 0.97). That is
precisely the collapse I worried about: the refinement starts from a random `Bernoulli(0.3)` adjacency
— far too dense for these sparse graphs — and five resampling sweeps from that init *stall on a dense
blob* for that seed, so the generated degree and orbit distributions are nothing like the reference.
Now look at seeds 123 and 456: `community_small` mmd_avg drops to 0.045/0.043, `ego_small` to
0.149/0.152, `enzymes` to 0.230/0.237 — perfectly respectable, with `mmd_orbit` essentially zero. So
GRAN is *not* a bad edge predictor; it is a *seed-fragile sampler*. The mean overall mmd_avg of 0.301
is one catastrophic seed averaged with two good ones. The diagnosis is sharp: the failure is in the
*generation dynamics*, not the model's capacity — an uncontrolled five-step resampling from an
arbitrary dense init has no mechanism to reliably walk a random graph back to a sparse, structured one,
so whether a seed lands well is a coin flip. And the independent-edge assumption means even the good
seeds leave higher-order correlations on the table. I want a generator whose *sampling process itself*
is principled — a controlled trajectory from noise to a clean graph, not a five-shot gamble.

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
adjacency, almost all zeros. Add a little Gaussian noise; a bit more; by the middle of the chain every
entry is some real number around the noise scale and the matrix is *dense*. Now ask: what is the degree
of node i in this object? How many triangles? The answers are undefined — there are no edges, just a
fog of continuous values. The sparsity that *was* the data is gone, destroyed by the very Gaussian
noise I added, and these are exactly the degree/clustering/orbit statistics I am scored on. The
denoiser would be reconstructing a sparse graph from something with no graph structure at all. That is
the wall, and it is the same wall in different clothes as GRAN's over-dense init: a process that passes
through dense, structureless intermediate states has no good way to land on a sparse structured graph.
So Gaussian noise is the wrong noise for a graph. The problem is not diffusion; it is that I would be
continuizing a thing that is discrete.

So I will not continuize it. Keep the graph discrete the whole way down: add noise that *edits the
graph* — flips an edge in or out — but leaves it, at every step, an honest discrete graph. Then the
intermediate states are always sparse and the structural notions stay defined, and denoising becomes a
clean question: given a corrupted graph, say which edges should really be there. For the binary
adjacency this is the simplest possible discrete corruption: at step t, *flip* each edge entry with
some probability, symmetrically. Concretely I tie the flip probability to a noise schedule. Take a
cosine schedule for the cumulative survival `alpha_bar_t` — `alpha_bar_t = cos²((t/T + s)/(1+s)·π/2)`
normalized to `alpha_bar_0 = 1`, with a small `s` so it corrupts gently near both ends — and set the
per-step flip probability `flip_prob = 0.5·(1 - alpha_bar_t)`. At t→0 nothing flips (the graph is
clean); as t grows toward T the flip probability rises toward 0.5, the point at which each edge entry
is an independent fair coin — i.e. the corruption limit is a `Bernoulli(0.5)` random graph,
independent of the clean input, which is exactly the property I need for a sampling prior. To apply the
corruption I draw a flip mask, symmetrize it (keep the upper triangle, mirror it) so the noisy
adjacency stays symmetric, and XOR it into the clean adjacency: `adj_noisy = |adj - flip_mask|`. Sample
a random timestep per graph, jump straight to the corrupted graph, and the whole chain trains in
parallel — no unrolling.

The denoiser takes the noisy adjacency and the timestep and predicts the *clean* edge logits. Because
the target of each edge is a binary label — edge or no-edge — the entire generative problem dissolves
into a pile of independent per-edge classifications: "is this edge really present?" No graph matching
(that was GraphVAE's `O(k⁴)` curse), no decoding a continuous adjacency, no alignment — just binary
cross-entropy of the predicted edge logits against the clean adjacency. That is the satisfying payoff
of keeping things discrete: where GRAN had to gamble on a refinement sweep and GraphVAE had to align
two unordered node sets, diffusion-with-an-x₀-target turns generation into supervised classification,
because the forward process already told me, for each edge, exactly what the clean answer was. I keep
the node-existence head from GRAN — a per-node MLP trained with BCE against the true node mask
`adj.sum(-1) > 0`, weighted at 0.5 against the edge loss — so the fixed-`N` adjacency can still drop
empty slots and represent graphs on fewer than `max_nodes` nodes. The total loss is
`edge_loss + 0.5·node_loss`, with a grad-norm clip at 1.0.

The denoiser itself I build as a graph transformer, because attention is a natural fit for edge
prediction: every pair of nodes already has an attention score, which is exactly the object I want to
turn into an edge decision. Each layer does multi-head self-attention over node features, but the
*edges modulate the attention* rather than being passive — I take the current (noisy) adjacency, embed
each entry into a per-head bias through a linear map, and add that bias into the attention scores, so an
edge between two nodes raises or lowers how much they attend to each other. Node features come from a
linear embedding of an identity input plus a time embedding (a small SiLU MLP of the normalized
timestep, broadcast over nodes), so the network knows *how noisy* the graph it is looking at is and can
denoise more aggressively early and gently late. After the transformer stack, the edge predictor reads
`[n_i, n_j, adj_noisy_ij]` per pair into a logit, symmetrizes `(L+Lᵀ)/2`, and zeroes the diagonal —
the same symmetric, self-loop-free output the loop requires. Residual connections and LayerNorm
throughout. This is the attentive edge predictor I already trusted from GRAN, now placed inside a
*defined* denoising process instead of an unanchored refinement.

Sampling is where the diffusion structure earns its keep over GRAN's gamble. I start from the
corruption limit — a `Bernoulli(0.5)` random symmetric adjacency, the prior the forward process
converges to — and walk t from T-1 down to 0. At each step I run the denoiser on the current adjacency
and the current timestep to get `edge_probs = sigmoid(edge_logits)`; for every step but the last I
*resample* `adj = Bernoulli(edge_probs)`, symmetrizing; at the final step I threshold at 0.5 for a
clean discrete output. Then I mask the adjacency by the node-existence prediction and read node counts
off it (clamped to ≥ 2). Notice the contrast with GRAN: GRAN started from an *arbitrary* `Bernoulli(0.3)`
init that does not match any corruption limit and took a fixed five sweeps with no schedule; here I
start from the *exact* prior the forward process defines and take 50 scheduled denoising steps, each
one a calibrated partial cleanup. The trajectory is anchored at both ends — known noise prior, known
clean target — and the schedule controls how much structure each step is allowed to commit. That is the
mechanism GRAN's collapse-prone seed-42 was missing.

I want to be honest about what this fill is and is not, because it is a deliberately reduced form of
the discrete-diffusion idea, forced by the harness contract. The fuller version of discrete graph
diffusion does several things this fill drops. It diffuses *node types* as well as edges with their own
transition matrix; here only the binary edges diffuse and node existence is a separate auxiliary BCE
head, not part of the diffusion. It uses *marginal* transition matrices `Q = αI + β·1mᵀ` whose limit is
the data's empirical edge frequency — so the noisy graphs stay as sparse as the data all the way down;
here the corruption is *uniform* edge-flipping whose limit is `Bernoulli(0.5)`, a dense random graph,
so the reverse process must spend its early steps merely re-sparsifying, exactly the inefficiency the
marginal prior was designed to remove. It samples by marginalizing an *analytic Bayes posterior*
`q(z_{t-1} | z_t, x)` over the network's clean-graph belief; here I skip the analytic posterior and
just resample from the predicted edge probabilities at each step, which is a coarser reverse step. And
it feeds the denoiser *structural and spectral features* (cycle counts, Laplacian eigenvalues) computed
from the noisy graph at each step to beat the 1-Weisfeiler-Leman expressivity ceiling — the one big
advantage discreteness unlocks — whereas this fill's denoiser is a plain edge-bias graph transformer
with none of those extras. So I am keeping the load-bearing idea — *discrete edge-flipping diffusion
with an x₀-predicting graph-transformer denoiser and a scheduled reverse process* — and dropping the
marginal prior, the analytic posterior, the node diffusion, and the structural features. I should
expect to pay for the uniform-prior and missing-features choices in exactly the orbit/clustering
statistics those pieces were meant to protect.

Reading GRAN's shape, here are my falsifiable expectations. The single biggest claim is about
*variance*: the scheduled 50-step reverse from the true noise prior should stop the seed-42 collapse
that GRAN suffered, because there is no longer an arbitrary dense init and an unanchored sweep — so I
expect DiGress's worst-seed mmd_avg to be *far* below GRAN's seed-42 (0.819 on ego, 0.814 on enzymes),
even if its best seed is no better than GRAN's good seeds. Concretely I expect the seed-42 ego/enzymes
numbers to come down from ~0.8 toward the 0.2–0.4 range, lifting the overall mean below GRAN's 0.301.
Where I am unsure, and where the uniform prior should bite: because the corruption limit is a dense
`Bernoulli(0.5)` graph rather than the sparse data marginal, the reverse process spends effort
re-sparsifying, and without the structural/spectral features the denoiser cannot perceive cycles
directly — so I expect `mmd_clustering` and `mmd_orbit` on the harder, sparser graphs (`ego_small`
seed-456, `enzymes`) to be the residual weakness, possibly *worse* than a model that nails sparsity
directly, even as the overall mean improves through reduced collapse. So the prediction is: DiGress
trades GRAN's catastrophic-seed fragility for a more *uniform* but not uniformly-better profile — lower
overall mean, much tamer worst seed, but a clustering/orbit residual on the sparse datasets that a
model with a better-matched prior could still beat. The distilled scaffold module is in the answer.
