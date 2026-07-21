GRAN's numbers told me exactly what its one-shot refinement bought and cost. Read the seed-42 column
across all three datasets: `community_small` mmd_avg 0.219, `ego_small` 0.819, `enzymes` 0.814 — every one
wrecked. The other two seeds: community drops to 0.045 and 0.043, ego to 0.149 and 0.152, enzymes to 0.230
and 0.237. So on every dataset the bad seed is roughly 3.5–5.5× worse than the good ones, and the overall
mean of 0.301 is one catastrophic seed averaged with two respectable ones. GRAN is *not* a bad edge
predictor; it is a *seed-fragile sampler*.

How the bad seed fails names the fix. The degree MMD explodes on seed-42 — 1.22 on ego against 0.25 on the
good seeds, 1.44 on enzymes against 0.29 — and the orbit MMD explodes from essentially zero to 1.12 on ego
and 0.97 on enzymes, while the clustering MMD barely moves with the seed (on enzymes 0.037 on the bad seed
against 0.39–0.42 on the good ones). That decomposition is mechanistic and matches how the harness scores:
degree MMD is a Gaussian-EMD kernel over the degree histogram and orbit MMD a plain Gaussian on *raw*
4-orbit counts — both unbounded and count-sensitive, so a dense blob (huge degrees, combinatorially many
4-node substructures) sends them to the moon — whereas clustering MMD compares histograms of a per-node
coefficient in `[0,1]`, which a blob just parks near 1, bounded. So seed-42 is a *density catastrophe* read
out through the two count-sensitive statistics: the refinement starts from a `Bernoulli(0.3)` adjacency —
expected degree `0.3·124 ≈ 37` on enzymes, far too dense — and five resampling sweeps stall on a dense blob.
The failure is in the *generation dynamics*, not capacity: an uncontrolled five-step sweep from an arbitrary
dense init has no schedule and no defined corruption process, so it has no reason to converge, and whether a
seed lands well is a coin flip. I want a generator whose *sampling process itself* is principled — a
controlled trajectory from noise to a clean graph, not a five-shot gamble.

The strongest paradigm I know for "iteratively turn noise into a sample" is diffusion: a fixed forward
process that corrupts a clean datapoint into noise through a Markov chain, and a learned network that
inverts one step at a time; to sample, start from the noise prior and denoise repeatedly. That is the right
answer to GRAN's failure structurally — its refinement was an *unanchored* fixed-point sweep with no defined
corruption process, so it had no reason to converge; diffusion replaces it with a *defined* corruption whose
exact reverse the network is trained to follow, step by calibrated step. And a subtlety worth keeping: I do
not train the network to predict the previous noisy state — that target is high-variance, it depends on
whatever noise I drew — but to predict the *clean* graph and reconstruct the reverse step from that, the
x₀-prediction DDPM uses to strip out the label noise.

The obvious port is GDSS's: embed the adjacency in continuous space and add Gaussian noise. Run that forward
process on one of these graphs and watch it break. At t=0 I have a sparse 0/1 adjacency — an `enzymes` row is
maybe 5 ones and 120 zeros. Add a little Gaussian noise, a bit more, and by mid-chain every entry is a real
number around the noise scale: those 120 structural zeros are now 120 small nonzeros, the matrix is *dense*.
What is the degree of node i now? A real sum, not a count. Triangles, 4-orbits? Undefined — there are no
edges, just a fog of continuous values. The sparsity that *was* the data is gone, destroyed by the noise I
added, and those are exactly the statistics I am scored on. So Gaussian noise is the wrong noise for a
graph — the same wall as GRAN's over-dense init: a process that passes through dense, structureless
intermediate states has no good way to land on a sparse structured graph. The problem is not diffusion; it
is continuizing a discrete thing.

So keep the graph discrete the whole way down: add noise that *edits* it — flips an edge in or out — but
leaves it an honest discrete graph at every step. Then the intermediate states stay sparse, the structural
notions stay defined, and denoising becomes "given a corrupted graph, say which edges should really be
there." The simplest discrete corruption for a binary adjacency: at step t, *flip* each edge entry with some
probability, symmetrically. I tie the flip probability to a cosine schedule for the cumulative survival —
`alpha_bar_t = cos²((t/T + s)/(1+s)·π/2)` normalized to `alpha_bar_0 = 1`, with `s ≈ 0.008` so it corrupts
gently near both ends — and set `flip_prob = 0.5·(1 - alpha_bar_t)`. Tabulating the process I am committing
to: at `t/T = 0`, `flip_prob = 0`, clean; at `t/T = 0.1`, `alpha_bar ≈ 0.972` and `flip_prob ≈ 0.014`, a
barely-perturbing 1.4% flip; at the midpoint `flip_prob ≈ 0.253`, a quarter of entries; at `t/T = 0.9`
about 0.488, and at `t/T = 1` exactly 0.5. So the schedule ramps the flip rate from 0 toward 0.5, spending
its gentleness at the clean end where a few wrong edges matter most.

The `flip_prob → 0.5` limit is the whole basis for the sampling prior, so pin it down. At `flip_prob = 0.5`
each entry flips with probability one-half independent of its clean value, so
`P(noisy=1) = P(clean=1)·0.5 + P(clean=0)·0.5 = 0.5` regardless of the clean input — the corrupted graph is
a `Bernoulli(0.5)` random symmetric graph, statistically independent of the datapoint. That is exactly the
property sampling needs: a fixed prior I can draw from without knowing the clean graph. To apply the
corruption I draw a flip mask, symmetrize it (upper triangle mirrored), and XOR it into the clean adjacency,
`adj_noisy = |adj - flip_mask|`. I sample a random timestep per graph and jump straight to the corrupted
state, so the whole chain trains in parallel — `train_step` stays a single forward pass as GRAN's did, now
conditioned on a timestep. Corrupt a 4-node ego star at the midpoint and roughly `0.25·6 = 1.5` of its six
upper-triangular entries flip: maybe a leaf detaches and a spurious edge appears, but the result is still an
honest 0/1 graph with a readable degree sequence and triangle count — the corrupted state never leaves the
space of graphs, which is the whole point, and the denoiser's job stays "which present edges are real, which
absent edges should be there," well-posed at every noise level. Contrast the Gaussian port, where the
midpoint turns every entry into a real number and the center's degree is a sum of continuous noise, not a
count.

The denoiser takes the noisy adjacency and the timestep and predicts the *clean* edge logits. Because each
edge's target is a binary label, the generative problem dissolves into independent per-edge classifications —
"is this edge really present?" — trained by plain BCE of the predicted edge logits against the clean
adjacency, with no graph matching, no continuous decode, no alignment. That is the payoff of staying
discrete: where GRAN gambled on a refinement sweep, diffusion with an x₀-target turns generation into
supervised classification, because the forward process already told me the clean answer for each edge. I
keep GRAN's node-existence head — a per-node MLP with BCE against `adj.sum(-1) > 0`, weighted 0.5 — so the
fixed-`N` adjacency can still drop empty slots. Total loss `edge_loss + 0.5·node_loss` with a grad-norm clip
at 1.0, unchanged because the loss geometry is the same per-entry BCE.

The denoiser is a graph transformer, because attention is a natural fit for edge prediction — every pair of
nodes already has an attention score, exactly the object I want to turn into an edge decision. Each layer
does multi-head self-attention over node features, but the *edges modulate the attention*: I embed each
(noisy) adjacency entry into a per-head bias through `Linear(1, n_heads)` and add it into the scores, so a
present edge tips the softmax toward or away from that neighbor before any value is read — the graph steers
the attention. Node features come from a linear embedding of an identity input plus a time embedding (a
small SiLU MLP of the normalized timestep, broadcast over nodes), so the network knows *how noisy* the graph
is and can denoise aggressively early and gently late — the time signal is what lets one network cover the
whole schedule from `flip_prob = 0.014` to `0.5`. After the stack, the edge predictor reads
`[n_i, n_j, adj_noisy_ij]` per pair into a logit, symmetrizes, and zeroes the diagonal — the same symmetric,
self-loop-free output the loop requires. This is the attentive edge predictor I trusted from GRAN, now
inside a *defined* denoising process instead of an unanchored refinement.

The budget stays affordable even though the core grew from three refinement blocks to four transformer
layers with a 4×-wide feed-forward. At `hidden_dim = 128` each layer is roughly `3·128²` for q/k/v, `128²`
for the projection, and the feed-forward `128·512 + 512·128 ≈ 131k` dominates — about 198k per layer; four
layers plus embeddings and heads is about 0.87M parameters, roughly twice GRAN's 0.46M and well inside the
`1.05×` cap. The one hyperparameter I move is the learning rate: the denoiser fits a *multi-task* target —
reconstruct the clean graph from every noise level from 1.4% to 50% flips at once — which is harder and more
conflicting than GRAN's single teacher-forced reconstruction, and a graph transformer with softmax attention
is more prone to an early destructive step. So I drop from `1e-3` to `2e-4`, trading a slower ramp for
stability; the grad clip at 1.0 is the same safety net.

Sampling is where the diffusion structure earns its keep. I start from the corruption limit — the
`Bernoulli(0.5)` random symmetric adjacency I verified is input-independent — and walk t from T-1 down to 0.
At each step I run the denoiser for `edge_probs = sigmoid(edge_logits)`; for every step but the last I
resample `adj = Bernoulli(edge_probs)`, symmetrizing; at the final step I threshold at 0.5. Then I mask by
the node-existence prediction and read node counts (clamped ≥ 2). The contrast with GRAN is structural:
GRAN started from an *arbitrary* `Bernoulli(0.3)` init that matches no corruption limit and took a fixed
five sweeps with no schedule; here I start from the *exact* prior the forward process defines and take 50
scheduled denoising steps, each a calibrated partial cleanup whose flip budget the schedule sets. The 50
passes cost more than GRAN's 5, but sampling runs once at the end of training, so it is a fixed one-time tax
I pay to trade a five-shot gamble for a fifty-step controlled descent anchored at both ends — known noise
prior, known clean target. That is the mechanism GRAN's collapse-prone seed was missing.

This fill is a deliberately *reduced* form of discrete graph diffusion, forced by the contract. The fuller
version diffuses *node types* too, with their own transition matrix; here only the binary edges diffuse and
node existence is a separate auxiliary BCE head. It uses *marginal* transition matrices `Q = αI + β·1mᵀ`
whose limit is the data's empirical edge frequency, so the noisy graphs stay as sparse as the data all the
way down; here the corruption is *uniform* edge-flipping whose limit is `Bernoulli(0.5)`, a dense random
graph. That difference is not cosmetic. My `Bernoulli(0.5)` prior on `enzymes` at `N = 125` has expected
degree `0.5·124 = 62` and about `0.5·125·124/2 ≈ 3875` edges; a real `enzymes` graph has a few hundred at
most, so the reverse chain starts as a nearly-complete graph and must *delete on the order of 3500 edges* in
its first few high-`flip_prob` steps before the structure could resemble a protein graph. A marginal prior
starting near the data's few-percent density would skip that entire re-sparsification phase; my uniform
prior spends its early budget there instead of on placing the right edges. The fuller version also samples by
marginalizing an *analytic Bayes posterior* `q(z_{t-1} | z_t, x)`; here I skip it and resample directly from
the predicted edge probabilities, a coarser reverse step that keeps the *same independent-edge assumption
GRAN had* — 50 steps spread the independence over time but each step still treats edges as conditionally
independent, so within-block correlations stay unmodeled. And the fuller version feeds the denoiser
*structural and spectral features* (cycle counts, Laplacian eigenvalues) from the noisy graph at each step
to beat the 1-Weisfeiler-Leman expressivity ceiling — the advantage discreteness unlocks — whereas this
denoiser is a plain edge-bias graph transformer that cannot perceive a triangle or 4-cycle directly, only
pairwise attention. That matters precisely for the two higher-order statistics I am scored on: clustering
counts triangles, orbit counts 4-node substructures. So I keep the load-bearing idea — *discrete
edge-flipping diffusion with an x₀-predicting graph-transformer denoiser and a scheduled reverse process* —
and drop the marginal prior, the analytic posterior, node diffusion, and structural features, expecting to
pay for the uniform-prior and missing-features choices in exactly the orbit/clustering statistics those
pieces protect.

So, reading GRAN's shape: the scheduled 50-step reverse from the true noise prior should tame the
catastrophic seed-42 collapse, since there is no longer an arbitrary dense init and an unanchored sweep —
the process is anchored and the schedule budgets each step. I expect the worst seed's ego/enzymes mmd_avg to
fall well below GRAN's ~0.8, and the orbit MMD in particular to come back toward the near-zero it held on
GRAN's stable seeds, lifting the overall mean below GRAN's 0.301. Where the uniform prior should bite:
because the corruption limit is a dense `Bernoulli(0.5)` graph rather than the sparse data marginal, the
reverse chain spends effort re-sparsifying, and without structural features the denoiser cannot perceive
cycles — so I expect the residual weakness to move to `mmd_clustering` and `mmd_orbit` on the sparse
`ego_small` and `enzymes`, and any residual seed fragility to surface through those cycle-sensitive
statistics rather than the degree/orbit blowup GRAN had, because the failure mode has moved from "dense init
the loss never saw" to "dense prior the reverse chain must undo." So: a tamer worst seed and a lower mean,
traded for a clustering/orbit residual on the sparse datasets that a better-matched prior could still beat.
