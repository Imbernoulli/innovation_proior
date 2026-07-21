DiGress did what I predicted at the variance end and exposed the weakness I flagged, and both point at the
same fix. The variance claim landed: the scheduled 50-step reverse killed GRAN's catastrophic seed —
seed-42 was 0.819 on ego under GRAN, 0.082 under DiGress, a tenfold improvement on the worst case — and the
overall mean dropped from 0.301 to 0.265. But reading the tables statistic by statistic, the net
improvement hides a trade. Orbit is where DiGress clearly won: ego orbit mean fell from 0.375 to 0.003,
enzymes orbit from 0.324 to 0.095, because killing the dense-blob collapse removed the combinatorial
4-orbit explosions seed-42 contributed. Degree improved on ego (0.573 → 0.269) but sat unchanged on enzymes
(0.674 → 0.682). And clustering got *worse*: ego clustering rose from 0.172 to 0.446, enzymes from 0.283 to
0.527. So DiGress traded GRAN's degree/orbit collapse for a clustering residual; the mean fell only because
the orbit gains outweighed the clustering losses.

`community_small` is quietly informative because *neither* rung moved it — GRAN's community mean was 0.102,
DiGress's 0.119, slightly worse. The culprit is the orbit term on the one hard seed: community seed-42's
orbit MMD was 0.395 under GRAN and 0.456 under DiGress, a residual both samplers failed to touch while the
other two community seeds sit near 10⁻⁴. So community carries a single stubborn seed-42-orbit spike riding
on an otherwise-solved dense two-community structure — it is already near a floor the *generation dynamics*
do not control, and I should expect a one-shot move to hold community roughly where it is, not win there.
The place a stability fix pays is where the *variance* lives: `ego_small` and `enzymes`.

And DiGress did not *reduce* the seed variance so much as *relocate* it. On ego the three seeds are 0.082,
0.080, 0.556 — two nearly identical and one 6.8× worse; on enzymes 0.786, 0.352, 0.166, a 4.7× spread; on
community 0.219, 0.091, 0.048. GRAN's bad seed was always 42; DiGress's is 456 on ego but still 42 on
enzymes, and the worst-to-best ratio inside a dataset is still 4–7×. Its worst case is now ego seed-456 at
0.556 — clustering a brutal 1.022, degree 0.645 — a seed GRAN handled fine at 0.152, and it collapses
through *clustering and degree* rather than degree and orbit. Meanwhile enzymes seed-42 stayed stubbornly
hard across both rungs (0.814 → 0.786), which reads as a capacity ceiling, not a fragility artifact I can
schedule away. So the scheduled trajectory made the worst *case* less catastrophic but the multi-step walk
is still a coin flip over seeds; the elaboration changed which face comes up, not whether the coin is fair.

What is the common thread under both GRAN's collapse and DiGress's residual? Both are sampling-process
pathologies driven by a *dense intermediate state*: GRAN starts from a too-dense `Bernoulli(0.3)` init,
DiGress's corruption limit is a dense `Bernoulli(0.5)` graph it must delete ~3800 edges from before it
resembles data. Each rung made the *generation dynamics* more elaborate — five refinement sweeps, then
fifty scheduled steps — precisely to tame that dense-intermediate problem, and each time the taming was
partial and left a seed that stalls. So the contrarian question: what if the elaborate multi-step process
is itself the source of the variance, and a *single-shot* generator that draws one latent and decodes the
whole graph in one pass simply has no trajectory to collapse?

Before committing, weigh it against the obvious alternative — keep DiGress and fix its prior. The residual
has a named cause (the uniform `Bernoulli(0.5)` limit forces re-sparsification), so the principled move is to
swap the uniform transition for a *marginal* one whose limit is the data's sparse edge frequency, add the
analytic Bayes posterior, maybe fold in cycle features. Price it: a real `enzymes` graph has degree ~5,
density ~`5/124 ≈ 0.04`, so a marginal prior would start the reverse chain at about `0.04·125·124/2 ≈ 310`
edges instead of the uniform prior's `3875` — skipping the deletion of ~3500 edges, exactly the
re-sparsification I blamed for the clustering residual, so it would likely help. But it *keeps the
multi-step stochastic trajectory*, the shared mechanism under both failures: a marginal prior starts closer
to the data but it is still a 50-step walk that can stall, and I would be paying more machinery — a marginal
transition matrix, an analytic posterior solver, and spectral features that mean recomputing Laplacian
eigenvalues from the *noisy* graph at every one of the 50 reverse steps — to make a fragile mechanism less
fragile rather than removing the fragility. The contrarian one-shot move is cheaper *and* attacks the root:
no trajectory, nothing to stall. The latent family is its natural home — a VAE draws `z ~ N(0,I)` once and
decodes the entire adjacency in a single deterministic pass. The price the latent family classically pays is
a permutation-broken reconstruction loss, which I will confront head-on; but the stability argument is
strong, because a one-shot decode cannot have the seed-to-seed collapse of a stochastic multi-step walk.

Set up the VAE: latent prior `p(z) = N(0,I)`, a recognition model `q(z|G)` emitting `(mu, log sigma²)`, the
reparameterization `z = mu + sigma·eps` so gradients flow through the sampling, and the bound
`L = E_q[-log p(G|z)] + KL[q(z|G) || p(z)]` with the Gaussian KL in closed form
`-1/2·Σ_j(1 + log sigma_j² - mu_j² - sigma_j²)`. The pieces I design per data type are the reconstruction
term and the encoder. The encoder should be *graph-aware*, not a flat MLP over the flattened adjacency the
default uses — the whole lesson of GRAN and DiGress is that edge-relevant representations come from message
passing, not a flat readout. So I encode with a GCN: identity node features `x = eye(N)`, two layers of
`X' = D^{-1/2}(A+I)D^{-1/2} X W` with self-loops and degree normalization, then a permutation-invariant
mean-pool into a graph vector, then linear heads to `(mu, log sigma²)`. With identity features the first
layer forms `A_norm · X = A_norm` itself, so node `i`'s pre-weight representation is its own
normalized-adjacency row — a neighborhood indicator weighted by `1/√(d_i·d_j)`; the `Linear(N, hidden)` maps
that to a learned function of *which neighbors it has, degree-weighted*, the second layer mixes over one
more hop, and the mean-pool collapses to a graph vector summarizing the neighborhood profile. That pool
separates the structure families the datasets vary over — a `community_small` graph's two dense blocks pool
to a "two-block, this-many-bridges" signature, an `ego_small` star's one dense center and many degree-1
leaves pool to something quite different — which is the property the decoder needs if a prior draw is to
land on a coherent graph type rather than a structureless blend. The degree-normalized aggregation is the
right bias here precisely because — unlike GRAN's attention over an *uncertain* adjacency, where I needed a
learned weighting — the encoder always sees the *true* clean adjacency, so a fixed normalized aggregation is
well-behaved, cheap, and permutation-equivariant before the pool makes it invariant. The decoder is an MLP
from `z` to the full `max_nodes²` logit matrix, symmetrized and zero-diagonalled — the one-shot symmetric,
self-loop-free output the loop requires.

The reconstruction loss is where the latent family classically breaks. A graph has no canonical node order,
so the same graph is one of up to `n!` matrices, and comparing the decoded adjacency to the target
*entrywise* punishes a correct graph with two nodes swapped as if it were wrong. The principled cure is
*graph matching*: find a one-to-one assignment between decoded slots and ground-truth nodes by maximizing a
pairwise structural similarity, then score in the aligned frame — the signature move of the latent-variable
approach to graphs, what makes the reconstruction term well-defined at all. But price it. The similarity is
indexed by *pairs of pairs* of nodes, an `O(k⁴)` object; on enzymes `125⁴ ≈ 2.4×10⁸` similarity entries
*per graph*, plus a power-iteration solver and a Hungarian `O(k³)` discretization, all *inside every forward
pass*. Carry it to the training budget: `2.4×10⁸` per graph × a batch of 32 × ~15 batches an epoch × 500
epochs is on the order of `5.8×10¹³` pairwise-similarity terms across training, before the power iteration
and Hungarian on top. Against that, the plain `train_step` is one GCN encode (two batched
`125×125`-by-`125×256` mat-muls), one MLP decode, one closed-form KL — a handful of
`125²·256 ≈ 4×10⁶`-flop operations per graph. The matched loss is four to five orders of magnitude more
compute per step and recurs every one of ~7500 steps; against a `1.05×`-parameter budget and a `train_step`
meant to return a scalar quickly it is simply not affordable — the same quartic wall that killed DeepGMG's
per-decision GNN. So I make the harness-forced simplification: drop the matching and train on **entrywise
reconstruction BCE** against the target in the order the loop padded it.

That is the order-sensitive loss the matching was invented to fix, so I owe an argument for why it is still
the right rung. The datasets are small and the harness fixes a consistent node ordering: each graph is
delivered in one padded order, the encoder produces a latent from that ordered adjacency, and the decoder
reproduces *that same* ordered adjacency — so within a single graph the loss is self-consistent even without
matching, and on these small regular datasets the structure (two-community, ego, enzyme motif) is strongly
correlated with the ordered adjacency the harness presents, enough to learn a reasonable latent of
*structure type*. What I give up is permutation *invariance* — the model wastes some capacity representing
orderings rather than structure — but I gain a `train_step` that is a single GCN encode plus a single MLP
decode plus a closed-form KL, no `O(k⁴)` matching and no inner solver, the only version that fits. So the
load-bearing contribution of the latent-variable graph generator — the graph-matching-aligned
reconstruction — is *exactly the part the harness omits*; what remains is a GCN-encoder VAE with entrywise
reconstruction, strictly simpler, and I choose it because the matched version cannot run here.

The parameter arithmetic confirms the plain version fits where the matched one does not. At
`hidden_dim = 256`, `latent_dim = 64`, the GCN encoder is tiny — `gcn1` a `Linear(125, 256)` at ~32k, `gcn2`
a `Linear(256, 256)` at ~66k — which is the point: the floor's flatten encoder was a `Linear(15625, 256)` at
4M just to ingest the adjacency, and I replaced it with a 32k message-passing ingest. The remaining cost is
the decoder's final `Linear(256, max_nodes²)`, which on enzymes is `256·15625 ≈ 4M` and dominates the
~4.25M total — the one-shot full-adjacency output is unavoidably `O(max_nodes²)` wide. That cost scales with
the dataset: on `community_small` where `max_nodes ≈ 20`, the output layer is `256·400 ≈ 100k`, negligible —
the parameter pressure only bites on enzymes. So the plain VAE lands at roughly half the floor's parameter
count and inside the budget, whereas the matched version's `O(k⁴)` per-step compute would blow the time
budget on enzymes regardless of parameters.

Two pieces complete the fill. A **node-existence predictor**: an MLP from `z` to `max_nodes` logits, BCE
against the true node mask `adj.sum(-1) > 0`, weighted 0.5 — this lets a fixed-`N` decoder represent a graph
on `n < N` nodes, and it is cleaner than DiGress's because the latent drives both the adjacency and the node
mask in one shot, so they are consistent by construction rather than read off a refined adjacency at
inference. And the **KL weight**, where a little arithmetic avoids the classic trap. At init the decoder
logits are near zero, sigmoid ~0.5, so reconstruction BCE is about `-log(0.5) ≈ 0.69`; as the decoder learns
the target is ~95% structural zeros, that mean BCE over `max_nodes²` entries falls toward ~0.05–0.1. The KL
is a *mean* over 64 latent dims, near 0 at init but *growing* as the encoder places different structure
types at different `mu`: for a latent that separates the families, `mu` is order 1 and the per-dimension KL
is about `0.5·(mu² + sigma² - 1 - log sigma²) ≈ 0.5`, a mean KL of the same order. So an *unweighted* KL of
~0.5 would sit on top of a reconstruction of ~0.05–0.1 and dominate the gradient, pulling every `mu` back to
0 and every `sigma` to 1 before the decoder can use the latent — the classic posterior collapse. Weighting
the KL at `0.001` drops its contribution to ~`5×10⁻⁴`, two orders below the reconstruction, while still
pulling gently toward `N(0,I)` so a prior draw at sampling time lands where the decoder has seen data. Total
loss `recon_loss + 0.5·node_loss + 0.001·KL`.

Sampling is why I expect this rung to be *stable* where the others were not. I draw `z ~ N(0,I)` once, decode
the full adjacency in a single pass, threshold at 0.5, mask by node existence; node counts are the masked
count, clamped ≥ 2. There is no iterative refinement to stall (GRAN) and no multi-step reverse chain to
re-sparsify (DiGress) — the whole generation is one deterministic decode of a single Gaussian draw, so the
seed-to-seed variance is exactly `Var_{z∼N(0,I)}[decode(z)]`, one draw's worth of noise bounded by how far
the decoder spreads over the prior. DiGress's sample is a 50-fold composition, each step reinjecting fresh
Bernoulli noise, so its variance accumulates and can *amplify* whenever a step lands the chain in the
off-distribution dense region — precisely the path by which ego seed-456 reached 0.556 while its siblings
sat near 0.08. Collapsing 50 noise injections to one cannot make the variance larger; it deletes the
amplification path entirely, because there is no second step for a bad first step to compound into. So this
rung should have the tightest seed spread of the three, and it directly targets DiGress's specific wound: a
one-shot decode has nothing to stall, so that re-sparsification collapse should not recur.

So, reading DiGress's shape: I expect a one-shot GCN-VAE to be the most *reliable* rung — the tamest worst
seed, no seed blowing up the way DiGress's ego seed-456 did, because there is no multi-step trajectory to
stall — and, with the amplification path gone, the lowest overall mean of the three. The biggest gains
should be exactly where DiGress struggled: `ego_small` clustering and degree, where a one-shot decode of a
graph-aware latent beats a reverse process that must re-sparsify a dense prior — so ego should be the rung's
clearest win. Where the dropped matching should bite: `enzymes`, the largest and most variable graphs, where
the entrywise order-sensitive loss has the most orderings to waste capacity on and the `max_nodes²` decoder
output is largest — and enzymes was the one dataset *neither* prior rung improved (seed-42 stayed ~0.8 on
both), which reads as a capacity ceiling my one-shot decode cannot remove, not a dynamics artifact. So I
expect enzymes to remain the hardest dataset, its degree term in particular staying near the wall it held
across both prior rungs, since a one-shot decode does not by itself fix an order-sensitive degree
reconstruction on the largest graphs. And `community_small` should stay roughly where GRAN and DiGress left
it — its only live residual is the shared seed-42 orbit spike neither sampler moved, and my change is to the
sampler's stability, not to how a dense-block generator lands 4-orbit counts; if community jumps, my read
that its residual is dynamics-independent was wrong. So the prediction is: GraphVAE wins on stability and
overall mean, decisive on `ego_small`, while enzymes stays the shared ceiling that the dropped
graph-matching would be needed to break.
