DiGress did what I predicted at the variance end and exposed exactly the weakness I flagged, and both
halves of that result point to the same fix. The variance claim landed: the scheduled 50-step reverse
from the true noise prior killed GRAN's catastrophic seed. Where GRAN's seed-42 was 0.819 on ego and
0.814 on enzymes, DiGress's seed-42 came in at 0.082 on ego — a tenfold improvement on the worst case —
and the overall mean dropped from 0.301 to 0.265. So the controlled diffusion trajectory bought
stability, as designed. But I should read the two tables against each other statistic by statistic
before I congratulate myself, because the net improvement hides a trade that is the whole story. Break
the change down by metric. Orbit is where DiGress clearly won: ego orbit mean fell from 0.375 to 0.003
— a hundredfold — and enzymes orbit from 0.324 to 0.095, because killing the dense-blob collapse
removed the combinatorial 4-orbit explosions that seed-42 was contributing. Degree improved on ego
(0.573 → 0.269, the collapse gone) but sat essentially unchanged on enzymes (0.674 → 0.682). And
clustering got *worse*: ego clustering mean rose from 0.172 to 0.446 — a factor of 2.6 the wrong way —
and enzymes clustering from 0.283 to 0.527, a factor of 1.9. So DiGress did not simply beat GRAN; it
traded GRAN's degree/orbit collapse for a clustering residual. The overall mean fell only because the
orbit gains outweighed the clustering losses.

One dataset I have not looked at yet is `community_small`, and it is quietly informative precisely
because *neither* rung moved it. GRAN's community mean mmd_avg was 0.102; DiGress's is 0.119 — DiGress
got slightly *worse* here, not better. Break that down and the culprit is the orbit term on the one hard
seed: community seed-42's orbit MMD was 0.395 under GRAN and 0.456 under DiGress, a residual that sits
there under both samplers while the other two community seeds have orbit near 10⁻⁴. So `community_small`
carries a single stubborn seed-42-orbit spike that five refinement sweeps and fifty scheduled denoising
steps both failed to touch, riding on top of an otherwise-solved dense two-community structure. That
tells me community is already near a floor the *generation dynamics* do not control — its residual is not
a stall I can remove by simplifying the sampler, it is a fixed feature of how a dense-block generator
lands that seed's 4-orbit counts. I should therefore expect a one-shot move to hold community roughly
where it is, not to win there; the place a stability fix can actually pay is where the *variance* lives,
and that is `ego_small` and `enzymes`.

Let me quantify DiGress's variance rather than eyeball it, because "moved the fragility" should be a
number. On ego the three seeds are 0.082, 0.080, 0.556 — two nearly identical and one 6.8× worse; on
enzymes they are 0.786, 0.352, 0.166 — a 4.7× spread top to bottom; on community 0.219, 0.091, 0.048 —
the bad seed 4.5× the best. So DiGress did not *reduce* seed variance so much as *relocate* it: GRAN's
bad seed was always 42, DiGress's is 456 on ego but still 42 on enzymes, and the worst-to-best ratio
inside a dataset is still 4–7×. The scheduled trajectory made the worst *case* less catastrophic (0.556
is not 0.819) but the multi-step walk is still a coin flip over seeds; the elaboration changed which face
comes up, not whether the coin is fair. That is the observation that turns me toward removing the walk
rather than refining it.

And the seed structure tells the same story from a different angle. GRAN's fragility was concentrated in
one seed (42) that blew up on every dataset; DiGress *moved* the fragility rather than removing it. Its
worst seed is now `ego_small` seed-456 at mmd_avg 0.556 — clustering a brutal 1.022, degree 0.645 —
where GRAN's ego seed-456 had been a tame 0.152. So a seed that GRAN handled fine, DiGress collapses on,
and it collapses through *clustering and degree* rather than degree and orbit. Meanwhile the enzymes
seed-42 case stayed stubbornly hard across both rungs: 0.814 on GRAN, 0.786 on DiGress, barely budged.
That persistence tells me enzymes is not a fragility artifact I can schedule away; it is a capacity
ceiling. So the honest read of DiGress is: it is a *better* rung — lower mean, the catastrophic-seed
collapse tamed — but it is not a solved problem, and the residual it leaves has a definite shape, a
clustering/degree weakness on the sparse datasets that surfaces as a fresh seed collapse when the
re-sparsification stalls.

So what is the common thread under both GRAN's collapse and DiGress's residual? Both are sampling-
process pathologies driven by a *dense intermediate state*: GRAN starts from a too-dense `Bernoulli(0.3)`
init, DiGress's corruption limit is a dense `Bernoulli(0.5)` graph — expected degree ~62 on enzymes,
some 3800 edges it must delete before it resembles data — and in both the generator has to fight its
way back to a sparse, structured object through a process that does not natively respect sparsity. Each
rung made the *generation dynamics* more elaborate — five refinement sweeps, then fifty scheduled
denoising steps — precisely to tame that dense-intermediate problem, and each time the taming was
partial and left a seed that stalls. Let me ask the contrarian question the two failures invite: what
if the elaborate multi-step generation process is itself the source of the variance, and a *single-shot*
generator that draws one latent and decodes the whole graph in one pass would simply not have a
multi-step trajectory to collapse?

Before I commit to that, let me weigh it against the obvious alternative, which is to keep DiGress and
fix its prior. The residual has a clear named cause — the uniform `Bernoulli(0.5)` limit forces
re-sparsification — so the "principled" next move is to swap the uniform transition for a *marginal*
one whose limit is the data's own sparse edge frequency, add the analytic Bayes posterior for a
sharper reverse step, maybe fold in cycle features. Let me price that concretely so I am not dismissing it on taste. A real
`enzymes` graph has degree in the single digits — call it ~5 — so density ~`5/124 ≈ 0.04`, and a
marginal prior whose stationary edge frequency matches the data would start the reverse chain at about
`0.04·125·124/2 ≈ 310` edges instead of the uniform prior's `0.5·125·124/2 ≈ 3875`. That skips the
deletion of roughly 3500 edges DiGress's uniform limit forces in its first high-flip steps — exactly the
re-sparsification I blamed for the clustering residual — so it would very likely help.
But look at what it keeps: it keeps the multi-step stochastic trajectory, which is the *shared* mechanism
under both failures — a marginal prior makes the trajectory start closer to the data, but it is still a
50-step stochastic walk that can stall on some seed, and I would be paying more machinery (a marginal
transition matrix, an analytic posterior solver, spectral features) to make a fragile mechanism less
fragile rather than removing the fragility. And the spectral features are not free: reaching the cycle
and clustering statistics means recomputing Laplacian eigenvalues from the *noisy* graph at every one of
the 50 reverse steps, an eigendecomposition per graph per step layered on top of the very walk I am
trying to get rid of. The contrarian one-shot move is cheaper *and* attacks the root: no trajectory,
nothing to stall. The latent-variable family is the natural home for that. A VAE
draws `z ~ N(0,I)` once and decodes the entire adjacency in a single deterministic forward pass — there
is no iterative sweep, no scheduled reverse chain, nothing to collapse. The price the latent family
classically pays is a permutation-broken reconstruction loss, which I will have to confront head-on. But
the *stability* argument is strong: a one-shot decode cannot have the seed-to-seed collapse that comes
from a stochastic multi-step walk, because there is no walk.

Let me set up the VAE properly so I know exactly what I am committing to. Latent prior `p(z) = N(0,I)`,
a recognition model `q(z|G)` emitting `(mu, log sigma²)`, the reparameterization `z = mu + sigma·eps`
with `eps ~ N(0,I)` so gradients flow through the sampling, and the bound
`L = E_q[-log p(G|z)] + KL[q(z|G) || p(z)]`, with the Gaussian KL in closed form
`-1/2·sum_j(1 + log sigma_j² - mu_j² - sigma_j²)`. The only piece I design per data type is the
reconstruction term `log p(G|z)` and the encoder. Two things matter for *this* task. First, the encoder
should be *graph-aware*, not a flat MLP over the flattened adjacency the default uses — because the
whole lesson of GRAN and DiGress is that edge-relevant representations come from message passing, not
from a flat readout. So I will encode with a GCN: identity node features `x = eye(N)`, two layers of
`X' = D^{-1/2}(A+I)D^{-1/2} X W` with self-loops and degree normalization, then a permutation-invariant
mean-pool over nodes into a graph-level vector, then linear heads to `(mu, log sigma²)`.

Let me actually trace what that GCN computes, because with identity node features it simplifies in a way
worth seeing. The first layer forms `A_norm · X`; with `X = eye(N)` that product is just `A_norm`
itself, so the pre-weight representation of node `i` is its own normalized-adjacency row — a length-`N`
vector whose entries are `1/√(d_i·d_j)` on `i`'s neighbors and zero elsewhere. The `Linear(N, hidden)`
then maps each such neighborhood-indicator row to a hidden vector, so after one layer node `i`'s
embedding is a learned linear function of *which neighbors it has, degree-weighted*. The second layer
mixes those over one more hop, and the mean-pool collapses to a graph vector that summarizes the degree/
neighborhood profile of the whole graph. Let me ground that on a concrete graph so I trust the pool
actually separates the structure types. Take a `community_small` graph: two dense blocks of, say, 8 and
9 nodes, many within-block edges and a few bridges. With `X = eye`, a node deep inside a block has ~7
nonzero first-layer entries at scale `1/√(d_i·d_j) ≈ 1/8` spread over its block-mates, while a bridge
node has entries in *both* blocks. The `Linear` maps "dense-within-one-block" rows and "spans-two-blocks"
rows to different hidden vectors, the second layer pulls each block's nodes toward a shared block
signature, and the mean-pool averages those into a graph vector that reads as "two dense blocks, this
many bridges" — exactly the axis `community_small` varies over. Now run an `ego_small` star through the
same machinery: one high-degree center whose row is dense and many degree-1 leaves whose rows each point
back only at the center, pooling to something that looks nothing like the two-block signature. So the
pooled latent does separate the structure families, which is the property the decoder needs if a prior
draw is to land on a coherent graph type rather than a structureless blend. The degree-normalized GCN
aggregation is the right inductive
bias here precisely because — unlike GRAN's attention over an *uncertain* adjacency, which was why I
needed a *learned* weighting there — the encoder always sees the *true* clean adjacency, so a fixed
normalized aggregation is well-behaved, cheap, and permutation-equivariant before the pool makes it
invariant. Second, the decoder: an MLP from `z` to the full `max_nodes × max_nodes` logit matrix,
symmetrized `(L+Lᵀ)/2` and zero-diagonalled — the symmetric, self-loop-free output the loop requires,
produced in one shot.

Now the reconstruction loss, which is where I have to be most careful, because this is exactly where the
latent family classically breaks. A graph has no canonical node order, so the same graph is one of up to
`n!` adjacency matrices, and if I compare the decoded adjacency to the target *entrywise*, a correct
graph emitted with two nodes swapped is punished as if it were completely wrong. The principled cure is
*graph matching*: find a one-to-one assignment between the decoded slots and the ground-truth nodes by
maximizing a pairwise structural similarity, then score the reconstruction in the aligned frame, so the
loss becomes permutation-aware. That matching is the signature move of the latent-variable approach to
graphs — it is what makes the reconstruction term well-defined at all — but let me price it before I
reach for it. The similarity is indexed by *pairs of pairs* of nodes, so it is an `O(k⁴)` object; on
enzymes, `k = 125` gives `125⁴ ≈ 2.4×10⁸` similarity entries *per graph*, and on top of that a
power-iteration solver to find the soft assignment plus a Hungarian discretization at `O(k³)`, all
*inside every forward pass*, for a batch of 32, for 500 epochs, across three datasets. Let me carry that
all the way to the training budget, because per-graph numbers flatter the method. Those `2.4×10⁸`
similarity entries are *per graph*; a batch is 32 graphs, `enzymes` at 587 graphs with an 80/20 split
leaves ~470 training graphs so ~15 batches an epoch, and the schedule is 500 epochs — so the matching
alone would touch on the order of `2.4×10⁸ · 32 · 15 · 500 ≈ 5.8×10¹³` pairwise-similarity terms across
training, before the power iteration (repeated mat-vecs on that affinity per graph) and the Hungarian
`O(k³) ≈ 2×10⁶` per graph on top. Against that, the plain `train_step` I am about to write is one GCN
encode (two batched `125×125` by `125×256` mat-muls), one MLP decode, one closed-form KL — a handful of
`125²·256 ≈ 4×10⁶`-flop operations per graph. The matched loss is four to five orders of magnitude more
compute per step and it recurs every one of the ~7500 steps; this is not a constant I can amortize. Set
against a `train_step` that is supposed to return a scalar loss quickly and a `1.05×`-parameter budget, that
matching is simply not affordable — it is the same quartic wall that killed DeepGMG's per-decision GNN,
in a different disguise. So I make the deliberate harness-forced simplification: drop the matching
entirely and train on the **entrywise reconstruction BCE** against the target in the order the loop
padded it.

I should be clear-eyed that this is the order-sensitive loss the matching was invented to fix, so I owe
an argument for why it is still the right rung here rather than a regression to the floor. The argument
is that the *datasets are small and the harness fixes a consistent node ordering*: each graph is
delivered in one padded order, the encoder produces a latent from that ordered adjacency, and the
decoder is asked to reproduce *that same* ordered adjacency — so within a single graph the loss is
self-consistent even without matching, and the model is effectively learning to autoencode the dataset's
particular orderings rather than the abstract graph. On these small, fairly regular datasets that is
enough to learn a reasonable latent of *structure type*, because the structure (two-community, ego,
enzyme motif) is strongly correlated with the ordered adjacency the harness presents. What I give up is
the permutation *invariance* of the objective — the model wastes some capacity representing orderings
rather than structure — but I gain a `train_step` that is a single GCN encode plus a single MLP decode
plus a closed-form KL, with no `O(k⁴)` matching and no inner solver, which is the only version that fits
the budget. This is the central same-vs-different note for this rung: the load-bearing contribution of
the latent-variable graph generator — the graph-matching-aligned reconstruction — is *exactly the part
the harness omits*; what remains is a GCN-encoder VAE with entrywise reconstruction, strictly simpler
than the matched version, and I am choosing it because the matched version cannot run here.

Let me confirm the plain version actually fits where the matched one does not, with the parameter
arithmetic. At `hidden_dim = 256` and `latent_dim = 64`, the GCN encoder is tiny — `gcn1` is a
`Linear(125, 256)` at ~32k, `gcn2` a `Linear(256, 256)` at ~66k — which is the whole point of using a
GCN: the default floor's flatten encoder was a `Linear(15625, 256)` at 4M just to ingest the adjacency,
and I have replaced that with a 32k message-passing ingest. The cost that remains is the *decoder's*
final `Linear(256, max_nodes²)`, which on enzymes is `256·15625 ≈ 4M` — the one-shot full-adjacency
output is unavoidably `O(max_nodes²)` wide, and that single layer dominates the ~4.25M total. Crucially
that cost *scales with the dataset*: on `community_small` where `max_nodes ≈ 20`, `max_nodes² = 400` and
the decoder output layer is `256·400 ≈ 100k`, negligible; the parameter pressure only bites on enzymes.
So the plain VAE lands at roughly half the floor's parameter count and comfortably inside the budget,
whereas the matched version's `O(k⁴)` per-step *compute* would blow the time budget on enzymes
regardless of parameters. The arithmetic confirms the choice: matching is unaffordable in compute, the
plain fill is affordable in both compute and parameters.

Two more pieces complete the fill. First, a **node-existence predictor**: an MLP from `z` to
`max_nodes` logits, trained with BCE against the true node mask `adj.sum(-1) > 0`, weighted at 0.5
against the edge reconstruction. This is what lets a fixed-`N` decoder represent a graph on `n < N`
nodes — the decoder predicts which slots are occupied — and it is a cleaner mechanism than DiGress's,
because here the latent directly drives both the adjacency and the node mask in one shot, so they are
consistent by construction rather than read off a refined adjacency at inference. Second, the **KL
weight**, and here a little arithmetic keeps me from the classic trap. Let me put numbers on the two
terms rather than trust a default. At initialization the decoder logits are near zero, so sigmoid gives
~0.5 per entry and the reconstruction BCE is about `-log(0.5) ≈ 0.69`; as the decoder learns that the
target is ~95% structural zeros, that *mean* BCE over `max_nodes²` entries falls toward the entropy of a
near-deterministic sparse prediction, on the order of 0.05–0.1. The KL is a *mean* over the 64 latent
dimensions, and it starts near 0 (the encoder emits `mu ≈ 0`, `logvar ≈ 0`) but *grows* as the encoder
learns to place different structure types at different `mu`: for a latent that actually separates
two-community from ego from enzyme, `mu` is of order 1, giving a per-dimension KL of about
`0.5·(mu² + sigma² - 1 - log sigma²) ≈ 0.5` and a mean KL of the same order. So once the model is
learning, an *unweighted* KL of ~0.5 sits right on top of a reconstruction of ~0.05–0.1 and would
dominate the gradient, pulling every `mu` back toward 0 and every `sigma` toward 1 before the decoder can
use the latent at all — the classic posterior collapse, where the model discards `z` and the samples
become structureless. Weighting the KL at `0.001` drops its contribution to ~`5×10⁻⁴`, two orders below
the reconstruction it must not swamp, while still applying a gentle pull toward `N(0,I)` so that a prior
draw at sampling time lands where the decoder has seen data. That is the β-VAE trade read off the actual
magnitudes of the two terms in *this* harness, not a borrowed constant. The total training loss is
`recon_loss + 0.5·node_loss + 0.001·KL`.

Sampling is the part that makes me believe this rung will be *stable* where the others were not, and it
is worth saying precisely why. I draw `z ~ N(0,I)` once, decode the full adjacency logits in a single
pass, threshold at 0.5, and mask by the node-existence prediction; node counts are the masked node
count, clamped to ≥ 2. There is no iterative refinement to stall (GRAN's failure mode) and no multi-
step reverse chain whose early steps must re-sparsify a dense sample (DiGress's failure mode). The whole
generation is one deterministic decode of a single Gaussian draw, so the seed-to-seed variance comes
*only* from the latent draw and the learned decoder, not from a stochastic multi-step trajectory that
can diverge. Let me make that quantitative rather than leave it as intuition, since it is the whole
thesis of the rung. A one-shot sample is a fixed deterministic function `G = decode(z)` of a single
Gaussian draw, so its seed-to-seed variance is exactly `Var_{z∼N(0,I)}[decode(z)]` — one draw's worth of
noise, bounded by how far the learned decoder spreads over the prior. DiGress's sample is a 50-fold
composition, each step reinjecting fresh Bernoulli noise into the running adjacency, so its variance
accumulates across the 50 steps and, worse, can *amplify* whenever a step lands the chain in the
off-distribution dense region — which is precisely the path by which `ego_small` seed-456 reached 0.556
while its two siblings sat near 0.08. Collapsing 50 noise injections down to one cannot make the variance
larger; structurally it should make it much smaller, and it deletes the amplification path entirely,
because there is no second step for a bad first step to compound into. That is the structural reason I
expect this rung to have the *tightest* seed spread of the
three — the elaborate generation dynamics that I added at steps 1 and 2 to fight collapse were
themselves a source of the collapse, and removing them removes the failure mode. And it directly targets
DiGress's specific wound: DiGress's worst case was ego seed-456 at 0.556, a re-sparsification stall;
a one-shot decode has nothing to stall, so if the stability argument is right that particular collapse
should not recur.

Reading DiGress's shape, here are my falsifiable expectations. The headline claim is *overall stability*:
a one-shot GCN-VAE should produce the lowest overall mean mmd_avg of the three rungs and, more tellingly,
the *tamest worst seed* — I expect no seed to blow up the way DiGress's ego seed-456 did (mmd_avg 0.556,
clustering 1.022), because there is no multi-step trajectory to stall. Concretely I expect the overall
mean to fall below DiGress's 0.265, with the biggest gains exactly on the statistics DiGress struggled
with: `ego_small` clustering and degree, where a one-shot decode of a graph-aware latent should beat a
reverse process that has to re-sparsify a dense prior — so I expect ego clustering to fall well back from
DiGress's 0.446 and the ego mean to be the rung's clearest win. Where I am unsure, and where the dropped
matching should bite: on `enzymes`, the largest and most variable graphs (up to 125 nodes), the entrywise
order-sensitive loss has the most orderings to waste capacity on and the `max_nodes²` decoder output is
largest, and — tellingly — enzymes was the one case where *neither* prior rung improved (seed-42 stayed
~0.8 on both), which reads as a capacity ceiling, not a dynamics artifact my one-shot decode can remove.
So I expect enzymes to remain the hardest dataset for this rung too — its mmd_avg should stay in the ~0.4
range, comparable to or only modestly better than DiGress, even as ego and community improve. So the
prediction is: GraphVAE wins on *stability and overall mean* — lowest worst-seed, lowest overall
mmd_avg, decisive on ego_small — while enzymes stays the shared ceiling that the dropped graph-matching
would be needed to break. This is where the ladder ends: the one-shot latent-variable generator,
graph-aware in its encoder, is the most *reliable* of the three within this harness, and its remaining
weakness on the largest graphs names exactly the permutation-matching machinery a further rung would
have to afford. Two more checkable calls fall out of the feedback I read. For `community_small`: its only
live residual is the shared seed-42 orbit spike that neither prior sampler moved, and my change is to the
sampler's *stability* rather than to how a dense-block generator lands 4-orbit counts, so I expect
community's mmd_avg to stay essentially where GRAN and DiGress left it, in the 0.10–0.12 band — neither
the clear win ego should be nor a regression; if it jumps, my read that its residual is
dynamics-independent was wrong. And for `enzymes` I expect the degree term specifically to stay near the
~0.68 wall it held across both prior rungs, since a one-shot decode of a graph-aware latent does not by
itself fix an order-sensitive degree reconstruction on the largest graphs. The distilled scaffold module
is in the answer.
