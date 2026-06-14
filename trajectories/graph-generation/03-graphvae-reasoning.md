DiGress did what I predicted at the variance end and exposed exactly the weakness I flagged, and both
halves of that result point to the same fix. The variance claim landed: the scheduled 50-step reverse
from the true noise prior killed GRAN's catastrophic seed. Where GRAN's seed-42 was 0.819 on ego and
0.814 on enzymes, DiGress's seed-42 came in at 0.082 on ego — a tenfold improvement on the worst case —
and the overall mean dropped from 0.301 to 0.265. So the controlled diffusion trajectory bought
stability, as designed. But now read where DiGress is *still* weak, because that is the next target.
Its `mmd_clustering` is high almost everywhere: 0.103 on community_small, but **0.446** mean on ego
(with seed-456 at a brutal 1.022) and **0.527** mean on enzymes. And `mmd_degree` on enzymes is 0.682.
That seed-456 ego collapse (clustering 1.022, degree 0.645, mmd_avg 0.556) is exactly the residual I
warned about: the uniform `Bernoulli(0.5)` corruption limit is a *dense* random graph, so the reverse
process spends its early steps merely re-sparsifying, and without structural features the denoiser
cannot perceive the clustering structure directly — so on a seed where the re-sparsification stalls, the
clustering and degree statistics blow up. DiGress traded GRAN's catastrophic-seed fragility for a
clustering/degree residual on the sparse datasets, which is a *better* trade but not a solved problem.

So what is the common thread under both GRAN's collapse and DiGress's residual? Both are sampling-
process pathologies driven by a *dense intermediate state*: GRAN starts from a too-dense `Bernoulli(0.3)`
init, DiGress's corruption limit is a dense `Bernoulli(0.5)` graph, and in both the generator has to
fight its way back to a sparse, structured object through a process that does not natively respect
sparsity. Each rung made the *generation dynamics* more elaborate — five refinement sweeps, then fifty
scheduled denoising steps — to try to tame that. Let me ask the contrarian question the two failures
invite: what if the elaborate multi-step generation process is itself the source of the variance, and a
*single-shot* generator that draws one latent and decodes the whole graph in one pass would simply not
have a multi-step trajectory to collapse? The latent-variable family is the natural home for that. A
VAE draws `z ~ N(0,I)` once and decodes the entire adjacency in a single deterministic forward pass —
there is no iterative sweep, no scheduled reverse chain, nothing to stall. The price the latent family
classically pays is a permutation-broken reconstruction loss, which I will have to confront head-on.
But the *stability* argument is strong: a one-shot decode cannot have the seed-to-seed collapse that
comes from a stochastic multi-step walk, because there is no walk.

Let me set up the VAE properly so I know exactly what I am committing to. Latent prior `p(z) = N(0,I)`,
a recognition model `q(z|G)` emitting `(mu, log sigma²)`, the reparameterization `z = mu + sigma·eps`
with `eps ~ N(0,I)` so gradients flow through the sampling, and the bound
`L = E_q[-log p(G|z)] + KL[q(z|G) || p(z)]`, with the Gaussian KL in closed form
`-1/2·sum_j(1 + log sigma_j² - mu_j² - sigma_j²)`. The only piece I design per data type is the
reconstruction term `log p(G|z)` and the encoder. Two things matter for *this* task. First, the encoder
should be *graph-aware*, not a flat MLP over the flattened adjacency the default uses — because the
whole lesson of GRAN and DiGress is that edge-relevant representations come from message passing, not
from a flat readout. So I will encode with a GCN: identity node features `x`, two layers of
`X' = D^{-1/2}(A+I)D^{-1/2} X W` with self-loops and degree normalization, then a permutation-invariant
mean-pool over nodes into a graph-level vector, then linear heads to `(mu, log sigma²)`. The
degree-normalized GCN aggregation is the right inductive bias here precisely because, unlike GRAN's
attention over an *uncertain* adjacency, the encoder always sees the *true* clean adjacency, so a fixed
normalized aggregation is well-behaved and cheap. Second, the decoder: an MLP from `z` to the full
`max_nodes × max_nodes` logit matrix, symmetrized `(L+Lᵀ)/2` and zero-diagonalled — the symmetric,
self-loop-free output the loop requires, produced in one shot.

Now the reconstruction loss, which is where I have to be most careful, because this is exactly where the
latent family classically breaks. A graph has no canonical node order, so the same graph is one of up to
`n!` adjacency matrices, and if I compare the decoded adjacency to the target *entrywise*, a correct
graph emitted with two nodes swapped is punished as if it were completely wrong. The principled cure is
*graph matching*: find a one-to-one assignment between the decoded slots and the ground-truth nodes by
maximizing a pairwise structural similarity, then score the reconstruction in the aligned frame, so the
loss becomes permutation-aware. That matching is the signature move of the latent-variable approach to
graphs — it is what makes the reconstruction term well-defined at all — but it costs `O(k⁴)` (the
similarity is indexed by pairs of pairs) and needs a power-iteration solver plus a Hungarian
discretization inside every forward pass. Within *this* harness, with `max_nodes` up to 125 on enzymes
and a 1.05× parameter budget and a 500-epoch schedule shared across three datasets, that `O(k⁴)`
matching is simply not affordable, and the contract gives me one `train_step` that must return a scalar
loss quickly. So I make the deliberate harness-forced simplification: drop the matching entirely and
train on the **entrywise reconstruction BCE** against the target in the order the loop padded it.

I should be clear-eyed that this is the order-sensitive loss the matching was invented to fix, so I owe
an argument for why it is still the right rung here rather than a regression. The argument is that the
*datasets are small and the harness fixes a consistent node ordering*: each graph is delivered in one
padded order, the encoder produces a latent from that ordered adjacency, and the decoder is asked to
reproduce *that same* ordered adjacency — so within a single graph the loss is self-consistent even
without matching, and the model is effectively learning to autoencode the dataset's particular orderings
rather than the abstract graph. On these small, fairly regular datasets that is enough to learn a
reasonable latent of *structure type*, because the structure (two-community, ego, enzyme motif) is
strongly correlated with the ordered adjacency the harness presents. What I give up is the permutation
*invariance* of the objective — the model wastes some capacity on orderings — but I gain a `train_step`
that is a single GCN encode plus a single MLP decode plus a closed-form KL, with no `O(k⁴)` matching and
no inner solver, which is the only version that fits the budget. This is the central same-vs-different
note for this rung: the load-bearing contribution of the latent-variable graph generator — the graph-
matching-aligned reconstruction — is *exactly the part the harness omits*; what remains is a GCN-encoder
VAE with entrywise reconstruction, strictly simpler than the matched version, and I am choosing it
because the matched version cannot run here.

Two more pieces complete the fill. First, a **node-existence predictor**: an MLP from `z` to
`max_nodes` logits, trained with BCE against the true node mask `adj.sum(-1) > 0`, weighted at 0.5
against the edge reconstruction. This is what lets a fixed-`N` decoder represent a graph on `n < N`
nodes — the decoder predicts which slots are occupied — and it is a cleaner mechanism than DiGress's,
because here the latent directly drives both the adjacency and the node mask in one shot, so they are
consistent by construction rather than read off a refined adjacency. Second, the **KL weight**: I keep
it tiny, `0.001`, because with a strong reconstruction target and a small latent the KL term, if given
full weight, would dominate early and collapse the posterior toward the prior before the decoder learns
any structure — the classic posterior-collapse failure. A down-weighted KL lets the latent carry real
structural information while still regularizing it toward `N(0,I)` enough that sampling from the prior
produces sensible graphs. The total training loss is `recon_loss + 0.5·node_loss + 0.001·KL`.

Sampling is the part that makes me believe this rung will be *stable* where the others were not, and it
is worth saying precisely why. I draw `z ~ N(0,I)` once, decode the full adjacency logits in a single
pass, threshold at 0.5, and mask by the node-existence prediction; node counts are the masked node
count, clamped to ≥ 2. There is no iterative refinement to stall (GRAN's failure mode) and no multi-
step reverse chain whose early steps must re-sparsify a dense sample (DiGress's failure mode). The whole
generation is one deterministic decode of a single Gaussian draw, so the seed-to-seed variance comes
*only* from the latent draw and the learned decoder, not from a stochastic multi-step trajectory that
can diverge. That is the structural reason I expect this rung to have the *tightest* seed spread of the
three — the elaborate generation dynamics that I added at steps 1 and 2 to fight collapse were
themselves a source of the collapse, and removing them removes the failure mode.

Reading DiGress's shape, here are my falsifiable expectations. The headline claim is *overall stability*:
a one-shot GCN-VAE should produce the lowest overall mean mmd_avg of the three rungs and, more tellingly,
the *tamest worst seed* — I expect no seed to blow up the way DiGress's ego seed-456 did (mmd_avg 0.556,
clustering 1.022), because there is no multi-step trajectory to stall. Concretely I expect the overall
mean to fall below DiGress's 0.265, with the biggest gains exactly on the statistics DiGress struggled
with: `ego_small` clustering and degree, where a one-shot decode of a graph-aware latent should beat a
reverse process that has to re-sparsify a dense prior. Where I am unsure, and where the dropped matching
should bite: on `enzymes`, the largest and most variable graphs (up to 125 nodes), the entrywise
order-sensitive loss has the most orderings to waste capacity on and the `max_nodes²` decoder output is
largest, so I expect enzymes to remain the hardest dataset for this rung too — its mmd_avg should stay in
the ~0.4 range, comparable to or only modestly better than DiGress, even as ego and community improve.
So the prediction is: GraphVAE wins on *stability and overall mean* — lowest worst-seed, lowest overall
mmd_avg, decisive on ego_small — while enzymes stays the shared ceiling that the dropped graph-matching
would be needed to break. This is where the ladder ends: the one-shot latent-variable generator,
graph-aware in its encoder, is the most *reliable* of the three within this harness, and its remaining
weakness on the largest graphs names exactly the permutation-matching machinery a further rung would
have to afford. The distilled scaffold module is in the answer.
