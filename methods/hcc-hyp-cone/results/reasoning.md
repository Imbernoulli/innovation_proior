Let me start from the thing that actually keeps breaking. I have a retrieval-style screener: encode a protein pocket and every candidate ligand into a shared space, score by similarity, rank, and the wins all live in the top ~1% of the list. The contrastive recipe I inherited — the DrugCLIP one — works and it scales, because inference is just a dot product against a cached matrix of molecule embeddings, so I can sweep billions of compounds. Concretely Gao et al. score `s(p,m) = g_φ(p)·f_θ(m)` with UniMol encoders for pocket and ligand, and train an in-batch symmetric InfoNCE: for each pocket the matched ligand is the positive, the other ligands in the batch are negatives, `L^p_k = −(1/N) log[ exp(s(p_k,m_k)/τ) / Σ_i exp(s(p_k,m_i)/τ) ]`, plus the mirror image with molecule as query, averaged. Fine. But two failures keep showing up and they're the whole reason I'm rethinking the scoring objective at all.

The first is graded affinity. Within one assay I don't just have "binder / decoy" — every screened ligand carries a real number, a pIC50, and I care about the *order*, because lead optimization is about telling a 100 nM compound from a 10 nM one. DrugCLIP's loss has exactly one positive per query per batch; it has no place to put "this binder is stronger than that binder." So I already know I'll need a within-assay ranking term, the listwise kind: sort the assay's ligands by measured affinity and maximize the probability of that ordering under a Plackett–Luce model, `p_{i,k} = exp(s_{i,k}) / Σ_{j≥k} exp(s_{i,j})`, with a position decay so the head of the list matters more. That's a known move; LigUnity bolts it on. It buys me ordering. It does not buy me the second thing.

The second failure is the one I can't fix with a loss term alone, and it's the one that makes me stare at the *geometry*. Activity cliffs. Two ligands that are almost the same molecule — one has an amino substituent removed, say — and yet bind ~80-fold differently. On PDB 5EHR there's exactly such a pair. A structure encoder, doing its job, maps near-identical molecules to near-identical vectors. And in Euclidean space that's fatal, because distance is linear: if `||v_1 − v_2||` is small in the encoder's native space, it stays small in the embedding, and any pocket-to-ligand score I read off is essentially the same for both. The measured numbers confirm it — a strong Euclidean pocket–ligand model gives the two members of such a pair scores that differ by less than 0.05; it literally cannot separate them. And it's not a uniform problem: these same models discriminate fine on *dissimilar* ligand pairs and degrade exactly as the pairs get structurally closer. So the failure is localized to the near-identical regime, which is precisely where cliffs are.

Now, the naive patch is "just push the cliff pair apart in the loss." But think about what that costs. If I force structurally similar molecules to large Euclidean distance, I'm fighting the encoder's own smoothness everywhere — I'd distort the metric for *all* similar pairs, including the genuinely-similar-and-similar-affinity ones, and wreck generalization. What I actually want is geometry that has *room*: a space where two points can be a hair apart along one axis and yet far apart in geodesic distance along another, so that a tiny structural wiggle can be amplified into a large functional gap *only when I want it to be*, without uniformly stretching the metric. Euclidean space has no such room — it's flat, distances add up linearly, there's nothing to exploit.

So what kind of space has room? I keep coming back to negative curvature. The defining fact about hyperbolic space is that volume grows exponentially with radius instead of polynomially. That's why it embeds trees and hierarchies with low distortion — there's exponentially more room near the boundary than a flat space provides. And graded binding affinity *is* a kind of hierarchy: within an assay, ligands fan out from weak to strong, and across the library there's a coarse-to-fine structure of "binds this family / binds this target / binds this pocket conformation specifically." If a curved space naturally encodes hierarchy, maybe it naturally encodes the affinity hierarchy, and — this is the bet — maybe the exponential growth gives me exactly the room to separate cliffs cheaply.

Let me make that bet quantitative before I commit, because "exponential room" is a slogan until I can show a small structural difference becomes a large distance. I'll work on the Lorentz model of hyperbolic space, because Nickel & Kiela established it as the numerically stable realization (the Poincaré ball is equivalent but blows up near its boundary, and the Lorentz distance and exponential map have clean closed forms). The model: points live on the upper sheet of a two-sheeted hyperboloid in `R^{n+1}`, with the Lorentzian inner product `⟨x,y⟩_L = ⟨x_space, y_space⟩ − x_time·y_time`, constrained to `⟨x,x⟩_L = −1/κ` (curvature `−κ`), so `x_time = sqrt(1/κ + ||x_space||²)`. The geodesic distance is `d_L(x,y) = (1/√κ) arccosh(−κ⟨x,y⟩_L)`. And I lift Euclidean encoder outputs onto it with the exponential map at the origin: treat the encoder output as a tangent vector `v` at the hyperboloid vertex and fold it onto the surface. Parameterizing only the space components, that map simplifies beautifully — `x_space = sinh(√κ ||v||)/(√κ ||v||) · v` — with the time coordinate forced by the constraint. (I'm following the MERU construction here; that's the one that made hyperbolic contrastive learning actually train.)

Now the cliff calculation. Take two ligands as tangent vectors `v_1, v_2` at the origin, structurally similar, so suppose they have nearly equal radial norm `||v_1|| ≈ ||v_2|| = r` and a small angle `θ = ∠(v_1, v_2)` between them, `θ ≪ 1`. I want their geodesic distance after the exponential map. The hyperbolic law of cosines for the triangle with two sides of length `r` and included angle `θ` gives, in a unit-curvature hyperboloid, `cosh d = cosh²r − sinh²r cos θ`. Let me expand for small `θ`: `cos θ ≈ 1 − θ²/2`, so `cosh d ≈ cosh²r − sinh²r (1 − θ²/2) = (cosh²r − sinh²r) + sinh²r · θ²/2 = 1 + sinh²r · θ²/2`, using `cosh²r − sinh²r = 1`. So `cosh d − 1 ≈ (sinh²r/2) θ²`. Now invert: for small argument `arccosh(1 + ε) = √(2ε) + O(ε^{3/2})`. With `ε = (sinh²r/2)θ²`, `√(2ε) = √(sinh²r · θ²) = sinh r · θ`. Restoring the curvature factor (`d_L` carries a `1/√κ`),

```
d_H(exp_o(v_1), exp_o(v_2)) ≈ (sinh r / √κ) · θ + O(θ³).
```

There it is, and it's exactly the lever I was hoping for. The separation is `θ` *amplified by `sinh r`*. In Euclidean space the same configuration gives separation `≈ r·θ` — linear in `r`. Here it's `sinh r`, which grows exponentially in `r`. So if two ligands occupy a large hyperbolic radial level and differ by even a tiny angle, their geodesic distance is enormous compared to the Euclidean image of the same angle. That's the room. A cliff pair — nearly identical structure (tiny `θ`) but very different function — can be separated if the geometry gives the stronger tier access to larger radial scale and tighter angular control, because the angular sliver is then amplified at the radius where it matters. I'm not distorting the metric uniformly; I'm letting the curvature do the amplification where the affinity hierarchy asks for it. Good. The bet has teeth. Now the whole design reduces to: make the radial coordinate carry binding-strength tier information, make angular position carry identity, and let the cliffs separate through the product of the two.

So how do I actually *control* radial depth and angular spread per ligand, as a function of its affinity? I need a structured prior on the manifold, not just "embed and hope." This is where I remember entailment cones. Ganea, Bécigneul & Hofmann built these to encode a partial order in hyperbolic space: attach to each point a nested cone opening away from the origin, and say `y` is "below" `x` in the order iff `y` lies inside `x`'s cone. The cone's opening — its half-aperture — has a closed form, and the crucial qualitative property on the Poincaré ball is

```
aper(x) = arcsin( K · (1 − c||x||²) / (√c ||x||) ),
```

which *shrinks* as `||x||` grows: a point farther out projects a narrower cone. Read that in my setting — a pocket pushed deep toward the boundary is "more certain / more specific," and it should admit only a tight set of ligand directions. That's exactly the inductive bias I want for a specific binding pocket. And since the Poincaré ball and Lorentz hyperboloid are isometric, this aperture transfers; on the hyperboloid it becomes `ω(x) = arcsin(2K / (√κ ||x_space||))` with `K` a small constant fixing the boundary near the origin. So I have a per-pocket admissible-direction cone, narrowing with the pocket's radial depth, computed in closed form.

But a single on/off cone is binary — "inside or outside" — and I established I need *graded* tiers, because affinity is graded. So I have to do something Ganea and MERU don't: turn the cone into a *hierarchy* of tiers indexed by affinity, and use *both* the radial dimension (the geodesic distance from pocket to ligand) and the angular dimension (the cone) as graded constraints. Let me build that.

First I need the angular measurement, and the argument order matters. For two manifold points `x` and `y`, the Lorentz helper `oxy_angle(x, y)` computes the exterior angle at the first point `x` in the hyperbolic triangle O-x-y. Via the hyperbolic law of cosines, writing `g(t) = cosh(√κ · d(·,·))` for each side and using that `g(d(x,y)) = −κ⟨x,y⟩_L`, `g(d(O,x)) = √κ · x_time`, `g(d(O,y)) = √κ · y_time`, the first-argument angle works out to

```
φ(x, y) = arccos( (y_time + x_time · κ⟨x,y⟩_L) / (||x_space|| · sqrt((κ⟨x,y⟩_L)² − 1)) ).
```

So I cannot treat the angle as symmetric. The aperture `ω_i` is attached to the pocket, computed as `half_aperture(pocket_i)`, while the implemented angular residual uses `φ_{ij} = oxy_angle(ligand_j, pocket_i)`, the first-argument angle with the ligand in the vertex slot. Swapping those arguments would be a different loss. The constraint I actually impose is the one the code computes: `φ_{ij} ≤ η_{ij}·ω_i`, with the pocket supplying the aperture and the ligand-pocket first-argument angle supplying the measured lean. The radial constraint is separate: the pocket-ligand geodesic distance must stay below the ligand's tier cap, `d_{ij} ≤ r_{ij}`. Two knobs per ligand: `r` as a pocket-centered geodesic cap and `η` as the scaled angular tolerance, both set by the affinity tier.

Now the tiers. I bucket each ligand by its affinity. The natural thresholds for pIC50 are the standard activity tiers — something like 5, 7, 9 (pIC50 5 is the conventional ~10 µM "active" cutoff; each step up is a decade of potency), so three thresholds give four buckets `b ∈ {0,1,2,3}`, weakest to strongest. Then per ligand,

```
r_k = r_0 + b·Δr,        η_k = η_0 − b·Δη,
```

with `r_0, η_0` the base radius and angular scale for the weakest tier and `Δr, Δη > 0` per-tier increments. Stare at the signs, because getting them backwards would invert the prior. A *stronger* binder (larger `b`) gets a *larger* `r_k`: the one-sided radial hinge permits a larger pocket-ligand geodesic cap for the stronger tier. This is a cap, not a force to sit exactly at that radius, but it gives high-affinity ligands room to occupy the larger radial scales where the `sinh r` angular amplification can matter. And a stronger binder gets a *smaller* `η_k`: a tighter angular tolerance, because a strong, specific binding event should align more decisively with the pocket's admissible direction than a weak one. So the implemented signs are strong = larger radial cap and tighter angular tolerance; weak = smaller radial cap and wider angular tolerance. The two knobs move in opposite directions with affinity, spreading tiers by both distance and angular selectivity, which is exactly the two-axis separation the cliff analysis said I needed (`sinh r · θ` uses both `r` and `θ`).

The losses are then one-sided hinges — only penalize *violations*, don't pull a ligand that already satisfies its tier:

```
L_rad = (1/√N) Σ_{i,j} max( d_{i,j} − r_{i,j},  0 ),
L_ang = (1/√N) Σ_{i,j} max( φ_{i,j} − η_{i,j}·ω_i,  0 ),
L_cone = λ_rad·L_rad + λ_ang·L_ang.
```

Why the `1/√N` and not `1/N`? Assays vary enormously in how many ligands they contain. A plain `1/N` mean would let a single huge assay dominate the gradient; no normalization at all would do the same the other way. The square root is the in-between: a sub-linear scaling so that bigger assays still contribute more (they have more real signal) but not linearly more, which keeps a handful of giant assays from drowning out the many small ones. I'll use the same `√` discipline throughout. And `λ_rad = λ_ang = 0.5` — I have no prior reason to favor the radial over the angular constraint, they're two halves of the same cone, so weight them equally.

Two regularizers fall out once I think about what can go wrong with the cone. The angular hinge `max(φ − η·ω, 0)` is zero as soon as `φ ≤ η·ω`, so the optimizer has no pressure to do better than *just* touching the boundary — and worse, it could collapse angles toward the axis trivially (everything aligned, no discrimination). To prevent the angular structure from collapsing, I add a margin `m` beyond the boundary: `R_ang = (1/√N) Σ max(φ − η·ω + m, 0)`, which keeps pushing until the ligand is comfortably *inside* the cone by `m`, not just on its edge. A small `m = 0.15` rad is enough to create a buffer without over-constraining. That's a soft "be decisively inside" pressure on top of the hard "be inside" hinge.

The second regularizer is about *which* threshold-selected ligands to emphasize. The cone losses treat all of a pocket's ligands by their tier, but the metric I'm ultimately judged on — BEDROC at α = 80.5, and EF at 1% — is dominated by the *very top* of the ranked list. The top ~2% of ranks carry ~80% of BEDROC's weight. So I should weight the loss to mirror that: within each assay, rank ligands by pocket distance, and exponentially down-weight by rank, `w_j = exp(−β·(rank_j − 1)/L_i)`. The natural choice for `β` is `β = 80.5` — the *same* focus parameter as BEDROC, so I'm shaping the training weighting to match the evaluation's early-enrichment emphasis. Then `R_het = (1/C) Σ_assays Σ_{j: v_j < v_th} −w_j log p_{i,j}`, where `p_{i,j}` is the intra-assay softmax probability of the pocket retrieving ligand `j`, and `C` is the number of assays whose threshold mask is nonempty. The threshold convention has to stay explicit: the HCC positive skip treats pIC50 `<5` as weak for contrastive positives, while this heterogeneity term follows the literal mask `v < 5`, so `R_het` is a threshold-mask term rather than a strong-pIC50-binder term. `R_ang` and `R_het` are auxiliary, so a small weight each, `λ_ang_reg = λ_het = 0.10`.

Now back to the contrastive and ranking core, because the cone hierarchy is the *new* part but it sits on top of a retrieval objective that still has to find binders at all. I keep DrugCLIP's symmetric in-batch contrastive structure and the listwise ranking, but I apply them to the hyperbolic embeddings, and I have to be careful about the similarity I feed the softmax. Geodesic distance is the "correct" hyperbolic similarity, but at *inference* I need a plain dot product so retrieval stays a matmul over a cached matrix — that constraint is non-negotiable, it's the whole reason retrieval-screening is feasible. So I'll score with the inner product of the spatial components, `s = h_poc_space · h_mol_space`, both at training (inside the softmax logits) and at inference, and let the cone/ranking losses do the geometric shaping. The on-manifold structure is learned via the cone losses; the dot product is a cheap monotone proxy for closeness that preserves it well enough near a pocket. (This is the practical compromise: train with geometry, retrieve with a matmul.)

Let me assemble the per-pathway contrastive+ranking loss carefully, because the assay bookkeeping is fiddly and the masks matter. The harness hands me, per batch, the projected pocket / molecule / protein embeddings and: `batch_list[i] = (s,e)` giving the contiguous column span of pocket `i`'s ligands, `act_list[i]` their affinities, and masking metadata. I form the logit matrix `logits = h_poc_space @ h_mol_space.T · logit_scale`, where `logit_scale = exp(τ_param)` is the learned inverse temperature (I detach it in the loss so the temperature doesn't get gamed). Then two masks, both set entries to `−inf`: a false-negative mask where pocket `i` and ligand `j` share a UniProt id (same target — an "off-diagonal" that's actually a true binder shouldn't be a negative), and a duplicate mask where ligand `j`'s SMILES is in pocket `i`'s set of known binders. Masking these out is essential or I'd be training against my own positives.

The pocket→ligand contrastive piece: for each pocket `i` and each true binder `k` in its span, I want a one-positive InfoNCE where only column `k` is the positive and everything else (post-mask) is a negative. I build that by adding a row mask that is `0` at `k` and `−inf` elsewhere, then `log_softmax` over the full masked row, and take `−lprob[k]`. I skip a ligand if it's a weak binder when the assay has more than one ligand (a sub-threshold pIC50 < 5 isn't a positive worth contrasting on), and I normalize each contribution by `1/√L_i`. The ligand→pocket mirror: each ligand column should retrieve its own pocket; I `log_softmax` over the transposed sim matrix and take the NLL of each ligand against its pocket index, summed per assay over `√L_i`. The listwise ranking piece, only when an assay has more than two ligands: for each ligand `k` in affinity order, mask out the ligands that are *not* meaningfully weaker than it — concretely, mask `idx` if `act[k] − log10(3) ≤ act[idx]`, i.e. only keep as "should-rank-below" the ligands at least 3-fold weaker, so I don't penalize near-ties within measurement noise — then `log_softmax` and accumulate `−lprob[k]` with the position decay `1/(log(k+2)·√L_i)`. That decay is the Plackett–Luce `μ_k = 1/(√B log(k+1))` shape: head of the list weighted more, normalized by assay size. The three pieces sum to the pathway loss `loss_pocket + loss_mol + loss_rank`.

Why a *third* tower, the protein sequence? The pocket is a single sampled conformation; the sequence (via ESM-2) is a complementary, conformation-free view of the same target, and adding it as a query pathway gives the loss another target-conditioned view without changing ligand caching. So I run the identical contrastive+ranking pathway a second time with the *protein-sequence* embedding as the query against the same ligands, and assemble `loss_hcc = α_poc·loss_poc_pathway + α_prot·loss_seq_pathway`, with both weights defaulting to 1. But — and this is a deliberate asymmetry — I apply the hyperbolic cone supervision *only* to the pocket branch. The pocket carries the structural, geometrically meaningful signal; the cone hierarchy is about pocket–ligand spatial alignment. Over-constraining the sequence branch geometrically would fight its role as an auxiliary query signal. So sequence participates in contrastive+ranking but not in the cone losses.

Total objective:

```
L_total = α_poc·loss_poc_pathway + α_prot·loss_seq_pathway
          + γ_cone · L_cone + λ_het · R_het + λ_ang_reg · R_ang.
```

with `γ_cone = 0.1` — the cone hierarchy is a structural *prior* shaping the space, not the primary retrieval signal, so it's down-weighted relative to the contrastive+ranking core; if I made it dominate, the model would satisfy the geometry at the expense of actually finding binders.

Now the projection heads and the parameters the geometry needs, because there are subtle initialization traps. Each backbone feature (512-d pocket, 512-d ligand, 480-d sequence) goes through a small nonlinear head into the shared 128-d space: `Linear(in, in) → ReLU → Linear(in, 128)`. Then comes the dangerous part — the exponential map. CLIP-style initialization makes the Euclidean projection output have norm `≈ √n`, and the exp map scales by `sinh`, so `sinh(√κ · √n)` for `n = 128` is astronomically large; the embeddings would explode and training would diverge immediately. The fix, which I take from how MERU tamed this: introduce a learnable per-tower scale `α`, initialized to `1/√n = 128^{−0.5}` so that the *scaled* Euclidean embedding has expected unit norm at init, and learn it in log space (so it can't collapse all embeddings to zero), and clamp `exp(α) ≤ 1` — i.e. `α ≤ 0` — so the scale can shrink but never blow the exp map back up. So each projection is `u = head(feat) · exp(α)`, then `h = exp_map0(u, κ)`. The curvature `κ` is itself learnable (init `log 1`), clamped to a sane band `[log 0.1, log 10]` so it can't collapse to Euclidean (`κ→0`) or detonate numerically. And `logit_scale` (the inverse temperature in the contrastive softmax) I init to `log 13`, the standard CLIP-ish starting temperature.

One last quirk to be careful about. The exp map returns *only* the 128-d space components — the time coordinate is implicit, recovered from the constraint. So the 128-d vector I get out *is* the on-manifold spatial representation. In the contrastive/cone losses, when I take the inner product for the softmax logits, I drop index 0 of that space vector (`emb[:, 1:]`) — a small implementation convention where the projector output is treated as `[lead, space...]` and the lead coordinate is excluded from the similarity, keeping the similarity computation aligned with how the manifold coordinates are bookkept. At *inference*, though, I score with the *full* 128-d dot product (`pocket_reps @ mol_reps.T`, take the max over a target's pockets, optionally add the sequence contribution), because that's the cached-retrieval matmul and it uses the embeddings as produced. The score function is the cheap dot product; everything geometric was paid for at training time.

Let me write the scoring module, filling exactly the stubs — projection heads + geometry params in `__init__`, the three `project_*` lifts onto the hyperboloid, `compute_loss` assembling the HCC pathways + cone hierarchy + regularizers, and the matmul `score`:

```python
"""Full hyperbolic scoring: HCC contrastive-ranking + cone hierarchy on the Lorentz model."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unimol.losses import lorentz as L


class CustomScoring(nn.Module):
    """Lorentz-hyperbolic embeddings + HCC contrastive ranking + cone hierarchy.

    Projects each backbone feature into a 128-d space, scales by a clamped learnable
    factor (so the exponential map cannot overflow), and lifts onto the hyperboloid
    via exp_map0 with a learnable curvature. Trains with an in-batch contrastive +
    listwise-ranking core (pocket->lig and seq->lig) plus radial+angular cone-tier
    constraints driven by per-ligand affinity buckets.
    """

    def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128):
        super().__init__()
        # Projection heads: Linear(in,in) -> ReLU -> Linear(in,embed_dim).
        self.mol_project = nn.Sequential(
            nn.Linear(mol_dim, mol_dim), nn.ReLU(), nn.Linear(mol_dim, embed_dim)
        )
        self.pocket_project = nn.Sequential(
            nn.Linear(pocket_dim, pocket_dim), nn.ReLU(), nn.Linear(pocket_dim, embed_dim)
        )
        self.protein_project = nn.Sequential(
            nn.Linear(protein_dim, protein_dim), nn.ReLU(), nn.Linear(protein_dim, embed_dim)
        )

        # Per-tower scale (log space), init so scaled embedding has ~unit norm; clamp exp<=1
        # to keep the exponential map from overflowing.
        self.mol_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())
        self.pocket_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())
        self.protein_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())

        # Learnable curvature (log space), clamped to [0.1, 10] so it stays hyperbolic.
        self.curv = nn.Parameter(torch.tensor([1.0]).log(), requires_grad=True)
        self._curv_minmax = {"max": math.log(10.0), "min": math.log(0.1)}

        # Inverse temperature for the contrastive softmax (CLIP-style init).
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))

        # Cone-hierarchy tiers (pIC50 thresholds) and weights.
        self.alpha_poc = 1.0
        self.alpha_prot = 1.0
        self.bounds = torch.tensor([5.0, 7.0, 9.0], dtype=torch.float32)
        self.chl_r0 = 0.5           # base radial cap (weakest tier)
        self.chl_dr = 0.5           # +radius per tier: stronger binders get a larger cap
        self.chl_eta0 = 0.7         # base angular scale (weakest tier, widest cone)
        self.chl_deta = 0.2         # -angle per tier: stronger binders get a tighter cone
        self.lambda_rad = 0.5
        self.lambda_ang = 0.5
        self.gamma_chl = 0.1        # cone is a prior, down-weighted vs the retrieval core
        self.lambda_angu = 0.10     # R_ang weight
        self.lambda_het = 0.10      # R_het weight

    def _clamp_params(self):
        self.mol_alpha.data = torch.clamp(self.mol_alpha.data, max=0.0)
        self.pocket_alpha.data = torch.clamp(self.pocket_alpha.data, max=0.0)
        self.protein_alpha.data = torch.clamp(self.protein_alpha.data, max=0.0)
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)

    def _project_to_hyperboloid(self, feat, proj_head, alpha):
        u = proj_head(feat) * alpha.exp()                 # scale before lifting
        with torch.autocast(u.device.type, dtype=torch.float32):
            h = L.exp_map0(u, self.curv.exp())            # lift onto the hyperboloid
        return h

    def project_mol(self, mol_feat):
        self._clamp_params()
        return self._project_to_hyperboloid(mol_feat, self.mol_project, self.mol_alpha)

    def project_pocket(self, poc_feat):
        self._clamp_params()
        return self._project_to_hyperboloid(poc_feat, self.pocket_project, self.pocket_alpha)

    def project_protein(self, prot_feat):
        self._clamp_params()
        return self._project_to_hyperboloid(prot_feat, self.protein_project, self.protein_alpha)

    def _compute_hcc_pair(self, emb_poc, emb_mol, batch_list, act_list,
                          uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
                          logit_scale):
        """Contrastive + listwise-ranking loss for one query pathway (pocket or seq)."""
        B = emb_poc.size(0)
        emb_poc = emb_poc[:, 1:]                          # spatial-component similarity
        emb_mol = emb_mol[:, 1:]
        logits = torch.matmul(emb_poc, emb_mol.T) * logit_scale

        N_mol = emb_mol.size(0)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        if uniprot_poc is not None and uniprot_mol is not None:
            for i in range(B):                            # same-target false negatives
                for j in range(N_mol):
                    if uniprot_poc[i] == uniprot_mol[j]:
                        mask[i, j] = True
        if pocket_lig_smiles is not None:
            for i in range(B):                            # duplicate-ligand false negatives
                bad = pocket_lig_smiles[i]
                for j in range(N_mol):
                    if lig_smiles[j] in bad:
                        mask[i, j] = True
        minus_inf = torch.finfo(logits.dtype).min
        sim_masked = logits.masked_fill(mask, minus_inf)

        # pocket -> ligand: one-positive InfoNCE per true binder; listwise ranking by affinity
        loss_mol_list, loss_rank_list = [], []
        for i in range(B):
            s, e = batch_list[i]
            acts = act_list[i]
            L_i = e - s
            out_i = sim_masked[i, s:e]
            for k in range(s, e):
                row_mask = torch.full_like(sim_masked[i], minus_inf)
                row_mask[k] = 0                            # isolate column k as the positive
                lprobs = F.log_softmax(row_mask + sim_masked[i], dim=-1)
                if L_i > 1 and acts[k - s] < 5:            # skip weak binders as positives
                    continue
                loss_mol_list.append(-lprobs[k] / math.sqrt(L_i))
            if L_i > 2:
                for k_rel in range(L_i - 1):
                    m = torch.zeros_like(out_i)
                    for idx in range(L_i):
                        if idx == k_rel:
                            continue
                        # only count ligands at least 3-fold weaker as "should rank below"
                        if acts[k_rel] - math.log10(3) <= acts[idx]:
                            m[idx] = minus_inf
                    lprobs_rank = F.log_softmax(m + out_i, dim=-1)
                    loss_rank_list.append(
                        -lprobs_rank[k_rel] / (math.log(k_rel + 2) * math.sqrt(L_i))
                    )
        loss_mol = torch.stack(loss_mol_list).sum() if loss_mol_list else torch.tensor(0.0, device=logits.device)
        loss_rank = torch.stack(loss_rank_list).sum() if loss_rank_list else torch.tensor(0.0, device=logits.device)

        # ligand -> pocket: each ligand retrieves its own pocket
        idx2poc = []
        for i, (s, e) in enumerate(batch_list):
            idx2poc += [i] * (e - s)
        targets = torch.tensor(idx2poc, dtype=torch.long, device=logits.device)
        lprobs_pocket_all = F.log_softmax(sim_masked.T, dim=-1)
        loss_pocket_list = []
        for i, (s, e) in enumerate(batch_list):
            L_i = e - s
            if L_i == 0:
                continue
            rows = list(range(s, e))
            loss_tmp = F.nll_loss(lprobs_pocket_all[rows], targets[rows], reduction="none")
            loss_pocket_list.append(loss_tmp.sum() / math.sqrt(L_i))
        loss_pocket = torch.stack(loss_pocket_list).sum() if loss_pocket_list else torch.tensor(0.0, device=logits.device)

        total = loss_pocket + loss_mol + loss_rank
        return {"loss": total, "loss_pocket": loss_pocket, "loss_mol": loss_mol,
                "loss_rank": loss_rank, "sim_masked": sim_masked}

    def compute_loss(self, mol_emb, poc_emb, prot_emb, batch_list, act_list,
                     uniprot_poc=None, uniprot_mol=None,
                     pocket_lig_smiles=None, lig_smiles=None):
        kappa = self.curv.exp().detach()
        logit_scale = self.logit_scale.exp().detach()

        # === Cone hierarchy (pocket <-> ligand only) ===
        poc_space = poc_emb[:, 1:]
        lig_space = mol_emb[:, 1:]
        poc_idx = []
        for i, (s, e) in enumerate(batch_list):
            poc_idx += [i] * (e - s)
        poc_idx = torch.tensor(poc_idx, device=poc_emb.device)

        poc_sel = poc_space[poc_idx]
        dist = L.pairwise_dist(poc_sel, lig_space, curv=kappa).diagonal()   # radial: pocket->ligand
        device = dist.device
        phi = L.oxy_angle(lig_space, poc_space[poc_idx], curv=kappa)        # angle at first arg
        omega = L.half_aperture(poc_space[poc_idx], curv=kappa)             # pocket cone aperture

        act_flat = torch.tensor([x for sub in act_list for x in sub],
                                device=poc_emb.device, dtype=torch.float32)
        bounds = self.bounds.to(poc_emb.device)
        bucket = torch.bucketize(act_flat, bounds)                          # affinity tier
        r_k = self.chl_r0 + bucket.float() * self.chl_dr                    # stronger -> larger cap
        eta_k = self.chl_eta0 - bucket.float() * self.chl_deta              # stronger -> tighter tolerance
        Nl = dist.size(0)
        L_rad = F.relu(dist - r_k).sum() / math.sqrt(Nl)
        L_ang = F.relu(phi - eta_k * omega).sum() / math.sqrt(Nl)
        loss_cone = self.lambda_rad * L_rad + self.lambda_ang * L_ang

        # angular margin: push decisively inside the cone, not just onto its boundary
        m_margin = 0.15
        R_ang = F.relu(phi - eta_k * omega + m_margin).sum() / math.sqrt(Nl)

        # heterogeneity: weight threshold-selected entries by distance rank (beta = BEDROC focus 80.5)
        R_het = torch.zeros(1, device=device)
        cnt_het = 0
        beta = 80.5
        offset = 0
        for i_poc, (s, e) in enumerate(batch_list):
            L_i = e - s
            if L_i < 1:
                continue
            d_i = dist[offset:offset + L_i].detach()
            rank = (d_i.unsqueeze(0) < d_i.unsqueeze(1)).float().sum(1) + 1
            w = torch.exp(-beta * (rank - 1) / L_i)
            logits_row = torch.matmul(poc_space[i_poc:i_poc + 1], lig_space.T) * logit_scale
            row_probs = F.softmax(logits_row[0, s:e], dim=-1)
            pos_mask = act_flat[offset:offset + L_i] < 5                    # literal v < v_th mask
            if pos_mask.any():
                R_het += -(w[pos_mask] * row_probs[pos_mask].log()).sum() / (w[pos_mask].sum() + 1e-9)
                cnt_het += 1
            offset += L_i
        R_het = R_het / max(cnt_het, 1)
        loss_reg = self.lambda_het * R_het + self.lambda_angu * R_ang

        # === HCC for both query pathways (pocket and sequence) ===
        loss_dict_poc = self._compute_hcc_pair(
            poc_emb, mol_emb, batch_list, act_list,
            uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles, logit_scale,
        )
        loss_dict_prot = self._compute_hcc_pair(
            prot_emb, mol_emb, batch_list, act_list,
            uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles, logit_scale,
        )
        loss_hcc = self.alpha_poc * loss_dict_poc["loss"] + self.alpha_prot * loss_dict_prot["loss"]

        total_loss = loss_hcc + self.gamma_chl * loss_cone + loss_reg
        return total_loss, {
            "loss": total_loss.item(),
            "loss_hcc": loss_hcc.item(),
            "loss_cone": loss_cone.item(),
            "loss_reg": loss_reg.item(),
            "sim_masked": loss_dict_poc["sim_masked"],
        }

    def score(self, mol_reps, pocket_reps, prot_reps=None):
        """Cached-retrieval scoring: full-embedding dot product, max over a target's pockets."""
        poc_scores = (pocket_reps @ mol_reps.T).max(axis=0)
        if prot_reps is not None:
            prot_scores = (prot_reps @ mol_reps.T).max(axis=0)
            return poc_scores + prot_scores
        return poc_scores
```

Let me close the loop on the causal chain. I started with a Euclidean contrastive screener that scales but cannot separate activity cliffs — structurally identical ligands with wildly different affinities collapse onto each other, because flat distance is linear and there's no room. Negative curvature gives that room: I derived, via the hyperbolic law of cosines and the `arccosh(1+ε)=√(2ε)` expansion, that two embeddings at radial depth `r` and small angle `θ` separate by `(sinh r/√κ)·θ` — angular difference amplified exponentially in depth — so if I make radial tier and angular position carry affinity-sensitive structure, cliffs can separate without distorting local structural similarity everywhere. To control distance and angular spread per ligand I borrowed the entailment cone (aperture narrowing with radial norm, from Ganea, lifted to the Lorentz model by MERU) and turned its single binary cone into an *affinity-tiered* radial+angular hierarchy: bucket ligands by pIC50, give stronger binders a larger pocket-ligand radius cap and a smaller angular tolerance, and penalize only violations with one-sided hinges, plus a margin regularizer so angles go decisively inside the boundary and a threshold-mask heterogeneity regularizer weighted by the same focus `β=80.5` the BEDROC metric uses. That cone hierarchy rides on top of DrugCLIP's symmetric in-batch contrastive loss and a Plackett–Luce listwise ranking term, run on both a pocket-query pathway and an auxiliary protein-sequence pathway (the latter without cone supervision), with false-negative and duplicate masks. The exponential map needs a clamped learnable per-tower scale to avoid overflow and a learnable, bounded curvature; training scores with a spatial inner product while inference stays a cheap cached-matrix dot product, preserving billion-scale retrieval. The geometry is paid for at training time; the screen pays off at the top of the list.
