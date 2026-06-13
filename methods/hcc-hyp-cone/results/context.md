# Context: geometry and objective design for contrastive protein–ligand virtual screening (circa 2023–2025)

## Research question

Virtual screening ranks a large compound library against a protein target to surface likely
binders. The retrieval framing treats this as nearest-neighbor search: encode the target
(its binding pocket, optionally its sequence) and every candidate ligand into a shared
embedding space, then rank candidates by a cheap similarity to the target embedding. Given
pretrained backbone encoders that are fine-tuned jointly — a 3D structure encoder producing a
512-dim pocket feature and a 512-dim ligand feature, and a protein-sequence language model
producing a 480-dim feature — the open question is the *scoring module*: how to project those
features into a shared space, what **geometry** that space should have, and what **training
loss** should shape it so that active binders are pushed to the very top of the ranked list.

Two things make this hard beyond ordinary contrastive retrieval. First, within a single assay
the ligands are not just "binder vs non-binder" — they carry *graded* affinity (a pIC50 value),
and the practical payoff is concentrated in the earliest fraction of the ranked list (the
top ~1%), so the objective must be sensitive to fine-grained ordering, not just binary
membership. Second, and most stubbornly, there are **activity cliffs**: pairs of ligands that
are almost identical in chemical structure yet differ by one or two orders of magnitude in
binding strength. A representation that respects chemical similarity will place such a pair
close together — exactly where the screen needs them far apart. A solution has to keep
structurally similar molecules from collapsing onto each other *whenever their affinities
diverge*, without distorting the genuine structural similarity it relies on elsewhere, and
without giving up the matmul-only inference that makes billion-scale screening feasible.

## Background

The retrieval-as-screening paradigm rests on a 3D molecular encoder. UniMol (Zhou et al.,
ICLR 2023) is an SE(3)-equivariant 3D graph transformer: atoms are tokens, a pairwise
distance representation is fed in as an attention bias, and a special `[CLS]`-like atom at the
geometric center pools a fixed-length representation of a molecule or a pocket. It is
pretrained by masked-atom-type prediction and coordinate denoising on large unlabeled
structure data, so it gives a strong, transferable 512-dim embedding for both pockets and
ligands. On the sequence side, ESM-2 (Lin et al., Science 2023) is a protein language model
trained on evolutionary-scale sequence data; its per-residue hidden states (here a 480-dim
model) give a complementary, structure-free view of a target.

The geometric prior in play is **hyperbolic geometry**. A space of constant negative curvature
has volume that grows *exponentially* with radius, rather than polynomially as in Euclidean
space; this is why it embeds tree-like and hierarchical data with far lower distortion than a
flat space (Nickel & Kiela 2017, 2018). The Lorentz (hyperboloid) model realizes an
n-dimensional hyperbolic space as the upper sheet of a two-sheeted hyperboloid in R^{n+1},
equipped with the Lorentzian inner product

```
⟨x, y⟩_L = ⟨x_space, y_space⟩ − x_time · y_time,     x, y ∈ R^{n+1},
```

with points constrained to ⟨x, x⟩_L = −1/κ (curvature −κ, κ > 0), equivalently
`x_time = sqrt(1/κ + ||x_space||²)`. The geodesic distance is
`d_L(x, y) = (1/√κ)·arccosh(−κ⟨x, y⟩_L)`. To turn a Euclidean encoder output into a point on
this manifold, one uses the exponential map at the origin: treat the encoder output as a
tangent vector at the hyperboloid vertex and "fold" it onto the surface. Nickel & Kiela 2018
established the Lorentz model as the numerically stable choice (the Poincaré ball, while
equivalent, is prone to instability near the boundary), and the closed forms of its distance
and exponential map are simple.

A second, sharper piece of hyperbolic machinery is the **entailment cone** (Ganea, Bécigneul &
Hofmann, ICML 2018). To encode a partial order — "this entails that," "this is a special case
of that" — Ganea et al. attach to each point a nested, geodesically-convex cone opening away
from the origin; a point lies "below" another in the order iff it sits inside that other's
cone. The cone's *half-aperture* (its opening angle) has a closed form. On the Poincaré ball,
for a point x_b with curvature parameter c,

```
aper_b(x_b) = arcsin( K · (1 − c||x_b||²) / (√c · ||x_b||) ),
```

with a small constant K fixing the boundary condition near the origin. The key qualitative
fact: the aperture *shrinks* as ||x_b|| grows — a point farther from the origin (more
"specific," more certain) projects a narrower cone. The Poincaré-ball and Lorentz models are
isometric, so this aperture transfers to the hyperboloid.

Two empirical facts about *Euclidean* retrieval set up the problem. (1) In a flat embedding
space, distance grows linearly, so two ligands that a structure encoder maps close together
*stay* close — there is no geometric room to separate an activity-cliff pair without bending
the metric, which degrades the smoothness the encoder relies on. Documented activity-cliff
cases bear this out: for instance, a known pair on PDB 5EHR (a ligand and its derivative with
one amino substituent removed) has an ~80-fold experimental affinity gap while a strong
Euclidean pocket–ligand model assigns the two nearly identical scores — the score difference is
often below 0.05, i.e. the model cannot tell them apart. (2) The same Euclidean models'
pairwise affinity-discrimination accuracy *degrades* precisely as candidate ligands become more
chemically similar, while their accuracy on dissimilar pairs is fine — the failure is specific
to the near-identical regime, which is where cliffs live.

## Baselines

**DrugCLIP (Gao et al., NeurIPS 2023).** Reframed structure-based virtual screening as dense
retrieval. UniMol encoders produce pocket embeddings `g_φ(x^p)` and molecule embeddings
`f_θ(x^m)`; similarity is the dot product `s(x^p, x^m) = g_φ(x^p)·f_θ(x^m)` (cosine if
normalized), in a shared **Euclidean** space. Training is in-batch symmetric InfoNCE: over a
batch of N paired (pocket, molecule), the diagonal pairs are positives and all off-diagonal
pairs are negatives. The Pocket-to-Mol loss for a pocket `x^p_k` is

```
L^p_k = − (1/N) · log [ exp(s(x^p_k, x^m_k)/τ) / Σ_i exp(s(x^p_k, x^m_i)/τ) ],
```

and the symmetric Mol-to-Pocket loss `L^m_k` swaps roles; the batch loss is
`L = (1/2) Σ_k (L^p_k + L^m_k)`. Inference is offline: cache all molecule embeddings, encode a
query pocket once, and the only online cost is a dot product against the cache — this is what
makes screening over billions of molecules tractable. **Gap:** the embedding space is flat, so
it cannot geometrically separate activity cliffs (see Background); each query has exactly one
positive per batch, so there is no notion of *graded* affinity; and the objective is purely
binary membership, with no within-assay ranking signal.

**Listwise ranking on top of contrastive (LigUnity, Feng et al. 2025; Plackett–Luce / ListNet,
Cao et al. 2007).** To add fine-grained ordering, a listwise ranking term sorts each assay's
ligands by measured affinity and maximizes the probability of producing that order under the
Plackett–Luce model: the probability of placing ligand v_{i,k} at step k from the remaining set
R_{i,k} = {k, …, B} is

```
p_{i,k}(v_{i,k}) = exp(s_{i,k}) / Σ_{j ∈ R_{i,k}} exp(s_{i,j}),     s_{i,k} = ⟨h_{u_i}, h_{v_{i,k}}⟩/τ,
```

and the per-assay loss is a position-decayed sum `L_rank = − Σ_k μ_k log p_{i,k}(v_{i,k})` with
a decay `μ_k = 1/(√B · log(k+1))` that weights the head of the list more. This recovers
within-assay affinity order. **Gap:** it remains a Euclidean objective; it orders ligands but
adds no geometric structure that *separates* near-identical structures by affinity, and gives
no explicit handle on the graded radial/angular structure a curved space could offer.

**Hyperbolic contrastive vision–language (MERU, Desai et al., ICML 2023).** Showed how to do
CLIP-style contrastive learning on the Lorentz hyperboloid: encode to Euclidean, scale by a
learnable per-tower scalar α (initialized to 1/√n so embeddings have unit norm at init, learned
in log space, clamped so exp(α) ≤ 1 to keep the exponential map from overflowing), then lift via
the exponential map at the origin, which — parameterizing only the space components — reduces to

```
x_space = sinh(√c ||v_space||) / (√c ||v_space||) · v_space,
```

with the time component recovered from the constraint. The curvature −c is itself a learnable
parameter. MERU also adapted the entailment cone into a loss with a learnable curvature: the
cone half-aperture on the hyperboloid is `aper(x) = arcsin(2K / (√c ||x_space||))` (the Lorentz
image of Ganea's Poincaré aperture, K = 0.1), and the *exterior angle* of the hyperbolic
triangle O-x-y — measuring how far y "leans" off the cone axis of x — has the closed form, via
the hyperbolic law of cosines,

```
ext(x, y) = arccos( (y_time + x_time·c⟨x,y⟩_L) / (||x_space|| · sqrt((c⟨x,y⟩_L)² − 1)) ).
```

The entailment loss `max(0, ext(x,y) − aper(x))` is zero when y already sits inside x's cone and
otherwise pulls it in, enforcing a partial order. **Gap:** MERU is built for image/text with a
*binary* "text entails image" relation; it has no notion of a continuous, graded order (affinity
tiers), no within-assay listwise ranking, and is a two-tower model, not a target-as-pocket-and-
sequence screening setup. Its cone is a single on/off constraint, not a tiered hierarchy.

## Evaluation settings

Training uses a curated assay-level dataset assembled from ChEMBL, BindingDB, and PDBBind,
organized so that affinity values are only compared *within* an assay (because IC50/Kd/Ki and
conditions are not cross-comparable). Targets whose UniProt IDs appear in the screening test
sets are strictly excluded; evaluation is therefore zero-shot (no target-specific training).
Optimization is Adam with learning rate 1×10⁻⁴.

Virtual-screening benchmarks: **DUD-E** (102 targets, each with experimentally verified actives
and ~50 property-matched decoys per active), **LIT-PCBA** (15 targets with 400K+ confirmed
inactives — realistic, no synthetic decoy bias), and challenging decoy sets in the same family.
Metrics, averaged over targets: **AUROC** (probability a random active outranks a random
inactive); **BEDROC** with focus α = 80.5 (Boltzmann-enhanced discrimination of ROC, where the
top ~2% of ranks carry ~80% of the weight — an early-enrichment metric); **Enrichment Factor**
EF at 0.5%/1%/5% cutoffs (fold-improvement in actives retrieved in the top fraction versus
random); and ROC enrichment at fixed false-positive rates. Affinity-ranking benchmarks (JACS,
Merck congeneric series) use Pearson r and Spearman ρ between predicted and experimental
affinity. Higher is better throughout.

## Code framework

The backbone encoders, data pipeline, training loop, and evaluation are fixed. The scoring
module is a single `nn.Module` that owns the projection from backbone features to the shared
embedding space and the training loss; everything geometric and loss-related is the empty slot.
The harness supplies, per training step, the projected pocket / molecule / protein embeddings
and the assay bookkeeping: `batch_list` (each pocket's contiguous span of ligands), `act_list`
(per-ligand affinity values), and the masking metadata (`uniprot_*` for same-target false
negatives, `*_smiles` for duplicate ligands). At evaluation it asks for numpy scores so that
the cached-embedding, dot-product retrieval stays a matmul.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Lorentz-model primitives that already exist (exp_map0, geodesic distance, cone aperture,
# triangle exterior angle); whether and how to use them is part of the design.
from unimol.losses import lorentz as L


class CustomScoring(nn.Module):
    """Scoring module for contrastive protein–ligand virtual screening.

    Backbone features (fine-tuned jointly): mol [B,512], pocket [B,512], protein [B,480].
    The projection geometry and the training loss are exactly what is to be designed.
    """

    def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128):
        super().__init__()
        # TODO: projection heads from backbone dims into the shared embedding space,
        #       plus whatever parameters the embedding geometry and loss require.
        pass

    def project_mol(self, mol_feat):      # [B,512] -> [B, embed_dim]
        # TODO: project a molecule feature into the shared embedding space.
        pass

    def project_pocket(self, poc_feat):   # [B,512] -> [B, embed_dim]
        # TODO: project a pocket feature into the shared embedding space.
        pass

    def project_protein(self, prot_feat): # [B,480] -> [B, embed_dim]
        # TODO: project a protein-sequence feature into the shared embedding space.
        pass

    def compute_loss(self, mol_emb, poc_emb, prot_emb, batch_list, act_list,
                     uniprot_poc=None, uniprot_mol=None,
                     pocket_lig_smiles=None, lig_smiles=None):
        # batch_list[i] = (s, e): pocket i's ligands occupy columns [s, e).
        # act_list[i]   = affinity values for those ligands.
        # TODO: the training objective we will design.
        pass

    def score(self, mol_reps, pocket_reps, prot_reps=None):
        # numpy arrays; must stay a cheap matmul so cached retrieval scales.
        # TODO: the evaluation scoring rule.
        pass
```

The final scoring module fills exactly these stubs: the projection heads and embedding
parameters in `__init__` / the three `project_*` methods, the objective in `compute_loss`, and
the retrieval rule in `score`.
