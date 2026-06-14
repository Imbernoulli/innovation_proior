# Context: learning a low-dimensional embedding from neighborhood relations

## Research question

We have a large collection of high-dimensional points — images especially — and we want a map
`G_W: R^D -> R^d` with `d` far smaller than `D`, such that a *simple* distance in the
low-dimensional output space (plain Euclidean, say) reflects which inputs we consider "similar."
The notion of similarity we are handed is not a clean numeric distance in pixel space; it is a
*neighborhood relation* supplied from outside: prior knowledge, manual labeling, temporal
adjacency in a video, or the fact that two images are two distorted versions of the same object.
Concretely we want three things at once. First, the output map must be a genuine **function** we
can apply to brand-new inputs we never saw while training — not just an assignment of coordinates
to the training set. Second, the map must be free to learn **invariances to complicated
nonlinear transformations** of the input (lighting, shift, rotation, geometric distortion), so
that two heavily transformed copies of the same object land near each other even though their
raw pixels are far apart. Third, it must remain faithful even for samples whose neighborhood
relationships are *unknown* at test time — we only ever supplied the relation for training pairs.

The reason this is hard is that the only supervision is relational ("these two are neighbors;
these two are not") — there are no class targets to regress onto, and there may be a huge,
open-ended number of categories with very few examples each, some categories not even present at
training time. So the objective cannot be a per-sample loss against a label; it has to be a loss
defined on *pairs* of inputs, driven entirely by the supplied neighbor / non-neighbor relation.
A solution would have to turn that relational signal into a single reusable embedding function
that generalizes off the training set, learns its own invariances, and spreads the data out over
the output manifold rather than piling it up.

## Background

The classical tools for mapping high-dimensional data to a low-dimensional space are linear:
**Principal Component Analysis** (project onto the subspace of maximum variance) and
**Multi-Dimensional Scaling** (find a low-dimensional embedding that best preserves pairwise
input distances). Both produce a *linear* embedding, and both presuppose a meaningful,
computable distance in the input space — for images, Euclidean distance in pixel space, which is
notoriously bad: two shifted or relit images of the same object can be far apart in pixels while
two unrelated images are close.

A wave of nonlinear spectral methods then attacked the problem: **Isomap** (Tenenbaum et al.
2000), **Locally Linear Embedding (LLE)** (Roweis & Saul 2000), **Laplacian Eigenmaps** (Belkin
& Niyogi 2003), **Hessian LLE** (Donoho & Grimes 2003). All share a three-step recipe: identify
each point's neighbors, build a Gram-like matrix from that neighborhood structure, and solve an
eigenvalue problem. **Kernel PCA** (Schölkopf et al. 1998) nonlinearly maps inputs to a feature
space via a kernel and extracts principal components there. These broke the linearity barrier and
recover curved manifolds beautifully on the training set.

But two limitations are load-bearing here. (1) **They embed the training set, not a function.**
The output is a set of coordinates for the points you already have; a genuinely *new* point
whose relationship to the training data is unknown cannot be placed without recomputing (or
approximating an out-of-sample extension that itself assumes a computable kernel/distance in
input space). (2) **The neighborhood information is not a learned invariant map.** LLE linearly
combines a point with its neighbors, which only makes sense for nearly registered, very similar
images; Isomap, Kernel PCA, and out-of-sample extensions depend on a useful distance or kernel;
Laplacian Eigenmaps and Hessian LLE can start from a supplied neighbor list, but they still do not
produce a reusable function for a new sample whose relation to the training set is unknown. None
of them can be told "be invariant to this nonlinear distortion" and learn that invariance from
data. A further observed pathology: several of these methods **tend
to cluster points in output space, sometimes densely enough to count as degenerate** — points
pile into a few tight clumps rather than spreading evenly over the manifold, which is often the
opposite of what one wants from a representation.

There is also a framing from a different corner of the field that turns out to matter: the
**energy-based model (EBM)** view. A probabilistic model assigns a *normalized* probability to
every configuration; an energy-based model instead assigns an *unnormalized* scalar **energy** to
each configuration and makes decisions by comparing energies (pick the low-energy one). The
practical appeal is that there is **no partition function to compute** — no intractable
normalizer — which buys a lot of freedom in the choice of architecture. The structural catch is
this: in a properly normalized probabilistic model, pushing the probability of one configuration
*up* automatically pushes others *down*, because the total must sum to one. An unnormalized energy
has no such bookkeeping — driving the energy of some configurations down does nothing, on its own,
to the energy of the rest. This asymmetry between normalized and unnormalized models is a known
fact of the EBM framework.

Finally, the **squared L2 / Euclidean** distance and the **hinge/margin** loss are both standard,
well-understood primitives by this time. A hinge `max(0, m - s)` is the workhorse of large-margin
methods (SVMs): it is active only up to a threshold `m` and exactly flat beyond it, which is how
margins are enforced without paying an unbounded cost once the constraint is satisfied.

## Baselines

**Siamese networks (Bromley, Guyon, LeCun, Säckinger & Shah, 1993).** The original twin-tower
architecture: two identical sub-networks that *share the same weights*, each consuming one member
of a pair, with their outputs joined by a module that measures the distance between the two
feature vectors. Introduced for signature verification — two signatures go in, the network
extracts an 80-byte feature vector from each, a joining neuron measures their distance, and a
signature is accepted as genuine if that distance falls below a threshold. Core idea: let one
shared feature extractor learn higher-order features from low-level input, and judge by distance
in feature space rather than by classification, so the system handles identities never seen at
training time. **Gap:** it supplies the shared-weight architecture and the "compare by distance"
idea, but its training is set up around the verification threshold; it does not come with a loss
that has a principled account of what keeps the learned features from degenerating, nor a
treatment of how non-matching pairs should be handled in the objective.

**Discriminative similarity-metric learning (Chopra, Hadsell & LeCun, 2005).** Casts metric
learning as an energy-based model on the same shared-weight siamese architecture. Define the
energy of a pair as the output distance `E_W(X1, X2) = ‖G_W(X1) - G_W(X2)‖`, and train so the
energy is low for "genuine" pairs (same person) and high for "impostor" pairs (different people),
asking for a margin between them. The decisive observation it records is what happens if one is
naive about it: minimizing the energy averaged over same-category pairs alone leads to a
catastrophic collapse, because the energy and the loss can both be driven to zero by making
`G_W` a constant function. It ties this directly to the EBM asymmetry above: a normalized
probabilistic model
would not have this failure, because raising one pair's probability lowers others'; an
unnormalized energy has no such automatic counter-pressure. **Gap:** the pairing supervision is
class-based (genuine/impostor identity labels), which limits it to settings where such categorical
labels exist; it is framed as verification rather than as general-purpose dimensionality
reduction / manifold learning; and it does not address the goal of *evenly covering* a
low-dimensional output manifold or of learning from purely relational, label-free neighbor
information (temporal proximity, known invariances).

**Spectral manifold-embedding methods (LLE / Isomap / Laplacian Eigenmaps / Hessian LLE; Kernel
PCA).** Covered under Background. As baselines for "produce a low-dimensional embedding from
neighborhood structure," their **gaps** are concrete: they embed the *training set* rather than
producing a reusable function, so they cannot place a new point without recomputation; they still
need either a useful input-space distance/kernel or a supplied neighbor list rather than learning
the invariant relation as a parametric map; they cannot be instructed to learn invariance to a
specified nonlinear transformation; and they are prone to piling points into dense clusters in
output space.

## Evaluation settings

The natural yardsticks for a learned invariant embedding, all available before the method:

- **MNIST** handwritten digits (LeCun et al.), 28x28 grayscale — used to learn a low-dimensional
  (e.g. 2D) embedding and inspect how the output organizes by digit identity and by deliberately
  injected nuisance transformations (e.g. horizontal shifts), checking whether the map is
  invariant to the nuisance and faithful for held-out samples whose neighbor relations were not
  given.
- **NORB** (LeCun et al.) — small objects (e.g. airplanes) imaged under controlled, systematically
  varied **lighting and azimuth/elevation**, the canonical testbed for learning invariance to
  pose and illumination, with a small fully connected network as the embedding function.
- **Neighborhood construction protocol:** form the relational supervision from prior knowledge —
  e.g. group images known to be transformations of the same object, or use temporal proximity in
  a sequence, or a simple Euclidean nearest-neighbor rule *only* to seed local neighbor lists —
  then evaluate the embedding by how well output-space Euclidean distance recovers the intended
  similarity structure and how evenly the data covers the output manifold.
- **Comparison protocol:** hold the embedding function and the neighbor relations fixed and
  contrast against a classical embedding (notably LLE) on the same data, looking at out-of-sample
  placement and at whether invariance to the targeted transformation is achieved.

## Code framework

The substrate that already exists: a shared-weight twin (siamese) harness that runs the *same*
parametric network `G_W` over both members of a pair, computes the output-space Euclidean
distance between the two embeddings, and feeds that distance — together with the supplied
relation label `Y` (0 = similar/neighbor, 1 = dissimilar/non-neighbor) — into a pair loss whose
gradient is backpropagated through *both* shared instances (the total gradient is the sum of the
two contributions, since the weights are tied). The optimizer is plain stochastic gradient
descent. Everything about *what the pair loss should be* is exactly the open question; the loss
module is the single empty slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GW(nn.Module):
    """The shared embedding function G_W: R^D -> R^d, d << D.
    Differentiable in W; the SAME instance (same weights) is applied to both
    members of every pair. Architecture (a conv net / small MLP) is generic
    and not the question here."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # generic feature extractor -> low-dimensional output
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PairLoss(nn.Module):
    """Loss defined on a PAIR of embeddings and the relation label Y.
    Y = 0 if the pair is similar (neighbors), Y = 1 if dissimilar.
    Returns a scalar to be minimized by SGD."""

    def __init__(self):
        super().__init__()

    def forward(self, z1, z2, Y):
        # output-space Euclidean distance between the two embeddings
        D = torch.norm(z1 - z2, dim=-1)          # D_W = ||G_W(x1) - G_W(x2)||
        # TODO: the pair objective we will design here, as a function of (D, Y).
        raise NotImplementedError


def train(gw, pair_loss, pairs, opt):
    """Existing siamese training loop. Each batch is (x1, x2, Y)."""
    for x1, x2, Y in pairs:
        z1 = gw(x1)                # shared weights ...
        z2 = gw(x2)                # ... applied to both members
        loss = pair_loss(z1, z2, Y)
        opt.zero_grad()
        loss.backward()            # gradient flows into W through both instances
        opt.step()
```

The twin towers, the shared weights, the Euclidean-distance cost module, the relation label `Y`,
and the SGD loop all predate the method; the pair objective `PairLoss.forward` is the slot to be
filled.
