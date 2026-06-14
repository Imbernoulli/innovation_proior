Let me start from what I actually have and what I actually want, because the usual machinery doesn't fit. I have a pile of high-dimensional points — images — and a relation handed to me from outside: for some pairs I'm told "these two are similar," meaning two views of the same object, or two frames close in time, or two images a human grouped together. I am *not* given a clean numeric distance in input space; pixel distance is exactly the thing I distrust, since two shifted or relit photos of one object sit far apart in pixels while two unrelated photos can sit close. And I want to come out the other side holding a *function* G_W from R^D to a low-dimensional R^d, d much smaller than D, such that ordinary Euclidean distance in the output reflects that handed-down similarity — and crucially the function has to apply to brand-new inputs I never trained on, and it has to be free to learn whatever nonlinear invariance (to lighting, shift, distortion) the data demands.

So before anything else: what's already on the table, and exactly where does each piece run out? The classical reducers, PCA and MDS, give me a linear map and lean on an input-space distance — PCA maximizes variance, MDS preserves pairwise input distances — both of which presuppose the very metric I've decided I can't trust, and both linear when the manifold I care about is curved. The nonlinear spectral crowd — LLE, Isomap, Laplacian Eigenmaps, Hessian LLE, Kernel PCA — does recover curved manifolds: find each point's neighbors, build a Gram-type matrix, solve an eigenproblem. Gorgeous on the training set. But two things kill them for my purpose. First, the output is *coordinates for the points I already have*, not a function; drop in a new image whose neighbors I don't know and there's nowhere to put it without redoing the eigenproblem. Second, the neighborhood structure is either tied to a useful input-space distance/kernel or handed in as a fixed list; none of these methods learns a reusable invariant map from that relation. I can't hand them "be invariant to this distortion" and have them learn it. And there's a nagging side-observation about several of them: they pile points into a few dense clusters in output space, sometimes degenerately so, rather than spreading them evenly. I don't just want *an* embedding; I want one that covers the manifold.

OK so a function that generalizes off the training set, learns its own invariances, judges by output distance. That immediately says: a parametric network for G_W, trained by gradient descent, and — since my only supervision is *relational*, "these two go together" — the loss has to be a function of a *pair*, run through the same network. Bromley and colleagues already built the architecture I need for that: the siamese net, two identical sub-networks sharing one set of weights, each eating one member of a pair, their outputs joined by a module that measures distance. They used it for signatures — feed two signatures, extract a feature vector from each, compare the distances, accept if below threshold. That's exactly my twin-tower: one shared G_W applied to both members, weights tied, and because the weights are tied the same feature extractor maps everything, the metric is automatically symmetric, and I can run it on a never-seen input later. The total gradient is just the sum of the two instances' contributions since they share W. Good — I'll take that scaffold.

Now the only real question: what is the pair loss? Let me write the obvious thing and see if it's enough, because I should never assume difficulty I haven't hit. Define the output-space distance for a pair as the Euclidean distance between the two embeddings,

  D_W(X1, X2) = ‖G_W(X1) − G_W(X2)‖₂.

This is the natural cost-module output and it's a *simple* distance, which is the whole point — I want a complicated learned map but a trivial metric on top of it, otherwise I've just moved the complexity around. My supervision says "similar pairs should be close." Close means small D_W. So minimize D_W on similar pairs. I want the gradient to behave well at the bottom, so instead of D_W itself I'll square it — call the similar-pair loss

  L_S = ½ D_W².

Quick check that the squaring is the right call and not laziness. The gradient is ∂L_S/∂W = D_W · ∂D_W/∂W, which is *linear in the distance D_W*: the pull grows in proportion to how far apart the pair is and dies smoothly to zero as D_W → 0. If I'd used the bare |D_W| instead, the gradient would be (∂D_W/∂W) with a fixed magnitude even an infinitesimal distance from the rest point — a constant-strength force that never lets go, a kink at the minimum. The quadratic gives me a clean, vanishing-at-the-bottom restoring force. So ½D_W² it is, for the similar pairs.

Let me actually picture what this is doing physically, because I think the mechanics will tell me whether I'm done. The outputs G_W(X) are points — masses — in R^d. The similar-pair loss ½D_W² is, term for term, the potential energy of a spring: ½ K X² with the displacement X being the distance D_W and ∂D_W/∂W playing the role of K in the gradient comparison. And its rest length is *zero* — the energy is minimized at D_W = 0 — so it's an attract-only spring. Any positive separation produces an attractive force pulling the two outputs together; the more I separate them, the harder it pulls; right on top of each other, no force. So my objective, as written, strings an attract-only spring between every pair I've labeled "similar," and asks SGD to relax the whole spring network to its lowest energy.

Now stare at that. A web of springs that *only attract*, rest length zero, and I'm minimizing total energy. Where does that relax to? Every spring wants its two endpoints coincident. There is a configuration that makes *every single one* of them happy simultaneously, with exactly zero energy: put all the output points at the same location. If G_W maps every input to one constant vector c, then for every similar pair D_W = ‖c − c‖ = 0, every L_S = 0, the total loss is 0 — a global minimum. The network doesn't have to learn anything about the images; it just has to learn to ignore them and emit a constant. That's not a near-miss or a bad local optimum; the constant map is a global minimizer of the objective I wrote, and it's degenerate — a single point, no manifold, no representation. Collapse.

And it's worse than "there's a bad global optimum I might avoid with luck." Suppose a point b1 is connected by attract-only springs to b2 and to b3, both labeled similar to b1. Moving b1 toward b2 can increase the b1–b3 distance, so a single local motion can trade one spring against another. But the global objective is just a sum of nonnegative spring energies, and there is a configuration that makes every similar-pair term in a connected component exactly zero at once: make all of their outputs identical. Mapping every input to one constant vector does that for all components simultaneously. The attract-only network has no internal opposition; nothing pushes back against reducing every observed similar distance. So minimizing this loss doesn't just *permit* collapse, it makes the collapsed constant map a global minimizer. Wall.

I want to be precise about *why* this happened, because the precision is the clue to the fix. It is not that I chose the wrong distance or the wrong power on D_W. It's that my loss only ever asks for energy to be *lowered* (on similar pairs) and never asks for energy to be *raised* anywhere. Think of D_W as an energy I'm assigning to a pair-configuration — low energy = "the model thinks these are similar." I'm shaping that energy from one side only: I push it down where I want it low. But I never push it up anywhere, so the trivial way to make the energy low *everywhere I'm looking* is to make G_W flat, which makes the energy low *everywhere, period*. There's no reference level, no counter-pressure.

Now compare to a model that doesn't have this disease, to see what it has that I don't. A properly normalized probabilistic model — one where the scores over all configurations sum to one — cannot collapse this way, and the reason is the normalization itself: if I push the probability of one pair-configuration *up*, the constraint that everything sums to one *automatically* pushes the probability of other configurations *down*. The competition is built into the normalizer. My distance-based, unnormalized objective is an energy with no partition function — which I *wanted*, because computing a normalizer over the space of images is intractable and dropping it buys me huge architectural freedom — but that same missing normalizer is exactly what removed the automatic counter-pressure. So I get the freedom of an unnormalized energy at the cost of having to *manually* supply the thing the normalizer was doing for free: I must explicitly push some energies *up*.

Up where? I only have one other kind of supervision: the pairs I'm told are *dissimilar* — non-neighbors, different objects. Those are exactly the configurations whose energy *should* be high. So the fix is forced, not invented: alongside the attract-only term that lowers D_W on similar pairs, add a term that *raises* D_W on dissimilar pairs — a repulsion. In the spring picture, dissimilar pairs get springs that *push apart*. That's what gives the network something to fight against the collapse: if everything piles onto one point, every dissimilar pair is at distance zero and the repulsion term is screaming. The constant map, which was the global *minimum* of the attract-only loss, becomes a *high*-loss configuration the moment a repulsion term is present, because at the collapsed point every dissimilar pair maximally violates it.

Let me design the repulsion carefully, because the obvious version is also a trap. Naively, "push dissimilar pairs apart" says: maximize D_W on them, i.e. add a term like −D_W, or reward large distance without limit. But an unbounded reward for separation is a disaster — the loss would be dominated by, and unbounded below from, shoving already-far dissimilar pairs infinitely farther apart; the optimizer would happily blow the embedding up to infinity and ignore the attractive structure entirely. I don't actually *want* dissimilar pairs infinitely far; I just want them *not too close*. There should be some separation `m` beyond which I'm satisfied and stop pushing. That is precisely a margin, and the standard primitive for "active up to a threshold, flat beyond it" is the hinge: max(0, m − D_W). It's nonzero — pushing — only while D_W < m, and exactly zero once D_W ≥ m. So a dissimilar pair already farther apart than the margin contributes nothing; one that's too close gets shoved out to the margin and no further.

And I'll square the hinge, for the same reason I squared the similar term: the dissimilar loss

  L_D = ½ {max(0, m − D_W)}²

has gradient ∂L_D/∂W = −(m − D_W) · ∂D_W/∂W for D_W < m, and 0 for D_W ≥ m. Read that as a force. It's repulsive (the minus sign), its magnitude is linear in the displacement (m − D_W) — Hooke again, but now a spring with rest length `m`: it pushes the two outputs apart whenever they're closer than m, hardest when they're coincident (D_W = 0, force ∝ m), and the force smoothly vanishes exactly at D_W = m. Past m, nothing — the spring is "repulse-only," and once stretched beyond its rest length there's no force at all, attractive or otherwise. A bare (unsquared) hinge would give a constant-magnitude repulsive force with a kink at the margin; the square gives me the linear, vanishing-at-the-margin force that matches the attractive side and is smooth to optimize.

So now every point sits in a mechanical equilibrium of two kinds of springs: attract-only springs (rest length 0) to its neighbors, m-repulse-only springs (rest length m) to its non-neighbors. The neighbors pull it in; the non-neighbors push it out to at least m; the system relaxes to a configuration that satisfies both as well as possible. Collapse is now energetically forbidden — squashing everything to a point maximizes every repulsion term. The two forces in balance are also exactly what spreads the points to *cover* the manifold instead of clumping: attraction keeps the local neighborhood structure tight, repulsion keeps distinct things at least m apart, so the embedding fills out rather than collapsing into dense clusters. The margin `m` sets the overall scale of the output embedding — the radius of the repulsion — which is the one free knob I've introduced and it has a clean meaning.

Let me assemble the full pair loss. With a binary label Y = 0 for a similar (neighbor) pair and Y = 1 for a dissimilar pair, the two branches have to be selected by the label:

  L(W, Y, X1, X2) = (1 − Y) · ½ (D_W)²  +  Y · ½ {max(0, m − D_W)}².

One loss, defined over a pair, that lowers the energy of neighbors and raises (up to a margin) the energy of non-neighbors. The total objective sums this over all labeled pairs — and the count of pairs can be as large as the square of the number of samples, which is fine because I train it stochastically, one pair (or a minibatch of pairs) at a time, backpropagating through the two shared instances of G_W and summing their gradient contributions. Let me sanity-check the two regimes against the spring story one more time. Similar pair, Y=0: loss ½D_W², gradient D_W·∂D_W/∂W — attractive, grows with distance, zero at coincidence. ✓ Dissimilar pair, Y=1, D_W < m: loss ½(m−D_W)², gradient −(m−D_W)·∂D_W/∂W — repulsive, strongest when coincident, zero at the margin. ✓ Dissimilar pair already beyond the margin, D_W ≥ m: max(0, m−D_W) = 0, loss 0, gradient 0 — leave it alone. ✓ Constant collapse G_W ≡ c: every similar term 0, but every dissimilar term ½m² — the collapsed state is now high loss rather than a minimizer. ✓ The contrastive term is what makes the constant map a non-solution.

I should pin down why the architecture really has to be the shared-weight siamese and not, say, two free networks. The map G_W is supposed to be one reusable function I can run on any future input and compare against any stored embedding; if the two towers had different weights I'd have two different maps and the notion of "distance between embeddings" wouldn't be well defined for a new point I run through only one of them. Tying the weights makes G_W a single function, makes D_W symmetric in its two arguments by construction, and means there's just one set of parameters to learn from the summed gradient of both instances. The only requirement I've placed on G_W is differentiability in W — the inside can be anything: a small fully connected net for low-resolution object images, a convolutional net for pixel images, since a convnet's shared-weight, multi-layer structure is itself how it learns shift-invariant local features while staying robust to geometric distortion, which is exactly the kind of invariance I wanted the embedding to absorb.

Let me write it as the algorithm I'd actually run. Build the pairs: for each input, use the supplied neighbor relation to mark its similar set (Y=0) and pair it with the rest as dissimilar (Y=1). Then loop to convergence: for each pair, push both members through the shared G_W, compute D_W; if Y=0 take a gradient step that *decreases* D_W (the ½D_W² branch), if Y=1 take a step that *increases* D_W toward the margin (the ½(m−D_W)² branch, doing nothing once past m); backprop through both shared instances; SGD update. The whole thing is one network, one distance, one gated pair loss.

Now let me land this in the code shape I'd hand off — the siamese harness with the pair-loss slot filled by exactly the gated objective I just derived.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GW(nn.Module):
    """Shared embedding G_W: R^D -> R^d, d << D. The SAME weights are applied to
    both members of every pair (siamese). Differentiable in W; the body is a
    generic feature extractor (small MLP here; a conv net for pixel images)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveLoss(nn.Module):
    """Gated pair loss: attract neighbors, repel non-neighbors up to a margin.
        L = (1 - Y) * 1/2 * D_W^2  +  Y * 1/2 * max(0, m - D_W)^2 ,
    with D_W = ||G_W(x1) - G_W(x2)||_2 and Y = 0 (similar) / 1 (dissimilar).
    The Y-branch is the contrastive term that makes the constant map a
    high-energy state, so the attract-only objective can no longer collapse."""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, Y):
        D = torch.norm(z1 - z2, dim=-1)                  # D_W = ||G_W(x1) - G_W(x2)||_2
        sim = 0.5 * D.pow(2)                             # similar: 1/2 D_W^2  (attract-only spring)
        dis = 0.5 * F.relu(self.margin - D).pow(2)       # dissimilar: 1/2 max(0, m - D_W)^2  (m-repulse-only)
        loss = (1.0 - Y) * sim + Y * dis                 # gate by the relation label
        return loss.mean()


def train(gw, criterion, pairs, opt):
    """Siamese training loop. Each batch is (x1, x2, Y); weights are shared, so
    the gradient into W is the sum of contributions from both instances of G_W."""
    for x1, x2, Y in pairs:
        z1 = gw(x1)                                      # shared G_W ...
        z2 = gw(x2)                                      # ... applied to both members
        loss = criterion(z1, z2, Y.float())
        opt.zero_grad()
        loss.backward()                                  # backprop through both shared instances
        opt.step()
```

If I strip the construction back to the attract-only baseline, two distorted views of the same input are just a similar pair with Y=0, and their projected embeddings z1, z2 play the role of G_W(X1), G_W(X2). Keeping only that branch means minimizing their squared distance with no dissimilar-pair term at all. I am back to the zero-rest-length spring network with no repulsion, and the constant map is a global minimizer. In PyTorch, that bare invariance term is the ordinary mean squared error between the two embeddings; the averaging and the missing ½ only change scale, not the collapse argument:

```python
import torch.nn.functional as F


def naive_invariance_loss(z1, z2):
    # Y = 0 branch alone: attract-only, no repulsion.
    # Global optimum is G_W = const (collapse) -- this is the degenerate baseline.
    return F.mse_loss(z1, z2)
```

The causal chain, start to finish: I wanted a generalizing function that maps high-dimensional inputs to a low-dimensional space where simple Euclidean distance reflects a handed-down similarity relation, learning its own invariances — which the classical reducers can't do (linear, input-distance-dependent or fixed-neighbor-list, training-set-only). The siamese shared-weight net gives me the reusable, symmetric function and a pair-based loss. The natural first move — square the output distance and minimize it on similar pairs — turned out to be an attract-only spring network with the constant collapse as a global minimizer, and no internal force opposing it. The reason was structural: an unnormalized energy lowered from one side has no counter-pressure, unlike a normalized probability where raising one score lowers others; the missing normalizer that bought me architectural freedom is exactly what I had to replace by hand. So I added an explicit contrastive term that *raises* the energy of dissimilar pairs — a margin hinge ½max(0, m−D_W)² acting as an m-rest-length repulse-only spring — which forbids collapse (it maximally fires at the collapsed point), spreads the embedding to cover the manifold, and stops wasting effort once pairs are farther apart than the margin m. Gating the two terms by the binary relation label gives the single contrastive loss (1−Y)·½D_W² + Y·½max(0,m−D_W)², trained by SGD through the shared towers — and stripping the contrastive term back off recovers the naive invariance-only loss with the same collapsed constant map as a minimizer.
