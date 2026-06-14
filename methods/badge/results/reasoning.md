Let me start from what actually hurts. I have a big pool of unlabeled examples and a deep net, and labels cost money, so each round I get to pick a batch of B examples, pay for their labels, retrain, and repeat — and I want to hit a target accuracy with as few labels as possible. The reason I'm forced into batches at all is worth pinning down, because it shapes everything. Classic active learning queries one point, refits, queries the next. For a network with millions of parameters that's hopeless on both ends: one labeled example barely moves the model, and retraining a deep net to convergence after every single query is computationally absurd. So I have to acquire B at a time. And the moment I do that, a failure mode appears that the one-at-a-time world never had: if I score each example on its own and take the top B, nothing stops the B winners from being near-duplicates of each other. They'd all sit in the same confusing corner of input space, all "informative" by the same measure, and a single label would have told the model what all B were going to tell it. I'd have paid for B labels and learned roughly one label's worth.

There's a second constraint I keep underestimating, and it's specific to this setting. In ordinary supervised learning I tune hyperparameters on validation data for free. Here, if I change a hyperparameter, the algorithm queries a *different* set of points, and I have to buy those labels. A hyperparameter sweep is a label-budget sweep. So whatever I build has to "just work" at fixed settings — every knob I add that needs per-dataset tuning is a knob I can't actually afford to turn. That pushes me hard toward a method with *no* free tradeoff parameter.

So what's on the table. Two camps, and the uncomfortable fact is that each only works in part of the world. Uncertainty sampling: score each point by how unsure the model is — least confidence 1 − max_i p_i, or the margin p_(1) − p_(2) between the top two class probabilities, or the entropy H(p) = −Σ_i p_i log p_i — and grab the most uncertain B. The intuition is sound: confidently-classified points teach nothing, the action is near the decision boundary. And empirically this is strong at small batch and with simple models. But it's exactly the method that piles up duplicates: the score is a function of *one* example, there's no term that even looks at the other chosen points, so at large batch it floods the batch with one cluster of uncertain look-alikes. Plus deep-net softmax outputs are notoriously overconfident, so the raw uncertainty numbers are shaky. The other camp is representative or diversity sampling — cover the unlabeled distribution so that training on the batch approximates training on the whole pool, regardless of labels. The cleanest version of this is Core-Set: bound the risk of the subset-trained model by training error plus generalization error plus a "core-set loss," show that minimizing the core-set loss reduces to the k-Center objective — pick B centers minimizing the largest distance from any pool point to its nearest center — and solve it greedily by furthest-first traversal, all in the network's penultimate-layer feature space z(x). This is great at large batch and when the architecture has strong inductive biases so z is meaningful. But it's pure geometry: it has no idea whether a point is informative, so it'll happily spend budget covering regions the model already gets right, and when z isn't meaningful — hard data, weak net — its "representative" batch can be *worse* than random. And which regime am I in? That depends on the data statistics I don't know in advance. So I can't even reliably pick which camp to join.

People have tried to get both. There's a meta-approach, Active Learning by Learning (Hsu & Lin), that runs a bandit to choose at each round whether to fire the diversity rule or the uncertainty rule — but it can only pick among the rules it's handed, it never builds one criterion that has both properties, and it inherits whatever weakness the chosen base rule has right now. What I actually want is a single quantity per example that simultaneously encodes "the model is uncertain about this" and lets me build a batch that's "spread out," with no coefficient I have to tune to balance them.

Let me think about what "the model is uncertain / this point is informative" even means for a net trained by gradient descent. The whole machine moves by gradients: an example that's informative is one that, once I know its label, induces a large gradient of the loss and therefore a large update to the parameters. So the natural currency of informativeness is gradient magnitude. If I knew the true label y of x, I'd compute the gradient g_x^y = ∂/∂θ ℓ(f(x;θ), y) and call ‖g_x^y‖ the example's pull on the model. That's basically the expected-gradient-length idea (Settles): score by gradient size. EGL even handles the unknown label by averaging over it under the model, EGL(x) = Σ_y p(y|x)‖∇_θ ℓ(x,y)‖. But two things bug me about EGL for my purpose. First, it's still a per-example scalar score — it has the exact duplicate-batch disease, no diversity term at all. Second, it averages a full-parameter gradient over all K labels, which is a lot of computation, and the full-θ gradient is a giant object. Let me not take the whole gradient. The cheapest informative slice is the gradient with respect to just the *last* layer's weights — and as I'll see, that slice is special.

Set up the network as f(x;θ) = softmax(W z(x;V)), where z(x;V) ∈ R^d is the penultimate-layer embedding and W ∈ R^{K×d} is the final linear layer, row W_i for class i. Cross-entropy loss against a label y is ℓ_CE = −log p_y = log(Σ_j e^{W_j·z}) − W_y·z. I want ∂ℓ_CE/∂W_i. The first term: ∂/∂W_i log Σ_j e^{W_j·z} = (e^{W_i·z}/Σ_j e^{W_j·z}) z = p_i z. The second term −W_y·z contributes −z to the i = y block and nothing elsewhere, i.e. −I(y=i) z. So the i-th block of the last-layer gradient is

  (g_x^y)_i = (p_i − I(y = i)) z(x;V).

Stare at this. The whole gradient embedding is an outer product: g_x^y = (p − e_y) ⊗ z, the probability-residual vector r = p − e_y tensored with the penultimate embedding z. That decomposition is the thing. The z factor is *exactly* the representation space Core-Set lives in — it carries where the example sits, its diversity/identity. The r factor carries how wrong/uncertain the model is — if the model is confident and right, p ≈ e_y, so r ≈ 0; if it's unsure, p is spread out and r is large. One object, and it already contains both of the things the two camps were fighting over: representation in z, uncertainty in r. I didn't have to balance them — they're multiplied together coordinate-blockwise inside a single vector.

But there's the obvious snag: this needs the true label y, which is the whole point — I don't have it. EGL's answer is to average over y. Let me try something cheaper and see if it's defensible: just hallucinate the model's own current prediction, ŷ = argmax_i p_i, plug it in, and use g_x = g_x^{ŷ}. One gradient per example, no averaging. Is that a principled thing to do or am I just being lazy? Let me compute the norm and find out, because the norm is what I'm going to use as the uncertainty signal.

  ‖g_x^y‖^2 = Σ_i (p_i − I(y=i))^2 ‖z‖^2.

Expand the inner sum: Σ_i (p_i − I(y=i))^2 = Σ_i p_i^2 − 2 Σ_i p_i I(y=i) + Σ_i I(y=i)^2 = Σ_i p_i^2 − 2 p_y + 1. So

  ‖g_x^y‖^2 = (Σ_i p_i^2 + 1 − 2 p_y) ‖z‖^2.

Now look at how this depends on the choice of label y: the only y-dependent piece is −2 p_y. To *minimize* ‖g_x^y‖ over y I want to *maximize* p_y — which means argmin_y ‖g_x^y‖ = argmax_y p_y = ŷ. So the label I hallucinated is exactly the one that gives the *smallest* possible gradient norm. That's not laziness, that's a guarantee: for the unknown true label y, ‖g_x‖ = ‖g_x^{ŷ}‖ ≤ ‖g_x^y‖. The hallucinated-label gradient norm is a *lower bound* on the gradient norm the example will really induce once I see its label. So when I pick a point because its g_x is large, I'm guaranteed it will produce at least that much update — I can't be fooled into thinking a point is informative when it isn't; the bound only ever understates. And conversely, look at the magnitude as a function of confidence: if the model is sharp, p ≈ e_ŷ, then Σ p_i^2 ≈ 1 and 2 p_ŷ ≈ 2, so Σ p_i^2 + 1 − 2 p_ŷ ≈ 0 — the gradient embedding nearly vanishes for confident points. If p is flat, that quantity is large. So ‖g_x‖ is small for confident examples and large for uncertain ones: the magnitude of this one cheap hallucinated gradient *is* a conservative uncertainty score. The lazy choice turns out to be the conservative-estimate choice. Good — I'll commit to last-layer gradient embedding at the hallucinated label, g_x = (p − e_ŷ) ⊗ z.

Now the batch. I have, for every pool point, a vector g_x whose *length* means uncertainty and whose *direction* (mostly inherited from z and from which class block r lights up) means identity/representation. I want a batch of B points that are individually long (uncertain) and collectively spread out (diverse), and — the binding constraint — with no tradeoff knob. What's the natural mathematical object for "a set that is high-quality and diverse at once"? A determinantal point process. Sample a size-B set Y with probability ∝ det(L_Y), where L_Y is the Gram matrix of the chosen {g_x}. Why is that the right object? Because the determinant of a Gram matrix of vectors equals the squared product of their lengths times the squared volume of the parallelepiped they span. The length part rewards big gradients — uncertainty. The volume part is large only when the vectors point in genuinely different directions — diversity — and collapses to zero the instant two chosen vectors are parallel. So det(L_Y) is *simultaneously* a quality score and a diversity score, fused, with no coefficient weighting one against the other. And it even self-adjusts to batch size for free: when B is small, the volume constraint is easy to satisfy, so the length factor dominates and the sampler behaves like uncertainty sampling; when B is large, spanning a B-dimensional volume with non-parallel vectors gets hard, so the diversity factor takes over. That's exactly the regime-adaptive behavior I was failing to get by hand — small-batch → uncertainty, large-batch → diversity — and here it's automatic. This is the criterion I want.

So why don't I just sample from a k-DPP over the gradient embeddings and call it done? Cost. Sampling from a k-DPP is genuinely expensive — exact samplers are high-order polynomial in the batch size and the embedding dimension, MCMC samplers have a mixing-time problem, and at the large batch sizes I care about it simply blows up memory. Wall. The criterion is right but I can't afford to draw from it. So I need something that *behaves* like the k-DPP — favors long, mutually-spread-out vectors — but is cheap and, ideally, has no knobs.

Here's where an old, unrelated tool snaps into place. k-means++ seeding. Its job in life is to initialize Lloyd's algorithm: pick the first center uniformly, then repeatedly sample the next center from the data with probability proportional to its *squared distance to the nearest already-chosen center* — D^2 weighting — with the guarantee that the resulting potential is within 8(ln k + 2) of optimal. I never thought of it as a sampler for active learning. But look at what D^2 weighting does in *my* gradient-embedding space. After I've chosen some points, the next point is drawn with probability ∝ (distance to the nearest chosen point)^2. A point that's far from everything already in the batch is much more likely to be picked — that's diversity, and the squaring sharpens it. And the magnitude is in there too, once I'm careful about where distances are measured from: a long gradient vector tends to be far from the others, and if I seed the very first point by its length rather than uniformly, I plant the batch on the single most uncertain example and let D^2 spread out from there. So k-means++ seeding in gradient space pulls toward exactly the same kind of set a k-DPP would — high-magnitude and diverse — but it's a handful of cheap passes over the data with no matrix determinant, no inverse, no MCMC, and no tradeoff hyperparameter. It's the affordable surrogate for the determinant.

Let me make the seeding concrete so I trust it. The first center: instead of uniform, I take the point with the largest ‖g_x‖ — the most uncertain example — so the batch starts from maximal uncertainty. Then for each subsequent pick I maintain, for every pool point, D(x) = the distance from g_x to its nearest already-chosen gradient embedding, and I sample the next point with probability D(x)^2 / Σ D^2. Repeat until I have B. Each new center is, in expectation, far from the current batch (diversity) and, because gradient length contributes to those distances, biased toward long vectors (uncertainty). Already-chosen points get distance zero so they can't be re-picked. That's the whole acquisition rule. No coefficient anywhere — the only "tradeoff" is the geometry of the D^2 rule interacting with batch size, which is exactly the free, automatic tradeoff I wanted.

I should sanity-check that this k-means++ surrogate really tracks the k-DPP and isn't just a vibe. The thing both samplers are supposed to produce is batches that are *diverse* (large Gram determinant of the selected gradient embeddings) and *high-magnitude* (large average ‖g_x‖). Measuring those two quantities on the batches each sampler returns, the k-means++ batches sit right on top of the k-DPP batches on both axes — and, if anything, k-means++ tends to find batches that are a touch more diverse and higher-magnitude than the k-DPP, since the k-DPP's extra stochasticity occasionally lets a mediocre set through. Meanwhile furthest-first traversal — the k-Center sampler that Core-Set uses — produces batches that are markedly *less* diverse and lower-magnitude in this gradient space, which is the tell that pure k-Center is leaving informativeness on the table. So the cheap seeding gives up essentially nothing in batch quality relative to the expensive determinant, while costing far less. That settles it: k-means++ seeding over hallucinated last-layer gradient embeddings.

Let me try to understand *why* this beats plain uncertainty sampling in a case I can actually compute, because the lower-bound argument tells me the magnitudes mean something but not why the diversity *helps the updates*. Take binary logistic regression, the stripped-down case: Y = {−1,+1}, predictive probability p_w(y|x) = σ(y w·x), and the hallucinated label is just ŷ = sign(w·x). The gradient of the logistic loss at the hallucinated label is ĝ_x = ∂/∂w log(1 + e^{−ŷ w·x}) = (1 − p_w(ŷ|x))·(−ŷ x), and at the true label g̃_x = (1 − p_w(y|x))·(−y x). Now restrict attention to the points right on the margin, S_w = {x : w·x = 0} — the genuinely uncertain ones, where p_w(+1|x) = p_w(−1|x) = 1/2. On that set the hallucinated and true gradients differ only by a sign: ĝ_x = s_x g̃_x with s_x ∈ {±1}. A sign flip doesn't change a Gram matrix's determinant, so a DPP (and hence my k-means++ surrogate) over the hallucinated gradients samples the same diverse sets as one over the true-label gradients. Why does that matter? Because sampling the *uncertain* points is already known to be doing something specific — uncertainty sampling on the margin set behaves like preconditioned stochastic gradient descent on the expected 0-1 loss (Mussmann & Liang) — and sampling those gradients *diversely*, via a determinant, is known to *reduce the variance* of the resulting mini-batch gradient estimate (the determinantal mini-batch diversification line). Put the two together: restricted to the low-margin region, my method is doing the same descent on the 0-1 loss that uncertainty sampling does, but with a lower-variance gradient direction because the batch is diverse rather than redundant. So the diversity isn't a separate goal bolted on — it makes the *uncertainty* updates better. That's the intuition I was missing, and it's exactly why grabbing one cluster of near-identical uncertain points (plain uncertainty sampling) is wasteful: their gradients are nearly collinear, so the batch's combined update is high-variance and mostly one direction repeated.

Now let me write the thing as an algorithm and then as code I'd actually ship. Each round, with the current model θ_t: for every unlabeled x, get the softmax p and the penultimate embedding z, form the residual r = p − e_ŷ (ŷ = argmax p), and the gradient embedding is g_x = r ⊗ z. Then run k-means++ seeding on {g_x} to pick B points: first the largest-‖g‖ point, then D^2-weighted draws. Query those, add to the labeled set, retrain from scratch, repeat.

One implementation worry before I code it: the gradient embedding g_x = r ⊗ z is a K·d-dimensional vector per example. For K classes and a few-hundred-dimensional z that's fine to materialize, and the most transparent code does exactly that — build the [n, K·d] matrix of gradient embeddings and run k-means++ on it. But I can avoid ever forming the outer products, which matters when K·d is large, by exploiting the factorization. The only thing k-means++ ever needs is *distances* between gradient embeddings, and for outer-product vectors those collapse. For g_a = r_a ⊗ z_a and g_b = r_b ⊗ z_b,

  ⟨g_a, g_b⟩ = Σ_{i,j} (r_a)_i (z_a)_j (r_b)_i (z_b)_j = (Σ_i (r_a)_i (r_b)_i)(Σ_j (z_a)_j (z_b)_j) = (r_a·r_b)(z_a·z_b),

so the K·d-dimensional inner product is just the product of a K-dimensional inner product and a d-dimensional one. In particular ‖g_a‖^2 = ‖r_a‖^2 ‖z_a‖^2, and the squared distance I need everywhere is

  ‖g_a − g_b‖^2 = ‖g_a‖^2 + ‖g_b‖^2 − 2⟨g_a,g_b⟩ = ‖r_a‖^2‖z_a‖^2 + ‖r_b‖^2‖z_b‖^2 − 2 (r_a·r_b)(z_a·z_b).

So I keep z and r separately, precompute the per-point squared norms ‖z‖^2 and ‖r‖^2, and compute any distance from one z·z and one r·r dot product — never touching the K·d space. The first center is the argmax of ‖g‖^2 = ‖r‖^2 ‖z‖^2, the most uncertain point, exactly as designed. Numerically the distance-squared can come out slightly negative from floating point, so I clip it at zero before the square root.

Here's the straightforward, materialize-the-embedding version first, since it's the clearest statement of the method:

```python
import numpy as np
from scipy import stats


def query(self, n):
    # unlabeled candidates
    idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

    # gradient embedding g_x = (p - e_yhat) (x) z(x), flattened to [m, emb_dim * K].
    # yhat = argmax_i p_i is the model's own (hallucinated) label; block i is
    # (p_i - I(yhat=i)) * z(x). Its norm is a lower bound on the true-label
    # gradient norm and is small for confident points -> magnitude = uncertainty.
    gradEmb = self.get_grad_embedding(self.X[idxs_unlabeled],
                                      self.Y.numpy()[idxs_unlabeled]).numpy()

    # k-means++ seeding in gradient space: long + spread-out batch, no hyperparameters.
    chosen = kmeans_pp(gradEmb, n)
    return idxs_unlabeled[chosen]


def kmeans_pp(X, k):
    # X: [m, D] gradient embeddings. Returns k indices, D^2-seeded.
    m = X.shape[0]
    norms2 = (X ** 2).sum(axis=1)
    # first center: the largest-norm (most uncertain) gradient embedding
    first = int(np.argmax(norms2))
    chosen = [first]
    # D2[x] = squared distance from x to its nearest chosen center
    D2 = ((X - X[first]) ** 2).sum(axis=1)
    D2[first] = 0.0
    while len(chosen) < k:
        total = D2.sum()
        if total == 0:                              # all remaining coincide with chosen
            remaining = list(set(range(m)) - set(chosen))
            ind = int(np.random.choice(remaining))
        else:
            probs = D2 / total                      # sample proportional to D^2
            ind = int(stats.rv_discrete(values=(np.arange(m), probs)).rvs())
            while ind in chosen:
                ind = int(stats.rv_discrete(values=(np.arange(m), probs)).rvs())
        chosen.append(ind)
        newD = ((X - X[ind]) ** 2).sum(axis=1)      # distances to the new center
        D2 = np.minimum(D2, newD)                   # keep nearest-center distance
        D2[chosen] = 0.0                            # never re-pick a chosen point
    return chosen
```

And the version I'd really run, factoring the outer product so I never build the K·d embeddings — same algorithm, distances computed from the separate z (embedding) and r (probability-residual) factors:

```python
import numpy as np
from scipy import stats


def _distance(R, Z, center):
    # squared distance from every (r,z) to a chosen (r0,z0), using
    # ||r (x) z - r0 (x) z0||^2 = ||r||^2||z||^2 + ||r0||^2||z0||^2
    #                             - 2 (r . r0)(z . z0).
    (r_vec, r_n2), (z_vec, z_n2) = R, Z
    (r0_vec, r0_n2), (z0_vec, z0_n2) = center
    dist = r_n2 * z_n2 + r0_n2 * z0_n2 - 2.0 * (r_vec @ r0_vec) * (z_vec @ z0_vec)
    return np.sqrt(np.clip(dist, a_min=0, a_max=None))   # clip FP-negative dist^2


def query(self, n):
    idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
    # penultimate embeddings z and softmax probabilities p for the pool
    embs, probs = self.get_embedding(self.X[idxs_unlabeled],
                                     self.Y.numpy()[idxs_unlabeled],
                                     return_probs=True)
    embs, probs = embs.numpy(), probs.numpy()
    m = len(idxs_unlabeled)

    # r = e_yhat - p  (sign vs (p - e_yhat) is irrelevant: norms and Gram are even)
    z_n2 = np.sum(embs ** 2, axis=-1)               # ||z||^2 per point
    yhat = np.argmax(probs, axis=-1)
    r = -1.0 * probs
    r[np.arange(m), yhat] += 1.0                    # residual; large when uncertain
    r_n2 = np.sum(r ** 2, axis=-1)                  # ||r||^2 per point

    R = (r, r_n2)
    Z = (embs, z_n2)

    chosen, chosen_list, mu, D2 = set(), [], None, None
    for _ in range(n):
        if len(chosen) == 0:
            # first center: largest ||g||^2 = ||r||^2 * ||z||^2  (most uncertain)
            ind = int(np.argmax(r_n2 * z_n2))
            mu = [((r[ind], r_n2[ind]), (embs[ind], z_n2[ind]))]
            D2 = _distance(R, Z, mu[0]).ravel().astype(float)
            D2[ind] = 0.0
        else:
            newD = _distance(R, Z, mu[-1]).ravel().astype(float)
            D2 = np.minimum(D2, newD)               # nearest-chosen-center distance
            D2[list(chosen)] = 0.0
            Dsq = D2 ** 2
            total = Dsq.sum()
            if total == 0:
                ind = int(np.random.choice(list(set(range(m)) - chosen)))
            else:
                dist = Dsq / total                  # sample proportional to D^2
                sampler = stats.rv_discrete(name='custm', values=(np.arange(m), dist))
                ind = int(sampler.rvs(size=1)[0])
                while ind in chosen:
                    ind = int(sampler.rvs(size=1)[0])
            mu.append(((r[ind], r_n2[ind]), (embs[ind], z_n2[ind])))
        chosen.add(ind)
        chosen_list.append(ind)

    return idxs_unlabeled[chosen_list]
```

Let me trace the causal chain end to end so I'm sure nothing's hand-waved. I'm forced into batch acquisition because single-point active learning is infeasible for deep nets, and batching independently-scored points yields redundant batches; on top of that I can't afford a tradeoff hyperparameter, because tuning it would itself burn labels. Uncertainty sampling captures informativeness but scores points in isolation, so it floods a batch with duplicates; representative/Core-Set sampling captures diversity in penultimate space but ignores informativeness and fails when that representation is poor. I needed one per-example object carrying both. Because a net learns by gradients, "informative" is "induces a large update," and the last-layer cross-entropy gradient has the clean closed form (p_i − I(ŷ=i)) z — an outer product of a probability-residual (uncertainty) with the penultimate embedding (representation), so both properties live in a single vector. I don't know the true label, so I hallucinate the model's prediction ŷ; computing the norm, ‖g_x^y‖^2 = (Σ p_i^2 + 1 − 2p_y)‖z‖^2, shows ŷ = argmin_y‖g_x^y‖, so the hallucinated-label gradient norm is a lower bound on the true one — a conservative uncertainty estimate that's tiny for confident points and large for uncertain ones. To build a long-and-diverse batch the ideal object is a k-DPP over {g_x}, whose Gram determinant fuses magnitude (quality) and spanned volume (diversity) with no tradeoff knob and even adapts to batch size on its own — but k-DPP sampling is too expensive. k-means++ seeding's D^2 rule, started from the largest-norm point, is a cheap surrogate that pulls toward the same high-magnitude, spread-out sets and empirically matches the k-DPP's batch diversity and magnitude at a fraction of the cost, with no hyperparameters. The binary-logistic-regression case shows why this helps: on the margin the hallucinated gradients equal the true ones up to sign, so diverse sampling of them is lower-variance descent on the 0-1 loss — the diversity improves the uncertainty updates rather than competing with them. And the outer-product factorization ⟨g_a,g_b⟩ = (r_a·r_b)(z_a·z_b) lets me run the whole seeding from the separate z and r factors without ever materializing the K·d gradient embeddings.
