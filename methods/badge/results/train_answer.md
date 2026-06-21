The problem is pool-based batch active learning for deep classifiers. We start with a small labeled set and a large unlabeled pool, and at each round we may buy labels for only B examples, retrain the model from scratch, and repeat. The goal is to reach a target accuracy with as few labels as possible. Because deep nets have millions of parameters, selecting and retraining on one example at a time is infeasible, so we are forced to acquire whole batches. But once we score examples independently and take the top B, nothing prevents the selected batch from containing many near-duplicates from the same uncertain region. A single label would have resolved the model's confusion about all of them, yet we pay for B. At the same time, any tradeoff hyperparameter is dangerous in active learning: changing it changes which points are queried, so tuning it consumes the very label budget we are trying to save. A practical rule must therefore fuse informativeness and diversity in a single, knob-free mechanism.

Existing methods only cover one side of this tension. Uncertainty sampling, whether by least-confidence, margin, or entropy, ranks points by predictive uncertainty and is cheap and effective when B is small. But the score looks at one example at a time, so as the batch grows it increasingly packs redundant points from the same ambiguous cluster, and deep softmax probabilities are often overconfident anyway. Core-Set and other representative methods select a geometrically diverse batch in the penultimate-layer feature space, which avoids redundancy, but they ignore whether the covered points would actually change the classifier; they can waste labels on regions the model already understands, and they depend on the representation already being meaningful. k-DPPs elegantly combine high quality with diversity through the determinant of a Gram matrix, but exact or MCMC sampling is too expensive for large pools and large batch sizes. What is missing is a cheap per-example object that already contains both uncertainty and representation, together with a fast batch-selection rule that needs no tuning.

The method is BADGE, Batch Active learning by Diverse Gradient Embeddings. The key observation is that a network learns through gradients, so the natural measure of how informative an example will be is the magnitude of the loss gradient it induces. For a softmax network f(x; theta) = softmax(W z(x; V)) with penultimate embedding z in R^d and final weights W in R^{K x d}, the last-layer cross-entropy gradient with respect to class i has the simple closed form (p_i - I(y = i)) z(x; V). Collecting all class blocks gives g_x^y = (p - e_y) tensor z: an outer product of the probability residual, which is large when the model is uncertain, and the penultimate embedding, which encodes where the example lives in representation space. Both of the properties we wanted are already multiplied inside one vector.

Because the true label is unknown, BADGE uses the model's own prediction yhat = argmax_i p_i. A short calculation shows that the squared norm of the gradient embedding is (sum_i p_i^2 + 1 - 2 p_y) ||z||^2, which is minimized by any class with maximal probability. Thus ||g_x^yhat|| is a lower bound on the norm of the gradient that the true label would induce. Confident correct examples have residual near zero and hence small gradient norm; uncertain examples have larger residual and larger norm. So the gradient embedding norm is a conservative, label-free uncertainty signal.

To select a batch, BADGE uses k-means++-style D^2 seeding in this gradient-embedding space. The first selected point is the one with the largest ||g_x||^2, the most uncertain example. Each later point is sampled with probability proportional to its squared distance to the nearest already-selected gradient embedding. The squared-distance weighting naturally rewards points that are far from the current batch, which enforces diversity, while the underlying gradient norm biases the selection toward uncertain points. There is no explicit coefficient balancing uncertainty against diversity; the geometry of the D^2 rule provides an automatic, batch-size-adaptive tradeoff. For small batches the length factor dominates, so the method behaves like uncertainty sampling; for large batches the volume constraint dominates, so it behaves like a diversity method. This gives most of the benefit of a k-DPP at a tiny fraction of the cost.

The gradient embedding has K d dimensions per example, but it never needs to be materialized. The only quantities used by k-means++ are pairwise distances, and for outer-product vectors the inner product factorizes as (r_a . r_b)(z_a . z_b), where r = p - e_yhat. The squared distance between two gradient embeddings is therefore ||r_a||^2 ||z_a||^2 + ||r_b||^2 ||z_b||^2 - 2 (r_a . r_b)(z_a . z_b), which is computed from the separate z and r vectors and their squared norms. This makes the method scale to large numbers of classes and large embeddings without ever forming a K d-dimensional vector.

```python
import numpy as np
from scipy import stats


class BadgeSampling(Strategy):
    """BADGE: Batch Active learning by Diverse Gradient Embeddings.
    Selects a batch that is both uncertain and diverse by k-means++ seeding
    in last-layer gradient-embedding space."""

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BadgeSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def _distance(self, R, Z, center):
        (r, r_n2), (z, z_n2) = R, Z
        (r0, r0_n2), (z0, z0_n2) = center
        dist = r_n2 * z_n2 + r0_n2 * z0_n2 - 2.0 * (r @ r0) * (z @ z0)
        return np.sqrt(np.clip(dist, a_min=0.0, a_max=None))

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        embs, probs = self.get_embedding(
            self.X[idxs_unlabeled],
            self.Y.numpy()[idxs_unlabeled],
            return_probs=True,
        )
        embs = embs.numpy()
        probs = probs.numpy()
        m = len(idxs_unlabeled)

        z_n2 = np.sum(embs ** 2, axis=-1)
        yhat = np.argmax(probs, axis=-1)
        r = -1.0 * probs
        r[np.arange(m), yhat] += 1.0
        r_n2 = np.sum(r ** 2, axis=-1)

        R = (r, r_n2)
        Z = (embs, z_n2)

        chosen = set()
        chosen_list = []
        mu = None
        D2 = None

        for _ in range(n):
            if len(chosen) == 0:
                ind = int(np.argmax(r_n2 * z_n2))
                mu = [((r[ind], r_n2[ind]), (embs[ind], z_n2[ind]))]
                D2 = self._distance(R, Z, mu[0]).ravel().astype(float)
                D2[ind] = 0.0
            else:
                newD = self._distance(R, Z, mu[-1]).ravel().astype(float)
                D2 = np.minimum(D2, newD)
                D2[list(chosen)] = 0.0
                weights = D2 ** 2
                total = weights.sum()
                if total == 0.0:
                    remaining = list(set(range(m)) - chosen)
                    ind = int(np.random.choice(remaining))
                else:
                    dist = weights / total
                    sampler = stats.rv_discrete(name="custm", values=(np.arange(m), dist))
                    ind = int(sampler.rvs(size=1)[0])
                    while ind in chosen:
                        ind = int(sampler.rvs(size=1)[0])
                mu.append(((r[ind], r_n2[ind]), (embs[ind], z_n2[ind])))
            chosen.add(ind)
            chosen_list.append(ind)

        return idxs_unlabeled[chosen_list]
```
