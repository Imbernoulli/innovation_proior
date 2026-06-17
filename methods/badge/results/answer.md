# BADGE, distilled

BADGE (Batch Active learning by Diverse Gradient Embeddings) is a pool-based batch
active-learning rule for deep classifiers. For each unlabeled example it forms a last-layer
gradient embedding using the model's own predicted, or hallucinated, label; then it chooses a
batch with k-means++-style `D^2` seeding in that gradient space. The embedding norm is a
conservative uncertainty signal, and distances between embeddings give within-batch diversity,
with no hand-tuned uncertainty/diversity coefficient.

## Problem it solves

Each active-learning round must choose `B` unlabeled examples to label. Independent uncertainty
scores can return many near-duplicates from one ambiguous region, while pure representative
selection can cover the pool without asking whether those examples would move the classifier.
Because changing an active-learning hyperparameter changes which labels are bought, a practical
batch rule should avoid a tunable tradeoff parameter.

## Key idea

For a softmax network `f(x; theta) = softmax(W z(x; V))`, penultimate embedding
`z in R^d`, last-layer weights `W in R^{K x d}`, and class probabilities `p`, the
cross-entropy gradient with respect to the final layer has class block

```text
(g_x^y)_i = (p_i - I(y = i)) z(x; V),      so      g_x^y = (p - e_y) tensor z.
```

The residual vector `p - e_y` carries uncertainty, and `z` carries representation. Since the
true label is unknown, use `yhat = argmax_i p_i` and define `g_x = g_x^yhat`. Its squared norm is

```text
||g_x^y||^2 = (sum_i p_i^2 + 1 - 2 p_y) ||z||^2.
```

The only label-dependent term is `-2 p_y`, so the minimum over labels is attained by any largest
probability class: `argmin_y ||g_x^y|| = argmax_y p_y`. If there is a tie, every tied maximizer is
also a minimizer; the code uses NumPy's deterministic first-maximum tie rule. Therefore
`||g_x^yhat|| <= ||g_x^y||` for the unknown true `y`. Confident examples have residual near zero;
uncertain examples have larger residual norm.

To make a batch both high-magnitude and diverse, an ideal object is a `k`-DPP over gradient
embeddings, because a Gram determinant rewards vector lengths and spanned volume. Exact or MCMC
`k`-DPP sampling is expensive, so BADGE uses the `D^2` recurrence from k-means++ seeding in
gradient space. Standard k-means++ starts its first center uniformly and has the
`8(ln k + 2)` expected-potential guarantee; the canonical BADGE implementation modifies the first
center to be the largest-norm gradient embedding, so that theorem should not be quoted as a
formal guarantee for BADGE's exact sampler.

## Algorithm

```text
Input: net f, unlabeled pool U, initial labels M, rounds T, batch size B.
S <- M random examples from U, with queried labels; train theta_1 on S.
for t = 1..T:
    for each x in U \ S:
        yhat(x) <- argmax_i f(x; theta_t)_i
        g_x <- d/d theta_out CE(f(x; theta_t), yhat(x))
    choose S_t by k-means++-style seeding on {g_x : x in U \ S}:
        first center <- argmax_x ||g_x||^2
        each later center <- sample x with probability D(x)^2 / sum_u D(u)^2
        where D(x) = min distance from g_x to an already chosen center
    query labels for S_t
    S <- S union S_t
    train theta_{t+1} on S
return theta_{T+1}
```

If all remaining embeddings have zero distance to the chosen set, the mathematical `D^2`
distribution is undefined and any remaining tied point is equivalent. The pinned reference code
does not special-case that degenerate event.

## Working code

Faithful core of `JordanAsh/badge` at commit `a2d18acd372cf0f61d9e75bfb0c879c107fbf9f6`.
The implementation uses `r = e_yhat - p`, the negative of the paper-gradient residual
`p - e_yhat`; norms, pairwise distances, and Gram determinants are unchanged by this global sign.

```python
import numpy as np
from scipy import stats


def distance(X1, X2, mu):
    Y1, Y2 = mu
    X1_vec, X1_norm_square = X1
    X2_vec, X2_norm_square = X2
    Y1_vec, Y1_norm_square = Y1
    Y2_vec, Y2_norm_square = Y2
    dist = (
        X1_norm_square * X2_norm_square
        + Y1_norm_square * Y2_norm_square
        - 2 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec)
    )
    # Numerical errors may cause the distance squared to be negative.
    assert np.min(dist) / np.max(dist) > -1e-4
    return np.sqrt(np.clip(dist, a_min=0, a_max=None))


def init_centers(X1, X2, chosen, chosen_list, mu, D2):
    if len(chosen) == 0:
        ind = np.argmax(X1[1] * X2[1])
        mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
        D2 = distance(X1, X2, mu[0]).ravel().astype(float)
        D2[ind] = 0
    else:
        newD = distance(X1, X2, mu[-1]).ravel().astype(float)
        D2 = np.minimum(D2, newD)
        D2[chosen_list] = 0
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name="custm", values=(np.arange(len(Ddist)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in chosen:
            ind = customDist.rvs(size=1)[0]
        mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
    chosen.add(ind)
    chosen_list.append(ind)
    return chosen, chosen_list, mu, D2


class BadgeSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BadgeSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        embs, probs = self.get_embedding(
            self.X[idxs_unlabeled],
            self.Y.numpy()[idxs_unlabeled],
            return_probs=True,
        )
        embs = embs.numpy()
        probs = probs.numpy()

        m = (~self.idxs_lb).sum()
        mu = None
        D2 = None
        chosen = set()
        chosen_list = []
        emb_norms_square = np.sum(embs ** 2, axis=-1)
        max_inds = np.argmax(probs, axis=-1)

        probs = -1 * probs
        probs[np.arange(m), max_inds] += 1
        prob_norms_square = np.sum(probs ** 2, axis=-1)
        for _ in range(n):
            chosen, chosen_list, mu, D2 = init_centers(
                (probs, prob_norms_square),
                (embs, emb_norms_square),
                chosen,
                chosen_list,
                mu,
                D2,
            )
        return idxs_unlabeled[chosen_list]
```

## Relation to prior methods

- **Uncertainty sampling:** the gradient norm is an uncertainty score, but the batch is selected
  with distances between examples rather than independent top-`B` ranking.
- **Core-Set / k-Center:** Core-Set uses only representation geometry; BADGE uses the
  representation multiplied by a probability residual, so uncertainty and representation live in
  one embedding.
- **EGL:** expected gradient length averages over possible labels and remains a scalar
  per-example score; BADGE uses one lower-bound gradient and then diversifies the batch.
- **k-DPP:** BADGE approximates the same quality/diversity geometry with a cheaper seeding rule.
