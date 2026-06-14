# BADGE, distilled

BADGE (Batch Active learning by Diverse Gradient Embeddings) is a pool-based batch
active-learning acquisition rule for deep networks. It represents each unlabeled example by the
gradient of the loss with respect to the network's last layer, computed at the model's own
predicted ("hallucinated") label, and selects each batch by running k-means++ seeding in that
gradient-embedding space. The gradient's *magnitude* encodes predictive uncertainty (it is a
lower bound on the true-label gradient norm), and the k-means++ `D^2` seeding makes the batch
*diverse* — so every batch is simultaneously uncertain and diverse, with no hyperparameter
trading the two off.

## Problem it solves

Choosing, each round, a batch of `B` unlabeled examples to label so a deep classifier reaches
high accuracy with few total labels. Scoring examples independently (uncertainty sampling)
yields redundant near-duplicate batches; pure representative/diversity selection ignores
informativeness. Active learning also cannot afford a tunable tradeoff coefficient, because
re-tuning it changes which points are queried and thus costs labels. BADGE needs none.

## Key idea

For a softmax network `f(x;θ) = softmax(W z(x;V))` with penultimate embedding `z ∈ R^d`,
last-layer weights `W ∈ R^{K×d}`, and softmax output `p`, the cross-entropy gradient w.r.t. the
last layer factors block-wise:

```
(g_x^y)_i = (p_i - I(y = i)) z(x;V)        =>   g_x^y = (p - e_y) ⊗ z .
```

It is an outer product of a probability-residual vector `r = p - e_y` (uncertainty) and the
penultimate embedding `z` (representation/diversity) — both properties in one vector.

**Hallucinated label and the lower bound.** The true label is unknown, so use the model's
prediction `ŷ = argmax_i p_i` and set `g_x = g_x^{ŷ}`. The norm is

```
‖g_x^y‖^2 = ( Σ_i p_i^2 + 1 - 2 p_y ) ‖z‖^2 ,
```

whose only `y`-dependence is `-2 p_y`, so `argmin_y ‖g_x^y‖ = argmax_y p_y = ŷ`. Hence
`‖g_x‖ ≤ ‖g_x^y‖` for the unknown true `y`: the hallucinated-label gradient norm is a lower
bound on the true-label one. It is near zero for confident points (`p ≈ e_ŷ ⇒ r ≈ 0`) and large
for uncertain ones — a conservative uncertainty score that can only understate informativeness.

**Diversity by seeding, not by a determinant.** The ideal selector is a `k`-DPP over `{g_x}`:
`P(Y) ∝ det(L_Y)`, `L` the Gram matrix, whose determinant fuses magnitude (quality) and spanned
volume (diversity) with no tradeoff coefficient and adapts to batch size automatically (small
`B` → magnitude dominates → uncertainty; large `B` → linear-independence dominates → diversity).
But `k`-DPP sampling is too expensive (high-order-polynomial exact samplers; MCMC mixing).
BADGE substitutes **k-means++ seeding** (Arthur & Vassilvitskii 2007): start from the
largest-norm gradient embedding, then sample each next point with probability proportional to
its squared distance to the nearest already-chosen point (`D^2` weighting). This pulls toward
long, mutually-distant vectors — empirically matching the `k`-DPP's batch diversity and
magnitude at far lower cost — with no hyperparameter.

**Why diversity helps the updates (binary logistic regression).** On the margin set
`S_w = {x : w·x = 0}` the hallucinated and true-label gradients agree up to a sign, so DPP /
k-means++ sampling of the hallucinated gradients equals sampling of the true ones (a Gram
determinant is sign-invariant). Uncertainty sampling on `S_w` is preconditioned SGD on the
expected 0-1 loss (Mussmann & Liang 2018) and diverse (determinantal) gradient sampling reduces
the variance of the mini-batch update (Zhang et al. 2017) — so BADGE performs the same descent
direction as uncertainty sampling but with lower variance, i.e. diversity improves the
uncertainty updates rather than competing with them.

## Algorithm

```
Input: net f, unlabeled pool U, initial labels M, rounds T, batch size B.
S <- M random examples from U, with queried labels;  train θ_1 on S.
for t = 1..T:
    for each x in U \ S:
        ŷ(x) <- argmax_i f(x;θ_t)_i                 # hallucinated label
        g_x  <- ∂/∂θ_out ℓ_CE(f(x;θ_t), ŷ(x))       # last-layer gradient embedding
    S_t <- k-means++ seeding on { g_x : x in U \ S } to pick B points;  query their labels.
    S   <- S ∪ S_t
    train θ_{t+1} on S
return θ_{T+1}
```

k-means++ seeding (Arthur & Vassilvitskii): `c_1 ~` largest-norm point (here, vs. uniform, to
seed on maximal uncertainty); then `c_t` sampled with probability `D(x)^2 / Σ_x D(x)^2`, where
`D(x) = min_{c chosen} ‖g_x - c‖`. Guarantee on the seeding potential: `E[φ] ≤ 8(ln k + 2) φ_OPT`.

## Working code

Drop-in `query(n)` for the pool-based harness. Two equivalent forms.

Straightforward form — materialize the `K·d` gradient embeddings and run k-means++ on them:

```python
import numpy as np
from scipy import stats


def query(self, n):
    idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
    # g_x = (p - e_yhat) (x) z(x), flattened to [m, emb_dim * n_classes].
    # block i = (p_i - I(yhat=i)) * z(x); norm = lower bound on true-label grad norm.
    grad_emb = self.get_grad_embedding(self.X[idxs_unlabeled],
                                       self.Y.numpy()[idxs_unlabeled]).numpy()
    chosen = kmeans_pp(grad_emb, n)            # D^2 seeding: long + diverse batch
    return idxs_unlabeled[chosen]


def kmeans_pp(X, k):
    m = X.shape[0]
    norms2 = (X ** 2).sum(axis=1)
    first = int(np.argmax(norms2))             # most uncertain point seeds the batch
    chosen = [first]
    D2 = ((X - X[first]) ** 2).sum(axis=1)     # squared dist to nearest chosen center
    D2[first] = 0.0
    while len(chosen) < k:
        total = D2.sum()
        if total == 0:                         # remaining points coincide with chosen
            ind = int(np.random.choice(list(set(range(m)) - set(chosen))))
        else:
            probs = D2 / total                 # sample proportional to D^2
            ind = int(stats.rv_discrete(values=(np.arange(m), probs)).rvs())
            while ind in chosen:
                ind = int(stats.rv_discrete(values=(np.arange(m), probs)).rvs())
        chosen.append(ind)
        D2 = np.minimum(D2, ((X - X[ind]) ** 2).sum(axis=1))
        D2[chosen] = 0.0
    return chosen
```

Factored form — never builds the `K·d` outer products, using
`⟨r_a⊗z_a, r_b⊗z_b⟩ = (r_a·r_b)(z_a·z_b)`, hence
`‖g_a-g_b‖^2 = ‖r_a‖^2‖z_a‖^2 + ‖r_b‖^2‖z_b‖^2 - 2(r_a·r_b)(z_a·z_b)`:

```python
import numpy as np
from scipy import stats


def _distance(R, Z, center):
    # squared distance of every (r, z) to a chosen (r0, z0) via the factorization
    (r_vec, r_n2), (z_vec, z_n2) = R, Z
    (r0_vec, r0_n2), (z0_vec, z0_n2) = center
    dist = r_n2 * z_n2 + r0_n2 * z0_n2 - 2.0 * (r_vec @ r0_vec) * (z_vec @ z0_vec)
    return np.sqrt(np.clip(dist, a_min=0, a_max=None))    # clip FP-negative dist^2


def query(self, n):
    idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
    embs, probs = self.get_embedding(self.X[idxs_unlabeled],
                                     self.Y.numpy()[idxs_unlabeled], return_probs=True)
    embs, probs = embs.numpy(), probs.numpy()
    m = len(idxs_unlabeled)

    z_n2 = np.sum(embs ** 2, axis=-1)                     # ||z||^2
    yhat = np.argmax(probs, axis=-1)
    r = -1.0 * probs
    r[np.arange(m), yhat] += 1.0                          # r = e_yhat - p (uncertainty)
    r_n2 = np.sum(r ** 2, axis=-1)                        # ||r||^2

    R, Z = (r, r_n2), (embs, z_n2)
    chosen, chosen_list, mu, D2 = set(), [], None, None
    for _ in range(n):
        if len(chosen) == 0:
            ind = int(np.argmax(r_n2 * z_n2))            # largest ||g||^2: most uncertain
            mu = [((r[ind], r_n2[ind]), (embs[ind], z_n2[ind]))]
            D2 = _distance(R, Z, mu[0]).ravel().astype(float)
            D2[ind] = 0.0
        else:
            D2 = np.minimum(D2, _distance(R, Z, mu[-1]).ravel().astype(float))
            D2[list(chosen)] = 0.0
            Dsq = D2 ** 2
            total = Dsq.sum()
            if total == 0:
                ind = int(np.random.choice(list(set(range(m)) - chosen)))
            else:
                sampler = stats.rv_discrete(name='custm',
                                            values=(np.arange(m), Dsq / total))
                ind = int(sampler.rvs(size=1)[0])
                while ind in chosen:
                    ind = int(sampler.rvs(size=1)[0])
            mu.append(((r[ind], r_n2[ind]), (embs[ind], z_n2[ind])))
        chosen.add(ind)
        chosen_list.append(ind)
    return idxs_unlabeled[chosen_list]
```

## Relation to prior methods

- **Uncertainty sampling** (least-confidence / margin / entropy): BADGE's gradient *norm* is an
  uncertainty score, but BADGE adds batch diversity via seeding, removing the duplicate-batch
  pathology of independent scoring.
- **Core-Set / k-Center** (Sener & Savarese 2018): operates in penultimate space `z` for
  diversity only; BADGE uses the gradient embedding `r ⊗ z`, which contains `z` *and*
  uncertainty `r`, and seeds with `D^2` (k-means++) rather than furthest-first (k-Center).
- **EGL** (Settles et al. 2008): scores by expected gradient length but per-example, with no
  diversity and an average over all `K` labels; BADGE uses one hallucinated-label gradient
  (cheaper, lower-bound-justified) and builds a diverse batch.
- **k-DPP** (Kulesza & Taskar 2011): the ideal quality-diversity selector BADGE approximates;
  k-means++ seeding matches its batch diversity/magnitude at far lower cost.
