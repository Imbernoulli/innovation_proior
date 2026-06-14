**Problem.** BAIT's Fisher A-optimal objective is right — fuse uncertainty with pool-aware diversity, no
tradeoff knob — but its machinery (rank-`k` per-class Fisher, a `d·k × d·k` pool matrix, random projection,
pseudo-inverses, forward/backward greedy) is too heavy for this CPU harness and collapsed on letter (26
classes). I want the same fused property — long *and* diverse batches — but cheaply: no matrix inverse, no
projection, nothing that blows up when the number of classes is large.

**Key idea (BADGE — diverse gradient embeddings).** A net learns by gradients, so informativeness =
"induces a large update." The last-layer cross-entropy gradient factors as an outer product
`g_x = (p − e_ŷ) ⊗ z` — the probability-residual `r` (uncertainty) tensored with the penultimate embedding
`z` (representation): both desiderata fused in one rank-one vector, no coefficient. Hallucinating the model's
own label `ŷ = argmax p` gives the *minimum*-norm gradient, so `‖g_x‖` is a lower bound on the true-label
gradient norm — conservative uncertainty, tiny for confident points. The ideal diverse-and-long batch is a
k-DPP over `{g_x}` (Gram det = lengths × spanned volume), but it's too expensive; k-means++ seeding (first
center = largest norm, then `D²`-weighted draws) is a cheap surrogate that targets the same high-magnitude,
spread-out sets with no hyperparameter and no matrix work.

**This task's implementation (vs. materialized BADGE).** The scaffold does *not* call `get_grad_embedding`;
it calls `self.get_embedding(..., return_probs=True)` for `z` and `p` separately, forms `r = e_ŷ − p` (sign
is irrelevant — norms/Gram are even), and runs k-means++ on the *factored* `(r, z)` pair via a `_distance`
helper using `⟨g_a,g_b⟩ = (r_a·r_b)(z_a·z_b)`, so it never materializes the `d·k`-dim embeddings. New
centers are sampled with `scipy.stats.rv_discrete` over the `D²` distribution; FP-negative `dist²` is
clipped at 0. This scales in the number of classes exactly where BAIT's projection choked.

**Hyperparameters.** None (the only "tradeoff" is the `D²`-vs-batch-size geometry, automatic). First center
by `argmax ‖r‖²‖z‖²`; `n` `D²`-weighted draws.

**What to watch.** Decisive test is letter: no projection/inverse, so BADGE should *not* collapse there and,
fusing BALD's uncertainty with cross-class diversity, should be the strongest rung on letter (above BALD's
0.836/0.716). Expect ≈ parity on spambase (diversity adds little to a balanced binary boundary) and a gain
over BAIT on splice. The bar: best aggregate rung — win letter while holding spambase/splice at the front.

```python
# EDITABLE region of badge/query_strategies/custom_sampling.py (lines 28-54) — step 5: BADGE (factored k-means++)
class CustomSampling(Strategy):
    """BADGE — Batch Active learning by Diverse Gradient Embeddings.
    Selects batches that are diverse and uncertain in gradient embedding space
    via k-means++ seeding."""

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        from scipy import stats
        from sklearn.metrics import pairwise_distances

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        embs, probs = self.get_embedding(
            self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], return_probs=True
        )
        embs = embs.numpy()
        probs = probs.numpy()

        # BADGE: k-means++ in (gradient-embedding x probability-residual) space
        m = len(idxs_unlabeled)
        emb_norms_square = np.sum(embs ** 2, axis=-1)
        max_inds = np.argmax(probs, axis=-1)

        prob_residuals = -1.0 * probs
        prob_residuals[np.arange(m), max_inds] += 1.0
        prob_norms_square = np.sum(prob_residuals ** 2, axis=-1)

        # k-means++ initialization
        chosen = set()
        chosen_list = []
        mu = None
        D2 = None

        def _distance(X1, X2, center):
            Y1, Y2 = center
            X1_vec, X1_norm_sq = X1
            X2_vec, X2_norm_sq = X2
            Y1_vec, Y1_norm_sq = Y1
            Y2_vec, Y2_norm_sq = Y2
            dist = (X1_norm_sq * X2_norm_sq + Y1_norm_sq * Y2_norm_sq
                    - 2.0 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec))
            return np.sqrt(np.clip(dist, a_min=0, a_max=None))

        for _ in range(n):
            if len(chosen) == 0:
                ind = np.argmax(emb_norms_square * prob_norms_square)
                mu = [((prob_residuals[ind], prob_norms_square[ind]),
                        (embs[ind], emb_norms_square[ind]))]
                D2 = _distance(
                    (prob_residuals, prob_norms_square),
                    (embs, emb_norms_square),
                    mu[0],
                ).ravel().astype(float)
                D2[ind] = 0
                chosen.add(ind)
                chosen_list.append(ind)
            else:
                newD = _distance(
                    (prob_residuals, prob_norms_square),
                    (embs, emb_norms_square),
                    mu[-1],
                ).ravel().astype(float)
                D2 = np.minimum(D2, newD)
                D2[list(chosen)] = 0
                D2_sq = D2 ** 2
                total = D2_sq.sum()
                if total == 0:
                    # Fallback: random from remaining unlabeled
                    remaining = list(set(range(m)) - chosen)
                    ind = np.random.choice(remaining)
                else:
                    Ddist = D2_sq / total
                    customDist = stats.rv_discrete(
                        name="custm", values=(np.arange(len(Ddist)), Ddist)
                    )
                    ind = customDist.rvs(size=1)[0]
                    while ind in chosen:
                        ind = customDist.rvs(size=1)[0]
                mu.append(((prob_residuals[ind], prob_norms_square[ind]),
                           (embs[ind], emb_norms_square[ind])))
                chosen.add(ind)
                chosen_list.append(ind)

        return idxs_unlabeled[chosen_list]
```
