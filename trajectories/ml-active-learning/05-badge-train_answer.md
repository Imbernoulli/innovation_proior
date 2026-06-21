BAIT confirmed exactly the split I predicted. On **spambase** the Fisher A-optimal objective reached 0.929 / 0.911 — above BALD's 0.905 / 0.892 and edging least confidence's 0.927 / 0.908 — and on **splice** 0.813 / 0.756, above BALD's 0.795 / 0.737: filling the diversity hole with a pool-aware information criterion genuinely helped where diversity binds. But on **letter** BAIT collapsed — mean 0.791 / 0.671, below random's 0.816 / 0.724, and seed 42 has no result at all because the run did not complete. That is the CPU adaptation biting precisely where I said: 26 classes makes $d\cdot k$ largest, so the random projection to 128 dims discards the most Fisher geometry, the streaming full-pool accumulation is slowest, and the time/memory budget can fail outright. So the lesson is sharp. The A-optimal *objective* is right — uncertainty fused with pool-aware diversity beats per-point uncertainty — but the *machinery* to compute it exactly (rank-$k$ Fisher factors, a $d\cdot k\times d\cdot k$ pool matrix, projections, pseudo-inverses, forward/backward greedy) is too heavy to survive this harness on the hardest dataset. I want the same fused property — long *and* diverse batches, no tradeoff knob — but obtained cheaply: no matrix inverse, no projection, nothing that blows up when the number of classes is large.

The right currency is hiding in how the net trains. The machine moves by gradients, so an example is informative exactly when, once I know its label, it induces a large gradient and therefore a large parameter update. The full-parameter gradient is a giant object I would have to average over the unknown label, so I take the cheapest informative slice — the gradient with respect to the *last* layer. With $f(x)=\operatorname{softmax}(Wz(x))$, $z$ the penultimate embedding and $W$ the final linear layer, the cross-entropy gradient in the block for class $i$ is

$$(g_x^y)_i = (p_i - \mathbf{1}[y=i])\cdot z(x),$$

so the whole last-layer gradient embedding is an outer product $g_x^y = (p - e_y)\otimes z$ — the probability-residual $r = p - e_y$ tensored with the penultimate embedding $z$. This decomposition is the thing the ladder kept separating. The $z$ factor is *exactly* the representation space — where the example sits, its diversity and identity, the same penultimate features. The $r$ factor carries how wrong/uncertain the model is — if it is confident and right, $p\approx e_y$ so $r\approx 0$; if unsure, $p$ is spread and $r$ is large. One rank-one object already containing both desiderata: representation in $z$ (BAIT's pool-Fisher diversity) and uncertainty in $r$ (what least confidence and BALD scored), multiplied together blockwise with no coefficient to balance — and rank-one per point, not the rank-$k$ Fisher BAIT had to project. That is the cheapness I need.

The obvious snag: $g_x^y$ needs the true label $y$, which I do not have. The expensive answer averages over $y$ under the model; the cheap one hallucinates the model's own current prediction $\hat y = \arg\max_i p_i$ and uses $g_x = g_x^{\hat y}$, one gradient per example, no averaging — and it is defensible, not lazy. The gradient norm is

$$\|g_x^y\|^2 = \Big(\textstyle\sum_i p_i^2 + 1 - 2p_y\Big)\|z\|^2,$$

whose only $y$-dependent piece is $-2p_y$. Minimizing $\|g_x^y\|$ over $y$ maximizes $p_y$, so $\arg\min_y \|g_x^y\| = \arg\max_y p_y = \hat y$: the hallucinated label gives the *smallest possible* gradient norm, meaning $\|g_x\| = \|g_x^{\hat y}\|$ is a *lower bound* on the gradient norm the example will really induce once I see its true label. That is a guarantee, not laziness — when I pick a point because its $g_x$ is large, it will produce at least that much update, the bound only ever understates, so I cannot be fooled into thinking a confident point is informative. And the magnitude tracks confidence: if the model is sharp $p\approx e_{\hat y}$ and $\sum p_i^2 + 1 - 2p_{\hat y}\approx 0$ so the embedding nearly vanishes; if $p$ is flat the quantity is large. So $\|g_x\|$ is the same conservative uncertainty score least confidence and BALD read, now living in a direction-bearing vector that also encodes representation.

I propose **BADGE — diverse gradient embeddings**. For every pool point I have a vector $g_x$ whose *length* means uncertainty and *direction* means identity, and I want a batch individually long and collectively spread out, with no tradeoff knob. The ideal mathematical object is a determinantal point process: sample a size-$n$ set with probability $\propto \det(L_Y)$, the Gram matrix of the chosen $\{g_x\}$. A Gram determinant equals the squared product of lengths times the squared spanned volume — length rewards big gradients (uncertainty), volume is large only when the vectors point in different directions (diversity) and collapses the instant two are parallel — so $\det(L_Y)$ *is* quality and diversity fused with no coefficient, and it self-adjusts to batch size (small $n$: volume slack, acts like uncertainty sampling; large $n$: spanning gets hard, diversity takes over). But sampling a k-DPP is genuinely expensive — high-order polynomial, MCMC mixing, memory blow-up at the batch sizes I care about. So I use a cheap surrogate: **k-means++ seeding**, whose $D^2$ weighting picks the first center then samples each next center with probability proportional to its squared distance to the nearest already-chosen center. In gradient space a point far from everything already chosen is much more likely picked — diversity, sharpened by squaring — and if I seed the first point by its *length* rather than uniformly, I plant the batch on the most uncertain example and let $D^2$ spread from there. This pulls toward exactly the high-magnitude, diverse set a k-DPP would, with no determinant, inverse, MCMC, projection, or hyperparameter, and it scales gracefully in the number of classes where BAIT's projection choked, since the only thing it ever needs is *distances* between embeddings.

Those distances collapse because the embeddings are outer products. For $g_a = r_a\otimes z_a$ and $g_b = r_b\otimes z_b$, $\langle g_a, g_b\rangle = (r_a\cdot r_b)(z_a\cdot z_b)$, so

$$\|g_a - g_b\|^2 = \|r_a\|^2\|z_a\|^2 + \|r_b\|^2\|z_b\|^2 - 2(r_a\cdot r_b)(z_a\cdot z_b).$$

I never form the $k\cdot d$-dimensional vectors: I keep $z$ and $r$ separately, precompute the per-point squared norms $\|z\|^2$ and $\|r\|^2$, and compute any distance from one $z\cdot z$ and one $r\cdot r$ dot product. The first center is $\arg\max \|r\|^2\|z\|^2$, the most uncertain point. This factored form is the load-bearing difference from materialized BADGE: the scaffold does *not* call `get_grad_embedding`; it calls `self.get_embedding(..., return_probs=True)` to get $z$ and $p$ separately, forms the residual $r = e_{\hat y} - p$ (sign is irrelevant — norms and the Gram matrix are even in $r$), and runs factored k-means++ on the $(r,z)$ pair through a `_distance` helper, sampling each new center with `scipy.stats.rv_discrete` over the $D^2$ distribution, clipping FP-negative $\text{dist}^2$ at 0. Why this beats plain uncertainty sampling where I can compute it: in binary logistic regression restricted to the margin set $w\cdot x = 0$ (the genuinely uncertain points), the hallucinated and true-label gradients differ only by a sign, which a Gram determinant ignores, so diverse sampling of the hallucinated gradients samples the same sets as the true ones — and sampling those gradients *diversely* is lower-variance descent on the 0/1 loss than grabbing one cluster of near-collinear uncertain points. The diversity makes the *uncertainty* updates better; it is not a separate goal bolted on. That is exactly the redundant-batch waste that limited least confidence and BALD, fixed by construction rather than by a coefficient.

The decisive test is **letter**, where BAIT collapsed because its heavy machinery could not survive 26 classes: BADGE's factored seeding has no projection and no matrix inverse, so it should *not* collapse, and because the fused embedding captures both the confusable-glyph uncertainty BALD scored and the across-class diversity thin-budget letter needs, I expect it to be the *strongest* rung on letter — above BALD's 0.836 / 0.716 and seed-42 0.893. On **spambase** I expect rough parity with BAIT and least confidence (a balanced binary boundary leaves little for diversity to add), and on **splice** a gain over BAIT's 0.813 / 0.756. The bar this endpoint must clear is to be the best *aggregate* rung — win letter outright while holding spambase and splice at the front — which would make it the only rung that fuses uncertainty and diversity cheaply enough to survive every dataset, the thing the whole climb was reaching for.

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
