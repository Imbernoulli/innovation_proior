BALD did what the noise-subtraction argument predicted and confirmed the limit I was carrying. On **letter** it repaired the regression — mean 0.8357 / 0.716, back above random and well above least confidence's collapsed 0.7955 / 0.668, with seed 42 up at 0.893 — exactly the $-\mathbb{E}_\theta[H[y\mid x,\theta]]$ term subtracting the irreducible 26-class boundary noise raw uncertainty was chasing. But on **spambase** BALD came in at 0.905 / 0.892, *below* least confidence's 0.927 / 0.908: a near-balanced binary problem has almost no irreducible noise to separate, and ten coarse dropout passes are a noisier estimate than one clean softmax max. The reason BALD is no longer dominant everywhere is the structural hole I named: it scores each point in isolation, with no term that looks at the other chosen points, so the $n$ highest-MI rows can be near-duplicates clustered in one stretch of the boundary — I pay for $n$ labels and learn far fewer labels' worth. And I cannot fix that by tuning a uncertainty/diversity coefficient, because changing a knob queries a *different* set whose labels I would have to buy. So this rung needs one principled objective that keeps BALD's per-point informativeness *and* makes the batch diverse, with no free coefficient.

Rather than pick a design criterion by taste, I derive the actual error I am minimizing and read off which scalar of the model's information it forces. Treat `self.clf` as a probability model: a softmax is $p(y\mid x,\theta)$, fitting is maximum likelihood with loss $\ell = -\log p(y\mid x,\theta)$, and the theory of how good an MLE is becomes available. The MLE's error is governed by the Fisher information $I(x;\theta) = \mathbb{E}_y[\nabla^2\ell(x,y;\theta)]$, the expected Hessian of the per-point loss — classically the MLE is asymptotically normal with covariance the inverse Fisher, so the Fisher is the *precision* a labeled point buys about $\theta$. There is a structural gift to check: for multiclass logistic regression the first-derivative block for class $p$ is $(\pi_p - \mathbf{1}[y=p])\,x$, linear in the label indicator, so its second derivative kills the label entirely, leaving $\nabla^2\ell = xx^\top\otimes(\operatorname{diag}(\pi)-\pi\pi^\top)$ — label-independent. That is what lets me talk about a point's Fisher *before* paying for its label, the whole game.

So I want a labeling set $S$ minimizing the MLE error, the Fisher is the currency, and I need a scalar of $\sum_{x\in S} I(x;\theta)$. Experimental design offers a menu — D-optimality (maximize $\det$), A-optimality (minimize $\operatorname{tr}$ of the inverse) — and is agnostic about which, so I derive the error instead of guessing. In Bayesian linear regression I can do it in closed form: the Bayes risk of the ridge estimate in the pool second-moment metric $\Sigma$ telescopes (cross terms cancel in pairs) to $\text{BayesRisk}(S)=\sigma^2\operatorname{tr}(\Lambda_S^{-1}\Sigma)$ with $\Lambda_S = \sum_{x\in S} xx^\top + \lambda\sigma^2 I$. Two things fall out I did not put in: the criterion is a *weighted trace* — weighted by the pool second moment $\Sigma$, not a determinant and not a bare $\operatorname{tr}(\Lambda^{-1})$; and the right side contains *no labels*, so the risk of labeling $S$ depends only on its features and the pool, an oracle-free objective minimizable before paying for a single label. The frequentist MLE analysis lands on the same functional, $\operatorname{tr}(I_S(\theta)^{-1} I_U(\theta))/m$, with the per-point Fisher for $xx^\top$ and the pool Fisher $I_U$ for $\Sigma$. Two independent derivations converging is what makes the object trustworthy.

I propose **BAIT — Fisher-information A-optimal selection** — choosing

$$S^\star = \arg\min_{S\subset U,\,|S|\le B}\;\operatorname{tr}\!\Big(\big(\textstyle\sum_{x\in S} I(x;\theta)\big)^{-1} I_U(\theta)\Big).$$

Why the trace beats the determinant, and therefore why this should out-diversify the rungs around it: under D-optimality $\det(I(x;\theta)^{-1} I_U) = \det(I(x;\theta)^{-1})\cdot\det(I_U)$, and $\det(I_U)$ is a constant in $x$ that factors straight out — the determinant is structurally *blind* to the pool Fisher, looking only at each candidate's own Fisher, never at which directions the pool weights. The trace does not collapse: $\operatorname{tr}((\sum_S I)^{-1} I_U)$ genuinely couples the selected Fisher to the pool Fisher and preferentially shores up the directions $I_U$ says matter. And $\operatorname{tr}(M^{-1})$ is the sum of inverse eigenvalues, dominated by the *smallest* — A-optimality pours effort into the directions of lowest information, the weak spots a determinant (a product, dominated by large eigenvalues) ignores. For active learning that is the right instinct: fix what you are worst at, weighted by how much the pool cares.

Now the part where this task's implementation diverges hard from the textbook method, because the harness runs on CPU under a 64 GB limit the full algorithm would blow through. The per-point Fisher restricts to the *last layer*, $I(x;\theta^L) = V_x V_x^\top$, where $V_x$ is a $(d\cdot k)\times k$ factor whose columns are the per-class last-layer loss gradients each scaled by $\sqrt{\text{class prob}}$, reproducing the exact $x^L x^{L\top}\otimes(\operatorname{diag}(\pi)-\pi\pi^\top)$. (BADGE, which I face next, uses *one* column — the gradient at the single hallucinated label — and does *not* scale by $\sqrt p$; its embedding is a rank-one shadow of this rank-$k$ Fisher, discarding $k-1$ directions and the probability weighting.) The scaffold advertises a `get_exp_grad_embedding` helper returning exactly these factors, but I deliberately do *not* call it: materializing the full $[n\_pool, k, d\cdot k]$ tensor and accumulating the $d\cdot k\times d\cdot k$ pool Fisher would exhaust the letter budget. Instead I rebuild the factors myself in *streaming* `DataLoader` batches — penultimate embeddings and softmax per batch, assembled into $V_x$ on the fly — and accumulate the pool Fisher and the labeled "seed" Fisher batch-by-batch divided by their counts, never holding the whole tensor at once. Second, $d\cdot k$ is still large (for letter, 26 classes times penultimate width is thousands), so before selection I **random-project** the factors down to a fixed `bait_proj_dim = 128` with one fixed Gaussian projection (seeded, scaled by $1/\sqrt{128}$); Johnson–Lindenstrauss preserves the inner products the trace is built from, so the A-optimal geometry survives while the matrices to invert become $128\times128$. Third, even projected, scoring the entire pool every greedy step is too slow, so I run BAIT on an **entropy-filtered candidate pool** — compute predictive entropy for every unlabeled point in the same streaming pass, keep the top $\max(4n, 512)$ most-uncertain, and let the Fisher selection diversify *within* that shortlist (uncertainty pre-filters, A-optimality diversifies). Fourth, the representation changes every round as the harness retrains, so this is not the convex theory's one-shot SDP; I recompute and re-solve each round.

The selection is greedy because exact combinatorial minimization is intractable, but the trace functional is *not* submodular, so plain forward greedy can commit early to points that become redundant later and cannot take them back. So I **oversample** — greedily add $2n$ points forward — then run a **backward** pass removing points one at a time, each time deleting the one whose removal hurts the objective least, down to $n$: forward casts a wide net, backward prunes the ones redundant in company. The per-step argmin would be a $128\times128$ inversion per candidate per step, which I avoid with the low-rank structure: each $I(x)=V_x V_x^\top$ is rank $k$, so the Woodbury identity turns $(M + V_x V_x^\top)^{-1}$ into an update needing only a $k\times k$ solve, and the cyclic trace lets me precompute $M^{-1} I_U M^{-1}$ once per step and score every candidate with a small batched matmul. One numerical adaptation the CPU port forces: the projected, candidate-filtered Fisher can be rank-deficient and ill-conditioned, so I use the *pseudo*-inverse `torch.linalg.pinv` everywhere a plain inverse appears and `nan_to_num` the scores — `torch.inverse` would throw on singular matrices here. The seed/candidate normalization matches the derived risk: scale the seed Fisher into $M_0$ by $n_\text{Labeled}/(n_\text{Labeled}+n)$ and each candidate $V_x$ by $\sqrt{n/(n_\text{Labeled}+n)}$ so the new batch joins in the correct proportion.

The expectations against BALD's numbers, genuinely uncertain because the adaptations cut both ways. On **spambase** and **splice** the full Fisher objective with the pool term should beat per-point-isolated BALD, since A-optimality directly targets the diversity hole — spambase back above 0.905 / 0.892 toward least confidence's 0.927 / 0.908, splice above 0.795 / 0.737. The sharpest worry is **letter**, where BALD was strongest, because letter is exactly where the CPU adaptations bite hardest: 26 classes means the largest $d\cdot k$, so the 128-dim projection discards the most geometry and the streaming full-pool Fisher is slowest, the run the time budget can fail to finish. I expect BAIT to *underperform* BALD on letter and possibly not complete every seed there — a missing or degraded letter result is the projection-and-time adaptation showing through, not the A-optimal objective failing. The bar: beat BALD on spambase and splice where diversity binds, at the cost of letter where the port handicaps it. If it loses everywhere, the entropy pre-filter plus 128-dim projection has degraded the geometry past usefulness, and the next rung must recover diversity more cheaply.

```python
# EDITABLE region of badge/query_strategies/custom_sampling.py (lines 28-54) — step 4: BAIT (CPU-adapted)
class CustomSampling(Strategy):
    """BAIT — Batch Active Learning via Information Matrices (Fisher embeddings).
    CPU-adapted version of the original BAIT algorithm.

    This implementation keeps the Fisher-matrix objective, but makes the
    selection pass tractable on MLS-Bench's CPU setup by:
    1. building Fisher statistics in streaming batches,
    2. projecting very high-dimensional Fisher embeddings before selection,
    3. running BAIT on an entropy-filtered candidate pool instead of the full
       unlabeled set.
    """

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.lamb = args.get('lamb', 1)
        self.max_proj_dim = int(args.get('bait_proj_dim', 128))
        self.candidate_pool = int(args.get('bait_candidate_pool', 0))
        self.selection_batch_size = int(args.get('bait_selection_batch_size', 256))
        self.seed = int(args.get('seed', 42))

    def _make_projection(self, full_dim):
        import torch

        if full_dim <= self.max_proj_dim:
            return None

        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed)
        projection = torch.randn(
            full_dim,
            self.max_proj_dim,
            generator=generator,
            dtype=torch.float32,
        )
        projection /= np.sqrt(float(self.max_proj_dim))
        return projection

    def _build_batch_embeddings(self, embedding, probs, projection):
        import torch

        n_lab = probs.shape[1]
        coeffs = -probs.unsqueeze(1).expand(-1, n_lab, -1).clone()
        diag = torch.arange(n_lab)
        coeffs[:, diag, diag] += 1.0
        coeffs *= torch.sqrt(probs.clamp_min(1e-12)).unsqueeze(-1)

        fisher = coeffs.unsqueeze(-1) * embedding.unsqueeze(1).unsqueeze(2)
        fisher = fisher.reshape(embedding.shape[0], n_lab, -1)
        if projection is not None:
            fisher = torch.matmul(fisher, projection)
        return fisher.contiguous()

    def _candidate_pool_size(self, n, total):
        default_size = max(4 * n, 512)
        if self.candidate_pool > 0:
            default_size = self.candidate_pool
        return min(total, default_size)

    def _collect_statistics(self, idxs_unlabeled, n):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader

        model = self.clf.eval()
        device = next(model.parameters()).device
        n_lab = int(torch.max(self.Y).item() + 1)
        emb_dim = model.get_embedding_dim()
        full_dim = emb_dim * n_lab
        projection = self._make_projection(full_dim)
        target_dim = full_dim if projection is None else projection.shape[1]

        fisher = torch.zeros(target_dim, target_dim, dtype=torch.float32)
        init = torch.zeros(target_dim, target_dim, dtype=torch.float32)
        n_labeled = max(int(np.sum(self.idxs_lb)), 1)
        unlabeled_scores = np.empty(len(idxs_unlabeled), dtype=np.float32)
        pool_to_unlabeled = np.full(self.n_pool, -1, dtype=np.int64)
        pool_to_unlabeled[idxs_unlabeled] = np.arange(len(idxs_unlabeled))

        loader = DataLoader(
            self.handler(self.X, self.Y, transform=self.args['transformTest']),
            shuffle=False,
            **self.args['loader_te_args']
        )

        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(device)
                logits, embedding = model(x)
                probs = F.softmax(logits, dim=1).cpu()
                batch_xt = self._build_batch_embeddings(embedding.cpu(), probs, projection)
                fisher += torch.sum(
                    torch.matmul(batch_xt.transpose(1, 2), batch_xt),
                    dim=0,
                ) / float(self.n_pool)

                idxs_np = idxs.numpy()
                labeled_mask = torch.from_numpy(self.idxs_lb[idxs_np])
                if labeled_mask.any():
                    init += torch.sum(
                        torch.matmul(
                            batch_xt[labeled_mask].transpose(1, 2),
                            batch_xt[labeled_mask],
                        ),
                        dim=0,
                    ) / float(n_labeled)

                unlabeled_mask = ~labeled_mask
                if unlabeled_mask.any():
                    unlabeled_rows = pool_to_unlabeled[idxs_np[unlabeled_mask.numpy()]]
                    batch_probs = probs[unlabeled_mask]
                    entropy = -torch.sum(
                        batch_probs * torch.log(batch_probs.clamp_min(1e-12)),
                        dim=1,
                    )
                    unlabeled_scores[unlabeled_rows] = entropy.numpy()

        candidate_count = self._candidate_pool_size(n, len(idxs_unlabeled))
        if candidate_count == len(idxs_unlabeled):
            candidate_local = np.arange(len(idxs_unlabeled))
        else:
            candidate_local = np.argpartition(unlabeled_scores, -candidate_count)[-candidate_count:]
        candidate_local = candidate_local[np.argsort(unlabeled_scores[candidate_local])[::-1]]
        candidate_global = idxs_unlabeled[candidate_local]
        pool_to_candidate = np.full(self.n_pool, -1, dtype=np.int64)
        pool_to_candidate[candidate_global] = np.arange(len(candidate_global))
        candidate_xt = torch.empty(
            len(candidate_global),
            n_lab,
            target_dim,
            dtype=torch.float32,
        )

        with torch.no_grad():
            for x, _, idxs in loader:
                idxs_np = idxs.numpy()
                candidate_rows = pool_to_candidate[idxs_np]
                keep_mask_np = candidate_rows >= 0
                if not keep_mask_np.any():
                    continue

                x = x.to(device)
                logits, embedding = model(x)
                probs = F.softmax(logits, dim=1).cpu()
                batch_xt = self._build_batch_embeddings(embedding.cpu(), probs, projection)
                keep_mask = torch.from_numpy(keep_mask_np)
                candidate_xt[torch.from_numpy(candidate_rows[keep_mask_np])] = batch_xt[keep_mask]

        return fisher, init, candidate_global, candidate_xt

    def _trace_scores(self, xt_batch, current_inv, fisher, add_identity):
        import torch

        rank = xt_batch.shape[-2]
        eye = torch.eye(rank, dtype=xt_batch.dtype).unsqueeze(0)
        sign = 1.0 if add_identity else -1.0
        info = current_inv @ fisher @ current_inv
        gram = torch.matmul(torch.matmul(xt_batch, current_inv), xt_batch.transpose(1, 2))
        inner = gram + sign * eye
        inner_inv = torch.linalg.pinv(inner)
        middle = torch.matmul(torch.matmul(xt_batch, info), xt_batch.transpose(1, 2))
        scores = torch.diagonal(
            torch.matmul(middle, inner_inv),
            dim1=-2,
            dim2=-1,
        ).sum(-1)
        finfo = torch.finfo(scores.dtype)
        return torch.nan_to_num(scores, nan=-finfo.max, posinf=finfo.max, neginf=-finfo.max)

    def _woodbury_update(self, current_inv, xt_sample, add_identity):
        import torch

        xt_sample = xt_sample.unsqueeze(0)
        rank = xt_sample.shape[-2]
        eye = torch.eye(rank, dtype=xt_sample.dtype).unsqueeze(0)
        sign = 1.0 if add_identity else -1.0

        current = current_inv.unsqueeze(0)
        inner = torch.matmul(torch.matmul(xt_sample, current), xt_sample.transpose(1, 2))
        inner_inv = torch.linalg.pinv(inner + sign * eye)
        updated = current - torch.matmul(
            torch.matmul(torch.matmul(current, xt_sample.transpose(1, 2)), inner_inv),
            torch.matmul(xt_sample, current),
        )
        return updated[0].contiguous()

    def _best_forward_index(self, xt_scaled, current_inv, fisher, selected_mask):
        import torch

        best_idx = None
        best_score = -float('inf')
        for start in range(0, len(xt_scaled), self.selection_batch_size):
            end = min(start + self.selection_batch_size, len(xt_scaled))
            batch = xt_scaled[start:end]
            scores = self._trace_scores(batch, current_inv, fisher, add_identity=True)
            batch_mask = selected_mask[start:end]
            if np.any(batch_mask):
                scores[torch.from_numpy(batch_mask)] = -torch.finfo(scores.dtype).max
            score, local_idx = torch.max(scores, dim=0)
            score = score.item()
            if score > best_score:
                best_score = score
                best_idx = start + local_idx.item()
        return best_idx

    def query(self, n):
        import gc
        import torch

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        fisher, init, candidate_global, xt_unlabeled = self._collect_statistics(
            idxs_unlabeled,
            n,
        )
        if len(candidate_global) <= n:
            return candidate_global

        n_labeled = int(np.sum(self.idxs_lb))
        K = n
        denom = float(max(n_labeled + K, 1))
        dim = xt_unlabeled.shape[-1]
        currentInv = torch.linalg.pinv(
            self.lamb * torch.eye(dim, dtype=torch.float32)
            + init * n_labeled / denom
        )
        xt_scaled = xt_unlabeled * np.sqrt(K / denom)

        indsAll = []
        selected_mask = np.zeros(len(candidate_global), dtype=bool)
        over_sample = 2

        for _ in range(min(int(over_sample * K), len(candidate_global))):
            ind = self._best_forward_index(xt_scaled, currentInv, fisher, selected_mask)
            if ind is None:
                break

            indsAll.append(ind)
            selected_mask[ind] = True
            currentInv = self._woodbury_update(
                currentInv,
                xt_scaled[ind],
                add_identity=True,
            )

        for _ in range(len(indsAll) - K):
            xt_selected = xt_scaled[indsAll]
            traceEst = self._trace_scores(xt_selected, currentInv, fisher, add_identity=False)
            delInd = torch.argmax(traceEst).item()
            currentInv = self._woodbury_update(
                currentInv,
                xt_scaled[indsAll[delInd]],
                add_identity=False,
            )
            del indsAll[delInd]

        gc.collect()
        return candidate_global[np.asarray(indsAll, dtype=int)]
```
