Least confidence did roughly what the boundary argument said, and where it *under*-delivered is the tell. On seed 42 it beat random everywhere — letter 0.867 vs 0.835, spambase 0.948 vs 0.935, splice 0.840 vs 0.771 — so chasing the model's least-confident points genuinely beats drawing blindly when the model is good. But the means expose the brittleness: on **letter** the least-confidence *mean* accuracy is 0.7955 and auc 0.668, *below* random's 0.816 / 0.724. That inversion has a specific cause I can name. The score $1-\max_y p(y\mid x)$ is a single deterministic forward pass through one network, and it cannot distinguish "I am unsure because I have not seen enough data here" (which a label fixes) from "I am unsure because this point is genuinely ambiguous, the label is intrinsically noisy" (which no label ever fixes). On letter, the confusable-but-irreducible boundary between similar glyphs eats queries that buy nothing. So the limitation that forces this rung is exact: I need to query only the uncertainty a label can resolve, separating it from the uncertainty a label cannot.

State what I actually want cleanly. "This label teaches me the most" means "this label most shrinks my uncertainty about the model parameters," so the myopic-best query maximizes the expected drop in parameter-posterior entropy,

$$\arg\max_x \; H[\theta\mid D] - \mathbb{E}_{y\sim p(y\mid x,D)}\big[H[\theta\mid y,x,D]\big].$$

The expectation over $y$ is unavoidable — I do not have the label yet, so I average over what it might be, weighted by the current predictive belief. This is the right object but I cannot afford it: $H[\theta\mid D]$ is the entropy of a posterior over thousands of network weights, and the expectation costs a full re-inference per imagined label, $O(N_x N_y)$ posterior updates. The fix is not to patch it but to read its form. Entropy minus expected-entropy-after-conditioning *is*, by definition, a mutual information — the conditional MI between the parameters and the unseen label, $I[\theta, y\mid x, D]$, an identity, not an approximation. And mutual information is symmetric, $I(A;B) = H(A) - H(A\mid B) = H(B) - H(B\mid A)$, so the same number equals

$$I[\theta, y\mid x, D] = H[y\mid x, D] - \mathbb{E}_{\theta\sim p(\theta\mid D)}\big[H[y\mid x,\theta]\big].$$

This is the move. The left side lives in parameter space; the right side has both entropies in *output* space, over the finitely many class labels, trivial even when $\theta$ is huge — and $\theta$ is conditioned only on $D$, never on a hypothetical $y$. There is no "pretend I saw label $y$ and re-infer" anywhere: I do inference once on the data I already have, then for every candidate compute two tiny class entropies. The $O(N_x N_y)$ re-inference collapses to $O(1)$ and the entropy of $\theta$ never appears, all without changing the number by one bit.

I propose **BALD — Bayesian Active Learning by Disagreement** — querying $\arg\max_x H[y\mid x,D] - \mathbb{E}_\theta[H[y\mid x,\theta]]$. Read the right side as a recipe and it says exactly what least confidence could not. The first term big means my *marginal* prediction, averaged over all $\theta$ in the posterior, is very uncertain. The second term small means each *individual* $\theta$ is confident. Together I want the $x$ where the parameter settings are each individually sure of the answer yet sure of *different* answers — confident disagreement. A point that is noisy-but-known, where every $\theta$ agrees "this is a genuine coin flip," has high $H[y\mid x,D]$ but *also* high $\mathbb{E}_\theta[H[y\mid x,\theta]]$, so the difference is small and I correctly skip it. A point where each $\theta$ is confident but they split has high $H[y\mid x,D]$ and *low* $\mathbb{E}_\theta[H[y\mid x,\theta]]$, so the difference is large and I query it. The subtraction is precisely the operation that separates epistemic uncertainty (which a label fixes) from aleatoric uncertainty (which it does not). Least confidence is essentially just the first term, the marginal uncertainty, which is why it chased the irreducible boundary noise that dragged letter under random; BALD is the first term *minus* the second. The containment is clean: if the observation noise is zero, every $\theta$ predicts deterministically, $H[y\mid x,\theta]=0$, the expectation vanishes, and the criterion reduces to plain marginal entropy — the uncertainty-sampling family. BALD is not a fourth competitor; it *contains* uncertainty sampling as the noise-free special case and adds the noise correction on top.

Now it has to land in this scaffold. The weight posterior $p(\theta\mid D)$ for the small net is hopelessly intractable and there is no probit-convolution closed form here, so I use the *model-agnostic* form, which needs only posterior samples of $\theta$ and the predictive softmax per sample. The harness gives exactly this through `self.predict_prob_dropout_split(X, Y, n_drop)`: it leaves dropout switched on at prediction time and runs the network `n_drop` times, each pass zeroing a different random subset of units, so each pass is effectively a different network drawn from an approximate weight posterior, returning a tensor $[n\_drop, \text{len}(X), n\_classes]$. So $T = $ `n_drop` dropout passes stand in for posterior samples $\theta^1,\dots,\theta^T$ — I never need the posterior in closed form, never its entropy, I just sample it and read off softmaxes. (The harness exposes only the dropout posterior, so the Gaussian-process path, the probit/squared-exponential closed form, and the nuisance-parameter and preference-learning extensions of the full Bayesian-disagreement story are all out; the MC estimator is the rung.) Plugging the samples in, the expectation over the posterior becomes an average over passes. The marginal predictive $p(y\mid x,D)=\mathbb{E}_\theta[p(y\mid x,\theta)]$ becomes the mean softmax $\bar p = \frac{1}{T}\sum_t p^t$, so the first term is the entropy of the mean, $H[\bar p] = -\sum_c \bar p_c\log\bar p_c$, high when the passes *together* are unsure; the second is the mean of the per-pass entropies, $\frac{1}{T}\sum_t H[p^t]$, high when each individual pass is unsure. The score is their difference,

$$I(x) \approx H[\bar p] - \frac{1}{T}\sum_t H[p^t],$$

high exactly when each pass is individually confident but the passes confidently disagree — the dropout networks vote different ways while each is sure of itself. Because it is entropy-of-an-average minus average-of-entropies it is non-negative by Jensen's inequality (entropy is concave), the right sign for an MI. One implementation point against the literal scaffold, where the sign silently breaks: I compute the *negative* $U = (\text{mean per-pass entropy}) - (\text{entropy of the mean}) = -I(x)$, sort $U$ ascending, and take the first $n$, since the smallest $U$ are the largest $I$. A handful of passes suffices to estimate two entropies; `n_drop = 10` is enough and cheap over the whole pool.

The expectations against least confidence's numbers: the cleanest is **letter**, where its mean inverted under random precisely because raw top-class uncertainty chased irreducible 26-class noise — the $-\mathbb{E}_\theta[H[y\mid x,\theta]]$ term is built to subtract exactly that, so I expect BALD to repair letter back above random, ideally above least confidence's seed-42 0.867. On **spambase**, already strong and carrying little irreducible noise, the split has little to separate, so I expect roughly a match, maybe a hair behind because ten coarse dropout passes are a noisier estimate than one clean softmax max. On **splice** a small gain. The structural caveat I carry forward: BALD separates resolvable from irreducible uncertainty *per point*, but like least confidence it has no term that looks at the other chosen points, so the $n$ highest-MI rows can still be near-duplicates. The next rung must fuse uncertainty with diversity.

```python
# EDITABLE region of badge/query_strategies/custom_sampling.py (lines 28-54) — step 3: BALD (MC dropout)
class CustomSampling(Strategy):
    """BALD — Bayesian Active Learning by Disagreement (MC Dropout).
    Selects samples where there is maximal disagreement across stochastic
    forward passes, approximating mutual information."""

    def __init__(self, X, Y, idxs_lb, net, handler, args, n_drop=10):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.n_drop = n_drop

    def query(self, n):
        import torch
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob_dropout_split(
            self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], self.n_drop
        )
        # Mean prediction across MC samples
        pb = probs.mean(0)
        # Total entropy: H[y | x, D]
        entropy1 = (-pb * torch.log(pb + 1e-10)).sum(1)
        # Expected entropy: E_theta[H[y | x, theta]]
        entropy2 = (-probs * torch.log(probs + 1e-10)).sum(2).mean(0)
        # BALD score = total entropy - expected entropy = mutual information
        U = entropy2 - entropy1
        return idxs_unlabeled[U.sort()[1][:n]]
```
