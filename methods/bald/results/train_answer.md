In pool-based active learning, labels are expensive and unlabeled inputs are plentiful. Each round the learner may select a small batch of pool points to label, after which the harness retrains the model. The principled myopic criterion for choosing the next point is to pick the input whose label most reduces uncertainty about the model parameters, written as argmax_x H[theta | D] - E_{y ~ p(y|x,D)}[H[theta | y, x, D]]. This expected drop in parameter-posterior entropy captures exactly how much a label would teach the model. But it is almost impossible to compute: for neural networks the parameter posterior is high-dimensional and intractable, and for nonparametric models such as Gaussian processes the parameter is an entire latent function, so its entropy is not even defined. Moreover, evaluating the expectation over y requires imagining every possible label, adding each hypothetical pair to the data, and re-running inference, which costs O(N_x N_y) posterior updates per round.

Existing cheap rules avoid that cost by looking only at predictions, but they throw away the distinction that matters. Least-confidence sampling, a form of uncertainty sampling, scores each candidate by one minus its highest predicted class probability. That is large whenever the model is unsure, but unsureness has two sources: genuine ignorance that more data can fix, and irreducible label noise that no amount of data can fix. A point sitting on a noisy decision boundary therefore looks just as attractive as a point where the model is genuinely conflicted, so the method chases noise. Maximum-entropy sampling makes the same mistake with the full predictive entropy H[y|x,D]. Query-by-committee tries to target disagreement among plausible models, but it collapses each committee member's full distribution to a hard vote, discarding confidence information. A split vote where every member is individually uncertain looks identical to a split where every member is confident but contradictory, so QBC reintroduces the noise problem. The Informative Vector Machine stays closer to the parameter objective but is built for subsampling already-labeled data and still needs O(N_x N_y) approximate updates.

The solution is BALD, Bayesian Active Learning by Disagreement. The crucial recognition is that the parameter-entropy objective is not an arbitrary utility; it is the conditional mutual information I[theta, y | x, D]. Mutual information is symmetric, so the same scalar can be rewritten as H[y | x, D] - E_{theta ~ p(theta|D)}[H[y | x, theta]]. The first term is the entropy of the marginal predictive distribution, averaged over the whole posterior. The second term is the expected entropy of the label when the parameters are fixed. Their difference is the epistemic uncertainty that a label can resolve: it is large exactly when the posterior contains confident but conflicting predictions, and small when the uncertainty is aleatoric noise that every model agrees is unavoidable. Because theta appears only inside an expectation conditioned on the data already observed, no hypothetical label requires a fresh posterior update. The criterion needs O(1) inference per round, and the entropies live over the finite set of class labels, so they are cheap and well defined even when theta is infinite-dimensional.

This output-space form also exposes why BALD contains the baselines as special cases. If the observation model is deterministic, H[y|x,theta] is zero for every theta and BALD reduces to maximum-entropy sampling. If instead of entropies one used hard committee votes, one would recover query-by-committee. The extra subtracted term is precisely the correction those methods lack. For Gaussian-process classifiers the two entropy terms can even be approximated in closed form from the posterior mean and variance of the latent function, but the same principle carries over to any model from which one can sample the posterior.

For a deep network, the weight posterior p(theta|D) is intractable, so the expectation is estimated with Monte Carlo dropout. Leaving dropout active during prediction gives T approximate posterior samples theta^1,...,theta^T; each pass returns a softmax vector p^t = p(y|x,theta^t). The marginal predictive distribution is the average softmax p_bar = (1/T) sum_t p^t, whose entropy estimates H[y|x,D]. The average of the per-pass entropies estimates E_theta[H[y|x,theta]]. Their difference is the BALD score, and by Jensen's inequality it is always non-negative. A typical default is T = 10 dropout passes, enough to estimate the two entropies cheaply across the whole pool.

The implementation below fills the query slot of the standard pool-based active-learning harness. It obtains the stochastic forward passes, computes the two entropy terms, ranks the unlabeled points by U = mean-per-pass-entropy - entropy-of-mean, which equals the negative mutual information, and returns the n points with smallest U, i.e., the largest BALD score.

```python
import numpy as np
import torch
from .strategy import Strategy


class BALDDropout(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, n_drop=10):
        super(BALDDropout, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.n_drop = n_drop

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob_dropout_split(
            self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], self.n_drop
        )
        pb = probs.mean(0)
        entropy1 = (-pb * torch.log(pb + 1e-10)).sum(1)
        entropy2 = (-probs * torch.log(probs + 1e-10)).sum(2).mean(0)
        U = entropy2 - entropy1
        return idxs_unlabeled[U.sort()[1][:n]]
```

This single rule turns the intractable parameter-space information-gain objective into a tractable output-space disagreement score. By keeping each posterior sample's confidence and subtracting the aleatoric component, BALD selects the points whose labels will actually resolve model uncertainty.
