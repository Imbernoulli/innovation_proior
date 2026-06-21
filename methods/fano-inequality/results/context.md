# Context: bounding error probability by residual uncertainty

## Research question

We have a random variable $X$ we care about — a transmitted message, an unknown parameter, a hidden label — and we never see it directly. What we see is a correlated observation $Y$, produced from $X$ through some noisy mechanism $p(y \mid x)$. From $Y$ we form a guess $\hat X = g(Y)$, and we ask: **how often must we be wrong?**

The quantity of interest is the probability of error $P_e = \Pr(\hat X \neq X)$. Two facts frame the problem. First, if $Y$ determined $X$ exactly — if $X$ were a deterministic function of $Y$ — we could guess with zero error. Second, the textbook information-theoretic measure of "how much of $X$ remains uncertain once $Y$ is known" is the conditional entropy $H(X \mid Y)$, and it is a basic fact that $H(X \mid Y) = 0$ **if and only if** $X$ is a function of $Y$. So the two endpoints agree: $H(X\mid Y)=0 \iff$ zero-error guessing is possible.

The question is how the *interior* behaves: when $H(X \mid Y)$ is small but nonzero, what can be said about $P_e$? We seek a relation between $P_e$ and $H(X\mid Y)$ that holds for **every** estimator $g$ at once — a converse, a "you cannot do better than this" statement, for any inference problem cast as guessing $X$ from $Y$. Two settings make the payoff concrete:

- **Communication.** A capacity quantity $C = \max_{p(x)} I(X;Y)$ is known to be *achievable*: one can transmit below $C$ with vanishing error. The matching impossibility statement — that rates above $C$ force the error away from zero — is a relation between the decoding error and the leftover uncertainty about the message given the channel output.
- **Statistics.** To show that *no* estimator of a parameter $\theta$ can beat a certain accuracy (a minimax lower bound), one reduces estimation to deciding among several candidate parameters and argues that the data cannot reliably tell them apart. That reduction calls for a tool turning "the data carries little information about which hypothesis holds" into a statement about how often the test errs.

A single inequality relating $P_e$ and $H(X\mid Y)$ would serve both.

## Background

The setting rests on Shannon's 1948 information theory, whose load-bearing pieces are the following.

**Entropy and conditional entropy.** Logs are base 2 unless otherwise stated. For a discrete random variable $X$ with pmf $p(x)$ on an alphabet $\mathcal X$,
$$H(X) = -\sum_{x} p(x) \log p(x),$$
the average uncertainty in $X$. For a pair $(X,Y)\sim p(x,y)$, the joint entropy is $H(X,Y) = -\sum_{x,y} p(x,y)\log p(x,y)$, and the conditional entropy is
$$H(X \mid Y) = \sum_y p(y)\, H(X \mid Y=y) = -\sum_{x,y} p(x,y)\log p(x\mid y),$$
the uncertainty about $X$ that remains after $Y$ is revealed. Mutual information is $I(X;Y) = H(X) - H(X\mid Y)$. Two structural properties matter:

- **Chain rule for entropy** (Cover–Thomas Thm 2.2.1): $H(X,Y) = H(X) + H(Y\mid X)$, and in conditional form $H(U,V\mid Z) = H(V\mid Z) + H(U\mid V,Z)$. A joint entropy can be peeled apart in either order.
- **Conditioning reduces entropy**: $H(X\mid Y) \le H(X)$, with equality iff $X \perp Y$. Extra information never increases uncertainty on average.
- A maximum-entropy fact: $H(X) \le \log|\mathcal X|$, with equality iff $X$ is uniform — the entropy of a variable known to live in a set of size $m$ is at most $\log m$.
- $H(X\mid Y) = 0 \iff X$ is a deterministic function of $Y$ (zero residual uncertainty means exact recoverability).

**Data-processing inequality** (Cover–Thomas Thm 2.8.1). If $X \to Y \to Z$ form a Markov chain (i.e. $Z$ depends on $X$ only through $Y$), then $I(X;Y) \ge I(X;Z)$; equivalently $H(X\mid Z) \ge H(X\mid Y)$. No processing of the observation — deterministic or randomized — can increase the information it carries about $X$. In particular any estimator $\hat X = g(Y)$ gives $X \to Y \to \hat X$, so $\hat X$ is at least as uncertain about $X$ as $Y$ is.

**The channel-coding problem.** A discrete memoryless channel is a kernel $p(y\mid x)$ used $n$ times to send one of $2^{nR}$ messages ($R$ is the rate, bits per channel use). The achievability half of the noisy-channel theorem establishes, by random coding, that for any $R < C$ there exist codes with error $\to 0$. The complementary direction concerns the regime $R > C$, and asks for a relation lower-bounding the decoding error in terms of a conditional entropy of the message given the channel output.

**The statistical minimax tradition.** In parameter estimation, the natural pessimistic benchmark is the minimax risk $\inf_{\hat\theta}\sup_\theta \mathbb E_\theta[\ell(\theta,\hat\theta)]$. A standard route to *lower* bounds reduces estimation to multiple-hypothesis testing: place several parameters $\theta_1,\dots,\theta_M$ at pairwise distance $\ge \epsilon$, and argue that if the data cannot reliably identify which one generated it, the estimator must be off by $\Omega(\epsilon)$ on average. Adjacent tools in this family bound an estimator's variance by information-like quantities — the Cramér–Rao bound $\operatorname{var}_\theta(\hat\theta)\ge 1/I(\theta)$ via Fisher information, and the Hammersley–Chapman–Robbins bound $\operatorname{var}_\theta(\hat\theta)\ge \sup_{\theta'}(\mathbb E_\theta\hat\theta-\mathbb E_{\theta'}\hat\theta)^2/\chi^2(P_{\theta'}\|P_\theta)$ — derived from $\chi^2$-divergence and the data-processing inequality. These tie a quadratic loss to an information quantity over a smooth parameter.

## Baselines

- **The qualitative recoverability criterion ($H(X\mid Y)=0\iff X=g(Y)$).** Zero conditional entropy is exactly the condition for exact recovery. It is an all-or-nothing statement covering the boundary case $H(X\mid Y)=0$.

- **Union/pairwise testing arguments.** To show two (or a few) hypotheses are confusable, bound the total variation or the divergence between the induced output distributions; if they are close, no test separates them well. These give two-point lower bounds: a *test*'s error in terms of a *distance between two distributions*.

- **Variance-based estimation bounds (Cramér–Rao, Hammersley–Chapman–Robbins).** For a *quadratic* loss and (near-)unbiased estimators, lower-bound $\operatorname{var}_\theta(\hat\theta)$ by the inverse Fisher information $1/I(\theta)$ (CR), or by $\sup_{\theta'}(\mathbb E_\theta\hat\theta - \mathbb E_{\theta'}\hat\theta)^2/\chi^2(P_{\theta'}\|P_\theta)$ (HCR), both obtained from the variational/$\chi^2$ representation of $f$-divergence plus data processing. They apply to squared error with (near-)unbiasedness over a smooth Euclidean parameter.

- **Achievability (random coding) for channels.** Random codebooks and typicality decoding achieve any $R<C$ with vanishing error. This is the positive half: it demonstrates that low error is *possible* below capacity.

## Evaluation settings

The natural yardsticks for a converse of this kind are analytic, not benchmark-numeric:

- **Discrete memoryless channels** (binary symmetric channel $\mathrm{BSC}_\delta$, binary erasure channel, the noisy typewriter) used $n$ times, with messages uniform over $\{1,\dots,2^{nR}\}$, rate $R$, average and maximal probability of error $P_e^{(n)}$, and capacity $C=\max_{p(x)}I(X;Y)$ as the dividing line. A converse is judged by whether it forces $P_e^{(n)}$ away from $0$ exactly when $R>C$, matching the achievability threshold.
- **Parametric estimation models** — e.g. the Gaussian location model $X\sim N(\theta, I_d)$, coin-flip $\mathrm{Ber}(\theta)$, and packings of a metric parameter space $\Theta$ — with loss $\ell(\theta,\hat\theta)$ a metric or its power, evaluated through the minimax risk $\inf_{\hat\theta}\sup_\theta \mathbb E_\theta[\ell]$ and the metric/covering–packing structure ($\epsilon$-packing number $M(\Theta,d,\epsilon)$).
- **Sanity checks for the inequality itself**: the no-observation case (best guess is the mode), and worst-case distributions that should make the bound tight.

## Code framework

A theorem does not need a canonical software implementation. The closest useful scaffold is a small numerical harness that checks a candidate inequality on concrete joint distributions, confirming that a derived bound is not violated and probing when it is tight. The primitives below already exist (entropy of a pmf, the conditional entropy of a joint table); the empty slot maps a joint distribution to a lower bound on error.

```python
import numpy as np

def entropy(p, base=2):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * (np.log(p) / np.log(base))).sum())

def conditional_entropy(joint, base=2):
    # joint[x, y] = p(x, y); returns H(X | Y)
    joint = np.asarray(joint, dtype=float)
    py = joint.sum(axis=0)
    h = 0.0
    for j, pyj in enumerate(py):
        if pyj > 0:
            h += pyj * entropy(joint[:, j] / pyj, base)
    return float(h)

def prob_error_of_estimator(joint, g):
    # g: array of length |Y| giving the guessed x for each y; P_e = Pr(g(Y) != X)
    joint = np.asarray(joint, dtype=float)
    pe = 0.0
    for j in range(joint.shape[1]):
        pe += joint[:, j].sum() - joint[g[j], j]
    return float(pe)

def error_lower_bound(joint, base=2):
    """Lower bound on P_e that holds for ANY estimator.
    TODO: fill in."""
    pass

# harness: pick a joint p(x,y), the optimal (MAP) estimator, compare P_e to the bound
# TODO: assert error_lower_bound(joint) <= prob_error_of_estimator(joint, map_rule(joint))
```
