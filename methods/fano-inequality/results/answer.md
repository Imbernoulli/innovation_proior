# Fano's inequality

## Problem it solves

You must guess a random variable $X$ from a correlated observation $Y$, forming $\hat X = g(Y)$. How small can the error probability $P_e = \Pr(\hat X \neq X)$ be? Fano's inequality answers with a universal lower bound — one that holds for *every* estimator $g$ — expressed through the residual conditional entropy $H(X\mid Y)$. It converts an information quantity into an operational error floor, and is the workhorse "converse" tool: it powers the converse to Shannon's channel-coding theorem and minimax lower bounds in statistics.

## Key idea

Leftover uncertainty obstructs reliable guessing. If $X$ still has much uncertainty after seeing $Y$ (large $H(X\mid Y)$), no rule can guess it reliably. The bridge from the *probability* $P_e$ to the *entropy* $H(X\mid Y)$ is the **error indicator** $E = \mathbf 1\{\hat X\neq X\}$: it is a Bernoulli$(P_e)$ variable, so its entropy is exactly $H(P_e)$, and the chain rule lets it be introduced for free and then re-expanded to expose both $H(P_e)$ and a $P_e\log(|\mathcal X|-1)$ term. The data-processing inequality then transfers the bound from the estimate $\hat X$ back to the observation $Y$, making it estimator-independent.

## Theorem (Fano's inequality)

Let $X$ take values in an alphabet $\mathcal X$ of size $|\mathcal X|\ge 2$, let the guess $\hat X$ also take values in $\mathcal X$, let $X \to Y \to \hat X$ be a Markov chain (any estimator $\hat X = g(Y)$, deterministic or randomized), and let $P_e = \Pr(\hat X \neq X)$. With logarithms in base 2,
$$H(P_e) + P_e \log(|\mathcal X|-1) \;\ge\; H(X\mid\hat X) \;\ge\; H(X\mid Y),$$
where $H(p) = -p\log p - (1-p)\log(1-p)$ is the binary entropy. The weaker, most-used operational forms are
$$1 + P_e\log|\mathcal X| \;\ge\; H(X\mid Y) \qquad\Longleftrightarrow\qquad P_e \;\ge\; \frac{H(X\mid Y) - 1}{\log|\mathcal X|},$$
combined with the trivial bound $P_e\ge 0$ when the numerator is negative.

Equivalent restatement: with $M = |\mathcal X|$, $H(X\mid Y) \le P_e\log(M-1) + H(P_e)$.

## Proof

Define the error indicator $E = \mathbf 1\{\hat X\neq X\}$, so $E\sim\mathrm{Bernoulli}(P_e)$ and $H(E)=H(P_e)$. Expand $H(E,X\mid\hat X)$ two ways with the chain rule for conditional entropy:
$$H(E,X\mid\hat X) = H(X\mid\hat X) + \underbrace{H(E\mid X,\hat X)}_{=\,0} = H(E\mid\hat X) + H(X\mid E,\hat X).$$
The term $H(E\mid X,\hat X)=0$ because $E$ is a deterministic function of $(X,\hat X)$. Equating the two expansions,
$$H(X\mid\hat X) = H(E\mid\hat X) + H(X\mid E,\hat X).$$
Bound each summand:

- $H(E\mid\hat X) \le H(E) = H(P_e)$ (conditioning reduces entropy).
- Split on $E$:
$$H(X\mid E,\hat X) = \Pr(E{=}0)\,H(X\mid\hat X,E{=}0) + \Pr(E{=}1)\,H(X\mid\hat X,E{=}1).$$
Given $E=0$, $X=\hat X$, so $H(X\mid\hat X,E{=}0)=0$. Given $E=1$, $X\neq\hat X$, so $X$ ranges over the $|\mathcal X|-1$ symbols other than $\hat X$ and $H(X\mid\hat X,E{=}1)\le\log(|\mathcal X|-1)$. Hence $H(X\mid E,\hat X) \le P_e\log(|\mathcal X|-1)$.

Therefore $H(X\mid\hat X) \le H(P_e) + P_e\log(|\mathcal X|-1)$. Finally, since $X\to Y\to\hat X$, the data-processing inequality gives $I(X;\hat X)\le I(X;Y)$, i.e. $H(X\mid\hat X)\ge H(X\mid Y)$. Chaining yields
$$H(P_e) + P_e\log(|\mathcal X|-1) \ge H(X\mid\hat X) \ge H(X\mid Y). \qquad\blacksquare$$

The weak form follows from $H(P_e)\le 1$ and $\log(|\mathcal X|-1)\le\log|\mathcal X|$. (If $\hat X$ ranges over a different/larger alphabet than $X$, replace $|\mathcal X|-1$ by $|\mathcal X|$ throughout.)

**Sharpness.** With no observation and $X$ over $\{1,\dots,m\}$ with mode probability $1-P_e$, the distribution $\big(1-P_e,\ \tfrac{P_e}{m-1},\dots,\tfrac{P_e}{m-1}\big)$ gives $H(X) = H(P_e) + P_e\log(m-1)$, meeting the bound with equality. The constants cannot be improved in general.

## Corollary (converse to the channel coding theorem)

For a discrete memoryless channel of capacity $C = \max_{p(x)} I(X;Y)$, with message $W$ uniform on $\{1,\dots,2^{nR}\}$ and $W\to X^n\to Y^n\to\hat W$, Fano gives $H(W\mid\hat W)\le 1 + P_e^{(n)}\,nR$. Then
$$nR = H(W) = H(W\mid\hat W) + I(W;\hat W) \le 1 + P_e^{(n)}nR + I(X^n;Y^n) \le 1 + P_e^{(n)}nR + nC,$$
using data processing and single-letterization $I(X^n;Y^n)\le\sum_i I(X_i;Y_i)\le nC$. Dividing by $n$ and letting $n\to\infty$ (so $P_e^{(n)}\to 0$ for any good code) forces $R\le C$; equivalently $P_e^{(n)}\ge 1 - C/R - 1/(nR)$, bounded away from $0$ whenever $R>C$.

## Corollary (minimax lower bounds — Fano's method)

For estimating $\theta$ under a metric loss $d$, take an $\epsilon$-separated set $\{\theta_1,\dots,\theta_M\}$, let $\theta$ be uniform over it, observe data $Z$, and round any estimator to the nearest hypothesis $\tilde\theta$. Fano in the form $P_e \ge 1 - \frac{I(\theta;Z)+\log 2}{\log M}$ gives, via the triangle inequality, a constant lower bound on $\Pr(d(\theta,\hat\theta)\ge\epsilon/2)$ whenever $I(\theta;Z)$ is small relative to $\log M$. Therefore $\mathbb E d(\theta,\hat\theta)$ is bounded below by a constant multiple of $\epsilon$, and so is the minimax risk over the original parameter space. This is the Le Cam/Tsybakov tradition of information-theoretic statistical lower bounds.

## Numerical sanity check

```python
import numpy as np

def entropy(p, base=2):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * (np.log(p)/np.log(base))).sum())

def conditional_entropy(joint, base=2):     # H(X|Y), joint[x,y]=p(x,y)
    joint = np.asarray(joint, float); py = joint.sum(0); h = 0.0
    for j, pyj in enumerate(py):
        if pyj > 0: h += pyj * entropy(joint[:, j]/pyj, base)
    return h

def map_rule(joint):                        # best (MAP) estimator x*(y)
    return np.argmax(np.asarray(joint, float), axis=0)

def prob_error(joint, g):                   # P_e of estimator g(Y)
    joint = np.asarray(joint, float)
    return float(sum(joint[:, j].sum() - joint[g[j], j] for j in range(joint.shape[1])))

def fano_floor(joint, base=2):              # (H(X|Y) - 1)/log|X|
    X = np.asarray(joint, float).shape[0]
    if X < 2:
        return 0.0
    weak = (conditional_entropy(joint, base) - 1.0) / (np.log(X)/np.log(base))
    return max(0.0, weak)

J = np.array([[0.30, 0.05, 0.05],
              [0.05, 0.20, 0.05],
              [0.05, 0.05, 0.20]])
assert prob_error(J, map_rule(J)) >= fano_floor(J) - 1e-12   # even the best estimator obeys the floor
```
