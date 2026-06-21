We are asked to guess a random variable X from a correlated observation Y, forming an estimator X-hat = g(Y). The error probability is P_e = Pr(X-hat != X). At the boundary, we already know that H(X|Y) = 0 if and only if X is a deterministic function of Y, which means zero residual uncertainty is equivalent to zero error. But that is only an all-or-nothing statement. In the much more common interior regime, H(X|Y) is small but positive, and we want a quantitative law that says leftover uncertainty forces a nonzero error floor no matter which g we choose. Pairwise hypothesis-testing bounds can show that two specific distributions are hard to distinguish, but they do not cleanly aggregate over many hypotheses and they do not express the error of every estimator through a single conditional-entropy quantity. Variance-based bounds like Cramer-Rao are tied to quadratic loss and smooth parameters, so they do not speak directly to the discrete guessing problem. What is missing is a universal converse: a lower bound on P_e in terms of H(X|Y) alone.

The method is Fano's inequality. It gives exactly that converse. Suppose X takes values in an alphabet of size M = |X| >= 2 and the guess X-hat also takes values in X. Let P_e = Pr(X-hat != X) and let H(p) = -p log p - (1-p) log(1-p) denote the binary entropy, with logarithms in base 2. Then Fano's inequality states

H(P_e) + P_e log(M - 1) >= H(X|X-hat) >= H(X|Y).

The right-hand inequality follows because X -> Y -> X-hat is a Markov chain, so data processing gives I(X; X-hat) <= I(X; Y) and therefore H(X|X-hat) >= H(X|Y). The left-hand inequality is the heart of the result. It converts the probability of error into an entropy via the error indicator E = 1{X-hat != X}, whose entropy is exactly H(P_e). The proof introduces E for free, expands the joint conditional entropy H(E, X | X-hat) two ways using the chain rule, and bounds each piece. One expansion gives H(X|X-hat) because E is determined once X and X-hat are known. The other expansion gives H(E|X-hat) + H(X|E, X-hat). Conditioning reduces entropy, so H(E|X-hat) <= H(E) = H(P_e). When E = 0 there is no uncertainty in X, and when E = 1 the true X lies among the M - 1 symbols different from the guess, contributing at most log(M - 1). Hence H(X|E, X-hat) <= P_e log(M - 1). Putting the pieces together yields the bound.

The most commonly used operational form is weaker but explicit. Since H(P_e) <= 1 and log(M - 1) <= log M, we obtain 1 + P_e log M >= H(X|Y), or equivalently

P_e >= (H(X|Y) - 1) / log M,

together with the trivial P_e >= 0 when the numerator is negative. This says that residual conditional entropy, measured in bits minus one, divided by the log alphabet size, is a hard floor on the error of every estimator. The constants are sharp in general: with no observation, the distribution that puts mass 1 - P_e on the mode and spreads the remaining mass uniformly over the other M - 1 symbols meets the bound with equality.

Fano's inequality is the standard engine behind two major kinds of converses. In channel coding, a message W uniform over 2^{nR} values satisfies H(W) = nR. Applying Fano to W -> X^n -> Y^n -> W-hat gives H(W|W-hat) <= 1 + P_e^{(n)} nR. Combining this with data processing and single-letterization forces nR <= 1 + P_e^{(n)} nR + nC, where C is channel capacity. Dividing by n and sending n to infinity shows that any sequence of codes with vanishing error must have R <= C. Thus rates above capacity are impossible. In statistics, Fano's method gives minimax lower bounds: pick an epsilon-separated set of M parameters, make theta uniform over them, observe data Z, and round any estimator to the nearest parameter. If the mutual information I(theta; Z) is small relative to log M, the testing error is bounded below by a constant, which translates into a lower bound on the estimation risk of order epsilon. The same inequality turns an information budget into an error floor in both settings.

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

def map_rule(joint):
    # best (MAP) estimator: for each y, pick the x with largest p(x, y)
    return np.argmax(np.asarray(joint, dtype=float), axis=0)

def prob_error(joint, g):
    # P_e of estimator g(Y)
    joint = np.asarray(joint, dtype=float)
    return float(sum(joint[:, j].sum() - joint[g[j], j] for j in range(joint.shape[1])))

def fano_floor(joint, base=2):
    # weak operational form: P_e >= (H(X|Y) - 1) / log|X|
    joint = np.asarray(joint, dtype=float)
    m = joint.shape[0]
    if m < 2:
        return 0.0
    h_xy = conditional_entropy(joint, base)
    weak = (h_xy - 1.0) / (np.log(m) / np.log(base))
    return max(0.0, weak)

# Example: the MAP estimator is the best possible, yet it still obeys Fano's floor.
J = np.array([[0.30, 0.05, 0.05],
              [0.05, 0.20, 0.05],
              [0.05, 0.05, 0.20]])
map_g = map_rule(J)
print("P_e (MAP):", prob_error(J, map_g))
print("Fano floor:", fano_floor(J))
assert prob_error(J, map_g) >= fano_floor(J) - 1e-12
```
