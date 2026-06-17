# Cycle-Following Strategy

Identify the box assignment with a permutation $\sigma$ of $\{1,\ldots,100\}$, where box $j$ contains number $\sigma(j)$.

The optimal strategy is cycle-following:

1. Prisoner $p$ first opens box $p$.
2. If that box contains number $q$, the prisoner next opens box $q$.
3. The prisoner keeps using the number just found as the label of the next box, stopping after at most $50$ boxes.

If the cycle of $\sigma$ containing $p$ has length $L$, then the boxes opened by prisoner $p$ are

$$
p,\sigma(p),\sigma^2(p),\ldots,\sigma^{L-1}(p).
$$

The box containing $p$ is the predecessor of $p$ on that cycle, namely $\sigma^{L-1}(p)$. Therefore prisoner $p$ finds their number on the $L$-th opening. Hence

$$
\text{prisoner }p\text{ succeeds} \iff L \le 50.
$$

All prisoners are freed if and only if every cycle of $\sigma$ has length at most $50$, equivalently if and only if the longest cycle is at most $50$.

A permutation of $100$ elements has at most one cycle of length greater than $50$. For fixed $k \in \{51,\ldots,100\}$, the number of permutations with a cycle of length exactly $k$ is

$$
\binom{100}{k}(k-1)!(100-k)! = \frac{100!}{k}.
$$

Therefore

$$
\Pr(\text{there is a cycle of length }k)=\frac{1}{k}.
$$

Since the long-cycle events are disjoint,

$$
\Pr(\text{failure})
= \Pr(\text{longest cycle}>50)
= \sum_{k=51}^{100}\frac{1}{k}
= H_{100}-H_{50}
\approx 0.6881721793.
$$

So the success probability is

$$
\boxed{
\Pr(\text{all freed})
= 1-(H_{100}-H_{50})
\approx 0.3118278207
}.
$$

For $2n$ prisoners allowed to open $n$ boxes each, the same formula gives

$$
\Pr(\text{success})=1-(H_{2n}-H_n).
$$

Since $H_{2n}-H_n \to \ln 2$, the success probability tends to

$$
1-\ln 2 \approx 0.3068528194.
$$

The finite success probabilities decrease monotonically to this limit, so the cycle-following success probability stays above $30\%$ for every $n$.

The cycle-following strategy is optimal. In the relaxed variant where opened boxes remain visible to later prisoners, a prisoner whose number is already visible succeeds immediately, and a prisoner stops once their number is found, each fresh opening samples uniformly from the still-unseen numbers. Thus the whole-group success probability in that relaxed game is independent of the rule used to choose fresh boxes, and the relaxed game upper-bounds the original game. Cycle-following attains the same success event in the original game as in the relaxed game: no cycle longer than $50$. Therefore no strategy can beat $1-(H_{100}-H_{50})$ for the $100$-prisoner case.
