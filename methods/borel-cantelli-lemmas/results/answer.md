# The Borel–Cantelli lemmas

## Problem

Given a sequence of events $A_1,A_2,\dots$ in a probability space $(\Omega,\mathcal F,P)$, decide — from the marginal probabilities $\{P(A_n)\}$ alone — whether, almost surely, **infinitely many** of the $A_n$ occur or only **finitely many**. The carrier of the question is the tail event
$$\{A_n \text{ i.o.}\} \;=\; \limsup_{n\to\infty} A_n \;=\; \bigcap_{n=1}^{\infty}\bigcup_{k=n}^{\infty} A_k \;=\; \{\omega : \omega\in A_k \text{ for infinitely many } k\}.$$

## Key idea

A dichotomy keyed on a single series $\sum_n P(A_n)$:

- **Convergence** of the series forces $P(\limsup A_n)=0$, with no assumption on how the events relate (the expected number of occurrences $\sum_n P(A_n)$ is finite, so the count is finite a.s.).
- **Divergence** of the series forces $P(\limsup A_n)=1$ **when the events are independent**; the inequality $1-x\le e^{-x}$ turns the product $\prod(1-P(A_k))$ from the complement into $e^{-\sum P(A_k)}\to 0$. Independence is essential — divergence alone does not suffice.

---

## First lemma (no independence)

**Theorem.** If $\displaystyle\sum_{n=1}^{\infty}P(A_n)<\infty$, then $P\!\left(\limsup_{n\to\infty}A_n\right)=0$ — almost surely only finitely many $A_n$ occur.

**Proof (tail of the union).** The sets $B_n=\bigcup_{k=n}^{\infty}A_k$ decrease to $\limsup_n A_n$, so by continuity from above and the union bound,
$$P\!\left(\limsup_n A_n\right)=\lim_{n\to\infty}P(B_n)\le \lim_{n\to\infty}\sum_{k=n}^{\infty}P(A_k)=0,$$
the last limit being $0$ because the tail of a convergent series vanishes. $\quad\blacksquare$

**Proof (counting occurrences).** Let $N=\sum_{k=1}^{\infty}1_{A_k}$ be the number of events that occur. By Tonelli, $E\,N=\sum_k P(A_k)<\infty$, so $N<\infty$ a.s. $\quad\blacksquare$

## Second lemma (independence)

**Theorem.** If the $A_n$ are independent and $\displaystyle\sum_{n=1}^{\infty}P(A_n)=\infty$, then $P\!\left(\limsup_{n\to\infty}A_n\right)=1$ — almost surely infinitely many $A_n$ occur.

**Proof.** It suffices to show $P\!\left(\bigcap_{k=n}^{\infty}A_k^c\right)=0$ for every $n$, since $\{A_n\text{ i.o.}\}^c=\bigcup_n\bigcap_{k\ge n}A_k^c$ is then a countable union of null sets. The $A_k^c$ are independent, so for $N\ge n$,
$$P\!\left(\bigcap_{k=n}^{N}A_k^c\right)=\prod_{k=n}^{N}\bigl(1-P(A_k)\bigr)\le \prod_{k=n}^{N}e^{-P(A_k)}=\exp\!\left(-\sum_{k=n}^{N}P(A_k)\right),$$
using $1-x\le e^{-x}$ ($x\in\mathbb R$, equality only at $0$). As $N\to\infty$ the exponent $\to-\infty$ (the tail of a divergent series), so by continuity from above $P\!\left(\bigcap_{k=n}^{\infty}A_k^c\right)=0$. Hence $P(\limsup_n A_n)=1$. $\quad\blacksquare$

**Necessity of independence.** On $\Omega=(0,1)$ with Lebesgue measure, the nested events $A_n=(0,1/n)$ have $\sum P(A_n)=\sum 1/n=\infty$, yet $\limsup_n A_n=\bigcap_n(0,1/n)=\varnothing$, so $P(\limsup A_n)=0$. Divergence alone is not enough; the events here pile all their mass on the same outcomes. The line that fails is the factorization — $P(\bigcap_{k=n}^N A_k^c)=1-1/N\not=\prod(1-1/k)$.

---

## Sharpenings (same engine)

**Pairwise-independence / quantitative count.** If the $A_m$ are pairwise independent and $\sum_m P(A_m)=\infty$, then with $S_n=\sum_{m=1}^n 1_{A_m}$,
$$\frac{S_n}{\sum_{m=1}^n P(A_m)}\xrightarrow{\;\text{a.s.}\;}1.$$
*Proof.* Uncorrelatedness gives $\operatorname{Var}(S_n)=\sum_m\operatorname{Var}(1_{A_m})\le \sum_m P(A_m)=ES_n$, so Chebyshev yields $P(|S_n-ES_n|>\delta ES_n)\le 1/(\delta^2 ES_n)\to 0$ ($S_n/ES_n\to1$ in probability). On the subsequence $n_k=\inf\{n:ES_n\ge k^2\}$, $T_k=S_{n_k}$ satisfies $k^2\le ET_k\le k^2+1$ and $P(|T_k-ET_k|>\delta ET_k)\le 1/(\delta^2k^2)$, which is summable; the first lemma gives $T_k/ET_k\to1$ a.s., and monotone sandwiching with $ET_{k+1}/ET_k\to1$ upgrades to $S_n/ES_n\to1$ a.s. Since $ES_n\to\infty$, infinitely-often follows. $\quad\blacksquare$

**Kochen–Stone (second-moment lower bound).** If $\sum_k P(A_k)=\infty$, then by Cauchy–Schwarz on $S_n=S_n 1_{\{S_n>0\}}$, $(ES_n)^2\le ES_n^2\,P(S_n>0)$, so
$$P\!\left(\limsup_n A_n\right)\ge \limsup_{n\to\infty}\frac{\bigl(\sum_{k=1}^n P(A_k)\bigr)^2}{\sum_{1\le j,k\le n}P(A_j\cap A_k)}.$$
The independent second lemma is the corner where this ratio $\to 1$.

## Application — strong law under a fourth moment

For i.i.d. $X_i$ with $EX_i=\mu$ and $EX_i^4<\infty$, $S_n=\sum_{i\le n}X_i$: assume $\mu=0$; expanding and discarding mean-zero terms, $ES_n^4=nEX_1^4+3n(n-1)(EX_1^2)^2\le Cn^2$, so $P(|S_n|>n\varepsilon)\le C/(n^2\varepsilon^4)$ is summable. The first lemma gives $|S_n|\le n\varepsilon$ eventually for every $\varepsilon$, i.e. $S_n/n\to\mu$ a.s. Conversely, if $E|X_i|=\infty$ then $\sum_n P(|X_n|\ge n)=\infty$ and the second lemma gives $|X_n|\ge n$ i.o., which precludes $S_n/n$ converging to a finite limit — a finite mean is necessary for the strong law.

---

## Numerical illustration

Pure theorems; the code only *exhibits* the dichotomy on sampled trajectories of independent events (it proves nothing).

```python
import math, random

def last_occurrence_and_count(prob_fn, N, seed=0):
    """One realization of independent events A_1..A_N with P(A_n)=prob_fn(n).
    Returns (number that occurred, largest index that occurred)."""
    rng = random.Random(seed)
    occurred, last = 0, 0
    for n in range(1, N + 1):
        if rng.random() < prob_fn(n):          # draw 1_{A_n} ~ Bernoulli(P(A_n))
            occurred += 1
            last = n
    return occurred, last

def running_frequency(prob_fn, N, seed=0):
    """S_n / E[S_n] for the divergent profile (S_n / sum_{k<=n} P(A_k) -> 1)."""
    rng = random.Random(seed)
    S, ES, snap = 0.0, 0.0, {}
    for n in range(1, N + 1):
        p = prob_fn(n)
        ES += p                                 # E[S_n] = sum_{k<=n} P(A_k)
        if rng.random() < p:
            S += 1
        if n in (1000, 10000, 100000, N):
            snap[n] = S / ES
    return snap

if __name__ == "__main__":
    # hinge of the divergent half: 1 - x <= e^{-x}, i.e. (1-x) - e^{-x} <= 0
    worst = max((1 - x) - math.exp(-x) for x in [i / 100 for i in range(101)])
    print(f"max of (1-x) - e^(-x) on [0,1]: {worst:.1e}   (<= 0 confirms 1-x <= e^(-x))")

    N = 200000
    summable  = lambda n: 1.0 / (n * n)         # sum 1/n^2 < inf  -> finitely often
    divergent = lambda n: 1.0 / n               # sum 1/n  = inf, indep -> infinitely often
    oc, lc = last_occurrence_and_count(summable, N)
    od, ld = last_occurrence_and_count(divergent, N)
    print(f"P(A_n)=1/n^2 : occurrences={oc:6d}, last index seen={lc:7d}   (settles -> finite)")
    print(f"P(A_n)=1/n   : occurrences={od:6d}, last index seen={ld:7d}   (keeps recurring)")
    print("divergent S_n / E[S_n] (should approach 1):",
          {n: round(v, 3) for n, v in sorted(running_frequency(divergent, N).items())})
```

Running it: `max of (1-x) - e^(-x) on [0,1]: 0.0e+00`; the summable profile's last occurrence is at a tiny index (finitely often), while the divergent profile keeps recurring out to large indices with $S_n/E[S_n]$ near $1$.
