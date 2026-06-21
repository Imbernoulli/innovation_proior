# Context: from single-event probabilities to statements that hold forever

## Research question

A probability assigns a number $P(A)\in[0,1]$ to a single event $A$. Many questions at the foundations of the subject are not about one event but about a whole *sequence* of them: events $A_1, A_2, A_3, \dots$, one for each stage $n$, and what happens *in the long run* along a single random outcome $\omega$. Does the property $A_n$ keep recurring forever, or does it eventually stop happening for good?

Concretely: think of $A_n$ as "something bad happens at stage $n$" — the $n$-th digit of a random real deviates from its expected frequency, the running average is still more than $\varepsilon$ away from its mean at step $n$, the $n$-th trial sets a new record. The question is: *for almost every outcome*, does $A_n$ happen only **finitely often**, or **infinitely often**? That is a statement about the *entire infinite tail* of the sequence and about a fixed $\omega$ — not about any single $n$.

The weak law of large numbers says $P(|\bar X_n - \mu| > \varepsilon) \to 0$: for each large $n$ the *probability* of a deviation is small. That is a statement about each marginal distribution, not about a fixed path $\omega$. The broad question here is: given the sequence of numbers $\{P(A_n)\}$ — and possibly some structural fact about how the $A_n$ relate — what can be concluded about the **probability that infinitely many $A_n$ occur**?

## Background

The setting is a probability space $(\Omega, \mathcal F, P)$: a sample space $\Omega$, a $\sigma$-field $\mathcal F$ of events closed under countable unions and complements, and a countably additive measure $P$ with $P(\Omega)=1$. The load-bearing facts about $P$ already in hand:

- **Monotonicity and subadditivity.** If $A\subseteq B$ then $P(A)\le P(B)$. For any countable collection, $P\!\left(\bigcup_k A_k\right) \le \sum_k P(A_k)$ — the union bound. This is unconditional: it needs nothing about how the $A_k$ relate.
- **Continuity along monotone sequences.** If $B_1\supseteq B_2\supseteq\cdots$ then $P\!\left(\bigcap_m B_m\right) = \lim_{m\to\infty} P(B_m)$ ("continuity from above"); dually for increasing sequences ("from below"). This is countable additivity restated, and it lets a probability of a limiting set be computed as a limit of probabilities.
- **Independence.** Events $A_1,\dots,A_k$ are independent if $P(A_{i_1}\cap\cdots\cap A_{i_j}) = \prod P(A_{i_\ell})$ for every sub-collection; a whole sequence is (mutually) independent if every finite sub-collection is. A standard fact: if $A_1,\dots,A_k$ are independent then so are their complements $A_1^c,\dots,A_k^c$. Independence makes intersection-probabilities *factor* into products.
- **Markov / Chebyshev.** For a nonnegative random variable $Y$ and $t>0$, $P(Y\ge t)\le EY/t$; applied to $(S-ES)^2$ this gives Chebyshev, $P(|S-ES|\ge t)\le \operatorname{Var}(S)/t^2$. For a sum of *uncorrelated* variables, the variance of the sum is the sum of the variances.
- **The behavior of a series.** A nonnegative series $\sum_n p_n$ either converges (finite sum) or diverges to $+\infty$. If it converges, its *tail* $\sum_{k\ge n} p_k \to 0$ as $n\to\infty$.
- **Expectation of a counting variable.** For indicator variables $1_{A_k}$ (equal to $1$ on $A_k$, $0$ off it), $E\,1_{A_k} = P(A_k)$, and by countable additivity of the integral (Tonelli, for nonnegative terms) $E\sum_k 1_{A_k} = \sum_k P(A_k)$. A nonnegative random variable with finite expectation is finite almost surely.

A motivating arithmetic fact sets the stakes. Borel had shown that "almost every" real number in $[0,1]$ is *normal* — each digit, and each finite digit-block, occurs with its expected asymptotic frequency. That is a statement that a sequence of "bad" events (the empirical digit-frequency is still off at stage $n$) occurs only finitely often for almost every $\omega$. Number-theoretic almost-everywhere statements of this kind live on the infinite tail; they are phrased neither with single-event probabilities nor with convergence-in-probability.

There is also an observed phenomenon about sequences of events. Take $\Omega=(0,1)$ with Lebesgue measure and the *nested* events $A_n=(0,1/n)$. The probabilities $P(A_n)=1/n$ sum to $+\infty$, so the events are large on average and never stop being sizable. Yet they shrink toward $\bigcap_n(0,1/n)=\varnothing$: no $\omega$ lies in infinitely many of them. So a divergent $\sum P(A_n)$ by itself is compatible with the bad event happening not even once eventually — the mass of these events is piled onto the same outcomes, overlapping as much as possible.

## Baselines

The prior art is the toolkit just described, used as far as it reaches.

**The weak law of large numbers (Bernoulli; Chebyshev's proof).** For i.i.d. $X_i$ with finite variance, $P(|\bar X_n - \mu|>\varepsilon)\le \sigma^2/(n\varepsilon^2)\to 0$. Core idea: Chebyshev on the average, whose variance is $\sigma^2/n$. It delivers: for each large $n$, the deviation is improbable — a statement about the *marginal* distribution of $\bar X_n$ at a single $n$.

**The union bound / subadditivity.** $P\!\left(\bigcup_{k\ge n} A_k\right)\le \sum_{k\ge n}P(A_k)$. Core idea: probability is countably subadditive. It delivers: an upper bound on the chance that *some* event from stage $n$ onward occurs.

**Continuity of measure along monotone limits.** Core idea: countable additivity lets $P$ pass through increasing/decreasing limits of sets. It delivers: the probability of a limiting set is the limit of the probabilities — a way to evaluate $P$ on a limit once the finite stages can be evaluated.

**Independence as a factorization rule.** Core idea: for independent events the probability of an intersection is the product of the probabilities. It delivers: it turns the joint probability of "$A_{n}^c$ and $A_{n+1}^c$ and … and $A_N^c$" into $\prod_{k=n}^N (1-P(A_k))$.

**Higher-moment tail control (Markov on $(S-ES)^{2m}$).** Core idea: a fourth-moment bound $ES_n^4\le Cn^2$ gives $P(|S_n|>n\varepsilon)\le C/(n^2\varepsilon^4)$, a probability that *decays fast in $n$*. It delivers: per-$n$ deviation probabilities small enough that their *sum over $n$* is finite.

## Evaluation settings

There is no benchmark dataset here; the natural yardsticks are whether the criterion (i) correctly classifies the standard test sequences and (ii) drives the canonical almost-sure theorems. The settings that would exercise it:

- **The nested cautionary sequence** $\Omega=(0,1)$, Lebesgue, $A_n=(0,1/n)$: divergent $\sum P(A_n)$ but $\limsup A_n=\varnothing$.
- **Independent coin-type sequences**, $A_n$ independent with prescribed $P(A_n)=p_n$ — e.g. $p_n=1/n^2$ (summable) versus $p_n=1/n$ (divergent) — where the dichotomy can be checked directly.
- **The strong law of large numbers under a moment assumption** (i.i.d. with finite fourth moment), as the headline almost-sure theorem a tail criterion should yield: $S_n/n\to\mu$ for almost every path.
- **Borel's normal numbers** on $([0,1],\text{Lebesgue})$ as the motivating arithmetic almost-everywhere statement.
- **Combinatorial sequences** such as record times and longest head-runs, where one wants almost-sure growth rates ($R_n/\log n$, $L_n/\log_2 n$).

The metric is binary correctness of the predicted $P(A_n\text{ i.o.})$ and the validity of the almost-sure conclusions derived from it.

## Code framework

A small numerical sandbox to *illustrate* (not prove) what a tail criterion predicts: simulate one random trajectory of events $A_1,\dots,A_N$ with chosen marginal probabilities, and watch the long-run behavior — how many of the $A_n$ occur and how far out they keep recurring. Pre-method primitives available: a random-number generator, the chosen probability profile $n\mapsto P(A_n)$, and an indicator draw at each stage. The slot to be filled is the quantity the criterion will turn out to hinge on.

```python
import math, random

def simulate_trajectory(prob_fn, N, seed=0):
    """One realization of independent events A_1..A_N with P(A_n)=prob_fn(n).
    Returns whatever summary of the long-run behavior the criterion will hinge on."""
    rng = random.Random(seed)
    # draw 1_{A_n} ~ Bernoulli(prob_fn(n)) for n = 1..N
    # TODO: accumulate the summary statistic that the tail criterion will be stated in terms of
    pass

def tail_quantity(prob_fn, N):
    """The deterministic quantity, computed from the marginal probabilities alone,
    that the criterion claims decides the long-run behavior."""
    # TODO: the quantity we will define here
    pass

if __name__ == "__main__":
    N = 200000
    profile_a = lambda n: 1.0 / (n * n)
    profile_b = lambda n: 1.0 / n
    # compare the trajectory summary against tail_quantity for each profile
    pass
```

The final code fills `tail_quantity` and the summary accumulated in `simulate_trajectory` with the concrete objects the derivation lands on, and adds the comparison that exhibits the predicted dichotomy on the two profiles above.
