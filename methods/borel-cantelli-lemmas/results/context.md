# Context: from single-event probabilities to statements that hold forever

## Research question

A probability assigns a number $P(A)\in[0,1]$ to a single event $A$. The pressing questions at the foundations of the subject, though, are not about one event but about a whole *sequence* of them: I have events $A_1, A_2, A_3, \dots$, one for each stage $n$, and I want to know what happens *in the long run* along a single random outcome $\omega$. Does the property $A_n$ keep recurring forever, or does it eventually stop happening for good?

Concretely: think of $A_n$ as "something bad happens at stage $n$" — the $n$-th digit of a random real deviates from its expected frequency, the running average is still more than $\varepsilon$ away from its mean at step $n$, the $n$-th trial sets a new record. The question I care about is *for almost every outcome*, does $A_n$ happen only **finitely often**, or **infinitely often**? That is a statement about the *entire infinite tail* of the sequence and about a fixed $\omega$ — not about any single $n$.

The tools of the day answer a weaker question. The weak law of large numbers says $P(|\bar X_n - \mu| > \varepsilon) \to 0$: for each large $n$ the *probability* of a deviation is small. But that says nothing about whether, for a *fixed* random $\omega$, the deviations eventually stop. A sequence can have $P(A_n)\to 0$ yet still hit $A_n$ infinitely often along almost every path. What is missing is a bridge: given the sequence of numbers $\{P(A_n)\}$ — and possibly some structural fact about how the $A_n$ relate — what can I conclude about the **probability that infinitely many $A_n$ occur**? The goal is a criterion, phrased purely in terms of $\{P(A_n)\}$, that pins this probability down. Whether such a criterion exists, what summary of the sequence $\{P(A_n)\}$ it should hinge on, and what structural hypothesis (if any) it needs, is the open problem.

## Background

The setting is a probability space $(\Omega, \mathcal F, P)$: a sample space $\Omega$, a $\sigma$-field $\mathcal F$ of events closed under countable unions and complements, and a countably additive measure $P$ with $P(\Omega)=1$. The load-bearing facts about $P$ that are already in hand:

- **Monotonicity and subadditivity.** If $A\subseteq B$ then $P(A)\le P(B)$. For any countable collection, $P\!\left(\bigcup_k A_k\right) \le \sum_k P(A_k)$ — the union bound. This is unconditional: it needs nothing about how the $A_k$ relate.
- **Continuity along monotone sequences.** If $B_1\supseteq B_2\supseteq\cdots$ then $P\!\left(\bigcap_m B_m\right) = \lim_{m\to\infty} P(B_m)$ ("continuity from above"); dually for increasing sequences ("from below"). This is exactly countable additivity restated, and it is what lets a probability of a limiting set be computed as a limit of probabilities.
- **Independence.** Events $A_1,\dots,A_k$ are independent if $P(A_{i_1}\cap\cdots\cap A_{i_j}) = \prod P(A_{i_\ell})$ for every sub-collection; a whole sequence is (mutually) independent if every finite sub-collection is. A standard fact: if $A_1,\dots,A_k$ are independent then so are their complements $A_1^c,\dots,A_k^c$. Independence is the structural hypothesis that makes intersection-probabilities *factor* into products.
- **Markov / Chebyshev.** For a nonnegative random variable $Y$ and $t>0$, $P(Y\ge t)\le EY/t$; applied to $(S-ES)^2$ this gives Chebyshev, $P(|S-ES|\ge t)\le \operatorname{Var}(S)/t^2$. For a sum of *uncorrelated* variables, the variance of the sum is the sum of the variances.
- **The behavior of a series.** A nonnegative series $\sum_n p_n$ either converges (finite sum) or diverges to $+\infty$. If it converges, its *tail* $\sum_{k\ge n} p_k \to 0$ as $n\to\infty$. This convergent/divergent split is the simplest possible binary fact attached to a sequence of probabilities.
- **Expectation of a counting variable.** For indicator variables $1_{A_k}$ (equal to $1$ on $A_k$, $0$ off it), $E\,1_{A_k} = P(A_k)$, and by countable additivity of the integral (Tonelli, for nonnegative terms) $E\sum_k 1_{A_k} = \sum_k P(A_k)$. A nonnegative random variable with finite expectation is finite almost surely.

A motivating arithmetic fact sets the stakes. Borel had shown that "almost every" real number in $[0,1]$ is *normal* — each digit, and each finite digit-block, occurs with its expected asymptotic frequency. That is precisely a statement that a sequence of "bad" events (the empirical digit-frequency is still off at stage $n$) occurs only finitely often for almost every $\omega$. Number-theoretic almost-everywhere statements of this kind are the natural pressure forcing the question above: they cannot even be *phrased* with single-event probabilities or with convergence-in-probability; they live on the infinite tail.

There is also a cautionary observed phenomenon about sequences of events. Take $\Omega=(0,1)$ with Lebesgue measure and the *nested* events $A_n=(0,1/n)$. The probabilities $P(A_n)=1/n$ sum to $+\infty$, so the events are "large on average and never stop being sizable." Yet they shrink toward the single point-free set $\bigcap_n(0,1/n)=\varnothing$: no $\omega$ lies in infinitely many of them. So a divergent $\sum P(A_n)$ by itself is consistent with the bad event happening *not even once eventually*. The mass of these events is piled onto the same outcomes — they overlap as much as possible. This is a concrete warning that a divergence criterion, if one exists, cannot be unconditional: the same divergent sum is compatible with opposite long-run verdicts depending on how the events overlap.

## Baselines

The prior art is the toolkit just described, used as far as it reaches.

**The weak law of large numbers (Bernoulli; Chebyshev's proof).** For i.i.d. $X_i$ with finite variance, $P(|\bar X_n - \mu|>\varepsilon)\le \sigma^2/(n\varepsilon^2)\to 0$. Core idea: Chebyshev on the average, whose variance is $\sigma^2/n$. What it delivers: for each large $n$, the deviation is improbable. Where it stalls: it is a statement about the *marginal* distribution of $\bar X_n$ at a single $n$. It says nothing about a fixed path $\omega$ — whether the deviations along that path eventually cease. Convergence in probability does not, by itself, control the joint behavior across all $n$ at once.

**The union bound / subadditivity.** $P\!\left(\bigcup_{k\ge n} A_k\right)\le \sum_{k\ge n}P(A_k)$. Core idea: probability is countably subadditive. What it delivers: an upper bound on the chance that *some* event from stage $n$ onward occurs. Where it stalls on its own: it is only an inequality in one direction and gives no *lower* bound, so it can show certain tail events are negligible but can never, unaided, show that a tail event is *certain*.

**Continuity of measure along monotone limits.** Core idea: countable additivity lets $P$ pass through increasing/decreasing limits of sets. What it delivers: the probability of a limiting set is the limit of the probabilities. Where it stalls on its own: it is purely a bookkeeping device — it tells you how to evaluate $P$ on a limit *once you can evaluate the finite stages*, but supplies no estimate of those stages.

**Independence as a factorization rule.** Core idea: for independent events the probability of an intersection is the product of the probabilities. What it delivers: it turns the joint probability of "$A_{n}^c$ and $A_{n+1}^c$ and … and $A_N^c$" into $\prod_{k=n}^N (1-P(A_k))$. Where it stalls on its own: a finite or infinite product of factors each just below $1$ is awkward to evaluate directly — whether $\prod_k(1-p_k)$ is $0$ or strictly positive is not read off by inspection, and naive bounding can lose the answer.

**Higher-moment tail control (Markov on $(S-ES)^{2m}$).** Core idea: a fourth-moment bound $ES_n^4\le Cn^2$ gives $P(|S_n|>n\varepsilon)\le C/(n^2\varepsilon^4)$, a probability that *decays fast in $n$*. What it delivers: per-$n$ deviation probabilities small enough that their *sum over $n$* is finite. Where it stalls: again it is a per-$n$ statement; on its own it does not convert "the summed deviation probabilities are finite" into a guarantee about a single path. The fast decay is suggestive — it is exactly the kind of quantity whose sum being finite ought to mean *something* pathwise — but the conversion is the missing step.

The common gap across all of these: each controls the behavior at a single stage $n$, or evaluates a limit once the stages are known, but none crosses from the sequence of per-stage numbers $\{P(A_n)\}$ to the **probability of the tail event "$A_n$ happens infinitely often."**

## Evaluation settings

There is no benchmark dataset here; the natural yardsticks are whether the criterion (i) correctly classifies the standard test sequences and (ii) drives the canonical almost-sure theorems. The settings that would exercise it:

- **The nested cautionary sequence** $\Omega=(0,1)$, Lebesgue, $A_n=(0,1/n)$: divergent $\sum P(A_n)$ but $\limsup A_n=\varnothing$. Any candidate divergence-criterion must respect this (it must *not* claim "infinitely often" here).
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
