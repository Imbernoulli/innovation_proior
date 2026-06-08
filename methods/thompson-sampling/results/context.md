# Context

## Research question

You are running a sequence of trials — patients arriving one at a time for one of two treatments, say — and the two treatments have *unknown* success probabilities. You cannot wait until you have collected a large, statistically comfortable sample before you act: each arriving individual must be assigned to *some* treatment *now*, and that assignment is itself one of your few data points. The data is meagre precisely because acting and learning are the same stream of events.

The precise problem: given the evidence of two small samples — of $n_1$ trials of the first treatment, $r_1$ succeeded and $s_1$ failed; of $n_2$ of the second, $r_2$ and $s_2$ — how should the *next* individuals be allocated between the two treatments, so as to lose as few of them as possible to the inferior treatment over the whole future course of the experiment? This is a question about *action under uncertainty from small data*, not about estimating a parameter to a tolerance. It matters most exactly when the classical advice ("collect more before you analyse") is most costly: when the individuals being treated are valuable, or the rate of data accumulation is slow, or both. A solution must (i) extract whatever guidance meagre data can give, (ii) keep the inferior treatment from sacrificing a stream of future individuals, and (iii) never freeze prematurely on a choice the data does not yet warrant.

## Background

**Probability of one unknown exceeding another.** The natural quantity to judge two treatments by is the probability $P$ that one treatment is better than the other *given the data so far*. If each unknown success probability $p_i$ is treated as a random quantity with a prior, and we observe successes and failures, Bayes' theorem ("the well-known Principle of Bayes") turns the prior into a posterior over $p_i$. With the natural state of ignorance — each $p_i$ *a priori* equally likely to lie in any two equal sub-intervals of $(0,1)$, i.e. a uniform prior — the posterior density after $r$ successes and $s$ failures in $n=r+s$ trials is $(n+1)!/(r!\,s!)\,p^{r}(1-p)^{s}$, a $\mathrm{Beta}(r+1,s+1)$ density. This is exactly the integrand of the incomplete Beta function.

**Pearson's incomplete-Beta machinery.** The tail probabilities of such densities — $\int_p^1$ of $p^{r}q^{s}$ — were already standard equipment. Karl Pearson had tabulated and studied the incomplete Beta function $I_p(u,v)=B_p(u,v)/B(u,v)$ (Phil. Mag. 1907; Biometrika 1924, 1928), and J. H. Müller (Biometrika 1930–31) had set out the $I_p(u,v)$ ratio notation with $u=r+1,\,v=s+1$. Pearson had also shown that such Beta tails equal the sum of the first several terms of a hypergeometric series, and had solved the related finite-urn sampling problem (probability that an urn of $N=R+S$ members contains no more than $R$ marked, given a sample of $r$ marked and $s$ unmarked). The earliest work in this direction is catalogued by Todhunter's *History of the Mathematical Theory of Probability* (1865). So the apparatus to compute "probability that one Beta-distributed quantity exceeds another, or a threshold" existed in pieces; what was missing was a *reduced, exact, hand-computable* evaluation of the probability that one unknown success probability exceeds a *second* unknown one, from two independent samples.

**Computation by hand.** Any such evaluation has to be tabulable. The relevant fact is the binomial-coefficient "pyramid" (Pascal's triangle): each interior entry is the sum of the two nearest entries in the row above (Glaisher 1917 tabulated these). Recurrences of this additive kind are what make a probability table buildable without a computer.

**The cost of acting on $P$.** Two crude disciplines are on the table for *using* such a probability to act. The "alternate case method" splits assignment evenly between the treatments until a decision is made. The opposite is to decide immediately and irrevocably for the apparently-better treatment. The latter has a clear failure mode: if the apparently-better treatment is in fact the worse, *every* future individual pays the full gap between the treatments — an expected sacrifice of $(1-P)$ per future individual, forever. With small samples $P$ is genuinely uncertain, so an irrevocable decision risks a permanent stream of avoidable losses. This observation frames everything: with meagre data, *how confidently* you act should track *how much* the data actually warrants.

## Baselines

- **Immediate irrevocable decision (the "make up your mind" rule).** Compute which treatment looks better and give *all* subsequent individuals that treatment. *Core math:* pick $\arg\max$ of the point estimates; expected per-individual sacrifice $=1-P$ if you have chosen the apparently-better arm. *Gap:* it throws away the uncertainty entirely. When $P$ is far from $1$ — exactly the small-sample regime — you commit a whole future population to a possibly-inferior treatment with no mechanism to recover.

- **The alternate case method (even split).** Assign individuals to the two treatments in equal proportion until enough data accrues to decide. *Core math:* fixed $1/2,1/2$ allocation regardless of the evidence. *Gap:* it ignores the evidence while it is being collected — it keeps sacrificing half of all individuals to the inferior treatment even once the data already points strongly one way. It is wasteful in precisely the cases where individuals are valuable.

- **Large-sample / "collect-then-analyse" statistics.** Defer action, gather a large sample, then apply asymptotic approximations to $P$. *Core math:* the many available large-$n$ approximations to incomplete-Beta tails (Pearson's series). *Gap:* it is inapplicable when you must act *before* a large sample exists, and "bounds to approximation have not been considered generally" for the small-sample case — so even the approximations come without error control where you most need them.

- **Hypothesis testing on the samples.** Use a significance criterion to decide whether the two probabilities differ. *Gap:* a test answers "are they distinguishable?", not "how should I allocate the next individuals?". A binary accept/reject discards the graded probability $P$ that an allocation rule could exploit, and it still demands a decision threshold rather than an action policy.

## Evaluation settings

The natural setting is *sequential allocation between two treatments with unknown binary outcomes* — the clinical-trial / research-planning scenario where individuals arrive over time, each is assigned to one of two treatments, and a "critical event" (success/failure) is observed per individual. The relevant data are the per-arm counts $(r_i,s_i)$ of successes and failures; the regime of interest is *small samples*, with the values $r_i,s_i$ small enough that exact hand tabulation is still plausible. The yardstick of a good allocation discipline is the *number of individuals sacrificed to the inferior treatment over the future course of the experiment* — i.e. cumulative avoidable loss — not the accuracy of a final point estimate. Over a long run with a real preference between treatments, the comparison is against the even-split and the immediate-decision disciplines above.

## Code framework

Sequential allocation from binary outcomes already has the basic bookkeeping: counting successes/failures per arm, computing a posterior over a single unknown probability from a uniform prior (Beta-shaped, via the incomplete-Beta apparatus), and the additive (Pascal-pyramid) recurrences for tabulating coefficients by hand. The open slot is the rule that turns the two arms' current evidence into the assignment of the next individual.

```python
import math

class Arm:
    """Tracks successes r and failures s for one treatment."""
    def __init__(self):
        self.r = 0   # successes (critical event occurred)
        self.s = 0   # failures
    def update(self, success: bool):
        if success: self.r += 1
        else:       self.s += 1
    def n(self):
        return self.r + self.s

def posterior_density(p, r, s):
    # uniform prior on (0,1); posterior of an unknown probability after
    # r successes, s failures.  Normaliser (n+1)!/(r! s!).
    n = r + s
    norm = math.factorial(n + 1) / (math.factorial(r) * math.factorial(s))
    return norm * (p ** r) * ((1 - p) ** s)

def prob_exceeds(r1, s1, r2, s2):
    # The probability that the second unknown probability exceeds the first,
    # given two independent samples.  This is the quantity we lack a clean,
    # hand-computable, exact evaluation of.
    pass  # TODO: derive the reduced exact formula

def allocate_next(arm1: Arm, arm2: Arm):
    # Decide which treatment the NEXT individual receives, given the evidence.
    pass  # TODO: the allocation rule

def run(stream):
    arm1, arm2 = Arm(), Arm()
    for individual in stream:
        choice = allocate_next(arm1, arm2)   # TODO
        outcome = individual.treat(choice)   # observe success / failure
        (arm1 if choice == 1 else arm2).update(outcome)
    return arm1, arm2
```
