Two treatments in front of me, and the individuals keep arriving. I do not get to run one treatment on a thousand people, run the other on a thousand more, tabulate, and then calmly decide. Each person who walks in has to be put on *some* treatment now, and that assignment is itself a data point. So the data is meagre not because I am lazy about collecting it but because collecting it and acting on it are the same act. And meagre data is exactly where the usual statistical advice — gather more before you analyse — is worst, because the cost of "more" is more individuals given the wrong treatment. I want to think about this as a problem of research planning: what do I *do* with the next person, given what I know so far.

Let me get the crude options on the table first, because they frame what a good rule would have to beat. One option: split everyone fifty-fifty between the two treatments until I feel I have enough to decide. Call it the alternate-case method. The other extreme: look at the data, see which treatment looks better, and from now on give *everyone* that treatment — decide once, irrevocably.

The irrevocable decision is the one that scares me, so let me price it. Suppose I commit to treatment one. Let $P$ be the probability, given the data I have, that treatment one really is the better one. Then with probability $1-P$ I have committed every future individual to the worse treatment, and each of them pays the gap between the treatments. The expected sacrifice is $(1-P)$ per future individual, *forever*. If my samples were huge, $P$ would be near $1$ and this would be fine. But my samples are small, so $P$ is genuinely uncertain — maybe $0.7$, maybe $0.6$ — and $1-P$ of a long future stream is a lot of people lost to a treatment I never gave a chance to prove itself. The irrevocable rule throws away exactly the thing I am least sure about: which arm is actually best.

The even split has the opposite fault. It keeps sacrificing half of everyone to the inferior treatment even after the data has started, clearly, to point one way. It refuses to use the evidence while the evidence is accumulating. In the cases I care about — valuable individuals, slow data — that waste is the whole cost.

So neither extreme is right, and staring at the two of them I can feel the shape of what I want. The even split ignores $P$ completely. The irrevocable decision treats $P$ as if it were $0$ or $1$. The honest thing is somewhere between: I should let *how confidently I act* track *how much the data actually warrants*, which is exactly what $P$ measures. Acting should be graded by $P$, not thresholded by it.

So take some increasing function $f(P)$ with $0 \le f(P) \le 1$, and treat a fraction $f(P)$ of individuals by the first method and $1-f(P)$ by the second — or, what is the same thing for one individual at a time, assign each arriving person to the first treatment *with probability* $f(P)$. When $P$ is near $1$, $f(P)$ is near $1$ and almost everyone gets the better-looking treatment; when $P$ is near $\tfrac12$, the data is not telling me much and I hedge close to an even split; and crucially, as data accumulates and $P$ drifts toward $0$ or $1$, $f(P)$ drifts with it, so the hedging *shrinks on its own*. I never have to decide when to stop exploring. The exploration extinguishes itself.

Now which $f$? It has to be increasing — more confidence, more assignment — and bounded in $[0,1]$. The most natural, least-arbitrary choice is the identity: $f(P) = P$. Assign each individual to the first treatment with exactly the probability that the first treatment is better. Let me check that this actually buys me something, because "natural" is not an argument.

Under $f(P)=P$, what is the expected per-individual sacrifice? Write $Q = 1-P$. With probability $P$ I assign to treatment one; the cost is incurred only when one is actually the worse arm, and the "badness" of that is carried by $Q$. With probability $Q$ I assign to treatment two; symmetrically the cost is carried by $P$. The expected sacrifice, in units of the gap, comes out proportional to $P\cdot Q + Q \cdot P = 2PQ$. So under probability-matching the expected sacrifice is $2PQ$. Now $2PQ = 2P(1-P)$, which is maximised at $P=\tfrac12$ where it equals $\tfrac12$, and strictly less than $\tfrac12$ whenever $P \ne \tfrac12$. Compare the alternate-case method: it assigns to the inferior arm half the time *regardless* of the evidence, a flat sacrifice of $\tfrac12$. So $2PQ \le \tfrac12$ says probability-matching never does worse than the even split, and does strictly better the moment the data shows any preference at all. And against the irrevocable decision: in the long run, if a real preference exists between the treatments, continuing to apportion by $f(P)=P$ instead of freezing immediately saves, sensibly, $1-P$ of the individuals subsequently treated — the very stream the irrevocable rule was sacrificing. That is the saving I was after, and $f(P)=P$ delivers it with no tuning and no stopping rule.

So the discipline is: act on the *probability that one arm beats the other*, by matching your assignment probability to it. Everything now hinges on actually computing $P$ — the probability that one unknown success probability exceeds another, from two small samples. If I cannot compute that for small $n$, the whole scheme is a slogan.

Let me set it up carefully. Two unknown probabilities $p_1, p_2$, one per treatment. I observe, independently, $r_i$ successes and $s_i$ failures in $n_i = r_i + s_i$ trials of treatment $i$. The probability of getting $r$ successes in $n=r+s$ trials at a fixed $p$ is $\binom{n}{r} p^{r} q^{s}$ with $q=1-p$. But $p$ is unknown — it is the thing I am uncertain about. So I treat it as a random quantity and put a prior on it. What prior? The honest state of ignorance: each $p_i$ *a priori* equally likely to lie in any two equal sub-intervals of $(0,1)$. That is the uniform prior on $(0,1)$. Now apply Bayes' theorem. The posterior density of $p$ after seeing $r$ successes and $s$ failures is proportional to prior times likelihood, i.e. proportional to $1 \cdot p^{r} q^{s}$. I just need the normalising constant: $\int_0^1 p^{r} q^{s}\,dp = \dfrac{r!\,s!}{(n+1)!}$ — that is the Beta integral $B(r+1,s+1)$. So the posterior density is

$$ \frac{(n+1)!}{r!\,s!}\, p^{r}\, q^{s}. $$

Stare at this. It is a Beta distribution with parameters $r+1$ and $s+1$. The data enter only through the counts; the prior contributes the $+1$ to each. Wonderful — the uniform prior is conjugate to Bernoulli sampling, so the posterior stays in the same family, and updating is trivial: one more success bumps the first parameter, one more failure bumps the second. I will lean on that. It means I never have to re-integrate; the posterior after $N$ observations is just $\mathrm{Beta}(r+1, s+1)$ with the running counts, and observing one more outcome takes $\mathrm{Beta}(\alpha,\beta)$ to $\mathrm{Beta}(\alpha+1,\beta)$ on a success or $\mathrm{Beta}(\alpha,\beta+1)$ on a failure. The parameters are nothing but pseudo-counts.

I will also want the *tail* of this posterior, the probability that $p$ exceeds some threshold, because that is the building block for "one exceeds the other". Compute $\int_p^1 \frac{(n+1)!}{r!\,s!}\, t^{r}(1-t)^{s}\,dt$. Integrate by parts repeatedly — peel off one factor of $(1-t)^{s}$ at a time — and it telescopes into a finite sum of binomial terms:

$$ \int_p^1 \frac{(n+1)!}{r!\,s!}\, t^{r}(1-t)^{s}\, dt \;=\; \sum_{a=0}^{r} \binom{n+1}{a}\, p^{a}\, q^{\,n+1-a}. $$

That is the regularized incomplete Beta written as a binomial tail — the same identity Pearson used, that an incomplete-Beta probability equals a sum of the first several terms of a hypergeometric series. Good: the single-arm tail is a finite, exact, hand-computable thing.

Now the real quantity. I want $P_{p_2 > p_1}$, the posterior probability that the second arm's success probability exceeds the first's. The two posteriors are independent, so

$$ P_{p_2 > p_1} = \int_0^1 \Big[\tfrac{(n_1+1)!}{r_1!\,s_1!} p_1^{r_1} q_1^{s_1}\Big] \left( \int_{p_1}^1 \tfrac{(n_2+1)!}{r_2!\,s_2!} p_2^{r_2} q_2^{s_2}\, dp_2 \right) dp_1 . $$

The inner integral I just learned to do — it is the tail of the second arm's posterior, a finite sum in powers of $p_1$. Substitute that sum in, and now the outer integral is, term by term, again of the form $\int_0^1 p_1^{r_1 + a}(1-p_1)^{s_1 + \cdots}\,dp_1$, each a Beta integral I can evaluate in closed form as a ratio of factorials. So the whole thing collapses to a *finite double sum of factorials* — no special functions left, nothing to approximate. Pushing the algebra through (collecting the factorials into binomial coefficients) the cleanest form is

$$ P_{p_2 > p_1} \;=\; \frac{\displaystyle\sum_{a=0}^{r_2} \binom{r_1 + r_2 - a}{\,r_1\,}\binom{s_1 + s_2 + 1 + a}{\,s_1\,}}{\displaystyle\binom{n_1 + n_2 + 2}{\,n_1 + 1\,}} . $$

This is what I was missing: a reduced, rational, *exact* evaluation of the probability that one unknown probability exceeds another, valid for any small counts, with no large-sample assumption. Let me sanity-check it for consistency. By the same construction, $P_{p_1 > p_2}$ is the same formula with the arms swapped, $\sum_{a=0}^{r_1}\binom{r_1+r_2-a}{r_2}\binom{s_1+s_2+1+a}{s_2}/\binom{n_1+n_2+2}{n_2+1}$. And since the event $p_1 = p_2$ has probability zero under continuous posteriors, $P_{p_2 > p_1} = 1 - P_{p_1 > p_2}$ — the two must sum to one. The bookkeeping closes.

There is also a useful approximation lurking, by Stirling. Write each factorial as $\sqrt{2\pi}\,m^{m+\frac12}e^{-m}$ times a bounded correction $Q(m)=e^{\theta/12m}$, $0<\theta<1$. Substituting into the factorial form shows $P_{p_2>p_1}$ tends, as $n_1$ grows with $r_2$ small, to the incomplete-Beta value $I_q(s_2+1, r_2+1)$ evaluated at the empirical $p = r_1/n_1$ — i.e. the small-sample exact formula and Pearson's large-sample machinery agree in the limit, and I can even bound the ratio between them by the largest and smallest term-ratios. So nothing is inconsistent with what was already known for large samples; I have filled in the small-sample hole and connected it to the rest.

For this to be a *usable* discipline I have to be able to build a table of these probabilities by hand, because in practice I will need $\psi(r_1,s_1,r_2,s_2)$ for many small argument values, again and again. The factorial formula is exact but tedious to recompute each time. So let me find recurrences. Define the numerator pieces — call them $N(r,s,r',s')$, sums of products of two binomial coefficients, and an auxiliary $D(n,n')=\binom{n+n'+2}{n+1}$ for the denominator. Pascal's rule, $\binom{m}{k}=\binom{m-1}{k-1}+\binom{m-1}{k}$, propagates straight through these sums: I get additive relations like $N(r,s,r',s') = N(r,s-1,r',s') + N(r,s,r',s'-1)$, exactly the pyramid by which the binomial coefficients themselves are tabulated — each value the sum of two already-listed neighbours. The symmetry relations let me list only cases with $r+s \ge r'+s'$ and positive arguments. So I can build the whole table from the bottom up by additions, doubling a final subtotal where the recurrence for $D$ requires it. The probability I want is then just the ratio $N/D$ read off the table. A short table of $n,n' \le 5$ suffices for the common small cases, and the rest follow from the recurrences. The hand-computability I demanded is real.

Step back and read what I have built. A patient arrives. I look at the running counts $(r_1,s_1)$ and $(r_2,s_2)$. I compute $P = P_{p_1 > p_2}$ from the table. I assign this patient to treatment one with probability $P$ and to treatment two with probability $1-P$ — probability-matching, $f(P)=P$. I observe success or failure and bump the appropriate count by one (the conjugate update). Repeat. The assignment automatically concentrates on the better arm as the evidence accumulates, hedges when the evidence is thin, and never needs a stopping rule. The expected sacrifice per individual is $2PQ$, never worse than the even split and strictly better whenever there is any preference, and in the long run it saves $1-P$ of the future stream that an immediate decision would have squandered.

Now, there is a cleaner way to *execute* the probability-matching rule that I want to pin down, because computing $P$ exactly is fine for two arms but does not obviously scale. Notice: assigning to arm one with probability $P = P_{p_1>p_2}$ is the same as the following procedure. Draw a sample $\tilde p_1$ from the posterior of arm one and an independent sample $\tilde p_2$ from the posterior of arm two, and assign to whichever arm's sample is larger. Why is that the same? Because the probability that $\tilde p_1 > \tilde p_2$ under the two independent posteriors *is* $P_{p_1 > p_2}$ — that is the very integral I computed. So instead of computing $P$ and then flipping a $P$-coin, I can just sample one plausible value of the success probability for each arm from its posterior and act greedily with respect to those samples. The randomness of the posterior draw *is* the randomness of the matching coin. And this version scales to any number of arms with no change: sample a plausible parameter for each arm, play the arm whose sample is largest. The probability that I play arm $k$ is then exactly the probability that arm $k$'s sampled parameter is the largest, which is exactly the posterior probability that arm $k$ is the optimal arm. Probability-matching, realised by posterior sampling.

I should be careful about *what* I sample. I sample a plausible *parameter* — a plausible value of the success probability $p_k$ — from the posterior, not a plausible *next outcome* (a $0$ or $1$). It is the parameter that has the comparison structure I need: I am asking "which arm could plausibly be the best", and that is a question about the means, not about a single Bernoulli draw. Sampling the binary outcome would collapse the very uncertainty I am trying to act on.

And I can see now why this beats the obvious alternatives, beyond just the two-arm sacrifice accounting. Suppose I were *greedy* — always play the arm with the larger posterior mean. Early on, a couple of lucky successes on a genuinely-inferior arm can make its posterior mean the largest, and then greedy keeps playing it, keeps getting reinforced near its (wrong) early mean, and never tries the other arm enough to discover the truth. Greedy can lock onto a bad arm because a confidently-wrong posterior is never challenged. Suppose instead I dither — play greedy but with probability $\epsilon$ pick a uniformly random arm. That fixes the locking, but it is blind: it spends the same exploratory effort on an arm I am already almost certain is bad as on an arm I am genuinely unsure about. With three arms whose posteriors say one is clearly best, one is clearly hopeless, and one is uncertain-but-maybe-best, dithering wastes half its exploration on the hopeless arm. Probability-matching by posterior sampling spends exploration *in proportion to the posterior probability of being optimal*: essentially nothing on the hopeless arm, a real share on the uncertain one. The exploration is targeted, and it vanishes exactly as fast as the posterior concentrates.

That qualitative story I would like to back with a guarantee on the *cumulative* sacrifice — the regret — over a long horizon $T$. Let the arms have true means $\mu_i$, let arm $1$ be the unique best with $\mu^\* = \mu_1$, and write $\Delta_i = \mu_1 - \mu_i$ for the gap of arm $i$. Plays of the best arm cost nothing; every play of a suboptimal arm $i$ costs $\Delta_i$ in expectation. So the expected regret is $\mathbb{E}[R(T)] = \sum_i \Delta_i\, \mathbb{E}[k_i(T)]$, where $k_i(T)$ is the number of times arm $i$ is played in $T$ steps. The whole game is to bound how often a suboptimal arm gets played. Take the two-arm case first; it already carries the essential difficulty.

Regret here is $\Delta\,\mathbb{E}[k_2(T)]$ with $\Delta = \mu_1 - \mu_2$. I want to show $k_2(T)$ is small. The clean idea: once arm $2$ has been played enough times, its own posterior is tightly concentrated around $\mu_2$, so arm $2$ will only be sampled larger than arm $1$ when arm $1$'s sample dips near $\mu_2$, which becomes rare. Set $L = 24(\ln T)/\Delta^2$ — the number of plays of arm $2$ after which a Chernoff–Hoeffding bound puts arm $2$'s empirical mean within $\Delta/2$ of $\mu_2$ with probability at least $1 - O(1/T^2)$. Split:

$$ \mathbb{E}[k_2(T)] \le L + \mathbb{E}\big[\text{plays of arm 2 after its } L\text{-th play}\big]. $$

The first $L$ plays contribute at most $\Delta L = 24(\ln T)/\Delta$ to the regret — and that is the $\ln T/\Delta$ term, the unavoidable logarithmic price of learning, matching the Lai–Robbins lower bound $\mathbb{E}[R(T)] \ge [\sum_i \Delta_i / D(\mu_i\Vert\mu_1) + o(1)]\ln T$ in its dependence on $T$. The work is in the second piece.

Here is where the randomized nature of the rule bites, and it bit me when I tried to copy the standard upper-confidence-bound argument. For a UCB algorithm the exploration is a *deterministic* bonus added to each empirical mean, shrinking as an arm is played; the argument there is that once arm $2$ has been played enough, its inflated estimate can no longer exceed arm $1$'s true mean, *regardless of how few times arm $1$ has been played*. That last clause fails completely for posterior sampling. If arm $1$ has been played only a handful of times, its posterior is nearly the uniform prior, and a uniform-ish sample $\tilde p_1$ can easily fall below $\mu_2$, so arm $2$'s well-concentrated sample beats it with *constant* probability — not polynomially small. So I cannot bound arm $2$'s play-probability uniformly over arm $1$'s play count; I have to actually track *how many times arm $1$ has been played*, because that is what controls how spread-out arm $1$'s posterior — and hence how beatable it — is. The randomness forces me to reason about the distribution of arm $1$'s play count. That is the new ingredient.

So model the gaps. After arm $2$ is well-concentrated (say after step $t_{j_0}$, with $j_0$ chosen so arm $1$ has been played enough that the analysis can start), look at the stretches of time *between consecutive plays of arm $1$*. Between the $j$-th and $(j{+}1)$-th play of arm $1$, every step is a play of arm $2$; the length of that stretch is what I must bound, because summing those lengths is $k_2$. Now, in that stretch arm $1$ is *not* being played, which means at every such step arm $1$'s sample failed to exceed arm $2$'s — and once arm $2$ is concentrated near $\mu_2$, "arm $1$ gets played" is essentially the event $\tilde p_1 > \mu_2 + \Delta/2$. With arm $1$ frozen at $j$ plays and $s$ of them successes, $\tilde p_1 \sim \mathrm{Beta}(s+1, j-s+1)$, so each step is an independent trial that "succeeds" (plays arm $1$) with probability $\Pr[\mathrm{Beta}(s+1,j-s+1) > y]$ for the threshold $y = \mu_2 + \Delta/2$. The number of steps until that happens is geometric. Its expectation is

$$ \mathbb{E}[X(j,s,y)] = \frac{1}{\Pr[\mathrm{Beta}(s+1,\,j-s+1) > y]} - 1 = \frac{1}{1 - F^{\beta}_{s+1,\,j-s+1}(y)} - 1, $$

the $-1$ because I do not count the step on which arm $1$ is finally played. And here the earlier tail identity earns its keep: the Beta tail $\Pr[\mathrm{Beta}(s+1,j-s+1) > y]$ equals a binomial cumulative $F^{B}_{j+1,\,y}(s)$, so $\mathbb{E}[X(j,s,y)] = 1/F^{B}_{j+1,y}(s) - 1$ — a thing I can estimate with binomial concentration.

Now assemble. The length $Y_j$ of the $j$-th between-plays stretch is controlled by $X(j, s(j), \mu_2 + \Delta/2)$ when arm $2$'s sample is not much larger than $\mu_2$, and by the crude horizon bound $T$ otherwise. That bad event has probability at most $2/T^2$ at each time after arm $2$ is saturated, so after summing over at most $T$ times and carrying the outside factor $T$ it contributes only $O(1)$. Therefore

$$ \mathbb{E}[k_2(T)] \le L + \sum_{j=0}^{T-1} \mathbb{E}\big[\min\{X(j,\,s(j),\,\mu_2 + \tfrac{\Delta}{2}),\,T\}\big] + O(1). $$

What remains is to sum the truncated geometric means over $j$, and this is a case analysis because the distribution of the number of successes in arm $1$ behaves very differently when arm $1$ has been played few versus many times. Let $D = y \ln(y/\mu_1) + (1-y)\ln((1-y)/(1-\mu_1))$ be the Bernoulli relative entropy from the threshold $y$ to $\mu_1$, and $R = \mu_1(1-y)/(y(1-\mu_1)) > 1$. Taking the expectation over the random number of successes $s(j)$, the bound splits into three regimes:

$$ \mathbb{E}\big[\mathbb{E}[\min\{X(j,s(j),y),T\}\mid s(j)]\big] \le \begin{cases} 1 + \dfrac{2}{1-y} + \dfrac{\mu_1}{\Delta'}\,e^{-Dj}, & j < \tfrac{y}{D}\ln R, \\[2mm] 1 + \dfrac{R^{y}}{1-y}\,e^{-Dj} + \dfrac{\mu_1}{\Delta'}\,e^{-Dj}, & \tfrac{y}{D}\ln R \le j < \tfrac{4\ln T}{\Delta'^2}, \\[2mm] \dfrac{16}{T}, & j \ge \tfrac{4\ln T}{\Delta'^2}, \end{cases} $$

with $\Delta' = \mu_1 - y = \Delta/2$. Read off where the cost lives. For large $j$ the term is $16/T$; summed over at most $T$ values of $j$ it is $O(1)$. The $e^{-Dj}$ terms form a geometric series with $\sum_j e^{-Dj}=O(1/\min\{D,1\})$, and Pinsker gives $D \ge 2\Delta'^2$, so the $\mu_1/\Delta'$ part is $O(1/\Delta'^3)$. The small- and middle-$j$ constants are the expensive part: $y\ln R/D$ is bounded by a term of order $(D+1)/(\Delta'D)$, and multiplying by the $1/(1-y)$ factor costs at most $O(1/\Delta'^4)$ after the same $D\ge 2\Delta'^2$ substitution. With $\Delta'=\Delta/2$, the post-$L$ play count is therefore $O(1/\Delta^4)$, and

$$ \mathbb{E}[k_2(T)] \le O\!\left(\frac{\ln T}{\Delta^2} + \frac{1}{\Delta^4}\right), \qquad \mathbb{E}[R(T)] = \Delta\,\mathbb{E}[k_2(T)] = O\!\left(\frac{\ln T}{\Delta} + \frac{1}{\Delta^3}\right). $$

So the cumulative sacrifice grows only logarithmically in the horizon, with the leading $\ln T / \Delta$ matching the information-theoretic floor — the exploration really does pay for itself and then stop. For $N$ arms the same machinery, now having to track interruptions of arm $1$'s plays by *every* suboptimal arm at once rather than just one, gives $\mathbb{E}[R(T)] \le O\big((\sum_{a=2}^{N} 1/\Delta_a^2)^2 \ln T\big)$ — still logarithmic in $T$, the dependence on the gaps worsened to account for the harder bookkeeping, but order-optimal in the horizon.

So the discipline is complete, and it is exactly the one I was reaching for at the start: act according to the probability that each action is the best one. Concretely — maintain a Beta posterior per arm from a uniform prior, $\mathrm{Beta}(r+1, s+1)$, updated by incrementing the success or failure count after each observation; to choose, draw one plausible success probability from each arm's posterior and play the arm whose draw is largest; observe, update, repeat. Sampling-and-playing-greedily *is* probability-matching, because the chance an arm's sample wins equals the posterior probability that arm is optimal — the very $P_{p_2>p_1}$ I derived in closed form for two arms. The posterior's own uncertainty supplies the exploration, targeted in proportion to the chance of being best and self-extinguishing as the posterior concentrates; and the price of that exploration, totalled over a long horizon, is only logarithmic.

```python
import math, random

def beta_bernoulli_thompson(arms, horizon, observe):
    """Probability-matching by posterior sampling for K Bernoulli arms.

    arms      : number of treatments / actions K
    observe   : observe(k) -> 1 (success) or 0 (failure) for chosen arm k
    Maintains Beta(S_k + 1, F_k + 1): uniform prior Beta(1,1), conjugate update.
    """
    S = [0] * arms              # successes per arm  (r_k)
    F = [0] * arms              # failures  per arm  (s_k)
    history = []

    for t in range(horizon):
        # sample a plausible success probability from each arm's posterior:
        # the draw's randomness is the randomness of the matching coin, so
        # P(arm k chosen) = P(arm k is optimal | data).  Sample the PARAMETER,
        # not a 0/1 outcome.
        theta = [random.betavariate(S[k] + 1, F[k] + 1) for k in range(arms)]

        # play greedily w.r.t. the sampled parameters == probability matching
        k = max(range(arms), key=lambda i: theta[i])

        r = observe(k)                      # 1 = success, 0 = failure
        if r == 1: S[k] += 1                # conjugate update: Beta(a+1, b)
        else:      F[k] += 1                #                   Beta(a, b+1)
        history.append(k)

    return S, F, history


def prob_first_exceeds_second(r1, s1, r2, s2):
    """Exact P(p1 > p2) from two independent uniform-prior Beta posteriors.

    Closed form for two arms (no sampling needed): a finite sum of binomials,
    the reduced rational evaluation of the probability one unknown probability
    exceeds another.
    P(p2 > p1) = sum_{a=0..r2} C(r1+r2-a, r1) C(s1+s2+1+a, s1) / C(n1+n2+2, n1+1).
    """
    C = math.comb
    n1, n2 = r1 + s1, r2 + s2
    p2_gt_p1 = sum(C(r1 + r2 - a, r1) * C(s1 + s2 + 1 + a, s1)
                   for a in range(r2 + 1)) / C(n1 + n2 + 2, n1 + 1)
    return 1.0 - p2_gt_p1               # P(p1 > p2) = 1 - P(p2 > p1)
```
