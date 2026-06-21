# Context: quantifying how fast error probability can be driven to zero below capacity

## Research question

Shannon's noisy-channel coding theorem settled the *qualitative* question: for a
memoryless channel there is a number — the capacity $C$ — such that for every
rate $R < C$ there exist block codes whose decoding-error probability can be made
as small as we like by taking the block length $N$ large enough, and for $R > C$
this is impossible. That is an existence statement about a limit. It does not say
*how fast* the error probability falls as $N$ grows, nor *how* that speed depends
on the rate $R$ or on the particular channel.

For an engineer deciding whether to put coding on a physical link, the operational
questions come first: at a chosen rate $R < C$, how large must the block length $N$
be to reach a target error probability $P_e$? Two modulation schemes might have the
same capacity yet behave differently at the rates one actually wants to run — which
is better for coding? These questions are about a *rate of decay*. The object of
interest is a function $E(R) > 0$, defined for $R$ in $0 < R < C$ and computable
from the channel's transition probabilities, such that the smallest achievable
error probability behaves like $P_e \approx e^{-N E(R)}$ — an exponent that turns
the asymptotic limit into a quantitative law and lets channels be compared on the
strength of their whole $E(R)$ curve, not just the single number $C$. The setting
spans any discrete memoryless channel and the amplitude-continuous channels (the
additive Gaussian-noise channel above all) with input power constraints.

## Background

**Capacity and the coding theorem (Shannon 1948).** For a discrete memoryless
channel with input letter $x$, output letter $y$, and transition law $P(y\mid x)$,
the average mutual information for an input distribution $p(x)$ is
$$I(p) = \sum_{x,y} p(x) P(y\mid x)\, \ln \frac{P(y\mid x)}{\sum_{x'} p(x') P(y\mid x')},$$
and the capacity is $C = \max_{p} I(p)$. Shannon proved that reliable
communication is possible for every $R < C$ and impossible above $C$. His
achievability argument is the seed of everything that follows: rather than
exhibit a good code, draw a *random* codebook — each of the $M = e^{NR}$ codewords
an independent string of $N$ letters drawn i.i.d. from a distribution $p$ — and
show the *ensemble-average* error probability tends to $0$. Since at least one
code in the ensemble is as good as the average, a good code exists. The decoder is
typical-set (joint-typicality) decoding, and the error analysis is a union bound
over the $M-1$ competing messages combined with the asymptotic equipartition of
typical sequences. The conclusion is $P_e \to 0$ for $R < C$.

**Maximum-likelihood decoding.** The error-minimizing decoder for equiprobable
messages declares the codeword $x_m$ maximizing $P(y\mid x_m)$. A decoding error
occurs precisely when some other codeword $x_{m'}$ satisfies
$P(y\mid x_{m'}) \ge P(y\mid x_m)$. This is the decoder whose error one bounds; the
error event for a given codeword depends on all the other codewords jointly.

**Union bound and pairwise (Bhattacharyya) bounding.** The standard handle on the
ML error event is the union bound: the probability that *some* competitor beats
$x_m$ is at most the sum over competitors of the pairwise confusion probabilities.
Each pairwise term — the probability that $x_{m'}$ looks at least as likely as
$x_m$ — is overbounded by a tilted sum
$\sum_y \sqrt{P(y\mid x_m)\,P(y\mid x_{m'})}$, the Bhattacharyya bound, which comes
from $1[P(y\mid x_{m'}) \ge P(y\mid x_m)] \le \sqrt{P(y\mid x_{m'})/P(y\mid x_m)}$.
Averaging this over the random ensemble yields a single fixed exponent in terms of
the channel.

**Chernoff/Hölder/Jensen bounding.** The standard inequalities of large-deviation
analysis are available for converting sums into exponents: bounding an indicator by
a nonnegative quantity raised to a power (Chernoff's method); Hölder's inequality
$\sum_j a_j b_j \le (\sum_j a_j^{1/\theta})^{\theta}
(\sum_j b_j^{1/(1-\theta)})^{1-\theta}$; Jensen's inequality relating the average of
a function to the function of the average; and the monotonicity of weighted power
means $\big(\sum_\ell q_\ell a_\ell^{r}\big)^{1/r}$ in $r$, which is the same Hölder
fact in another guise.

**The state of the art on the exponent.** Elias (1955), for the binary symmetric
channel, derived upper *and* lower bounds on the smallest achievable error
probability for codes of block length $N$ that both decay exponentially in $N$ for
every $R < C$, and that coincide over a substantial range of rates up to capacity.
Two facts from that work were by then part of the prevailing wisdom: almost all
randomly chosen codes are essentially as good as the best code ("most codes are
good"), and the special class of *linear* codes already achieves the same average
performance as the fully random ensemble. Fano (1961) stated, for general discrete
memoryless channels, the strongest then-known form: the minimum error probability
is squeezed between
$e^{-N[E_L(R) + o(1)]} \le P_e \le 2\, e^{-N E(R)}$ with $E_L(R), E(R) > 0$ below
capacity and $E_L(R) = E(R)$ in a band just beneath $C$. The cleanest exponential
statements available were channel-specific (the BSC, orthogonal signals in Gaussian
noise). The "cutoff rate" $R_0$ — the exponent's value extrapolated to zero rate,
and the rate above which sequential decoding's computation blows up — was the other
landmark on the $E(R)$ curve.

## Baselines

**Shannon's random-coding achievability (1948).** Random codebook + typical-set
decoding + union bound show ensemble-average $P_e \to 0$ for $R < C$; this yields
the existence of capacity and a vanishing-in-the-limit error.

**Elias' BSC exponent (1955).** For the binary symmetric channel, explicit
exponentially-decaying upper and lower bounds on $P_e(N)$, with random and linear
ensembles shown equivalent, matched over a range of rates.

**Fano's general bounds (1961).** For a general DMC, $P_e$ is sandwiched between two
exponentially-decaying bounds with a positive exponent below capacity, agreeing in a
high-rate band.

**Bhattacharyya / cutoff-rate union bound.** Union bound plus pairwise
Bhattacharyya bounding gives a closed-form exponent, essentially
$E_0(1,p) - R$ with $E_0(1,p) = -\ln \sum_j \big(\sum_k p_k \sqrt{P_{jk}}\big)^2$.
This is a single line of slope $-1$; its exponent reaches zero at the cutoff rate
$R_0$.

## Evaluation settings

The natural yardsticks are the standard channel models on which an exponent would be
exhibited and compared. The **discrete memoryless channel** specified by a
transition matrix $P_{jk} = \Pr(b_j \mid a_k)$ over a $K$-letter input alphabet and
$J$-letter output alphabet — with the **binary symmetric channel** (crossover
probability $q$, $P_{11}=P_{22}=1-q$, $P_{12}=P_{21}=q$) as the canonical special
case. The **very noisy channel**, where the output is nearly independent of the
input ($P_{jk} = q_j(1 + \epsilon_{jk})$ with $|\epsilon_{jk}| \ll 1$), as a regime
admitting a universal limiting curve. The **time-discrete additive Gaussian-noise
channel**, $y = x + z$ with $z \sim \mathcal{N}(0,1)$ per use and an average power
constraint $\frac{1}{N}\sum_n x_n^2 \le A$ (so $A$ is the power signal-to-noise
ratio per degree of freedom), with orthogonal signaling and the band-limited
Gaussian channel as reference points. The quantities of interest are the
exponent–rate curve $E(R)$ as a function of $R \in (0,C)$, the critical rate
$R_{\text{crit}}$ and cutoff rate $R_0$ that mark its breakpoints, and the capacity
$C$ that bounds its support. The relevant comparison is against Shannon's (1959)
exact exponent for the Gaussian channel and Elias' BSC exponent, in the rate ranges
where those are known.

## Code framework

For this problem, the useful scaffold is the minimal symbolic machinery for
describing a memoryless channel, computing Shannon's known capacity functional, and
leaving an explicit slot for a rate-dependent ML error decay bound. A tiny numerical
harness for a DMC can hold that slot.

```python
import numpy as np

def mutual_information(p, P):
    """I(p) for input dist p (len K) over transition matrix P (J x K).
    P[j, k] = Pr(output b_j | input a_k)."""
    q = P @ p                                    # output marginal, len J
    out = 0.0
    for k in range(len(p)):
        for j in range(P.shape[0]):
            if p[k] > 0 and P[j, k] > 0:
                out += p[k] * P[j, k] * np.log(P[j, k] / q[j])
    return out

def capacity(P, p_grid):
    """C = max_p I(p), the qualitative reliable-rate threshold."""
    return max(mutual_information(p, P) for p in p_grid)

def pairwise_ml_overlap(P, k, i):
    """Bhattacharyya-style overlap for confusing input letters k and i."""
    return np.sqrt(P[:, k] * P[:, i]).sum()

def achievable_decay_rate(R, P, p_grid):
    """A computable per-symbol decay rate for ML error at rate R.
    # TODO: fill in."""
    pass

def error_probability_bound(N, R, P, p_grid):
    """Upper bound on ML decoding error for block length N at rate R.
    # TODO: fill in."""
    pass
```
