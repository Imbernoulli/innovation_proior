## Research question

Erdős asked how evenly one can split the differences between two equal halves of $[2n]=\{1,\dots,2n\}$. For a partition $A\cup B=[2n]$ with $|A|=|B|=n$, let $M_k$ be the number of pairs $(a,b)\in A\times B$ with $a-b=k$, and set
$$
M(n)=\min_{A\cup B=[2n]}\max_{-2n<k<2n} M_k .
$$
The goal is to determine the limiting constant
$$
\mu=\lim_{n\to\infty}\frac{M(n)}{n}
$$
as sharply as possible, with rigorous upper and lower bounds. An upper bound can be supplied by an explicit family of partitions, or by a limiting density that can be discretized back into partitions. A lower bound must certify that every admissible partition, and hence every limiting density, has some shift with overlap at least the claimed value.

## Background

The first constraint is mass. Summing $M_k$ over all shifts counts every pair in $A\times B$ once, so $\sum_k M_k=n^2$. Since there are fewer than $4n$ possible shifts, $\max_k M_k>n/4$. A middle-block construction gives $M(n)\le n/2$. This leaves the constant between $1/4$ and $1/2$ and shows the basic pattern: global information about the overlap profile must force a pointwise peak.

The continuous viewpoint replaces the indicator of $A$ by a density. Work on $[-1,1]$, take $f:[-1,1]\to[0,1]$ with $\int_{-1}^1 f=1$, set $g=1-f$, extend outside the interval by zero when needed, and define
$$
M(x)=\int_{-1}^1 f(t)g(x+t)\,dt,\qquad -2\le x\le 2.
$$
Swinnerton-Dyer's reduction identifies $\mu$ with the infimum of $\|M\|_\infty$ over such densities, equivalently with the step-function formulation on $[0,2]$ used for constructive upper bounds. The problem can therefore be attacked as a variational problem over functions rather than as a direct search over $\binom{2n}{n}$ partitions.

The continuous overlap has two elementary moment constraints:
$$
\int_{-2}^2 M(x)\,dx=1
$$
and, with $E(M)=\int_{-2}^2 xM(x)\,dx$,
$$
\int_{-2}^2 x^2M(x)\,dx=\frac23+\frac12E(M)^2.
$$
Consequently the centered variance is $2/3-E(M)^2/2\le 2/3$. Packing a mass-$1$ function of height at most $\mu$ into one centered block gives the basic variance lower bound $\mu\ge 1/\sqrt8$; Moser's sharper use of the same moment obstruction gives
$$
\mu\ge \sqrt{4-\sqrt{15}}\approx0.35639395869.
$$
The limitation is that moments alone ignore the correlation structure that forces $M$ to come from a pair $f,1-f$.

## Baselines

**Erdős (1955): averaging.** The identity $\sum_k M_k=n^2$ gives $M(n)>n/4$, while a middle interval gives $M(n)\le n/2$. It proves the problem is linear in $n$ but leaves a factor-two gap.

**Erdős-Scherk, Świerczkowski, Motzkin-Ralston-Selfridge.** These refinements improved the discrete bookkeeping and narrowed the interval for the constant, but they did not produce a variational description of the limit or a systematic certificate for all partitions.

**Moser-Murdeshwar and Moser.** The function analogue supplies the mass and second-moment identities above. The method turns the question into an extremal problem for a bounded density of fixed mass and bounded variance. It reaches $\sqrt{4-\sqrt{15}}$ but has no way to exploit the fact that $M$ is a correlation of $f$ with the translated complement.

**Swinnerton-Dyer's reduction.** The asymptotic combinatorial constant equals the continuous infimum over step densities $f$ on $[0,2]$ with values in $[0,1]$ and $\int_0^2 f=1$ of
$$
\max_k \int f(x)\bigl(1-f(x+k)\bigr)\,dx.
$$
This legitimizes both constructive step-function upper bounds and analytic lower bounds on the continuous overlap.

**Haugland's step functions.** A symmetric step density on a uniform grid turns the overlap integral into a finite nonperiodic shift correlation of the step heights. Haugland's 21-step construction gave $0.382002\ldots$, and the 51-step construction gives
$$
\mu\le 0.3809268534330870.
$$
This is a constructive upper bound only; it does not certify that every density has a peak near that value.

## Evaluation settings

The natural benchmark is a rigorous interval for $\mu$: a lower certificate valid for every admissible density and an explicit density giving an upper bound. A lower-bound computation must be a relaxation in the correct direction, so every true overlap profile maps to a feasible point whose objective is no larger than its actual $\|M\|_\infty$. A numerical solver value is not enough; the certificate must survive roundoff. An upper-bound computation must evaluate the step-function overlap as a nonperiodic translation on $[0,2]$, with the complement taken only where the translated point remains in the interval.

## Code framework

The constructive side already has a minimal scoring harness for step densities.

```python
import numpy as np

def step_overlap_objective(left_half_heights):
    """
    Build a symmetric odd-step density on [0, 2] from the heights on [0, 1],
    then return the largest nonperiodic grid-shift overlap integral.
    """
    pass
```

The lower-bound side needs a generic certificate pipeline: collect universal facts about $M$, turn them into a finite relaxation over interval averages, and certify the relaxed optimum from the dual.

```python
def admissible_properties_of_M(R, T):
    """
    Return the function-level identities and inequalities to impose on M.
    Existing facts include total mass and the second-moment identity.
    """
    pass

def build_relaxation(N, properties, parameter_box):
    """
    Variables are interval averages of M on the positive and negative halves,
    plus any auxiliary coefficient variables required by the chosen properties.
    Minimize Omega with 0 <= averages <= Omega.
    """
    pass

def certify_lower_bound(relaxation, dual_certificate=None):
    """
    Solve the relaxation numerically, or verify a supplied dual certificate
    with explicit roundoff margins.
    """
    pass
```
