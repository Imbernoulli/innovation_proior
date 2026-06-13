## Research question

Erdos asked how small the largest cross-difference overlap can be when the integers
\([2n]=\{1,\ldots,2n\}\) are split into two equal parts. For a partition
\(A\cup B=[2n]\), \(|A|=|B|=n\), define
\[
M_k=\#\{(a,b)\in A\times B:a-b=k\},\qquad
M(n)=\min_{A\cup B=[2n]}\max_{-2n<k<2n}M_k .
\]
The asymptotic question is to determine the constant
\[
\mu=\lim_{n\to\infty}\frac{M(n)}{n}.
\]
A construction gives an upper bound by exhibiting partitions, or by exhibiting a limiting density that can be discretized back into partitions. A lower bound must certify that every admissible partition, hence every limiting density, has at least one shift whose overlap is no smaller than the proposed constant.

## Background

The first invariant is total mass. Summing \(M_k\) over all shifts counts every element of \(A\times B\) once, so \(\sum_k M_k=n^2\). Since there are fewer than \(4n\) shifts, \(\max_k M_k>n/4\). The elementary middle-block construction gives \(M(n)\le n/2\). This fixes the right scale but leaves a wide constant gap.

The useful asymptotic language replaces a partition by a density. On \([-1,1]\), take \(f:[-1,1]\to[0,1]\) with \(\int_{-1}^1 f=1\), put \(g=1-f\), extend both functions by zero outside \([-1,1]\), and define
\[
M(x)=\int_{-1}^1 f(t)g(x+t)\,dt,\qquad -2\le x\le2.
\]
Swinnerton-Dyer's reduction identifies \(\mu\) with the infimum of \(\|M\|_\infty\) over these densities. Equivalently, in the constructive form on \([0,2]\), one may take step functions \(f\in[0,1]\), \(\int_0^2 f=1\), and minimize the largest nonperiodic translate integral
\[
\max_s\int f(x)\bigl(1-f(x+s)\bigr)\,dx .
\]
The reduction is the bridge between finite partitions and a continuous variational problem over step functions.

The continuous overlap has moment identities that any lower-bound method can use. Its mass is
\[
\int_{-2}^2M(x)\,dx=1.
\]
If \(E(M)=\int_{-2}^2xM(x)\,dx\), then
\[
\int_{-2}^2x^2M(x)\,dx=\frac23+\frac12E(M)^2,
\]
so the centered variance is \(2/3-E(M)^2/2\le 2/3\). A mass-one function bounded by height \(\omega\) has its smallest possible centered variance when it is a centered block of height \(\omega\) and width \(1/\omega\), giving the baseline lower bound \(\omega\ge1/\sqrt8\). Moser sharpened this moment obstruction to
\[
\sqrt{4-\sqrt{15}}\approx0.35639395869.
\]
The moment approach stalls because it constrains \(M\) only through its mass and variance, and successive refinements in this vein have not pushed the lower bound past Moser's value.

## Baselines

**Erdos averaging.** The identity \(\sum_kM_k=n^2\) gives \(M(n)>n/4\), while the middle interval construction gives \(M(n)\le n/2\). The method is completely robust but blind to all geometry of the overlap profile.

**Erdos-Scherk, Swierczkowski, Motzkin-Ralston-Selfridge.** These works improve the finite bookkeeping around difference multiplicities and narrow the interval for the constant. They do not provide a variational object whose every feasible point can be attacked by analytic constraints.

**Moser-Murdeshwar and Moser.** The function analogue brings in mass and second-moment information. It proves that too flat an overlap profile cannot fit into the available variance budget, and Moser's refinement reaches \(\sqrt{4-\sqrt{15}}\). These bounds use only low-order integral data of \(M\).

**Swinnerton-Dyer reduction.** The asymptotic combinatorial constant equals the continuous infimum over bounded step densities. This makes two complementary tactics legitimate: search for explicit step densities for upper bounds, and certify universal inequalities for continuous overlaps for lower bounds.

**Haugland step functions.** Symmetric step densities on a uniform grid turn the upper-bound objective into a finite nonperiodic cross-correlation. A 21-step construction gives \(0.382002\ldots\); a 51-step construction gives
\[
\mu\le0.3809268534330870.
\]
This supplies the best constructive target in this frame, but it does not by itself explain why every density must have a comparable peak.

## Evaluation settings

The natural output is a rigorous interval for \(\mu\). On the upper side, the evidence is an explicit step density whose nonperiodic shift overlaps can be evaluated with upward-safe arithmetic. On the lower side, the evidence must be a certificate valid for every admissible overlap, so any numerical search output counts only as exploration unless the final bound is recovered with explicit roundoff margins.

## Code framework

The constructive side starts from a symmetric grid density and evaluates the largest nonperiodic translate overlap.

```python
import numpy as np

def symmetric_step_values(left_half_values):
    """Mirror grid values on [0, 1] to a symmetric density on [0, 2]."""
    pass

def step_overlap_objective(left_half_values):
    """Return the largest nonperiodic grid-shift overlap integral."""
    pass
```

The lower-bound side needs its own scaffold that yields a machine-checkable bound valid for every admissible overlap.

```python
def certify_lower_bound():
    """Produce a rigorous lower bound on mu, valid for every admissible overlap."""
    pass
```
