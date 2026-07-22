# Beyond the Golden Angle: A Boundary-Aware Divergence Schedule

## Problem

A sunflower head grows by depositing primordia (future seeds) one at a time
from the center outward, each one further out than the last. Primordium $k$
$(k=1,\dots,N)$ is deposited at a growth-law-determined radius $r_k$ and at an
angle $\theta_k$ (degrees) that you choose. Your job is to schedule the
angles $\theta_1,\dots,\theta_N$.

**Growth law.** The radii are *not* yours to pick -- they follow a two-regime
power law fixed by the input. Let $K_0=\mathrm{round}(\text{trans\_frac}\cdot N)$.
For $k\le K_0$ (the bulk):
$r_k = R\cdot(k/N)^{p_{bulk}}$.
For $k>K_0$ (the rim/transition region, only present when trans_frac<1), with
$r_{K_0}$ from the formula above and $t=(k-K_0)/(N-K_0)$:
$r_k = r_{K_0} + (R-r_{K_0})\cdot t^{p_{rim}}$.
Always $r_N=R$ exactly. The classic self-similar case is $p_{bulk}=0.5$ with
no transition (trans_frac=1); several instances instead bend the growth rate
partway to the rim, which the golden angle's optimality proof does not
account for.

**Lateral inhibition (deposition rule).** Primordium $k$ is only a valid
deposit if it keeps its distance from *every already-placed* primordium
$j<k$ at or above an inhibition radius
$d_{\min}(k)=\alpha\cdot(r_k-r_{k-1})$ (with $r_0=0$). If any pair violates
this for any $k$, the whole schedule is infeasible (score 0). This models
lateral inhibition: a new primordium cannot form where an existing one
already suppresses growth.

**Packing uniformity (objective).** Place all $N$ primordia at
$(r_k\cos\theta_k, r_k\sin\theta_k)$. Compute their Voronoi diagram, giving
every real primordium a bounded cell (its cell is intersected against a
mirror image of itself reflected outward across the disk rim, which gives
rim primordia a fair, bounded cell instead of an infinite one -- a standard
finite-boundary treatment). Let $\text{score\_frac}\cdot N$ be the count of
the *outermost* (largest-$k$) primordia; only their cell areas feed the
score, via the coefficient of variation of that subset:
$F=-\log_{10}(\mathrm{CV}^2+0.03)$, $\mathrm{CV}^2=\mathrm{Var}(\text{areas})/\mathrm{Mean}(\text{areas})^2$.
Larger $F$ = more uniform packing near the rim. The checker reports
$\text{Ratio}=\min(1,\,F/F_{\text{baseline}}\cdot 0.1)$ where $F_{\text{baseline}}$
comes from its own naive constant-step construction.

Why score only the rim window? That is exactly where a fixed divergence
constant (the famous $137.50776\ldots^\circ$ golden angle) is known to be
only *asymptotically* optimal: for a finite, bounded disk, the primordia near
the edge run out of future neighbors to statistically average against, and
if the growth law itself bends before reaching the rim, a single constant
angle cannot track it. A good schedule should behave like the golden angle
in the bulk (it genuinely is excellent there) but adapt near the boundary.

## Input (stdin)

```
N R alpha
p_bulk trans_frac p_rim
score_frac
```
All values are given to full float precision; $N\le 60$.

## Output (stdout)

$N$ lines, one real number per line: $\theta_1,\dots,\theta_N$ in degrees
(any real value; taken modulo 360). $\theta_k$ is the absolute angular
position scheduled for primordium $k$ -- the "divergence angle" at step $k$
is the implied difference $\theta_k-\theta_{k-1}$ (e.g. a constant divergence
schedule is $\theta_k=k\cdot c \bmod 360$ for a fixed $c$).

## Feasibility

Output must have exactly $N$ finite real numbers. Every primordium $k\ge 2$
must satisfy $\min_{j<k}\lVert p_k-p_j\rVert \ge d_{\min}(k)-10^{-6}$, checked
against ALL earlier primordia (not just $k-1$). Any violation scores 0.

## Scoring

$\text{Ratio}\in[0,1]$ as above, per test case; overall score is the mean
over 10 cases. A do-nothing constant-angular-step schedule scores about
$0.1$; a well-tuned schedule can substantially beat that, especially where
the growth law bends before the rim.

## Example (worked, illustrative only)

For $N=4$, $R=10$, no transition, if you output `137.507764`, `275.015528`,
`52.523292`, `190.031056` (i.e. $\theta_k=k\cdot 137.50776^\circ \bmod 360$),
each primordium sits at radius $r_k=10\cdot(k/4)^{0.5}$ at that angle. The
checker verifies the inhibition constraint pairwise, then computes the rim
Voronoi cells and their coefficient of variation to produce a Ratio.
