## Research question

Some systems have complex microscopic behavior that controls macroscopic effects, and in the
hardest cases the relevant fluctuations are not confined to one scale — they persist out to
macroscopic wavelengths with every intermediate scale mattering too. The sharpest clean instance is
a system near a **critical point**: the Curie point of a ferromagnet, or the liquid–vapor critical
point of a fluid. Approaching it, the correlation length ξ (the distance over which one degree of
freedom influences another) **diverges**, and the system looks self-similar across scales — drops and
bubbles of every size from atomic to macroscopic, "critical opalescence." The thermodynamic functions
become **non-analytic** there: the spontaneous magnetization vanishes as (Tc−T)^β with β≈1/3, the
correlation length grows as (T−Tc)^{−ν} with ν≈2/3 — both in three dimensions, both stubbornly *not*
the values 1/2 that the standard theory predicts.

The goal is a method that can actually **compute** this: explain how non-analytic, universal behavior
emerges from a partition function that is a sum of perfectly analytic Boltzmann factors; reproduce
the measured exponents; and explain why utterly different microscopic systems (a magnet, a fluid, an
alloy) share the *same* exponents. The obstacle is structural: a problem with fluctuations on all
length scales at once defeats every tool that works with one scale or few variables.

## Background

**The all-scales obstruction.** Analytic methods are most effective with one degree of freedom.
Ordinary numerical integration fails past ~10 variables; PDE methods past ~3; even Monte Carlo, which
can reach millions of variables, converges painfully slowly. A faithful simulation of a critical
system would need a lattice resolving every scale from atomic spacing up to the diverging ξ — a
hopeless number of coupled integrations. Special exact solutions exist (Onsager's 1944 two-dimensional
Ising model; the Thirring model; the Kondo solution) but they are isolated tricks, not a general
method.

**The non-analyticity puzzle.** For the Ising model — moments σ_i = ±1 on a lattice, energy favoring
aligned nearest neighbors, partition function Z = Σ exp(−H/kT) — the Boltzmann factor is analytic in
T for all T except 0, and a sum of analytic functions is analytic. Yet the magnet shows sharp
non-analytic behavior at Tc. The non-analyticity is real only in the thermodynamic limit of infinite
size, where the analyticity theorems no longer protect the infinite sum; but it has been a major
challenge to show *how* even an infinite sum produces such behavior.

**Mean-field / Landau theory and where it breaks.** Landau (1937) proposed that, restricting to
configurations of fixed magnetization density M, the free energy is analytic in M:
F = V{ R M² + U M⁴ }, with R ∝ (T−Tc) and U>0 constant near Tc. Minimizing over M gives M=0 above Tc
and, below, the M satisfying 0 = 2RM + 4UM³, i.e. M ∝ (Tc−T)^{1/2} — so β=1/2. The space-dependent
(Landau–Ginzburg) form F = ∫[ (∇M)² + R M² + U M⁴ − B M ] yields, from a δ-function field, a
correlation length ξ ∝ R^{−1/2} ∝ (T−Tc)^{−1/2}, so ν=1/2. Both disagree with experiment and with
Onsager. The hidden assumption is that analyticity survives once *space-dependent* fluctuations are
averaged out — that only atomic-scale fluctuations matter, exactly as in hydrodynamics; the
non-analyticity is then attributed solely to the final minimization over the single number M. This is
precisely the assumption that fails below four dimensions, where long-wavelength fluctuations on all
scales up to ξ are important. Four dimensions is the dividing line (an estimate due to Ginzburg also
locates it at four): above it Landau is right; below it the corrections are uncontrolled in general
but become *small* — proportional to (4−d) — just below four.

**The diagnostic facts, already on the table.** Before 1900, fluid experiments already showed β
closer to 1/3 than 1/2. Onsager (1944) solved the two-dimensional Ising model exactly and got ν=1
(not 1/2), a clean violation of mean field. Through the 1950s Domb, Sykes, Fisher and others extracted
three-dimensional exponents from very-high-order high-temperature series and found values in
disagreement with mean field but agreeing with experiment. By the 1960s a large experimental effort
had pinned the exponents down solidly, with Green, Fisher, Widom, and Kadanoff coordinating. The
exponents of different systems coincide — **universality** — and they obey relations among themselves
(scaling laws), and inequalities (Rushbrooke, Griffiths). The data demanded a theory beyond mean
field that could *predict* these numbers and their relations.

## Baselines

**Gell-Mann–Low renormalization group in QED (1954).** In quantum electrodynamics the measured charge
e is a long-distance property (pith balls, centimeters) while the natural scale is the electron's
Compton wavelength (~10^{−11} cm); the "bare" charge e₀ lives at short distance. Gell-Mann and Low
showed a family of effective charges e_λ, one for each momentum scale λ, interpolating between e (low
λ) and e₀ (high λ), obeying a differential equation λ de_λ/dλ = ψ(e_λ) whose ψ has a non-divergent
power series. Stueckelberg and Petermann (1953) had named the group relating different
reparametrizations the "renormalization group." The deep point: e_λ at one scale determines it at the
next, and the equation can predict that e expanded in powers of e₀ diverges. **Gap:** the
transformation carries a *fixed, single* coupling (the charge). It is a flow in a one-dimensional
space, tied to perturbative QED, with no route to the non-perturbative, strong-coupling regime of a
critical point.

**Kadanoff block spins (1966).** Near Tc, where ξ is huge, group the lattice into blocks — say
2×2×2 atoms — each acting as one effective moment. The block moments are assumed to interact through
the *same* nearest-neighbor form as the original spins, but with an effective temperature T_L and
field h_L for blocks of size L (in units of the atomic spacing). Kadanoff posited that T_{2L}, h_{2L}
are analytic functions of T_L, h_L, and that at Tc these reach L-independent fixed values. From this
single hypothesis he derived Widom's scaling laws and the exponent relations (e.g. 2−α = dν). **Gap:**
it is an *assumption*, not a calculation. It postulates that blocking preserves the two-parameter
nearest-neighbor form, and it supplies no method to compute the functions T_L → T_{2L}, h_L → h_{2L}.
So it explains the *form* of scaling and relates exponents, but cannot produce a single exponent from
the microscopic model.

**Widom scaling (1965).** A homogeneous (scaling) form for the equation of state near Tc that
accommodates non-mean-field exponents and forces relations among them. **Gap:** purely phenomenological
— no theoretical basis for the homogeneity it assumes (a fact a careful reader notices immediately).

**Onsager's exact 2D Ising solution (1944).** A complete solution for one model in one dimensionality,
proving mean field wrong (ν=1). **Gap:** a special transformation, not extensible to three dimensions,
to fluids, or to general models.

## Evaluation settings

The natural yardsticks already exist. The **Ising model** on a lattice (moments ±1, nearest-neighbor
coupling) is the paradigm; its **one-dimensional** chain is exactly soluble by the **transfer matrix**
(Z = tr T^N ≈ λ_max^N), giving free energy per spin f = −ln(2 cosh K) with K = J/kT, and — by
Frobenius — analytic for all finite T, hence no finite-temperature transition in 1D. The
**two-dimensional** Ising model supplies an exact non-mean-field benchmark. In three dimensions the
benchmarks are **high-temperature series expansions** (Domb–Sykes–Fisher) and **experiment**. The
objects to reproduce are the critical exponents (β, ν, …), the relations among them, and the
universality across systems. The diagnostic phenomena — diverging ξ, critical opalescence, and the
(Tc−T)^β magnetization curve — are the qualitative targets.

## Code framework

A 1D Ising chain has a coupling K = J/kT and an exact transfer-matrix free energy, available below as a
benchmark to check any proposed approach against.

```python
import math

def solve(K0, n_steps):
    """TODO: implement the proposed approach for the 1D Ising chain."""
    pass

def free_energy_per_spin_exact(K):
    """Exact transfer-matrix free energy per spin for the 1D Ising chain."""
    return -math.log(2.0 * math.cosh(K))
```
