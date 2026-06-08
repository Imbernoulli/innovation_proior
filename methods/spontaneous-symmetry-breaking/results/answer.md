# Spontaneous symmetry breaking and Goldstone's theorem

## Problem

A Lagrangian can possess an exact continuous symmetry while the physical states of the world fail to share it: a ferromagnet picks a direction, a superconductor breaks particle-number/gauge symmetry, the strong interactions have a near-symmetry under chirality that the spectrum does not display. The questions: can the *ground state* break a symmetry that the *laws* respect, with no symmetry-violating term put in by hand — and if so, what is the price paid in the spectrum of excitations?

## Key idea

Yes, and the price is a massless boson. If the potential `V` is invariant under a continuous symmetry but its minimum lies not at the symmetric point but on a *degenerate set* of vacua (the "Mexican hat": a circle/sphere of minima), the vacuum must select one point, **spontaneously breaking** the symmetry. Expanding the fields about that point, the directions transverse to the vacuum set (climbing the wall) are massive, while the directions *along* the degenerate set (the flat valley) have **zero** potential curvature, hence are **exactly massless**. Each independent flat direction — each spontaneously broken continuous generator — yields one massless spinless **Nambu–Goldstone boson**. This is forced in any relativistic local theory (Goldstone's theorem), independent of the detailed dynamics.

## The mechanism — complex scalar with the Mexican-hat potential

Take a complex scalar field with global U(1) symmetry `φ → e^{iα}φ`:

$$
\mathcal{L} = \partial_\mu\phi^*\,\partial^\mu\phi - V(\phi), \qquad
V(\phi) = -\mu^2|\phi|^2 + \lambda|\phi|^4, \qquad \mu^2>0,\ \lambda>0.
$$

**Degenerate vacua.** `V` depends only on `|φ|²`. With `r=|φ|`, `dV/dr = -2\mu^2 r + 4\lambda r^3 = 0` gives

$$
|\phi|^2 = v^2 \equiv \frac{\mu^2}{2\lambda}.
$$

The modulus is fixed, the phase is free: the minima form a **circle** `|φ| = v`. The vacuum must choose a phase; choosing it breaks the U(1) with no asymmetric input. Take `⟨φ⟩ = v` (phase 0).

**Expand about the vacuum.** Write `φ = v + (σ + iπ)/√2` with `σ, π` real. The kinetic term is canonical, `∂_\muφ^*∂^\muφ = \tfrac12(\partial\sigma)^2 + \tfrac12(\partial\pi)^2`. Using `μ² = 2λv²`, `V = λ(|φ|^2 - v^2)^2 - λv^4` with

$$
|\phi|^2 - v^2 = \sqrt2\,v\,\sigma + \tfrac12(\sigma^2+\pi^2).
$$

The quadratic part of `V` is `λ(√2 v σ)^2 = 2λv^2\sigma^2 = μ^2\sigma^2`; there is **no** `π^2` term. The first pure `π` contribution is quartic, and the `σπ^2` term is cubic, so neither is a mass. Reading masses off the curvature:

$$
\boxed{m_\sigma^2 = 2\mu^2} \quad\text{(radial mode — massive)}, \qquad
\boxed{m_\pi^2 = 0} \quad\text{(angular Goldstone mode — massless)}.
$$

**Why π is massless.** The angular direction runs along the circle of degenerate vacua, where `V` is exactly constant; zero curvature means zero mass. Equivalently, near the chosen vacuum the broken U(1) starts as a *shift* of the tangent field, `π → π + √2 v α`, which forbids a standalone `π^2` mass term. The Goldstone boson is the field excitation that walks along the valley of vacua — the field-theory analog of the ferromagnet's spin wave.

## Goldstone's theorem (general statement and proof)

**Statement.** In a relativistic local quantum field theory, if a continuous global symmetry with conserved Noether current `j_μ` (`∂^μ j_μ = 0`) is spontaneously broken — i.e. there is a field `φ` with `⟨0|\,i[Q,φ]\,|0⟩ = ⟨δφ⟩ ≠ 0` — then the spectrum contains a massless spinless boson for each broken generator.

**Proof (Goldstone–Salam–Weinberg).** Consider `J_μ(x) = ⟨0|j_μ(x)\,φ(0)|0⟩`. Lorentz covariance (φ scalar, `j_μ` a vector) forces

$$
J_\mu(x) = \partial_\mu J(x), \qquad J(x)\ \text{Lorentz-invariant.}
$$

Current conservation gives `0 = \partial^\mu J_\mu = \Box J`, so in momentum space

$$
\tilde J_\mu(k) = \lambda\, k_\mu\, \delta(k^2), \qquad \lambda \neq 0,
$$

with `λ ≠ 0` forced by the breaking condition `⟨δφ⟩ ≠ 0` (`∫ J_0` reconstructs `⟨[Q,φ]⟩`). The support at `k² = 0` means a **massless** physical state couples to both the broken current and the order parameter: the Nambu–Goldstone boson.

**Counting.** The number of massless bosons = dimension of the vacuum manifold = number of broken generators. (One real field, double well, *discrete* `φ→−φ`: no flat direction, massive only, no Goldstone. Complex field, circle: one. `O(N)` on a sphere `Σφ_i²=v²`: `N−1`.)

## Fermionic realization (Nambu–Jona-Lasinio)

The mechanism also generates **mass as a dynamical gap**, by analogy with the BCS energy gap. Start from massless fermions with a chirally symmetric four-fermion interaction (invariant under number `ψ→e^{iα}ψ` and chirality `ψ→e^{iγ_5α}ψ`):

$$
\mathcal{L} = -\bar\psi\gamma^\mu\partial_\mu\psi + g\big[(\bar\psi\psi)^2 - (\bar\psi\gamma_5\psi)^2\big].
$$

A self-consistent condensate `⟨\barψψ⟩ ≠ 0` spontaneously breaks chiral symmetry and gives the fermion a mass `M ∼ 2g⟨\barψψ⟩` determined by the gap equation (cutoff `Λ`)

$$
1 = \frac{2g\Lambda^2}{\pi^2}\left[1 - \frac{M^2}{\Lambda^2}\ln\!\left(1+\frac{\Lambda^2}{M^2}\right)\right],
$$

with a nontrivial `M ≠ 0` solution above a critical coupling. The bound states are a pseudoscalar `0⁻` (`\barψγ_5ψ`) at **mass 0** — the Goldstone boson — and a scalar `0⁺` (`\barψψ`) at mass `2M` (the massive "radial" partner). Identifying the `0⁻` with the **pion**: the nucleon mass is the dynamically generated gap, the pion is the (near-)Goldstone boson of (approximate) chiral symmetry, its small mass coming from a small explicit chiral-symmetry-breaking term, and the soft-pion relations (Goldberger–Treiman, `g_π ≈ 2M g_A G`) follow. The Bogoliubov–Valatin equations `Eψ_{p,+} = εψ_{p,+} + Δψ†_{−p,−}`, `E=√(ε²+Δ²)`, are formally the Dirac equation with the gap `Δ` in the mass slot — the bridge from superconductivity to particle mass.

## Worked symbolic check

```python
import sympy as sp

mu2, lam = sp.symbols('mu2 lambda', positive=True)
sigma, pi = sp.symbols('sigma pi', real=True)
r = sp.symbols('r', positive=True)

# vacuum: a circle of degenerate minima, |phi| = v
V_r = -mu2*r**2 + lam*r**4
v = [s for s in sp.solve(sp.diff(V_r, r), r) if s != 0][0]   # v = sqrt(mu2/(2 lambda))

# expand phi = v + (sigma + i pi)/sqrt(2)
mod2 = (v + sigma/sp.sqrt(2))**2 + (pi/sp.sqrt(2))**2         # |phi|^2
V = sp.expand(-mu2*mod2 + lam*mod2**2)

m2_sigma = sp.simplify(sp.diff(V, sigma, 2).subs({sigma:0, pi:0}))
m2_pi    = sp.simplify(sp.diff(V, pi,    2).subs({sigma:0, pi:0}))
print("m_sigma^2 =", m2_sigma)   # 2*mu2  -> radial mode MASSIVE
print("m_pi^2    =", m2_pi)       # 0      -> angular Goldstone mode MASSLESS
```

## Causal chain

Continuous symmetry + wrong-sign quadratic term ⇒ minimum slides onto a degenerate set of vacua ⇒ the vacuum picks one point, breaking the symmetry with no asymmetric input ⇒ radial fluctuation is massive (`m² = 2μ²`), angular fluctuation along the degenerate valley is exactly massless ⇒ current conservation + Lorentz covariance make this a theorem: one massless spinless boson per broken continuous generator ⇒ on the fermion side, mass becomes a dynamical gap and the massless pseudoscalar is the pion.
