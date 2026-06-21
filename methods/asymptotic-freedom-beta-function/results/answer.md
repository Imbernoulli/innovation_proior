# Asymptotic freedom: the one-loop β-function of a non-Abelian gauge theory

## Problem

Bjorken scaling in deep-inelastic scattering says the strong force must *weaken* at short
distance, so that the proton's constituents look free when probed hard — yet they are
permanently confined. In renormalization-group terms this requires a theory whose coupling
flows to zero in the ultraviolet, i.e. a **negative** one-loop β-function near g = 0. Every
renormalizable theory of scalars, fermions, and Abelian gauge fields screens (β > 0); the one
untested case is non-Abelian (Yang–Mills) gauge theory.

## Key idea

In a non-Abelian gauge theory the gauge bosons ("gluons") are themselves charged spin-1
particles. Beyond the ordinary dielectric **screening** that any charged virtual matter produces
(β > 0, as in QED), the charged gluons add a **paramagnetic** vacuum response from their spin —
**antiscreening** — of the opposite sign and larger magnitude. The one-loop β-function is the
competition between gluon antiscreening (∝ the adjoint Casimir C(G)) and quark screening (∝ the
fermion index R_net). When antiscreening wins, the coupling vanishes in the ultraviolet:
**asymptotic freedom**.

## The result

For a simple gauge group G with Dirac fermions in a representation R, the one-loop β-function is

    β(g) = μ dg/dμ = (g³ / 16π²) · [ −(11/3) C(G) + (4/3) R_net ] + O(g⁵),

where
- C(G) = quadratic Casimir of the adjoint representation (= N for SU(N)),
- R_net = Σ_r T(r) = sum of Dynkin indices of the fermion multiplets
  (= n_f · ½ for n_f Dirac fermions in the SU(N) fundamental).

For SU(N) with n_f fundamental flavors:

    β(g) = (g³ / 16π²) · [ −(11/3) N + (2/3) n_f ].

For SU(3) (color, i.e. QCD):

    β(g) = −(g³ / 16π²) · ( 11 − (2/3) n_f ) ≡ −(g³ / 16π²) · b₀,   b₀ = 11 − (2/3) n_f.

**Asymptotic freedom holds iff b₀ > 0**, i.e.

    R_net < (11/4) C(G)   ⟺ (SU(N), fundamentals)  n_f < (11/2) N   ⟺ (SU(3))  n_f < 16.5.

So a color SU(3) gauge theory is asymptotically free for up to 16 quark flavors; the physical
6 flavors give b₀ = 11 − 4 = 7 > 0.

**Running coupling.** Integrating β = −(g³/16π²) b₀ with t = ½ ln(s/M²) and b = b₀/16π²,

    ḡ²(t) = g² / ( 1 + 2 b g² t )  →  0  as t → ∞  (≈ 1/(2 b t) ~ 1/ln s).

The coupling vanishes logarithmically in the ultraviolet ⟹ matrix elements of currents approach
free-field (Bjorken-scaling) values, with **calculable logarithmic violations** set by b₀.
Toward the infrared (t < 0) the denominator passes through zero and ḡ² grows without bound —
perturbation theory loses control, consistent with confinement (the low-energy regime is outside
the one-loop result's validity).

## Derivation sketch (one loop, covariant gauge with Faddeev–Popov ghosts)

The coupling renormalization read off the quark–gluon vertex is g_bare = g Z₁/(Z₂ Z₃^{1/2}), so
with Z = 1 + δ,

    β(g) = g · Res[ 2δ₁ − 2δ₂ − δ₃ ] · (g²/16π²).

One-loop counterterm residues (coefficient of g²/16π²ε):

| counterterm | diagram(s) | residue |
|---|---|---|
| δ₂ | quark self-energy (QED-like × C(Q)) | −C(Q) |
| δ₁ | quark–gluon vertex: QED-like (C(Q) − ½C(G)) + 3-gluon graph (+3/2 C(G)) | −(C(Q) + C(G)) |
| δ₃ | gluon vacuum polarization: quark loop (−4/3 R_net) + gluon loop + 4-gluon tadpole + ghost loop (+5/3 C(G)) | (5/3) C(G) − (4/3) R_net |

The non-transverse ("bad") g^{μν} pieces of the gluon, tadpole, and ghost loops cancel among
themselves (via y Γ(y) + Γ(y+1) = 0 at y = 1 − D/2), leaving a transverse self-energy — the check
that the gauge sector and ghosts are assembled correctly. Assembling,

    Res[2δ₁ − 2δ₂ − δ₃] = 2(−C(Q)−C(G)) − 2(−C(Q)) − ((5/3)C(G) − (4/3)R_net)
                        = −(11/3) C(G) + (4/3) R_net,

with the quark Casimir C(Q) canceling (it cannot enter the universal running of g). The −2 C(G)
from the vertices and −5/3 C(G) from the self-energy combine to the −11/3 C(G).

**Checks:** gauge-parameter independence of β (the individual δ's are gauge-dependent, β is not);
identical β from the ghost–gluon, three-gluon, and four-gluon vertices (Slavnov–Taylor); the
Abelian limit C(G) → 0 reproduces QED screening (β > 0 from +4/3 R_net only).

**Physical reading.** With ε μ = 1 (c = 1), electric screening ⟺ diamagnetism. A charge-q,
spin-s particle contributes Δμ = [ −1/3 + (γ s)² ] q² to the vacuum permeability. A gluon has
s = 1 and (by gauge invariance) gyromagnetic ratio γ = 2, so Δμ = (−1/3 + 4) q² = +11/3 q² —
paramagnetic, antiscreening: the 11/3 is "4 (spin paramagnetism) − 1/3 (orbital diamagnetism)."
A spin-½ quark gives a screening −2/3 per flavor (the −2 n_f/3 coefficient). Antiscreening wins
unless there are too many quarks.

## Code (group factors → b₀ → running coupling)

```python
import numpy as np
from fractions import Fraction as F

def residues(C_G, C_Q, R_net):
    """One-loop counterterm residues (coeff of g^2/16pi^2 eps), covariant gauge + FP ghosts."""
    res_d2 = -C_Q                               # quark self-energy
    res_d1 = -(C_Q + C_G)                       # quark-gluon vertex
    res_d3 = F(5, 3)*C_G - F(4, 3)*R_net        # gluon vacuum polarization (5 diagrams)
    return res_d1, res_d2, res_d3

def one_loop_beta_coefficient(C_G, C_Q, R_net):
    """beta(g) = (g^3/16pi^2) * combo ;  combo = -(11/3)C_G + (4/3)R_net  (C_Q cancels)."""
    d1, d2, d3 = residues(C_G, C_Q, R_net)
    return 2*d1 - 2*d2 - d3

def su_n_data(N, n_f):
    C_G   = F(N)                                # adjoint Casimir
    C_Q   = F(N*N - 1, 2*N)                     # fundamental Casimir (cancels in beta)
    R_net = F(n_f, 2)                           # n_f Dirac fundamentals, T(fund)=1/2
    return C_G, C_Q, R_net

def run_coupling_sq(g0_sq, b0, t):
    """g^2(t) = g0^2 / (1 + 2 b g0^2 t),  b = b0/16pi^2,  t = (1/2) ln(s/M^2)."""
    b = b0 / (16 * np.pi**2)
    return g0_sq / (1.0 + 2.0 * b * g0_sq * t)

if __name__ == "__main__":
    N, n_f = 3, 6
    C_G, C_Q, R_net = su_n_data(N, n_f)
    combo = one_loop_beta_coefficient(C_G, C_Q, R_net)   # -7
    b0 = -combo                                          # 7  (beta = -(g^3/16pi^2) b0)
    print("combo = -(11/3)C_G + (4/3)R_net =", combo)
    print("b0 = 11 - 2 n_f/3 =", F(11) - F(2,3)*n_f, "->", b0)
    print("asymptotically free:", b0 > 0, " (need n_f < 16.5)")
    for t in [0.0, 5.0, 50.0, 500.0]:
        print(f"t={t:6.1f}  g^2(t)={run_coupling_sq(1.0, float(b0), t):.4f}")
```

Output: `combo = -7`, `b0 = 7`, asymptotically free, and `g²(t)` decreasing toward 0 as t → ∞.
