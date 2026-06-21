# The axial (Adler–Bell–Jackiw) anomaly

## Problem

In a Dirac theory the axial-vector current j^μ_5 = ψ̄γ^μγ₅ψ has, classically, the divergence ∂_μ j^μ_5 = 2im₀ ψ̄γ₅ψ, which vanishes in the massless limit where the Lagrangian is invariant under chiral rotations ψ → e^{iαγ₅}ψ. Current algebra / PCAC builds on this: matrix elements of ∂·A are computed by formal manipulation of loop integrals, Ward identities, and the equations of motion. Applied to π⁰ → γγ, this reasoning (Sutherland–Veltman, 1967) predicts the amplitude T(0) = 0 in the soft-pion limit — yet π⁰ → 2γ is the dominant decay, and Steinberger's (1949) explicit nucleon loop gives a finite nonzero answer. The puzzle: which formal step is illegitimate, and what is the exact correction?

## Key idea

The formal derivation of the axial Ward identity / PCAC relation silently **shifts an integration variable inside a fermion loop**. That shift is harmless only for loops that converge well enough. The smallest closed loop attached to the axial current — the triangle with one axial vertex γ^μγ₅ and two photon vertices (AVV), the smallest because Furry's theorem kills the pure-vector analog — is **linearly divergent** (its would-be quadratic piece dies since tr{γ₅γ^αγ^β} = 0). In a linearly divergent integral, a shift r → r + a leaves a finite, calculable surface term ∝ a. That surface term cannot simultaneously respect photon gauge invariance and the axial Ward identity; physics (two real photons cannot carry J = 1; heavy fermions must decouple) forces the gauge-invariant choice, dumping the entire surface term onto the axial divergence. The classical chiral symmetry is broken by quantization — and the breaking is a fixed one-loop number.

## The result

The anomalous axial-current divergence equation (unrenormalized fields, mass m₀, coupling e₀; Bjorken–Drell metric):

    ∂_μ j^μ_5(x) = 2im₀ j_5(x) + (e₀²/16π²) ε^{μνρσ} F_{μν}(x) F_{ρσ}(x) ,

equivalently, with α₀ = e₀²/4π and F̃^{μν} = ½ε^{μνρσ}F_{ρσ},

    ∂_μ j^μ_5 = 2im₀ j_5 + (α₀/4π) ε^{ξστρ} F_{ξσ} F_{τρ} = 2im₀ j_5 + (e₀²/8π²) F_{μν} F̃^{μν} .

In massless electrodynamics the right-hand side does not vanish: the axial current is not conserved despite the chiral invariance of the Lagrangian.

## Derivation (sketch of the full argument)

1. **Formal axial Ward identity.** (p−p′)^μ Γ^5_μ = 2m₀Γ^5 + S_F^{-1}γ₅ + γ₅S_F^{-1}. Split graphs into type (a) (axial vertex on the open line) and type (b) (axial vertex on a closed loop). Type (a) telescopes with the pure-algebra identity (★) — exact. Type (b) requires the shift r → r + p′ − p.

2. **The dangerous loop.** Power counting: a 2n-photon axial loop is convergent for n ≥ 2, so the shift is legal and no anomaly arises. The triangle (n = 1) is linearly divergent; the shift leaves a surface term (Jauch–Rohrlich, 1955).

3. **Compute the triangle.** Use Rosenberg's (1963) gauge-invariant AVV expression R_{σρμ}(k₁,k₂) = Σ_i A_i (ε-tensor structures), with the A_i finite Feynman-parameter integrals. Contracting with the axial momentum:

    −(k₁+k₂)^σ R_{σρμ} = 2m₀ R_{ρμ} + 8π² k₁^ξ k₂^τ ε_{ξτρμ},

   the extra 8π² being the anomaly. The pseudoscalar triangle gives R_{ρμ} = 8π² m₀ I₀₀(k₁,k₂) k₁^ξ k₂^τ ε_{ξτρμ}.

4. **Uniqueness — the anomaly cannot be removed.** The routing/subtraction ambiguity of the linearly divergent triangle is the single term R_{σρμ}[ζ] = R_{σρμ} + ζ ε_{τσρμ}(k₁−k₂)^τ (the unique structure: dimension of a mass, three-index pseudotensor, Bose-symmetric, no kinematic singularities). Then

    k₁^μ R[ζ] = −ζ k₁^σ k₂^τ ε_{τσρμ}   (vector divergence),
    −(k₁+k₂)^σ R[ζ] = 2m₀ R_{ρμ} + (8π² − 2ζ) k₁^ξ k₂^τ ε_{ξτρμ}   (axial divergence).

   ζ = 4π² makes the axial WI normal but violates photon gauge invariance; ζ = 0 keeps gauge invariance and carries the full 8π² anomaly. The transversality of two on-shell photons (l·(k₁+k₂)=0 ⇒ l^σε₁^ρε₂^μ R[ζ] = 0) and decoupling (lim_{m₀→∞} R[ζ] = 0) each independently force **ζ = 0**. The same 8π² controls the anomalous high-energy behavior (Weinberg power counting saturated, one power above naive), cross-checking the value.

5. **Nonrenormalization at one loop.** All AVVVV, AVVVVVV, … loops are convergent enough that their shifts are legal; only the bare triangle contributes, so the coefficient is fixed at lowest order.

## Application: the π⁰ → γγ low-energy theorem

The anomaly modifies the PCAC equation for the neutral axial current:

    ∂_μ F^5_{3μ}(x) = (f_π M_π²/√2) φ_π(x) + S (α₀/4π) ε^{ξστρ} F_{ξσ} F_{τρ} ,

with S = Σ_i (axial coupling)_i Q_i² counting the charges of the fermions in the loop. The Sutherland–Veltman vanishing of the first term at the soft point now leaves the amplitude determined entirely by the anomaly. In the Gell-Mann–Lévy σ-model (PCAC canonical, F = f⁻¹ = 2m/g), the explicit lowest-order calculation shows the puzzle directly — PCAC holds without any shift, k_σ F^{σμν} = −(2mg^{-1}μ²/(k²−μ²))T^{μν}, while T(0) = 4π²g/m ≠ 0 — and gauge invariance fails by the surface term, F₄ − ½k²F₃ = −4π², p_μ F^{σμν} = −4π² ε^{σαβν}p_β q_α. Reorganizing into physical quantities (Goldberger–Treiman, pion wave-function renormalization) gives the low-energy theorem

    F = −(α/π) · 2S · (g_r(0) / (m_N g_A)) ,   Γ(π⁰→2γ) = τ⁻¹ = (m_π³/64π) F² ,

yielding τ⁻¹ ≈ 9.7 eV against the measured 7.37 ± 1.5 eV — the entire observed neutral-pion lifetime is the anomaly. S is fixed by the constituent charges (S = ½ fits), turning the π⁰ lifetime into a probe of charge structure; the analogous U-spin construction gives η → 2γ.

A regulator engineered to enforce both gauge invariance and PCAC (mass-dependent Pauli–Villars–Gupta couplings, g_i/m_i = g/m, giving T_reg(0) = 4π²(g/m − g₁/m₁) = 0) does formally remove the anomaly, but the regulator fields fail to decouple (their couplings grow as g_i = (g/m) m_i), injecting unrenormalizable infinities (∝ (g_i/m_i)^{2n} m_i^{2n}) into the strong interactions — so the anomaly cannot be removed without sacrificing renormalizability (or unitarity). It is a genuine physical effect, not a regularization artifact.

## Worked constant (the load-bearing surface term)

```python
import sympy as sp
pi = sp.pi

# AVV triangle divergences as functions of the routing/subtraction freedom ζ:
#   R_{σρμ}[ζ] = R_{σρμ} + ζ ε_{τσρμ}(k1-k2)^τ
zeta = sp.symbols('zeta', real=True)
vector_div_coeff   = zeta              # ∝ k1^μ R[ζ]; must vanish for gauge invariance
axial_anomaly_coeff = 8*pi**2 - 2*zeta # extra term in (k1+k2)·R beyond 2 m0 R_pseudoscalar

zeta_star = sp.solve(sp.Eq(vector_div_coeff, 0), zeta)[0]      # gauge invariance -> 0
anomaly   = axial_anomaly_coeff.subs(zeta, zeta_star)          # -> 8*pi^2

e0 = sp.symbols('e_0', positive=True)
anomaly_operator_coeff = e0**2/(16*pi**2)   # coefficient of ε F F in ∂_μ j^μ_5

# π0 -> 2γ low-energy theorem from the anomaly-modified PCAC
alpha, S, g_r, m_N, g_A, mu = sp.symbols('alpha S g_r m_N g_A mu', positive=True)
F_pi = -(alpha/pi) * 2*S * (g_r/(m_N*g_A))
rate = mu**3/(64*pi) * F_pi**2              # Γ(π0->2γ) = τ^{-1}

print("zeta fixed by gauge invariance:", zeta_star)            # 0
print("axial anomaly coefficient:", anomaly)                   # 8*pi**2
print("operator coefficient of ε F F:", anomaly_operator_coeff)# e_0**2/(16*pi**2)
print("F(π0->2γ) =", F_pi)
print("rate =", sp.simplify(rate))
```
