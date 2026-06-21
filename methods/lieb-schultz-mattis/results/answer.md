# The twist-state theorem for the spin-½ antiferromagnetic chain

## Problem

For the one-dimensional spin-½ Heisenberg antiferromagnet H = Σ_i **S**_i·**S**_{i+1}, decide two questions that the Bethe ansatz cannot deliver and that variational energies cannot diagnose reliably: is the ground state unique, and is there a finite energy gap above it as N → ∞? The answer must hold for the *genuine* isotropic model, rigorously, not for a free-fermion caricature.

## Key idea

Two variational inequalities settle it, with an exactly soluble free-fermion model as the corroborating bridge.

1. **Soluble bridge (XY model).** The Jordan–Wigner string turns the spin-½ XY chain into free spinless fermions; Bogoliubov diagonalization gives the exact spectrum, showing a gap that closes only at the isotropic point, where a unique ground state coexists with linearly dispersing gapless excitations.
2. **Uniqueness (genuine Heisenberg).** After a sublattice rotation the Hamiltonian has all-non-positive off-diagonal matrix elements; a variational argument on |C| forces the ground-state amplitudes to be all nonzero and same-signed (Marshall–Peierls), hence the ground state is nondegenerate.
3. **Gaplessness (genuine Heisenberg).** A slowly winding twist of the ground state, Ψ_k = exp(ik Σ_n n S^z_n) Ψ_0 with k = 2π/N, has energy ≤ E_0 + 2π²/N and is orthogonal to the ground state **because the spin is half-integer**; so the gap vanishes as N → ∞.

---

## 1. The soluble XY model (Jordan–Wigner → free fermions)

Hamiltonian, with anisotropy γ:
H_γ = Σ_i [(1+γ) S^x_i S^x_{i+1} + (1−γ) S^y_i S^y_{i+1}].

Lowering operators a_i = S^x_i − iS^y_i, with S^z_i = a_i^† a_i − ½, are on-site fermionic ({a_i,a_i^†}=1, a_i²=0) but off-site bosonic ([a_i,a_j]=0) — a quadratic form in them is not diagonalizable by a linear canonical transformation. The **Jordan–Wigner transformation** repairs the statistics with a string:

c_i = exp[πi Σ_{j=1}^{i−1} a_j^† a_j] a_i, c_i^† = a_i^† exp[−πi Σ_{j=1}^{i−1} a_j^† a_j],

giving true fermions {c_i, c_j^†} = δ_ij, {c_i, c_j} = 0, with c_i^† c_i = a_i^† a_i. For free ends,

H_γ = ½ Σ_{i=1}^{N−1} [(c_i^† c_{i+1} + γ c_i^† c_{i+1}^†) + h.c.],  𝔑 = Σ_i c_i^† c_i = Σ_i (S^z_i + ½).

(For a cyclic chain the wrap-around bond carries the full string, giving a boundary factor exp(iπ𝔑)+1: periodic spin BC map to periodic/antiperiodic fermion BC according to the parity of 𝔑; the boundary term is O(1/N).)

**Bogoliubov diagonalization of a general real quadratic Fermi form.** For H = Σ_{ij}[c_i^† A_ij c_j + ½(c_i^† B_ij c_j^† + h.c.)] (A symmetric, B antisymmetric), seek η_k = Σ_i(g_ki c_i + h_ki c_i^†). With φ_k = g_k + h_k, ψ_k = g_k − h_k:

φ_k(A − B) = Λ_k ψ_k,  ψ_k(A + B) = Λ_k φ_k  ⇒  φ_k(A − B)(A + B) = Λ_k² φ_k.

Since (A+B)^T = A−B, (A−B)(A+B) is symmetric positive semidefinite ⇒ Λ_k² ≥ 0. Then

**H = Σ_k Λ_k η_k^† η_k + ½(Σ_i A_ii − Σ_k Λ_k).**

For the c-cyclic XY model the eigenvectors are plane waves φ_kj = √(2/N) sin kj or cos kj, k = 2πm/N, with

**Λ_k² = 1 − (1 − γ²) sin²k**,  taken Λ_k ≥ 0 (particle-hole convention, ground state = fermion vacuum η_k Ψ_0 = 0).

Ground-state energy:
E_0 = −½ Σ_k Λ_k ⟶ **E_0/N = −(1/π) ℰ(1 − γ²)** (complete elliptic integral),
with limits E_0/N = −1/π (isotropic, γ=0) and −½ (Ising, γ=1).

**Gap.** min_k Λ_k² = γ² at k = ±π/2, so the gap is |γ| — it closes **only** at γ = 0, where Λ_k = |cos k| ≈ |q| near k = π/2 + q (linear, gapless). For N even and not divisible by 4 the ground state is nondegenerate. So the isotropic XY model has a unique, gapless ground state — but this is a free model; the genuine Heisenberg chain (with S^z S^z giving a fermion interaction) needs the following.

## 2. Theorem 1 — nondegeneracy of the ground state

**Statement.** For the linear chain of spin-½'s with nearest-neighbor antiferromagnetic Heisenberg interactions, the ground state is nondegenerate (hence total S = 0).

**Proof.** Work in the S^z_total = 0 sector; basis = Ising configurations Φ_μ (N/2 up, N/2 down). Rotate the B sublattice by π about z (S^{x,y}_j → −S^{x,y}_j, S^z_j → S^z_j):

H' = Σ S^z_i S^z_j − ½ Σ (S^+_i S^-_j + S^-_i S^+_j),

so every off-diagonal matrix element in the Ising basis is ≤ 0. For Ψ_0 = Σ_μ C_μ Φ_μ (C real), Schrödinger reads (E − ε_μ)C_μ = ½ Σ_{μ'(μ)} C_{μ'}, ε_μ = ⟨Φ_μ|Σ S^z_i S^z_j|Φ_μ⟩.

*Lemma 1 (all C_μ ≠ 0).* Suppose C_μ = 0 for μ ∈ {μ_1,…,μ_r}. Some connected C_{μ'} at one such μ_p is nonzero (else H' block-decomposes, impossible by connectivity of the flip-flop), so 0 = Σ_{μ'(μ_p)} C_{μ'} needs C's of both signs. The trial state Ψ_0' = Σ_μ |C_μ| Φ_μ is not an eigenstate (|C_{μ_p}| = 0 but Σ|C_{μ'}| ≠ 0), so E_0' > E_0. But with all off-diagonal elements negative,
E_0' = Σ ε_μ C_μ² − ½ Σ_μ Σ_{μ'(μ)} |C_μ||C_{μ'}| ≤ Σ ε_μ C_μ² − ½ Σ_μ Σ_{μ'(μ)} C_μ C_{μ'} = E_0,
contradiction. Hence all C_μ ≠ 0.

*Lemma 2 (Marshall–Peierls).* For a ground state, equality E_0' = E_0 must hold, which requires every connected C_μ C_{μ'} > 0; connectivity ⇒ all C_μ share one sign.

Two ground states in the S^z = 0 sector would both be sign-definite ⇒ not orthogonal ⇒ impossible. Marshall: at least one ground state has S = 0; a second would carry an S^z = 0 member, excluded. **Ground state nondegenerate.** (Generalizes to any bipartite lattice, any dimension.)

## 3. Theorem 2 — absence of an energy gap

**Statement.** For the cyclic spin-½ Heisenberg chain there is an excited state with excitation energy → 0 as N → ∞.

**Proof.** Define the twist state

**Ψ_k = exp(ik Σ_n n S^z_n) Ψ_0 ≡ 𝒪^k Ψ_0.**

*Orthogonality (k = 2πm/N, m odd).* With U_z the one-site cyclic translation (U_z **S**_i U_z^{−1} = **S**_{i+1}, S_{N+1} = S_1) and [H, U_z] = 0, the nondegenerate ground state obeys U_z Ψ_0 = e^{iα} Ψ_0. Then

⟨Ψ_0|Ψ_k⟩ = ⟨Ψ_0|U_z 𝒪^k U_z^{−1}|Ψ_0⟩,  U_z 𝒪^k U_z^{−1} = 𝒪^k exp(ikN S^z_1) exp(−ik Σ_n S^z_n).

Ψ_0 is a singlet ⇒ exp(−ik Σ_n S^z_n) Ψ_0 = Ψ_0. Since S^z_1 = ±½, exp(ikN S^z_1) = diag(e^{ikN/2}, e^{−ikN/2}); for k = 2πm/N with **m odd**, kN/2 = πm and e^{±iπm} = −1, so exp(ikN S^z_1) = −1. Hence
**⟨Ψ_0|Ψ_k⟩ = −⟨Ψ_0|Ψ_k⟩ = 0.**
(The −1 requires half-integer spin: for integer S, e^{ikNS^z_1} = +1 and orthogonality fails.)

*Energy (k = 2π/N).* Since 𝒪^{−k} S^x_n 𝒪^k = S^x_n cos kn + S^y_n sin kn, 𝒪^{−k} S^y_n 𝒪^k = −S^x_n sin kn + S^y_n cos kn, 𝒪^{−k} S^z_n 𝒪^k = S^z_n,

𝒪^{−k} H 𝒪^k = H + (cos k − 1) Σ_n (S^x_n S^x_{n+1} + S^y_n S^y_{n+1}) + sin k Σ_n (S^x_n S^y_{n+1} − S^y_n S^x_{n+1}).

Expectation in Ψ_0: (i) ⟨H⟩ = E_0; (ii) cos k − 1 = −½(2π/N)² + O(N^{−4}), times an O(N) bounded bond sum, ≤ (2π/N)²·(N/2) + O(N^{−3}); (iii) the last sum ∝ ⟨Ψ_0|[Σ_n n S^z_n, H]|Ψ_0⟩ = 0. Therefore

**⟨Ψ_k|H|Ψ_k⟩ ≤ E_0 + 2π²/N.**

A state orthogonal to the nondegenerate ground state with energy within 2π²/N of E_0 exists for every even N ⇒ **no energy gap**. (Two dimensions, N×M torus, M = O(N^ν): 𝒪^k = exp(ik Σ_{n,m} n S^z_{n,m}) gives ⟨Ψ_k|H|Ψ_k⟩ ≤ E_0 + 2π²/N^{1−ν}; for a true N×N lattice the trial state is too crude to be decisive.)

## Result

The isotropic spin-½ Heisenberg antiferromagnetic chain has a **unique ground state** (Theorem 1) and a **gapless excitation spectrum** (Theorem 2). The conclusion rests on the half-integer value of the spin, entering through the sign exp(ikN S^z_1) = −1 that makes the twist state orthogonal to the ground state; for integer spin the argument does not apply.

## Optional worked numerical check (soluble XY model)

```python
import numpy as np
from scipy.special import ellipe

def xy_spectrum(N, gamma):
    """Exact single-particle energies Lambda_k of the c-cyclic spin-1/2 XY chain
    via Jordan-Wigner + Bogoliubov:  Lambda_k^2 = 1 - (1 - gamma^2) sin^2 k."""
    m = np.arange(-N // 2, N // 2)        # k = 2*pi*m/N
    k = 2 * np.pi * m / N
    Lambda = np.sqrt(1.0 - (1.0 - gamma**2) * np.sin(k)**2)   # particle-hole convention, >= 0
    return k, Lambda

def xy_ground_state_energy_per_site(N, gamma):
    """E_0 = -1/2 sum_k Lambda_k ; per site."""
    _, Lambda = xy_spectrum(N, gamma)
    return -0.5 * Lambda.sum() / N

def xy_gap(N, gamma):
    """Minimum excitation energy = |gamma|, closing only at the isotropic point gamma=0."""
    _, Lambda = xy_spectrum(N, gamma)
    return Lambda.min()

if __name__ == "__main__":
    N = 20000
    # Isotropic point: gapless, E_0/N -> -1/pi
    print("gamma=0  E0/N =", xy_ground_state_energy_per_site(N, 0.0),
          " (-1/pi =", -1/np.pi, "),  gap =", xy_gap(N, 0.0))
    # Ising limit: gapped, E_0/N -> -1/2
    print("gamma=1  E0/N =", xy_ground_state_energy_per_site(N, 1.0),
          " (-1/2),  gap =", xy_gap(N, 1.0))
    # Thermodynamic-limit ground energy from the elliptic integral: -(1/pi) E(1 - gamma^2)
    for g in (0.0, 0.5, 1.0):
        print(f"gamma={g}: -(1/pi)*ellipe(1-g^2) =", -(1/np.pi) * ellipe(1.0 - g**2))
```
