# The valence-bond-solid (VBS / AKLT) ground state

## Problem

Exhibit a concrete, isotropic, translationally invariant integer-spin antiferromagnetic chain whose
ground state can be written in closed form and *rigorously* shown to be: unique, with no broken
symmetry (no Néel order), exponentially decaying correlations, and a finite energy gap. This
realizes the conjectured "massive" integer-spin (Haldane) phase as a theorem rather than a
field-theory prediction.

## Key idea

Represent each spin-1 as two spin-½'s symmetrized. On every nearest-neighbor bond, lock one spin-½
from each of the two sites into a singlet (a *valence bond*); use exactly one bond per link. Because
each neighboring pair of physical spins then shares one singlet (spin 0) plus two free spin-½'s
(spin ≤ 1), the pair can have total spin 0 or 1 but **never 2**. So the state lies in the kernel of
the projector onto total spin 2 of every bond. Taking the parent Hamiltonian to be a sum of those
positive projectors makes H ≥ 0 with the valence-bond-solid (VBS) state at energy exactly 0 — an
exact ground state, with no diagonalization. The construction generalizes: on a coordination-z
bipartite lattice take s = z/2 and project out the top spin z of each bond (chain: z = 2 ⇒ s = 1).

## The model and the ground state

Parent Hamiltonian (spin-1 chain), with x = Sᵢ·Sᵢ₊₁:

  H = Σᵢ P₂(Sᵢ + Sᵢ₊₁),  P₂ = (1/6)x² + (1/2)x + 1/3.

P₂ is the orthogonal projector onto the total-spin-2 subspace of a neighboring pair: with
(Sᵢ+Sᵢ₊₁)² = 4 + 2x, x equals −2, −1, +1 on the J = 0, 1, 2 subspaces, and (1/6)x²+(1/2)x+1/3
equals 0, 0, 1 there. Equivalently H = Σ [Sᵢ·Sᵢ₊₁ − β(Sᵢ·Sᵢ₊₁)²] + const with β = −1/3 (an overall
factor 2 and an additive constant aside): the special bilinear–biquadratic point.

VBS ground state (Schwinger-boson form):

  |ψ⟩ = Πᵢ (a†ᵢ b†ᵢ₊₁ − b†ᵢ a†ᵢ₊₁) |0⟩,

each factor a singlet bond between the auxiliary spin-½'s of adjacent sites; a spin-1 is the
symmetric pair at each site. Matrix-product form (bond dimension D = 2): for σ ∈ {+1, 0, −1},

  B^{+1} = √(2/3) [[0,1],[0,0]],  B^0 = −(1/√3) [[1,0],[0,−1]],  B^{−1} = √(2/3) [[0,0],[−1,0]],

right-normalized (Σ_σ Bˢ Bˢ† = 1), and |ψ⟩ = Tr(B^{σ₁}…B^{σ_N}) on a ring.

## Rigorous results (one dimension, β = −1/3)

1. **Exact ground state.** H ≥ 0 (sum of projectors) and H|ψ⟩ = 0, so |ψ⟩ is a zero-energy ground
   state.
2. **Boundary structure / edge modes.** An open chain has a four-fold ground-state degeneracy = two
   free edge spin-½'s (a spin-1 triplet ⊕ a spin-0 singlet). A periodic chain has a **unique**
   ground state.
3. **Correlations.** From the geometric overlap series (each closed loop = 2, each bond contraction
   = 3⁻¹; open-chain overlap = δδ(3^L−1)/2 + δδ, ring normalization = 3^L + 3),

   ⟨S^a₀ S^b_r⟩ = δ^{ab} (−1)^r (4/3) 3^{−r}  for all r > 0,

   so correlations decay **exponentially** with correlation length ξ = 1/ln 3 ≈ 0.91; there is **no
   Néel order**. (Hidden "string" order is nonzero, value −4/9: every nonzero spin is followed,
   after a run of 0's, by a spin of opposite sign — diluted antiferromagnetism.)
4. **Uniqueness in infinite volume.** The infinite-volume expectation ω(A) = lim_L ⟨Ω,AΩ⟩/⟨Ω,Ω⟩
   equals a finite-volume expectation (the boundary factors cancel; same state for all four boundary
   choices). The finite-volume lemma — any state annihilated by every bond term is a combination of
   the four VBS states — plus this factorization gives a **unique** infinite-volume ground state.
   All truncated correlations decay: |ω(AB) − ω(A)ω(B)| ≤ 3^{−(d−2)} ‖A‖‖B‖.
5. **Finite gap (uniform in L).** Ground spaces of growing subchains are nested (a ground state of
   H_{1,n} is one of H_{1,n−1}), so P_n = 1 − Q_n increases in n. Telescoping
   P_L = Σ_{n≥l}(P_{n+1}−P_n) + P_l, bounding each increment locally (Cauchy–Schwarz), using that
   well-separated window projectors are orthogonal, and the key estimate ε(l) ≤ c·3^{−l} (a state
   orthogonal to the (n+1)-site ground space is nearly traceless; tracing costs 3^{−l}), gives
   P_L ≤ 16(l+1)ε(l) + C·H_{1,L}. Choosing l (independent of L) with 16(l+1)ε(l) < 1 yields
   H_{1,L} ≥ ε P_L with ε > 0 independent of L. Hence a **finite gap**, surviving to infinite
   volume.

So the spin-1 chain at β = −1/3 has, provably, a unique gapped disordered ground state with
exponentially decaying correlations and no broken symmetry — the first rigorous example of the
massive integer-spin phase.

## Generalizations

- **General lattice:** coordination z, s = z/2, H = Σ_{⟨ij⟩} P_z(Sᵢ+Sⱼ); the one-bond-per-link VBS
  is an exact ground state. In one dimension the VBS exists iff the spin is integer (so integer vs
  half-integer chains differ in kind). In dimension > 1 the criterion is lattice/coordination, not
  parity of 2s.
- **Hexagonal lattice (2D), s = 3/2:** H = Σ P₃(Sᵢ+Sⱼ) has an exact VBS ground state; a
  self-avoiding-walk (random-loop) bound gives exponential decay, ξ₀ = 1/ln(√6/2) ≈ 4.93 (≈ 3.54
  with the exact connective constant), hence no Néel order — a disordered exact ground state above
  one dimension.
- **Cayley tree (caution):** for coordination z ≥ 5 the VBS has Néel order (and is gapless / not
  unique); for z = 3 it is disordered. "VBS ⇒ disordered" holds only at low coordination.
- **Variational placement:** per-bond trial energies E_VBS = −4/3 − 2β, E_dimer = −1 − 8β/3,
  E_Néel = −1 − 2β. At the realistic Heisenberg point β = 0, E_VBS = −4/3 vs the true ≈ −1.40
  (higher by only ≈ 0.07), beating both alternatives — the VBS is a good picture of the realistic
  massive phase; a transition to the dimerized phase is expected near β = 1/2.
- **Majumdar–Ghosh:** the same telescoping argument proves the spin-½ MG chain has a gap above its
  two dimerized ground states (the symmetry-breaking side of the integer/half-integer dichotomy).

## Verification

Two independent computations confirm the result: exact diagonalization of H = Σ P₂ on a small ring
gives ground energy exactly 0 and a finite gap; the D = 2 matrix-product transfer matrix has leading
eigenvalue 1 and next eigenvalue −1/3, giving ξ = 1/ln 3 and ⟨S₀ᶻS_rᶻ⟩ = (4/3)(−1/3)^r.

```python
import numpy as np

Sz = np.diag([1.0, 0.0, -1.0])
Sp = np.array([[0, np.sqrt(2), 0], [0, 0, np.sqrt(2)], [0, 0, 0]], float)  # S^+
Sm = Sp.T.conj()
Sx, Sy = 0.5 * (Sp + Sm), -0.5j * (Sp - Sm)

def two_site_SS():
    return (np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)).real      # S_i . S_{i+1}

def P2():
    SS = two_site_SS()                                                      # spin-2 projector
    return (1/6) * (SS @ SS) + 0.5 * SS + (1/3) * np.eye(9)

def H_ring(N):
    dim = 3**N
    H = np.zeros((dim, dim))
    p2 = P2().reshape(3, 3, 3, 3)
    for i in range(N):
        j = (i + 1) % N
        full = np.zeros((dim, dim))
        for a in range(dim):
            idx = np.unravel_index(a, (3,) * N)
            for si in range(3):
                for sj in range(3):
                    amp = p2[si, sj, idx[i], idx[j]]
                    if amp == 0.0:
                        continue
                    nidx = list(idx); nidx[i] = si; nidx[j] = sj
                    b = np.ravel_multi_index(tuple(nidx), (3,) * N)
                    full[b, a] += amp
        H += full
    return H

def ed_demo(N=6):
    w = np.linalg.eigvalsh(H_ring(N))
    print(f"[ED N={N}] ground energy = {w[0]:.10f}  (VBS => 0)")
    print(f"[ED N={N}] gap = {w[1] - w[0]:.6f}")

sq = np.sqrt
B = {+1: sq(2/3) * np.array([[0, 1], [0, 0]], float),
      0: -1/sq(3) * np.array([[1, 0], [0, -1]], float),
     -1: sq(2/3) * np.array([[0, 0], [-1, 0]], float)}
sigma = {+1: 1.0, 0: 0.0, -1: -1.0}

def transfer(op=None):
    T = np.zeros((4, 4))
    for s in (+1, 0, -1):
        c = 1.0 if op is None else op[s]
        T += c * np.kron(B[s], B[s])
    return T

def mps_demo(rmax=6):
    T, Tz = transfer(), transfer(op=sigma)
    ev = sorted(abs(np.linalg.eigvals(T)))
    print(f"[MPS] leading eig = {ev[-1]:.6f} (=1); xi = {1/np.log(ev[-1]/ev[-2]):.6f} (1/ln3={1/np.log(3):.6f})")
    L = 200
    for r in range(1, rmax + 1):
        num = np.trace(np.linalg.matrix_power(T, L - r) @ Tz @ np.linalg.matrix_power(T, r - 1) @ Tz)
        den = np.trace(np.linalg.matrix_power(T, L))
        print(f"[MPS] <S0z S{r}z> = {num/den:+.6f}   (4/3)(-1/3)^{r} = {(4/3)*(-1/3)**r:+.6f}")

if __name__ == "__main__":
    ed_demo(6)
    mps_demo()
```
