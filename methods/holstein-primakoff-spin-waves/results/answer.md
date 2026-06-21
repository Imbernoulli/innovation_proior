# The Holstein–Primakoff transformation and linear spin-wave theory

## Problem

Find the elementary excitations and the field/temperature dependence of the magnetization of a Heisenberg ferromagnet,

    H = −J Σ_⟨ij⟩ S_i·S_j + g μ_B H Σ_i S_i^z   (+ dipolar terms),   J > 0,

where the localized single-spin-flip is not an energy eigenstate (the exchange term hops it), and Bloch's plane-wave magnons solve only the one-excitation sector with no systematic, extensible, operator-level framework for adding excitations, an external field, or the dipole–dipole interaction.

## Key idea

The number of spin deviations from full alignment at a site, n_i = S − S_i^z, is a non-negative integer bounded by 2S — exactly the spectrum of a boson number operator. Represent each spin by a boson, fixing S^z and letting the su(2) algebra determine S^±:

    S_i^z = S − a_i†a_i
    S_i^+ = √(2S − a_i†a_i) · a_i = √(2S) √(1 − a_i†a_i/2S) · a_i
    S_i^- = a_i† · √(2S − a_i†a_i) = √(2S) a_i† √(1 − a_i†a_i/2S)

with [a_i, a_j†] = δ_ij. The square-root factor √(2S − a†a) does two jobs at once: it makes S^- annihilate the fully reversed state |n=2S⟩ (so the boson Fock space is truncated to match the (2S+1)-dimensional spin), and it makes the spin commutation relations hold **exactly** (not just approximately). At low excitation density (⟨a†a⟩ ≪ S, i.e. low T or large S) the square root is expanded, √(1 − a†a/2S) ≈ 1 − a†a/4S, so to leading order S^+ ≈ √(2S) a, S^- ≈ √(2S) a†. Substituting into H turns the spin Hamiltonian into a **quadratic form in bosons (non-interacting magnons) plus a controlled 1/S series of magnon–magnon interactions**.

## The transformation is exact (algebra preserved)

With n̂ = a†a and f = √(2S − n̂), using [n̂, a] = −a and a a† = n̂ + 1:

- [S^z, S^+] = −[n̂, f a] = −f[n̂, a] = f a = S^+ (and [S^z, S^-] = −S^-).
- [S^+, S^-] = [f a, a† f] = f a a† f − a†(2S − n̂)a = (n̂+1)(2S−n̂) − n̂(2S−n̂+1) = 2(S − n̂) = 2 S^z.

So the map is an exact rewriting of the spin algebra for all 0 ≤ n ≤ 2S; the 1/S expansion is the only approximation, and it is controlled.

## Magnon spectrum (linear spin-wave theory)

Keeping terms through O(S) (linear approximation S^+ ≈ √(2S)a):

    H = −(NqJS²)/2 + H_1 + H_2 + O(1/S),
    H_1 = JS Σ_⟨ij⟩ (a_i† − a_j†)(a_i − a_j) = −JS Σ_⟨ij⟩(a_i†a_j + a_j†a_i) + qJS Σ_i a_i†a_i,

with N sites and coordination number q. Fourier transform a_j = N^{−1/2} Σ_k e^{ik·r_j} a_k (preserves [a_k, a_k'†]=δ_{kk'}) diagonalizes H_1:

    H_1 = Σ_k ε_k a_k†a_k,   ε_k = qJS(1 − γ_k),   γ_k = (1/q) Σ_δ cos(k·δ),

i.e. a free gas of bosonic magnons. For a 1D chain (δ = ±a): γ_k = cos(ka) and

    ε_k = 2JS(1 − cos ka) ≈ JS(ka)²  (k → 0),

reproducing Bloch's quadratic spin-wave dispersion. The O(S⁰) term H_2 = (J/4) Σ_⟨ij⟩ [a_i†a_j†(a_i−a_j)² + (a_i†−a_j†)²a_i a_j] is the leading magnon–magnon interaction.

## External field and magnetization

The Zeeman term is diagonal in the magnon basis:

    g μ_B H Σ_i S_i^z = g μ_B H (NS − Σ_k a_k†a_k)  ⇒  ε_k(H) = qJS(1 − γ_k) + g μ_B H,

a rigid, k-independent gap — no re-diagonalization. Each magnon lowers the total spin by one unit, so with Bose occupation n_B(ε) = 1/(e^{ε/k_BT} − 1),

    M(T,H) = g μ_B ( NS − Σ_k n_B(ε_k(H)) ).

At H=0, low T, ε_k ≈ JS(ka)² gives the saturation deficit Σ_k n_B ∝ T^{3/2} — Bloch's law as a corollary. Turning on H gaps out the cheap small-k magnons, suppressing the deficit; ∂M/∂H is the intrinsic susceptibility, whose large-field behavior — together with the dipole–dipole term, which under the same linearization adds further direction-dependent quadratic contributions to ε_k — falls off slowly with field (∼ H^{−1/2} at high field for the intrinsic volume susceptibility of the domain magnetization).

## Antiferromagnet (same framework, plus Bogoliubov)

For two sublattices with opposite references (S^z = S − a†a on A, S^z = b†b − S on B), the quadratic Hamiltonian acquires anomalous pair terms a_i b_j + a_i†b_j† that number-conserving Fourier cannot diagonalize. A Bogoliubov rotation α_k = u_k a_k − v_k b_{−k}†, β_k = u_k b_{−k} − v_k a_k† with u_k² − v_k² = 1 (bosonic) and the off-diagonal-killing condition γ_k(u_k²+v_k²) + 2u_kv_k = 0 gives u_k² = ½(1/√(1−γ_k²)+1), v_k² = ½(1/√(1−γ_k²)−1), and

    ε_k = qJS √(1 − γ_k²),   1D: ε_k = 2JS|sin ka|  (linear at small k),

with a zero-point energy lowering signaling that the Néel state is not the true ground state.

## Code: verifying the transformation

```python
import numpy as np

def spin_matrices(S):
    """Standard (2S+1)-dim spin-S matrices in the |S,m> basis, m = S,...,-S."""
    dim = int(round(2 * S + 1))
    m = np.array([S - k for k in range(dim)])
    Sz = np.diag(m).astype(complex)
    Sp = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):                        # S^+|m> = sqrt(S(S+1)-m(m+1))|m+1>
        Sp[k - 1, k] = np.sqrt(S * (S + 1) - m[k] * (m[k] + 1))
    return Sz, Sp, Sp.conj().T

def boson_matrices(S):
    """Truncated bosons on the same (2S+1)-dim space, n=0..2S, |n> <-> m=S-n."""
    dim = int(round(2 * S + 1))
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):                         # a|n> = sqrt(n)|n-1>
        a[n - 1, n] = np.sqrt(n)
    return a, a.conj().T

def msqrt(M):                                        # matrix sqrt of a PSD operator
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(np.clip(w.real, 0, None))) @ V.conj().T

def hp_exact(S):
    """S^z = S - a^dag a ; S^+ = sqrt(2S - a^dag a) a ; S^- = a^dag sqrt(2S - a^dag a)."""
    a, ad = boson_matrices(S)
    n = ad @ a
    root = msqrt(2 * S * np.eye(n.shape[0]) - n)     # truncation + algebra factor
    return S * np.eye(n.shape[0]) - n, root @ a, ad @ root

def hp_linear(S):
    """Leading order: S^+ ~ sqrt(2S) a, S^- ~ sqrt(2S) a^dag (valid for <n> << 2S)."""
    a, ad = boson_matrices(S)
    n = ad @ a
    return S * np.eye(n.shape[0]) - n, np.sqrt(2 * S) * a, np.sqrt(2 * S) * ad

def comm(A, B):
    return A @ B - B @ A

if __name__ == "__main__":
    for S in [0.5, 1.0, 1.5, 2.0]:
        Sz, Sp, Sm = spin_matrices(S)
        ez, ep, em = hp_exact(S)
        repro = max(np.abs(Sz - ez).max(), np.abs(Sp - ep).max(), np.abs(Sm - em).max())
        alg = np.abs(comm(ep, em) - 2 * ez).max()     # exact: [S^+,S^-] = 2 S^z
        lz, lp, lm = hp_linear(S)
        dim = int(round(2 * S + 1))
        top = np.abs(lp[0, 1] - Sp[0, 1])             # n small: linear map exact
        bot = np.abs(lp[dim - 2, dim - 1] - Sp[dim - 2, dim - 1])  # n~2S: linear map fails
        print(f"S={S}: reproduce={repro:.1e} algebra={alg:.1e} "
              f"linear top={top:.1e} bottom={bot:.1e}")
```

Output: the exact map reproduces the spin matrices and satisfies [S^+,S^-]=2S^z to machine precision (~1e-15) for every S; the linear approximation is exact at the top of the ladder (small n, the thermally occupied sector) and errs by O(1) near full occupancy (n≈2S) — precisely the validity window of linear spin-wave theory.
