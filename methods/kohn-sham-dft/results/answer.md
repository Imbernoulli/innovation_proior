# Kohn–Sham Density Functional Theory

## Problem

Find the ground-state energy and density of N interacting electrons in an external
potential v(r) without solving for the 3N-dimensional wavefunction. The
wavefunction approach hits an exponential wall — a chemically accurate trial Ψ
needs ~p^{3N} parameters — so it is limited to N of order ten. The density n(r)
has three components for any N and is therefore the variable to build on.

## Key idea

1. **The density determines everything (Hohenberg–Kohn).** The ground-state
   density fixes the external potential up to a constant, hence the Hamiltonian and
   all properties. So there is a *universal* functional F[n] (independent of v) with
   the variational principle E_v[n] = ∫v n dr + F[n] ≥ E₀, equality at the true n.

2. **Map the interacting system onto a non-interacting one of the same density.**
   Do not approximate the kinetic energy locally (that is what makes Thomas–Fermi
   predict no bonds). Instead split T[n] = T_s[n] + T_c[n], where T_s[n] is the
   kinetic energy of *non-interacting* electrons with density n (computed exactly
   from orbitals), and collect the small remainder plus the non-classical
   interaction into one exchange-correlation functional:
     F[n] = T_s[n] + U_H[n] + E_xc[n],
     E_xc[n] ≡ (T[n] − T_s[n]) + (U[n] − U_H[n]),  U_H[n] = ½∬ n n′/|r−r′|.
   Minimizing E_v[n] gives an Euler–Lagrange equation identical in form to that of
   non-interacting electrons in an effective potential, so the interacting density
   is reproduced by solving single-particle equations self-consistently.

3. **Approximate only E_xc, and do it locally (LDA).** E_xc is the smallest term,
   so a crude local model is tolerable. Use the exchange-correlation energy per
   electron of the uniform electron gas evaluated at the local density.

## The Kohn–Sham equations (Hartree atomic units, ℏ = m = e = 1)

Single-particle eigenproblem:

  ( −½∇² + v_eff(r) ) φ_i(r) = ε_i φ_i(r)

Density from the N lowest occupied orbitals:

  n(r) = Σ_i^{occ} |φ_i(r)|²

Effective potential (self-consistent — depends on n):

  v_eff(r) = v(r) + ∫ n(r′)/|r − r′| dr′ + v_xc(r),   v_xc(r) = δE_xc[n]/δn(r)

Solve by iteration: guess n → build v_eff → diagonalize → rebuild n → repeat to
self-consistency.

Total energy (band sum minus double counting):

  E = Σ_i ε_i − ½∬ n(r)n(r′)/|r−r′| dr dr′ − ∫ v_xc(r) n(r) dr + E_xc[n].

Limits: dropping E_xc and v_xc gives the Hartree self-consistent equations exactly;
approximating T_s itself by the local uniform-gas value n^{5/3} gives Thomas–Fermi.
With the exact E_xc the equations are formally exact — all many-body effects reside
in E_xc.

## Local density approximation

  E_xc^{LDA}[n] = ∫ e_xc(n(r)) n(r) dr,   v_xc^{LDA}(r) = d[n e_xc(n)]/dn |_{n=n(r)}

with e_xc(n) the xc energy per electron of the uniform electron gas. The exchange
part is analytic:

  E_x^{LDA}[n] = −(3/4)(3/π)^{1/3} ∫ n(r)^{4/3} dr,
  v_x^{LDA}(r) = −(3/π)^{1/3} n(r)^{1/3}.

This fixes the coefficient that the older local-exchange stand-in left as a free
parameter: v_x^{LDA} = (2/3) × the whole-Fermi-sphere local-exchange potential, the
2/3 forced by the functional derivative rather than chosen. The correlation part
e_c(n) is a known uniform-gas number from interpolation formulas (e.g. Wigner) and
improvable to higher precision; r_s = (3/4πn)^{1/3}.

## Why LDA works beyond slowly-varying densities (exchange-correlation hole)

Define the xc hole n_xc(r,r′) = g(r,r′) − n(r′), the depletion of other electrons
around an electron at r. By the adiabatic connection — scale the interaction
U → λU, 0 ≤ λ ≤ 1, holding the density fixed at the physical n, with the
Hellmann–Feynman theorem dE_λ/dλ = ⟨dH_λ/dλ⟩ — one gets the formally exact result

  E_xc[n] = ½ ∫ dr ∫ dr′ n(r) n̄_xc(r,r′)/|r − r′|,   n̄_xc = ∫₀¹ n_xc^λ dλ,

i.e. E_xc is each electron interacting with its own coupling-averaged hole. Every
λ-hole obeys the **sum rule** ∫ n_xc^λ(r,r′) dr′ = −1 (one screened electron), so
the average does too. Because the Coulomb kernel is isotropic, E_xc depends mainly
on the hole's spherical average and normalization, not its detailed shape. The LDA
hole (the uniform-gas hole) has the *exact* normalization, so it gives good E_xc
even for strongly inhomogeneous atoms, and its exchange/correlation errors cancel
systematically. Improvements that *break* the sum rule (e.g. a naive gradient
expansion of the hole) tend to do worse, which guides the construction of better
functionals.

## Worked numerical example: self-consistent 1D Kohn–Sham solver

Spin-paired electrons in a 1D harmonic well v(x) = x², atomic units, exchange-only
LDA (correlation omitted for brevity — it slots in as another additive piece of
e_xc and its potential). The bare 1/|x−x′| diverges in 1D, so the Hartree kernel
is softened to 1/√((x−x′)²+1); in 3D one uses the bare Coulomb kernel. Every block
maps to one step of the equations above.

```python
import numpy as np

def build_kinetic(x):                         # T_s: -1/2 d^2/dx^2 (finite diff.)
    n = len(x); h = x[1] - x[0]
    lap = (np.diag(np.full(n, 2.0))
           + np.diag(np.full(n-1, -1.0), 1)
           + np.diag(np.full(n-1, -1.0), -1)) / h**2
    return 0.5 * lap

def density(psi_gn, occ, x):                  # n(x) = sum_n f_n |phi_n|^2
    h = x[1] - x[0]; n = np.zeros_like(x)
    for i, f in enumerate(occ):
        if f:
            psi = psi_gn[:, i]
            psi = psi / np.sqrt(np.sum(psi**2) * h)
            n += f * psi**2
    return n

def exchange_lda(n, x):                        # E_x = -(3/4)(3/pi)^(1/3) int n^(4/3)
    h = x[1] - x[0]; c = (3.0/np.pi)**(1.0/3.0)
    E_x = -(3.0/4.0) * c * np.sum(n**(4.0/3.0)) * h
    v_x = -c * n**(1.0/3.0)                     # v_x = -(3/pi)^(1/3) n^(1/3)
    return E_x, v_x

def hartree(n, x):                             # classical electrostatics (1D-softened)
    h = x[1] - x[0]
    K = 1.0 / np.sqrt((x[:, None] - x[None, :])**2 + 1.0)
    v_H = K @ n * h
    return 0.5 * np.sum(n * v_H) * h, v_H

def occupations(num_electrons, num_states):    # Aufbau: 2 electrons per state
    occ = np.zeros(num_states); e = num_electrons; i = 0
    while e > 0 and i < num_states:
        occ[i] = min(2, e); e -= occ[i]; i += 1
    return occ

def solve_ks(x, v_ext, num_electrons, iters=200, mix=0.3, tol=1e-8):
    T = build_kinetic(x); h = x[1] - x[0]
    n = np.zeros_like(x); occ = occupations(num_electrons, len(x)); E_old = None
    for _ in range(iters):
        E_x, v_x = exchange_lda(n, x)
        E_H, v_H = hartree(n, x)
        v_eff = v_ext + v_H + v_x                          # v_eff = v + v_H + v_xc
        eps, psi_gn = np.linalg.eigh(T + np.diag(v_eff))   # KS single-particle eqn
        n = (1 - mix) * n + mix * density(psi_gn, occ, x)  # density mixing
        # E = sum f_n eps_n - E_H - int n v_xc dx + E_xc  (double-counting fix)
        E = np.sum(occ * eps[:len(occ)]) - E_H - np.sum(n * v_x) * h + E_x
        if E_old is not None and abs(E - E_old) < tol:
            break
        E_old = E
    return E, eps, n

if __name__ == "__main__":
    x = np.linspace(-8, 8, 401); v_ext = x**2
    for Ne in (2, 6):
        E, eps, n = solve_ks(x, v_ext, Ne); h = x[1] - x[0]
        print(f"N={Ne}: E={E:.5f} Ha, N_check={np.sum(n)*h:.4f}, "
              f"eps[:3]={np.round(eps[:3],4)}")
```

The loop converges and the density integrates to exactly N electrons: the
non-interacting auxiliary system reproduces the requested density, with the kinetic
energy carried exactly by the orbitals and only the small exchange-correlation term
approximated.
