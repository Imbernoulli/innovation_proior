# Convex relaxation of AC optimal power flow (SDP / SOCP)

## Problem

Pick generator outputs and bus voltage phasors that serve the load at minimum
generation cost subject to the **AC power-flow equations** and to bounds on
voltage magnitudes, generator outputs, and line flows. With bus admittance
matrix `Y` (off-diagonal `(i,j) = -y_ij`, diagonal = sum of incident
admittances + shunts), Ohm's law is `I = YV` and the net complex power injected
at bus `j` is `s_j = V_j I_j* = V^H Y_j^H V` with `Y_j := e_j e_j^H Y`. So every
quantity OPF constrains is a **quadratic form** `V^H M V` in the complex voltage
vector `V`:

    p_j = V^H Phi_j V,   q_j = V^H Psi_j V,   |V_j|^2 = V^H J_j V,

with `Phi_j = (Y_j^H + Y_j)/2`, `Psi_j = (Y_j^H - Y_j)/(2i)`, `J_j = e_j e_j^H`.
The matrices `Phi_j, Psi_j` are **indefinite**, so the feasible set in `V` is
nonconvex; OPF is a nonconvex QCQP and NP-hard in the worst case (a `±1`
reduction). Newton/interior-point solvers return a local optimum with no global
certificate; DC-OPF is convex but throws away losses, reactive power, and
voltage limits.

## Key idea

`V^H M V = tr(M · VV^H)`, so lift to the Hermitian matrix `W := VV^H`. Every
quadratic form becomes **linear** in `W`: `p_j = tr(Phi_j W)`,
`q_j = tr(Psi_j W)`, `|V_j|^2 = tr(J_j W)`. A Hermitian `W` equals `VV^H` for
some `V` iff `W ⪰ 0` **and** `rank W = 1`. The lift gathers *all* the
nonconvexity into the single rank constraint; `W ⪰ 0` is convex. **Drop the
rank constraint:**

    min_{W ⪰ 0}  tr(C W)
    s.t.  p_j ≤ tr(Phi_j W) ≤ p̄_j,
          q_j ≤ tr(Psi_j W) ≤ q̄_j,
          v_j ≤ tr(J_j W)   ≤ v̄_j,   (+ line-flow LMIs)

a **semidefinite program** — convex, polynomial-time. Its feasible set is a
superset of the true one (it contains higher-rank `W` no voltage produces), so
its value **lower-bounds** OPF.

**Exactness.** If the SDP optimum `W*` comes out **rank one**, then
`W* = V*(V*)^H`: `V*` is AC-feasible and attains the lower bound, hence is the
**certified global** AC-OPF optimum. Since OPF is NP-hard this cannot always
happen, so exactness is conditional. Reading it off the dual: the dual is again
an SDP, `max` of a linear function of multipliers `(x,r)` subject to a linear
matrix inequality `A(x,r) ⪰ 0`; Slater holds, so the gap between the relaxed
primal and its dual is zero. Complementary slackness then forces
`tr(A(x*,r*) W*) = 0`, so `rank W* ≤ dim ker A(x*,r*)`. The global-phase symmetry
`V ↦ e^{iφ}V` makes that null space inherently even-dimensional, so the
certificate is:

> **zero duality gap (relaxation exact, `V*` recoverable) if the zero eigenvalue
> of `A(x*,r*)` has multiplicity exactly two** — a sufficient certificate
> (the exact-iff statement is rank-one `W*`; multiplicity two implies it).

**Why it holds for real grids.** For the over-satisfaction-relaxed OPF the load
multipliers are sign-constrained (`λ_k, γ_k ≥ 0`). Because resistance is
nonnegative, `Re{Y}` has non-positive off-diagonals; on the resistive backbone
`A` reduces to `diag(T, T)` with `T` symmetric, irreducible (connected
network), non-positive off-diagonal — exactly Perron–Frobenius, whose smallest
eigenvalue is **simple**, giving multiplicity two on `A`. Reactive coupling adds
a skew block; the double eigenvalue persists while the coupling stays small
relative to the resistive backbone. **Lossless transformers** drop out of
`Re{Y}` and disconnect that graph (multiplicity jumps to four — the IEEE 30-bus
failure); a tiny `~1e-5` pu resistance reconnects it and restores the
certificate.

**Radial collapse.** A tree is chordal with edge-cliques only, so `W ⪰ 0`
reduces to a `2×2` PSD condition per edge, `W_jj W_kk ≥ |W_jk|²` — a **rotated
second-order cone**, the Jabr formulation in `(u, R, I)`. On a tree the SOCP is
exactly as tight as the SDP; on a mesh it is strictly weaker (the cycle-angle
constraints are gone).

## Algorithm

1. Build `Y`; add `~1e-5` pu resistance to lossless transformers so `Re{Y}` is
   connected. Form `Phi_j, Psi_j, J_j` from the rows of `Y`.
2. **General network (SDP):** declare Hermitian `W ⪰ 0`, fix the slack
   (`W_00 = |V_0|²`), impose the linear-in-`W` injection / voltage / line-flow
   bounds, minimize the linear cost surrogate, solve the SDP.
3. **Exactness test:** eigendecompose `W*`. If `rank W* = 1`, factor
   `W* = V*(V*)^H`, fix the slack angle to 0 — `V*` is the certified global
   optimum. Otherwise the value is only a lower bound (gap may be > 0).
4. **Radial network (SOCP):** solve the cheaper per-edge rotated-cone (Jabr)
   program; recover magnitudes from `u`, angles by tree traversal.

## Code

```python
import numpy as np
import cvxpy as cp

# --- Bus admittance matrix (MATPOWER convention): off-diag -y_ij, diag = sum ---
def build_Ybus(n, branches, shunts=None, tx_resistance=1e-5):
    Y = np.zeros((n, n), dtype=complex)
    for (i, j, r, x) in branches:                  # series impedance r + j x
        if r == 0.0:                               # lossless transformer ->
            r = tx_resistance                      # nudge so Re{Y} stays connected
        y = 1.0 / (r + 1j * x)
        Y[i, i] += y; Y[j, j] += y
        Y[i, j] -= y; Y[j, i] -= y
    if shunts is not None:
        for i, ysh in shunts:
            Y[i, i] += ysh
    return Y

# --- Per-bus Hermitian forms from s_j = V^H Y_j^H V, Y_j = e_j e_j^H Y ---------
def bus_forms(Y):
    n = Y.shape[0]
    Phi, Psi, J = [], [], []
    for j in range(n):
        e = np.zeros((n, 1)); e[j, 0] = 1.0
        Yj = e @ (e.T @ Y)                         # j-th row of Y, rest zero
        YjH = Yj.conj().T
        Phi.append(0.5 * (YjH + Yj))               # Re s_j = tr(Phi_j W)
        Psi.append((YjH - Yj) / (2j))              # Im s_j = tr(Psi_j W)
        J.append(e @ e.T)                          # |V_j|^2 = tr(J_j W)
    return Phi, Psi, J

# === (A) Full SDP relaxation: lift W = V V^H, keep W >= 0, DROP rank(W)=1 =====
#     Convex; if W* is rank one the relaxation is EXACT and V* is the certified
#     global AC-OPF optimum, else the value is only a lower bound on OPF.
def solve_sdp_opf(Y, p_lo, p_hi, q_lo, q_hi, v_lo, v_hi, cost_lin, slack=0):
    n = Y.shape[0]
    Phi, Psi, J = bus_forms(Y)
    W = cp.Variable((n, n), hermitian=True)        # the lifted matrix V V^H
    cons = [W >> 0]                                # convex half of W = V V^H
    cons += [cp.real(W[slack, slack]) == 1.0]      # slack |V_slack| = 1
    for j in range(n):                             # every quantity LINEAR in W
        pj = cp.real(cp.trace(Phi[j] @ W))
        qj = cp.real(cp.trace(Psi[j] @ W))
        vj = cp.real(cp.trace(J[j]   @ W))
        cons += [pj >= p_lo[j], pj <= p_hi[j],
                 qj >= q_lo[j], qj <= q_hi[j],
                 vj >= v_lo[j], vj <= v_hi[j]]
    obj = cp.Minimize(sum(cost_lin[j] * cp.real(cp.trace(Phi[j] @ W))
                          for j in range(n)))      # linear-in-W cost surrogate
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS)                      # any SDP solver (SeDuMi/MOSEK)

    Wv = W.value
    evals, evecs = np.linalg.eigh(Wv)             # exactness test: rank of W*
    rank = int(np.sum(evals > 1e-5 * evals[-1]))
    if rank == 1:                                 # relaxation EXACT
        V = np.sqrt(evals[-1]) * evecs[:, -1]     # factor W* = V V^H
        V *= np.exp(-1j * np.angle(V[slack]))     # fix slack angle to 0
        return prob.value, V, True                # certified global optimum
    return prob.value, None, False                # lower bound only; gap may be > 0

# === (B) SOCP relaxation in Jabr branch variables: EXACT for RADIAL networks ==
#     A tree is chordal with edge-cliques only, so W >= 0 reduces to a 2x2 PSD
#     (rotated second-order cone) per edge: 2 u_i u_j >= R_ij^2 + I_ij^2.
def solve_socp_radial(n, edges, G, B, P, Q, gens, v_slack):
    s2 = np.sqrt(2.0)
    u = cp.Variable(n)                            # u_i = |V_i|^2 / sqrt(2)
    R = {(i, j): cp.Variable() for (i, j) in edges}   # R_ij ~ Re(V_i V_j*)
    I = {(i, j): cp.Variable() for (i, j) in edges}   # I_ij ~ Im(V_i V_j*)
    cons = [u[0] == v_slack ** 2 / s2]
    for g, vg in gens.items():
        cons += [u[g] == vg ** 2 / s2]
    for (i, j) in edges:                          # rotated cone == 2x2 minor of W>=0
        # ||[sqrt2 R, sqrt2 I, u_i-u_j]|| <= u_i+u_j  <=>  2 u_i u_j >= R^2 + I^2
        cons += [cp.SOC(u[i] + u[j],
                        cp.vstack([s2 * R[i, j], s2 * I[i, j], u[i] - u[j]]))]
    def nbrs(i): return [j for (a, j) in edges if a == i] + \
                        [a for (a, j) in edges if j == i]
    def edge(i, j): return (i, j) if (i, j) in R else (j, i)
    sgn = lambda i, j: 1.0 if i < j else -1.0     # I_ij is antisymmetric
    for i in range(1, n):                         # LINEAR power balance
        cons += [-s2 * u[i] * G[i, :].sum()
                 + sum(G[i, j] * R[edge(i, j)]
                       + B[i, j] * sgn(i, j) * I[edge(i, j)]
                       for j in nbrs(i)) == P[i]]
        if i not in gens:
            cons += [s2 * u[i] * B[i, :].sum()
                     + sum(-B[i, j] * R[edge(i, j)]
                           + G[i, j] * sgn(i, j) * I[edge(i, j)]
                           for j in nbrs(i)) == Q[i]]
    prob = cp.Problem(cp.Maximize(sum(R[e] for e in edges)), cons)
    prob.solve(solver=cp.ECOS)
    V_mag = np.sqrt(s2 * np.array([u[i].value for i in range(n)]))
    return prob.value, V_mag                      # angles recovered by tree traversal
```
