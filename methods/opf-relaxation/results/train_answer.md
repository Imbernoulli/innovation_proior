Every few minutes a grid operator has to pick generator outputs so the network serves its load at minimum cost without violating any physics or any equipment rating. The cost is the benign part — a convex quadratic $\sum_k c_{k2}P_{Gk}^2+c_{k1}P_{Gk}+c_{k0}$ in the real power generated. The pain lives entirely in the constraint that the chosen operating point actually solve the AC power-flow equations. With the bus admittance matrix $Y$ (off-diagonal $(i,j)$ entry $-y_{ij}$, diagonal the sum of incident admittances plus shunts), Ohm's law is $I=YV$ and the net complex power injected at bus $j$ is voltage times conjugate current, $s_j=V_jI_j^{*}=V^H Y_j^H V$ with $Y_j:=e_je_j^H Y$. The crucial structural fact is that this is a *quadratic form* in $V$: every quantity OPF constrains takes the shape $V^H M V$ for some Hermitian $M$, namely $p_j=V^H\Phi_jV$, $q_j=V^H\Psi_jV$, and $|V_j|^2=V^H J_jV$ with $\Phi_j=\tfrac12(Y_j^H+Y_j)$, $\Psi_j=\tfrac1{2i}(Y_j^H-Y_j)$, $J_j=e_je_j^H$. The matrices $\Phi_j,\Psi_j$ are built from complex admittances and come out *indefinite*, so each quadratic equality carves out a curved nonconvex surface and the feasible set in $V$ is nonconvex — possibly even disconnected. OPF is a nonconvex QCQP, and it is genuinely hard, not cosmetically so: clamp every $|V_j|=1$, make the network purely resistive ($\mathrm{Im}\{Y\}=0$), kill reactive power, and force each $V_j\in\{-1,1\}$, and OPF collapses to minimizing $V^TYV$ over $\{-1,1\}^n$, a max-cut–flavored $\pm1$ problem that is NP-hard. So there is no hope of a general polynomial-time exact algorithm.

The standard tools all cope with the nonconvexity rather than removing it, and each falls short in a specific way. Newton–Raphson and interior-point solvers attack the KKT conditions of the nonconvex problem directly; but KKT is only *necessary* when the problem is nonconvex, so they return a local stationary point with no global certificate, and on a disconnected feasible region they can be trapped on the wrong island. Metaheuristics (genetic algorithms, particle swarm) merely search harder — still stochastic, still no certificate. The one genuinely convex, genuinely global tool, DC-OPF, buys its convexity by discarding the very physics that matters: it flattens $|V|\approx1$, linearizes $\sin\theta\approx\theta$, drops resistance and losses and reactive power, leaving a tidy LP whose optimum is not an AC-feasible operating point. Jabr's radial conic program is exact but only on trees, leaking on meshed networks where the dropped cycle-angle constraint matters. The tension is therefore sharp: convexity has so far been bought only by abandoning the AC model, and I want both — the true AC physics *and* a convex problem carrying a global certificate.

I propose the **SDP relaxation of AC-OPF** (with its **SOCP / Jabr** specialization for radial networks). The leverage is the identity $V^H M V=\mathrm{tr}(M\,VV^H)$, which says the quadratic form is *linear* in the rank-one outer product $VV^H$. So I change variables to the Hermitian matrix $W:=VV^H$. Under this lift every constraint and magnitude term becomes linear: $p_j=\mathrm{tr}(\Phi_j W)$, $q_j=\mathrm{tr}(\Psi_j W)$, $|V_j|^2=\mathrm{tr}(J_j W)$; the convex quadratic cost goes linear-in-$W$ through a Schur-complement epigraph, and the apparent-power limit, a sum of squares of linear-in-$W$ terms, becomes a linear matrix inequality. The entire problem is now linear in $W$ — but the nonconvexity cannot have evaporated, and it hasn't: a Hermitian $W$ equals $VV^H$ for some $V$ if and only if $W\succeq0$ *and* $\mathrm{rank}\,W=1$. The lift has taken nonconvexity that was smeared across every constraint and concentrated all of it into a single rank condition. The PSD half $W\succeq0$ is convex; the rank-one half is the lone troublemaker. The whole move is then: drop the rank constraint. What remains,
$$ \min_{W\succeq0}\ \mathrm{tr}(CW)\quad\text{s.t.}\quad \underline p_j\le\mathrm{tr}(\Phi_jW)\le\overline p_j,\ \ \underline q_j\le\mathrm{tr}(\Psi_jW)\le\overline q_j,\ \ \underline v_j\le\mathrm{tr}(J_jW)\le\overline v_j, $$
plus the line-flow LMIs, is a semidefinite program — convex, polynomial-time. Because I deleted a constraint, the relaxed feasible set is a *superset* of the true one (it admits higher-rank $W$ that no voltage produces), so minimizing the same objective over it yields a guaranteed *lower bound* on the true OPF optimum — already useful for certifying any local solution.

What makes the relaxation usable is a clean exactness criterion. If the SDP optimum $W^{\star}$ happens to come out rank one, then $W^{\star}=V^{\star}(V^{\star})^H$; $V^{\star}$ satisfies every original constraint and attains the lower bound, so it is the *certified global* AC-OPF optimum. Since OPF is NP-hard this cannot always happen, so exactness is conditional, and the right place to read it off is the dual. Dualizing the SDP gives again an SDP — maximize a linear function of multipliers $(x,r)$ subject to a linear matrix inequality $A(x,r)\succeq0$, where $A$ is an affine combination of $\Phi_j,\Psi_j,J_j$ weighted by the duals plus small Schur blocks. A strictly feasible dual point exists (take $\lambda_k=c_{k1}+1$ on generator buses, $\mu_k$ large enough that $A\succ0$, the $r$-blocks at identity), so Slater holds and the gap between the relaxed primal and its dual is zero. Complementary slackness for the PSD constraint then forces $\mathrm{tr}(A(x^{\star},r^{\star})W^{\star})=0$, and since both matrices are PSD every column of $W^{\star}$ lies in $\ker A(x^{\star},r^{\star})$, so $\mathrm{rank}\,W^{\star}\le\dim\ker A(x^{\star},r^{\star})$. I cannot ask for $\dim\ker A=1$, because of a symmetry: if $V^{\star}$ is optimal so is $e^{i\phi}V^{\star}$ for any global phase, which in the real embedding means $\ker A$ is inherently even-dimensional. The correct certificate is therefore that the zero eigenvalue of $A(x^{\star},r^{\star})$ has multiplicity *exactly two* — which forces $\mathrm{rank}\,W^{\star}\le2$, and a phase-rotated combination of the two eigenpairs recovers a rank-one optimum, with $V^{\star}=(\zeta_1+\zeta_2 i)(X_1+X_2 i)$ for two real scalars pinned by fixing the slack angle to zero and matching one active voltage bound. Solving the dual is also cheaper: $O(|N|+|L|)$ variables against the dense $O(n^2)$ entries of $W$, loads enter only the objective and topology only the LMIs, and the certificate is a single eigenvalue count.

Why does that multiplicity equal two for real power networks? Start with the purely resistive case ($\mathrm{Im}\{Y\}=0$, no reactive power), which is itself the NP-hard $\pm1$ reduction. There the reactive multipliers and the $r$-blocks vanish and $A$ collapses to a block-diagonal $\mathrm{diag}(T,T)$ with one real symmetric block $T$. The off-diagonals of $T$ carry the line admittances $y_{lm}$, whose real parts are nonnegative because resistance is a nonnegative physical quantity, so if the load multipliers $\lambda_k^{\star}\ge0$ every off-diagonal of $T$ is *non-positive*. A symmetric, irreducible (strongly connected network) matrix with non-positive off-diagonals is exactly the Perron–Frobenius setting: its smallest eigenvalue is *simple*, and stacking two copies gives multiplicity two on $A$. So the NP-hard DC case is convexifiable, not by luck but because resistance is nonnegative and the grid is connected. The sign assumption $\lambda_k^{\star}\ge0$ is supplied by relaxing the load equality $P_{Lk}=P_{Dk}$ to the inequality $P_{Lk}\ge P_{Dk}$ (over-satisfaction allowed), whose multiplier is sign-constrained; in normal operation the nodal price is nonnegative so the inequality is tight and the modified OPF coincides with the original. Restoring reactive power turns $A$ into
$$ A(x^{\star},r^{\star})=\begin{bmatrix}T & \bar T\\ -\bar T & T\end{bmatrix} $$
with $T$ symmetric and $\bar T$ skew-symmetric; the double eigenvalue at $\bar T=0$ persists by continuity while the reactive coupling stays small relative to the resistive backbone, the off-diagonal signs again coming from the $\Pi$-model (resistance $\ge0$, line inductive) plus $\lambda_k^{\star},\gamma_k^{\star}\ge0$. One genuine failure mode remains: benchmark transformers are modeled as *lossless*, so they vanish from $\mathrm{Re}\{Y\}$ and can disconnect the resistive graph, making $T$ reducible and pushing the multiplicity to four — exactly what happens on the IEEE 30-bus system, three regions bridged by lossless transformers, where the four smallest eigenvalues of $A$ come out $0,0,0,0$. The fix follows from the diagnosis: give each lossless transformer a tiny resistance, on the order of $10^{-5}$ per unit, far too small to move the operating point but enough to reconnect $\mathrm{Re}\{Y\}$, restore irreducibility, and split the eigenvalues back to $0,0$ plus two positive — multiplicity two, certificate restored, on every IEEE case from 14 to 300 buses.

For radial networks the SDP collapses to something cheaper. A tree is chordal with the maximal cliques being just its edges, and for a chordal graph $W\succeq0$ is equivalent to all maximal-clique principal submatrices being PSD. So $W\succeq0$ reduces to a $2\times2$ PSD condition per edge, $W_{jj}W_{kk}\ge|W_{jk}|^2$, which is precisely a rotated second-order cone — the Jabr formulation $2u_ju_k\ge R_{jk}^2+I_{jk}^2$ in the variables $u_i\propto|V_i|^2$, $R_{ij}\propto\mathrm{Re}(V_iV_j^{*})$, $I_{ij}\propto\mathrm{Im}(V_iV_j^{*})$, with bus angles recovered by walking the tree. On a tree this SOCP is exactly as tight as the full SDP; on a mesh it is strictly weaker, because the per-edge minors no longer pin down a globally PSD completion with consistent cycle angles — the same radial-versus-mesh boundary that limited the conic load-flow to begin with.

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
