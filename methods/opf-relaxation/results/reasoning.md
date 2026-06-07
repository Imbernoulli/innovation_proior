Let me start from what an operator actually has to do every few minutes: pick generator outputs so the grid serves its load at minimum cost, without violating any physics or any equipment rating. The cost is the easy part — a convex quadratic $\sum_k c_{k2}P_{Gk}^2+c_{k1}P_{Gk}+c_{k0}$ in the real power generated. The hard part is the constraint that the chosen operating point actually be a solution of the AC power-flow equations. So let me write those down carefully, because everything hinges on their structure.

I have buses $0,1,\dots,n$, with bus $0$ the slack ($V_0=1\angle0^\circ$). The lines have admittances $y_{ij}$, and I collect them into the bus admittance matrix $Y$: off-diagonal $(i,j)$ entry $-y_{ij}$, diagonal $(i,i)$ entry the sum of incident admittances. Ohm's law in matrix form is just $I=YV$ — the injected current vector is $Y$ times the voltage vector. The thing I actually constrain is *power*, not current, and the net complex power injected at bus $j$ is voltage times conjugate current, $s_j=V_jI_j^{*}$. So

$$ s_j = V_j (e_j^H Y V)^{*} = V_j\, V^H Y^H e_j = V^H (e_j e_j^H Y)^H V. $$

Let me name $Y_j:=e_je_j^H Y$ — that's the matrix whose $j$-th row is the $j$-th row of $Y$ and whose other rows are zero. Then $s_j=V^H Y_j^H V$. Stare at that. The power injection at every bus is a *quadratic form in $V$*. Not linear — quadratic. The real and reactive parts split off cleanly: $Y_j^H$ has a Hermitian part $\Phi_j=\tfrac12(Y_j^H+Y_j)$ and a skew-Hermitian part, and $\mathrm{Re}\,s_j=V^H\Phi_jV$, $\mathrm{Im}\,s_j=V^H\Psi_jV$ with $\Psi_j=\tfrac1{2i}(Y_j^H-Y_j)$. And the voltage-magnitude limit is also quadratic: $|V_j|^2=V^H(e_je_j^H)V=V^H J_jV$. Even the apparent line flow $|S_{lm}|^2$ is a sum of squares of such forms.

So the whole problem is: minimize a convex quadratic in $P_G$, subject to a pile of constraints each of the form $V^HMV \lessgtr$ const, for various Hermitian $M$. That's a quadratically constrained quadratic program. If the $M$'s were positive semidefinite I'd be happy — convex, done. But $\Phi_j$ and $\Psi_j$ come from $Y$, which is built from complex admittances; they're *indefinite*. An indefinite quadratic equality constraint carves out a curved nonconvex surface. So the feasible set in $V$-space is nonconvex.

How bad is "nonconvex" here — is it just cosmetically nonconvex, or genuinely hard? Let me try to break it on purpose. Suppose I clamp every voltage magnitude to one, $|V_j|=1$ for all $j$. Then I'm minimizing $\mathrm{Re}\{V^HYV\}$ over the torus of unit-modulus complex vectors. That region wraps around the origin but excludes it — definitely nonconvex. Now push harder: make the network purely resistive, $\mathrm{Im}\{Y\}=0$, kill reactive power, and force each $V_j\in\{-1,1\}$. Now I'm minimizing $V^TYV$ over $\{-1,1\}^n$ — a $\pm1$ quadratic. That's a partitioning / max-cut–flavored combinatorial problem, NP-hard. And these are *special cases* of OPF. So OPF is NP-hard in the worst case. There's no hope of a general polynomial-time exact algorithm; the indefinite quadratic constraints are doing real combinatorial damage.

That's sobering, because the standard tools all live with the nonconvexity rather than removing it. Newton–Raphson and interior-point solvers on the KKT conditions: but KKT is only *necessary* when the problem is nonconvex, so I get a local stationary point with no idea whether a better operating point exists elsewhere — and with a disconnected feasible region I might be stuck on the wrong island entirely. The metaheuristics (genetic algorithms, particle swarm) just search harder; still no certificate, still stochastic. And the one genuinely convex, genuinely global tool — DC-OPF — gets its convexity by *throwing away the physics I care about*: it flattens $|V|\approx1$, linearizes $\sin\theta\approx\theta$, drops resistance and losses and reactive power, leaving a tidy LP that solves economic dispatch but whose optimum isn't an AC-feasible point. So the tension is sharp: convexity has been bought only by abandoning the AC model. I want both — the true AC physics *and* a convex problem with a global certificate.

Where's the leverage? Let me look at the radial case, because someone already found convex structure there. On a tree distribution feeder, instead of carrying around the complex $V_j$ with their phase angles, introduce $u_i\propto|V_i|^2$ and per-branch quantities $R_{ij}\propto\mathrm{Re}(V_iV_j^{*})$ and $I_{ij}\propto\mathrm{Im}(V_iV_j^{*})$. In these variables the power-flow equations are *linear* — the quadratic mess in $V$ flattens out. The only nonlinearity that survives is one bilinear identity: $|V_i|^2|V_j|^2=\mathrm{Re}(V_iV_j^{*})^2+\mathrm{Im}(V_iV_j^{*})^2$, i.e. $u_iu_j\sim R_{ij}^2+I_{ij}^2$. And *that* I can write as a rotated second-order cone, $2u_iu_j\ge R_{ij}^2+I_{ij}^2$ — convex! A second-order cone program, polynomial-time. After solving, I recover the bus angles by walking the tree from the root, accumulating branch angle differences $\theta_{ij}=\arcsin(I_{ij}/(V_iV_j))$.

Why does this work only on a tree? Because on a tree the branch angle differences determine the bus angles uniquely — there's exactly one path from the root to each bus. On a *meshed* network the angle differences have to be consistent around every cycle: their sum around a loop must close up (an arctangent constraint linking the $R$'s and $I$'s). The conic formulation silently drops that constraint, so for a mesh the relaxation has extra freedom and need not correspond to any real voltage assignment. So this is a real foothold but a topology-limited one: convex and exact for radial, leaky for mesh.

Now — the radial trick replaced the voltages with products $V_iV_j^{*}$ and a magnitude $|V_i|^2$, and the bilinearity became a cone. That smells like the general semidefinite move for a QCQP. Let me try to do the same thing without assuming a tree. The obstruction to convexity is that $V^HMV$ is quadratic in $V$. But $V^HMV=\mathrm{tr}(M\,VV^H)$ — it's *linear* in the rank-one matrix $VV^H$. So let me change variables: define $W:=VV^H$, an $(n{+}1)\times(n{+}1)$ Hermitian matrix. Then every single constraint and the magnitude terms become linear in $W$:

$$ p_j=\mathrm{tr}(\Phi_j W),\quad q_j=\mathrm{tr}(\Psi_j W),\quad |V_j|^2=\mathrm{tr}(J_j W). $$

The objective, after pushing the quadratic cost through a Schur-complement epigraph trick, is linear in $W$ too; the apparent-power constraint $|S_{lm}|^2\le (S^{\max})^2$ is a sum of squares of linear-in-$W$ terms, which Schur's complement turns into a linear matrix inequality. So in the new variable $W$ the *entire problem is linear* — objective and all constraints. I've completely linearized OPF.

That can't be free; I've hidden the nonconvexity somewhere. Where? In the change of variables itself. A Hermitian $W$ equals $VV^H$ for some vector $V$ if and only if two things hold: $W\succeq0$ (positive semidefinite) and $\mathrm{rank}\,W=1$. The first is convex — the PSD cone. The second, $\mathrm{rank}\,W=1$, is the lone nonconvex constraint. So I've taken a problem whose nonconvexity was smeared across every constraint and concentrated *all* of it into a single rank condition on a matrix variable.

And once it's all in one place, I can see what to do: drop it. Keep $W\succeq0$, delete $\mathrm{rank}\,W=1$. What's left is

$$ \min_{W\succeq0}\ \mathrm{tr}(CW)\quad\text{s.t.}\quad \mathrm{tr}(\Phi_jW)\in[\underline p_j,\overline p_j],\ \mathrm{tr}(\Psi_jW)\in[\underline q_j,\overline q_j],\ \mathrm{tr}(J_jW)\in[\underline v_j,\overline v_j], $$

plus the line-flow LMIs — a linear objective over an intersection of linear constraints and the PSD cone. A semidefinite program. Convex. Polynomial-time. This is the SDP relaxation of AC-OPF.

But I dropped a constraint, so I have to be honest about what I get. The relaxation's feasible set is a *superset* of the true one (every real $W=VV^H$ is still feasible, plus higher-rank $W$'s that no voltage produces). Minimizing the same objective over a larger set gives a *lower bound* on the true OPF optimum. So the SDP value is a guaranteed lower bound — already useful as a certificate of how good any local solution is. The question is when it's *exact*.

Here's the clean part. Suppose I solve the SDP and the optimal $W^{\star}$ happens to come out rank one. Then $W^{\star}=V^{\star}(V^{\star})^H$ for some vector $V^{\star}$, I factor it out, and $V^{\star}$ satisfies every original constraint with equality of the trace identities — it's AC-feasible — and it achieves the lower bound. A point that is feasible for the original problem *and* attains a lower bound on the original optimum must be globally optimal. So: **if the SDP solution is rank one, the relaxation is exact and I have recovered the global AC-OPF optimum, with a certificate.** That's exactly the global-optimality guarantee I wanted, on the full AC model. The only thing standing between me and victory is whether $W^{\star}$ comes out rank one — and since OPF is NP-hard, it can't *always*. So the real question is: for which networks does it?

Let me get at exactness through duality, because the dual will both be cheaper to solve and hand me a usable certificate. The SDP is a convex program; form its Lagrangian dual. The PSD constraint $W\succeq0$ gets a matrix multiplier; the scalar constraints get scalar multipliers. Grinding through the standard SDP dualization, the dual is *again* an SDP — maximize a linear function of the multipliers subject to a linear matrix inequality $A(x,r)\succeq0$, where $A(x,r)$ is an affine combination of the constraint matrices $\Phi_j,\Psi_j,J_j$ weighted by the dual variables, plus a couple of $3\times3$ and $2\times2$ LMIs from the Schur complements. Strong duality holds here: both are SDPs, and I can exhibit a strictly feasible dual point (pick the multipliers $\lambda_k=c_{k1}+1$ on generator buses, $\mu_k$ large enough that $A\succ0$, the auxiliary $r$-blocks set to the identity so the Schur LMIs are strictly satisfied), so Slater's condition holds and the duality gap between the relaxed primal SDP and its dual is zero. Good — relaxed primal value $=$ dual value.

Now I have two gaps to keep straight. There's the gap between the *true* OPF and its relaxation (the rank story), and there's the gap between the relaxation and *its* dual (zero, by Slater). Chain them: true-OPF optimal value equals the relaxed-SDP value if and only if the relaxation is exact, i.e. iff the SDP has a rank-one optimal solution. And I can read exactness off the *dual* solution via complementary slackness. At optimum, complementary slackness for the PSD constraint forces $\mathrm{tr}(A(x^{\star},r^{\star})W^{\star})=0$, and since both matrices are PSD this means every column of $W^{\star}$ lies in the null space of $A(x^{\star},r^{\star})$. So the rank of $W^{\star}$ is at most the dimension of $\ker A(x^{\star},r^{\star})$. If that null space is small, $W^{\star}$ is forced to be low rank.

How small do I need it? Naively I'd want $\dim\ker A=1$ to force rank one. But there's a symmetry I have to account for. If $V^{\star}$ is optimal, so is $e^{i\phi}V^{\star}$ for any global phase $\phi$ — same $W^{\star}=VV^H$, same everything. In the real-embedded form $X=[\mathrm{Re}\,V;\mathrm{Im}\,V]$, this phase freedom means that whenever $[X_1;X_2]$ is in $\ker A$, so is the rotated $[-X_2;X_1]$. The null space is *inherently* even-dimensional — it can't be one-dimensional. So the right condition is $\dim\ker A(x^{\star},r^{\star})=2$, i.e. the smallest (zero) eigenvalue of $A$ has multiplicity exactly two. When that holds, $W^{\star}$ has rank at most $2$, and — accounting for that same phase symmetry — a rank-one solution can be extracted: if $W^{\star}$ has rank two with nonzero eigenpairs $(\rho_1,E_1),(\rho_2,E_2)$, then a suitably phased combination $(\rho_1+\rho_2)EE^T$ is itself a rank-one optimal $W$. And concretely, given the two-dimensional null space spanned by $[X_1;X_2]$, the voltage is recovered as $V^{\star}=(\zeta_1+\zeta_2 i)(X_1+X_2 i)$ for two real scalars $\zeta_1,\zeta_2$ pinned down by fixing the slack-bus angle to zero and matching one active voltage bound.

So the exactness certificate is purely a statement about the dual matrix: **the duality gap is zero (relaxation exact, global optimum recoverable) if $A(x^{\star},r^{\star})$ has a zero eigenvalue of multiplicity exactly two.** And solving the dual is the cheaper route anyway: its variables number $O(|N|+|L|)$, one per bus and line, against the dense $O(n^2)$ entries of $W$ in the primal; the loads and limits enter only the dual objective while the topology $Y$ enters only the LMIs, a clean separation; and the certificate is just an eigenvalue count on $A$.

This is the whole algorithm now. Solve the dual SDP for $(x^{\star},r^{\star})$. If the optimal value is $+\infty$, OPF was infeasible. Otherwise compute the multiplicity $\psi$ of the zero eigenvalue of $A(x^{\star},r^{\star})$. If $\psi\le2$, recover $V^{\star}$ from the null space and I have the certified global optimum. If $\psi>2$, the test is inconclusive — I might not be able to recover a unique voltage in polynomial time. Everything reduces to one question: *why would $\psi$ equal exactly two for real power networks?*

Let me actually try to prove it, starting with the simplest nontrivial case: a purely resistive DC network with constant active-power loads ($\mathrm{Im}\{Y\}=0$, no reactive power). This case is itself NP-hard (it's the $\pm1$ reduction from before), so if I can show zero duality gap here it's genuinely surprising and informative. With no reactive power, the reactive multipliers $\gamma_k$ vanish and the auxiliary $r$-blocks vanish, and $A(x^{\star},r^{\star})$ collapses to a block-diagonal form $\mathrm{diag}(T,T)$ with one real symmetric $n\times n$ block $T$ repeated. The off-diagonal $(l,m)$ entry of $T$ is, after the dust settles,

$$ T_{lm} = -\tfrac{y_{lm}}{2}\big(\lambda_{lm}^{\star}+\lambda_{ml}^{\star}\big)+\lambda_l^{\star}+\lambda_m^{\star}\ \text{-type terms} $$

— and here's where the *physics* enters. The line admittance $y_{lm}$ from the $\Pi$-model of a real line or transformer has nonnegative real part, because resistance is a nonnegative physical quantity. If the load multipliers $\lambda_k^{\star}$ are nonnegative, every off-diagonal entry of $T$ comes out *non-positive*. A symmetric matrix with non-positive off-diagonals, that is *irreducible* — which holds exactly when the network graph is strongly connected — is precisely the setting for the Perron–Frobenius theorem: its smallest eigenvalue is *simple*. And since $A=\mathrm{diag}(T,T)$ stacks two copies, $A$'s smallest eigenvalue has multiplicity exactly two. That's $\psi=2$. Zero duality gap. The NP-hard DC case is *convexifiable* — not by luck, but because resistance is nonnegative and the grid is connected.

The catch in that argument is the assumption $\lambda_k^{\star}\ge0$. Where does it come from? In the original OPF the load is an *equality*, $P_{Lk}=P_{Dk}$ — deliver exactly the demanded power — and an equality multiplier is free in sign. So I can't assume $\lambda_k\ge0$ directly. Let me relax the load constraint to an inequality: $P_{Lk}\ge P_{Dk}$, *over-satisfaction allowed*, and call this the modified OPF. Now the multiplier on an inequality is sign-constrained, $\lambda_k\ge0$ — exactly what I needed. So for the modified OPF, the Perron–Frobenius argument goes through and the duality gap is zero (and where $\lambda^{\star}$ has a zero entry I can perturb the constraint to $\lambda_k\ge\varepsilon$, run the argument, and let $\varepsilon\to0$; the rank-one solution has bounded norm thanks to the voltage limits, so it persists in the limit).

But did I cheat by changing the problem? I need the modified OPF and the original OPF to have the *same* solution. Physically they should: over-delivering power can only cost more, never less — the nodal price $\lambda_k$ is nonnegative in normal operation, so the optimum delivers exactly the demanded $P_{Dk}$ and the inequality is tight. Formally, the dual of the modified problem is the dual of the original plus the extra constraints $\lambda_k\ge0$; so if the original dual's optimum already happens to satisfy $\lambda_k^{\star}\ge0$, the two problems share a solution (and one can drop the over-constraining voltage/flow limits at a non-generator bus to *guarantee* $\lambda_k^{\star}\ge0$, since a load bus with known demand shouldn't be over-constrained). The only way they'd differ is an abnormal regime where power is effectively sold at a negative price — which a sanely operated grid avoids. So in normal operation, modified OPF $=$ OPF, and the zero-gap result transfers.

Now lift the restriction to resistive networks. With reactive power back, $A$ is no longer block-diagonal $\mathrm{diag}(T,T)$; it picks up an off-diagonal coupling block,

$$ A(x^{\star},r^{\star})=\begin{bmatrix}T & \bar T\\ -\bar T & T\end{bmatrix}, $$

with $T$ symmetric and $\bar T$ *skew-symmetric*. The skew-symmetric $\bar T$ can't have all-nonpositive entries, so I can't apply Perron–Frobenius to the full $A$ directly. But notice that when $\bar T=0$ the smallest eigenvalue of $A$ is again doubly repeated, by the resistive argument. So the eigenvalue multiplicity is a *continuity* question: how big can $\bar T$ get before the double eigenvalue splits? Scale the coupling, $\bar T\mapsto\omega\bar T$; at $\omega=0$ the multiplicity is two, and it stays two on an interval $[0,\omega^{\max}]$ with $\omega^{\max}>0$. If the actual coupling is small enough that $\omega=1$ falls inside that interval — i.e. the reactive coupling $\bar T$ is sufficiently smaller than the resistive backbone $T$ — the multiplicity is still two and the gap is still zero. For a real network operating in a normal regime the resistive structure dominates and this holds; the sign structure that makes $T$'s off-diagonals non-positive comes from two more physical facts about the $\Pi$-model — the off-diagonals of $\mathrm{Re}\{Y\}$ are non-positive (resistance $\ge0$) and the off-diagonals of $\mathrm{Im}\{Y\}$ are nonnegative (the line is inductive) — together with $\lambda_k^{\star},\gamma_k^{\star}\ge0$ from the over-satisfaction relaxation on both active and reactive loads.

One more wrinkle, and it's a real one, not a cosmetic one. The Perron–Frobenius step needed the resistive graph — the graph induced by $\mathrm{Re}\{Y\}$ — to be *strongly connected*, so that $T$ is irreducible. But real benchmark systems contain transformers modeled as *lossless* (zero resistance). A lossless transformer contributes nothing to $\mathrm{Re}\{Y\}$, so its line is *absent* from the resistive graph. If transformers tie together otherwise-separate regions, removing them disconnects the resistive graph, $T$ becomes reducible, and instead of a doubly-repeated smallest eigenvalue I get one repeated *four* times — the multiplicity $\psi$ is $4$, the condition $\psi\le2$ fails, and my certificate is void even though the true duality gap may still be zero. This is exactly the failure mode I'd hit on the IEEE systems: the 30-bus circuit is three regions bridged by lossless transformers, so $\mathrm{Re}\{Y\}$ splits into three pieces and the four smallest eigenvalues of $A$ come out $0,0,0,0$.

The fix follows straight from the diagnosis: give each lossless transformer a tiny resistance — on the order of $10^{-5}$ per unit — so it reappears in $\mathrm{Re}\{Y\}$ and reconnects the resistive graph. The perturbation is far too small to move the optimal operating point in any meaningful way, but it restores irreducibility, hence the simple Perron–Frobenius eigenvalue, hence multiplicity two. After it, the four smallest eigenvalues of $A$ separate into $0,0$ and two strictly positive ones — $\psi=2$, certificate restored, global optimum recovered. So the same physical principle — resistance is nonnegative, and a connected passive resistive network has a simple Perron–Frobenius mode — is what makes every IEEE benchmark (14, 30, 57, 118, 300 buses) convexifiable, once the modeling artifact of the lossless transformer is patched.

Let me also pin down the cleanest special case to make sure the machinery is right: loss minimization with the cost $f_k(P_{Gk})=P_{Gk}$ in the ideal limit of zero transmission losses. Then I can write down the dual solution explicitly — $r^{\star}=0$, $\lambda_k^{\star}=1$, all other multipliers zero — and check that $A(x^{\star},r^{\star})=\mathrm{diag}(\mathrm{Re}\{Y\},\mathrm{Re}\{Y\})$. Since $\mathrm{Re}\{Y\}$ is the admittance of a passive connected resistive network it is PSD with a single zero eigenvalue (the all-ones mode), so $A$ has a zero eigenvalue of multiplicity exactly two — the certificate holds, and the dual objective $\sum_k P_{Dk}$ matches the optimal (loss-free) generation. Real losses are nonzero but small, so the optimal point sits close to this loss-free dual point and $A$ keeps its two zero eigenvalues — which is the intuition for why small losses don't open a duality gap.

Finally, the radial case deserves its cheaper treatment. On a tree I don't need the full dense $W\succeq0$. The exactness of the SDP comes down to PSD-ness of $W$ together with rank-one-ness; for a *chordal* graph, $W\succeq0$ is equivalent to all its maximal-clique principal submatrices being PSD, and a tree is chordal with the maximal cliques being just the edges. So $W\succeq0$ reduces to: for every edge $(j,k)$, the $2\times2$ submatrix $\begin{bmatrix}W_{jj}&W_{jk}\\W_{kj}&W_{kk}\end{bmatrix}\succeq0$. That $2\times2$ PSD condition is exactly a rotated second-order cone — $W_{jj}W_{kk}\ge|W_{jk}|^2$ — which is the Jabr cone $2u_ju_k\ge R_{jk}^2+I_{jk}^2$ in the $(u,R,I)$ variables. So on a radial network I can drop the global SDP entirely and solve a much cheaper SOCP with one rotated cone per edge, and for a tree it's exactly as tight as the full SDP. On a mesh, the per-edge $2\times2$-PSD SOCP is a *strictly weaker* relaxation than the full SDP, because the edge submatrices no longer pin down a globally PSD completion with consistent cycle angles — which is precisely the radial-vs-mesh boundary I hit at the very start with the conic load-flow.

Let me write the code. Two routines: the SOCP relaxation in the Jabr branch variables (exact for radial), and the full BIM SDP relaxation in $W$ with the rank-one exactness test (for general networks).

```python
import numpy as np
import cvxpy as cp

# ---------------------------------------------------------------------------
# Build the bus admittance matrix Y (MATPOWER convention): off-diagonal -y_ij,
# diagonal sum of incident admittances (+ shunts). This is pure network data.
# ---------------------------------------------------------------------------
def build_Ybus(n, branches, shunts=None):
    Y = np.zeros((n, n), dtype=complex)
    for (i, j, y_ij) in branches:          # y_ij = 1 / (r + j*x)
        Y[i, i] += y_ij
        Y[j, j] += y_ij
        Y[i, j] -= y_ij
        Y[j, i] -= y_ij
    if shunts is not None:
        for i, ysh in shunts:
            Y[i, i] += ysh
    return Y

# ---------------------------------------------------------------------------
# Per-bus Hermitian forms: p_j = tr(Phi_j W), q_j = tr(Psi_j W),
# |V_j|^2 = tr(J_j W). These come straight from s_j = V^H Y_j^H V.
# ---------------------------------------------------------------------------
def bus_forms(Y):
    n = Y.shape[0]
    Phi, Psi, J = [], [], []
    for j in range(n):
        e = np.zeros((n, 1)); e[j, 0] = 1.0
        Yj = e @ (e.T @ Y)                 # j-th row of Y, rest zero
        YjH = Yj.conj().T
        Phi.append(0.5 * (YjH + Yj))               # Hermitian part  -> Re s_j
        Psi.append((YjH - Yj) / (2j))              # skew part        -> Im s_j
        J.append(e @ e.T)                          # picks out |V_j|^2
    return Phi, Psi, J

# ===========================================================================
# (A) Full SDP relaxation in the bus-injection model: lift W = V V^H, keep
#     W >= 0, DROP rank(W) = 1. Convex; if W* is rank one the relaxation is
#     exact and V* is the certified global AC-OPF optimum.
# ===========================================================================
def solve_sdp_opf(Y, p_lo, p_hi, q_lo, q_hi, v_lo, v_hi, cost_lin):
    n = Y.shape[0]
    Phi, Psi, J = bus_forms(Y)
    W = cp.Variable((n, n), hermitian=True)        # the lifted matrix V V^H
    cons = [W >> 0]                                 # the convex half of W = V V^H
    cons += [cp.real(W[0, 0]) == 1.0]              # slack |V_0| = 1
    for j in range(n):
        pj = cp.real(cp.trace(Phi[j] @ W))         # all power/voltage quantities
        qj = cp.real(cp.trace(Psi[j] @ W))         # are now LINEAR in W
        vj = cp.real(cp.trace(J[j]   @ W))
        cons += [pj >= p_lo[j], pj <= p_hi[j],
                 qj >= q_lo[j], qj <= q_hi[j],
                 vj >= v_lo[j], vj <= v_hi[j]]
    # generation cost: linear-in-W surrogate for sum_k c1_k * p_k (+ const)
    obj = cp.Minimize(sum(cost_lin[j] * cp.real(cp.trace(Phi[j] @ W))
                          for j in range(n)))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS)                       # any SDP solver (SeDuMi/SCS/MOSEK)

    Wv = W.value
    evals, evecs = np.linalg.eigh(Wv)              # exactness test: rank of W*
    rank = int(np.sum(evals > 1e-5 * evals[-1]))
    if rank == 1:                                  # relaxation EXACT
        V = np.sqrt(evals[-1]) * evecs[:, -1]      # factor W* = V V^H
        V = V * np.exp(-1j * np.angle(V[0]))       # fix slack angle to 0
        return prob.value, V, True
    return prob.value, None, False                 # lower bound only; gap may be > 0

# ===========================================================================
# (B) SOCP relaxation in the branch-flow (Jabr) variables: exact for RADIAL
#     networks. For a tree, W >= 0 reduces to a 2x2 PSD (rotated cone) per
#     edge, since a tree is chordal with edge-cliques only.
# ===========================================================================
def solve_socp_radial(n, edges, G, B, P, Q, gens, v_slack):
    s2 = np.sqrt(2.0)
    u = cp.Variable(n)                              # u_i = |V_i|^2 / sqrt(2)
    R = {(i, j): cp.Variable() for (i, j) in edges} # R_ij ~ Re(V_i V_j*)
    I = {(i, j): cp.Variable() for (i, j) in edges} # I_ij ~ Im(V_i V_j*)
    for (i, j) in list(R):
        R[j, i] = R[i, j]; I[j, i] = I[i, j]
    cons = [u[0] == v_slack ** 2 / s2]
    for g, vg in gens.items():
        cons += [u[g] == vg ** 2 / s2]
    # rotated second-order cone per edge == the 2x2 principal minor of W >= 0
    for (i, j) in edges:
        # ||[sqrt2 R, sqrt2 I, u_i-u_j]|| <= u_i+u_j  <=>  2 u_i u_j >= R^2 + I^2
        cons += [cp.SOC(u[i] + u[j],
                        cp.vstack([s2 * R[i, j], s2 * I[i, j], u[i] - u[j]]))]
    def nbrs(i): return [j for (a, j) in edges if a == i] + \
                        [a for (a, j) in edges if j == i]
    sgn = lambda i, j: 1 if i < j else -1
    for i in range(1, n):                           # linear power balance
        cons += [-s2 * u[i] * G[i, :].sum()
                 + sum(G[i, j] * R[i, j] + B[i, j] * sgn(i, j) * I[i, j]
                       for j in nbrs(i)) == P[i]]
        if i not in gens:
            cons += [s2 * u[i] * B[i, :].sum()
                     + sum(-B[i, j] * R[i, j] + G[i, j] * sgn(i, j) * I[i, j]
                           for j in nbrs(i)) == Q[i]]
    prob = cp.Problem(cp.Maximize(sum(R[i, j] for (i, j) in edges)), cons)
    prob.solve(solver=cp.ECOS)
    V = np.sqrt(s2 * np.array([u[i].value for i in range(n)]))   # |V_i| = (sqrt2 u_i)^{1/2}
    return prob.value, V                            # angles recovered by tree traversal
```

To recap the causal chain: minimizing generation cost subject to the AC power-flow laws forces me to constrain quadratic forms $V^HMV$ in the complex voltages, so OPF is a nonconvex QCQP — provably NP-hard via a $\pm1$ reduction. Rewriting $V^HMV=\mathrm{tr}(MW)$ with $W=VV^H$ makes every constraint linear in the matrix $W$ and gathers all nonconvexity into the single rank-one condition; dropping $\mathrm{rank}\,W=1$ leaves the convex constraint $W\succeq0$ and yields a polynomial-time SDP that lower-bounds OPF and is *exact* whenever its optimum is rank one. Reading exactness off the dual matrix $A$, the certificate is that $A$'s zero eigenvalue has multiplicity exactly two — which the Perron–Frobenius theorem guarantees because resistance is nonnegative (so $\mathrm{Re}\{Y\}$ has non-positive off-diagonals) and a connected passive resistive network has a simple smallest eigenvalue, once lossless transformers are nudged with $10^{-5}$ pu resistance to keep that graph connected. For radial networks the SDP collapses to one rotated second-order cone per edge — the Jabr formulation — which is exactly as tight as the SDP on a tree and strictly weaker on a mesh.
