# Context: convex relaxation of the AC optimal power flow problem

## Research question

Operate an electrical power network at minimum cost while respecting physics and equipment limits. Concretely: choose generator real- and reactive-power outputs and the resulting bus voltage phasors so as to minimize total generation cost, subject to the AC power-flow equations (Kirchhoff's and Ohm's laws) and to bounds on voltage magnitudes, generator outputs, and line flows. This is the optimal power flow (OPF) problem, posed by Carpentier in 1962, and it sits under economic dispatch, unit commitment, voltage/var control, loss minimization, network reconfiguration, and contingency-constrained operation.

The difficulty is that the bus power injections are *quadratic* in the complex voltages, so the feasible region is nonconvex and can even be disconnected. Half a century of solvers (Newton–Raphson, sequential quadratic / nonlinear programming, interior-point methods built on the Karush–Kuhn–Tucker conditions, plus a zoo of metaheuristics — genetic algorithms, particle swarm, simulated annealing) return only *local* optima with no certificate of global optimality, because the KKT conditions are merely necessary for a nonconvex problem. What a practitioner would want is a formulation that is genuinely *convex* — solvable in polynomial time with a global guarantee — yet still faithful to the true AC physics, together with a way to tell, after solving, whether the convex answer is in fact the global AC optimum.

## Background

**The AC power-flow equations.** Model the network as a graph on buses $N^+=\{0,1,\dots,n\}$ (bus 0 the slack bus, $V_0=1\angle 0^\circ$) with line admittances $y_{ij}\in\mathbb{C}$. Collect them into the $(n{+}1)\times(n{+}1)$ bus admittance matrix $Y$, whose off-diagonal $(i,j)$ entry is $-y_{ij}$ and whose diagonal $(i,i)$ entry is $\sum_{k\sim i}y_{ik}$ (plus shunts). Ohm's law gives the injected currents $I=YV$, and the net complex power injected at bus $j$ is

$$ s_j = V_j I_j^{*} = V_j\,(e_j^H Y V)^{*} = V^H Y_j^H V,\qquad Y_j := e_j e_j^H Y. $$

So $s_j$ is a *quadratic form* in the voltage vector $V$. Writing $\Phi_j=\tfrac12(Y_j^H+Y_j)$ (Hermitian part) and $\Psi_j=\tfrac1{2i}(Y_j^H-Y_j)$ (skew part), the real and reactive injections are $p_j=V^H\Phi_j V$ and $q_j=V^H\Psi_j V$, and the squared voltage magnitude is $|V_j|^2=V^H J_j V$ with $J_j=e_je_j^H$. Every quantity that OPF constrains is a quadratic form $V^H M V$ for some Hermitian $M$.

**Why OPF is a nonconvex QCQP, and NP-hard.** With a (convex) quadratic generation cost and these quadratic equality/inequality constraints, OPF is a quadratically constrained quadratic program (QCQP). The matrices $\Phi_j,\Psi_j$ are indefinite, so the constraint sets are nonconvex; the feasible region in $V$ can encircle the origin without containing it (hence be nonconvex even when connected) or break into disjoint pieces. The problem is NP-hard in the worst case: fixing all $|V_j|=1$ reduces OPF to minimizing $\mathrm{Re}\{V^H Y V\}$ over the unit-modulus torus, and adding $\mathrm{Im}\{Y\}=0$, zero reactive injection, and $V_j\in\{-1,1\}$ reduces it to a $\pm1$ quadratic optimization (a partitioning/max-cut–type combinatorial problem). Indefinite quadratic constraints alone already make the problem NP-hard.

**DC-OPF, the convex but lossy linearization.** The standard escape is to linearize: assume voltage magnitudes are flat ($|V_j|\approx1$), small angle differences ($\sin\theta\approx\theta$), and negligible line resistance, which decouples real power from voltage magnitude and drops losses and reactive power entirely. The resulting DC-OPF is a linear program — convex, fast, the backbone of electricity-market economic dispatch. Its price is model error: no losses, no voltage/var constraints, no reactive power. The open problem is to recover global optimality on the *full* AC model while staying convex.

**The radial conic insight (Jabr, 2006).** For a radial (tree) distribution network, the load-flow equations can be recast as a *conic* program. Introduce per-bus $u_i\propto|V_i|^2$ and per-branch $R_{ij}\propto\mathrm{Re}(V_iV_j^{*})$, $I_{ij}\propto\mathrm{Im}(V_iV_j^{*})$. The power-flow equations become *linear* in $(u,R,I)$, and the single bilinear identity $|V_i|^2|V_j|^2=\mathrm{Re}(V_iV_j^{*})^2+\mathrm{Im}(V_iV_j^{*})^2$ becomes a rotated second-order cone $2u_iu_j\ge R_{ij}^2+I_{ij}^2$. The load-flow problem is then a second-order cone program, solvable in polynomial time by interior-point methods, with the bus angles recovered afterward by walking the tree. The limitation Jabr leaves open: for a *meshed* network the angles must be consistent around every cycle (an arctangent constraint), which the conic relaxation drops — so the conic formulation is faithful only on radial topologies.

**Matrix-form tools for quadratic forms.** A scalar quadratic form obeys the algebraic identity $V^H M V=\mathrm{tr}(M\,VV^H)$, relating it to the outer-product matrix $VV^H$. Separately, by Schur's complement a bound on a sum of squares of linear terms, or on a convex quadratic cost, can be re-expressed as a linear matrix inequality, the building block of semidefinite programming. These are standard pieces of the QCQP / conic-optimization toolkit.

**Strong duality of SDPs.** The dual of a semidefinite program is again a semidefinite program, and under a Slater (strict-feasibility) condition strong duality holds, so the primal and dual optimal values coincide. The Perron–Frobenius theorem is a standard tool for such matrices: a symmetric, irreducible matrix with non-positive off-diagonals has a *simple* smallest eigenvalue (irreducibility meaning the graph it represents is strongly connected).

**Sign structure of the admittance data.** The off-diagonal entries of $\mathrm{Re}\{Y\}$ are $-\mathrm{Re}\{y_{ij}\}$, and line/transformer resistances are nonnegative physical quantities, so those off-diagonals are non-positive; $\mathrm{Re}\{Y\}$ is a passive admittance whose diagonal-plus-row-sum is nonnegative. One modeling artifact to note in the benchmark data: transformers are commonly modeled as *lossless* (zero resistance), so they contribute nothing to $\mathrm{Re}\{Y\}$ and are absent from the graph it induces.

## Baselines

- **Newton–Raphson / interior-point AC-OPF on the KKT conditions.** Solve the stationarity + complementarity conditions of the nonconvex AC-OPF directly with a second-order method (e.g. as in MATPOWER). Fast and accurate near a good initial point, and the de facto industry tool. Gap: the KKT conditions are only *necessary* for a nonconvex problem, so the solver converges to a local optimum with no global certificate, and can land on different local solutions from different starts.

- **DC-OPF (linear programming).** The linearized economic-dispatch LP described above. Gap: convex and global, but for an approximate model — no losses, no reactive power, no voltage-magnitude constraints — so its optimum is not an AC-feasible operating point.

- **Jabr radial conic program (2006).** The SOCP load-flow / power-flow formulation in $(u,R,I)$ for tree networks. Gap: exact only on radial topologies; on meshed networks the dropped cycle-angle (arctangent) constraints make it a relaxation that need not return an AC-feasible point.

- **Metaheuristics (genetic algorithms, particle swarm, evolutionary programming).** Global *search* heuristics applied to the nonconvex AC-OPF. Gap: no polynomial-time guarantee, no optimality certificate, and stochastic, non-reproducible solutions.

## Evaluation settings

The natural yardsticks are the standard IEEE benchmark transmission systems — 14-, 30-, 57-, 118-, and 300-bus cases — archived with the network topology (admittance matrix), generator cost curves, and the voltage/generation/line limits, as distributed with power-system toolboxes (MATPOWER, PSAT). Smaller hand-built 3- and 4-bus networks serve to expose the structure (disconnected feasible regions, the role of transformer resistance) in full detail. The objective is either total generation cost ($\sum_k c_{k2}P_{Gk}^2+c_{k1}P_{Gk}+c_{k0}$) or total active power loss. A solver is judged on whether it returns a globally optimal, AC-feasible operating point in polynomial time and provides a certificate of global optimality; for radial distribution feeders, the additional desiderata are a certificate of infeasibility when the operating point is far from normal and robustness where Newton's method fails to converge. Modeling tools of the era for posing and solving the convex programs include YALMIP with the SeDuMi SDP solver, and conic/QCP solvers such as Gurobi; CVXPY provides a Python modeling layer for the same conic programs.

## Code framework

A pre-method scaffold. The data and conic-solver primitives already exist; the solver itself is the empty slot.

```python
import numpy as np

# --- existing: read a MATPOWER-style case into network + admittance data ---
def load_case(casefile):
    """Parse buses, generators, branches, limits; build the bus admittance
    matrix Y, demands s_d, voltage/gen/line limits, cost coefficients."""
    ...  # standard toolbox functionality
    return Y, s_d, limits, cost

# --- existing: complex-quadratic-form bookkeeping -------------------------
def injection_quadratic_forms(Y):
    """For each bus j return the Hermitian matrices (Phi_j, Psi_j, J_j) such
    that  p_j = V^H Phi_j V,  q_j = V^H Psi_j V,  |V_j|^2 = V^H J_j V."""
    n1 = Y.shape[0]
    Phi, Psi, J = [], [], []
    for j in range(n1):
        e = np.zeros((n1, 1)); e[j] = 1.0
        Yj = e @ (e.T @ Y)                 # j-th row of Y, else 0
        Phi.append(0.5 * (Yj.conj().T + Yj))
        Psi.append((Yj.conj().T - Yj) / (2j))
        J.append(e @ e.T)
    return Phi, Psi, J

# --- THE SLOT: a tractable global approach to the nonconvex AC-OPF ----------
def solve_opf(Y, s_d, limits, cost):
    """Return an operating point (bus voltages V) that minimizes generation
    cost subject to the AC power-flow constraints, ideally with some way to
    judge after the fact how close it is to the global optimum. Left empty
    here."""
    # TODO
    raise NotImplementedError
```
