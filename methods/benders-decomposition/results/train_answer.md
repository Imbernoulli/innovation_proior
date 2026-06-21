Large optimization models often mix variables that play very different roles. Some variables encode discrete strategic choices, network design decisions, capacity investments, or first-stage commitments; once those choices are fixed, the remaining variables describe flows, dispatch, recourse, or operational responses that frequently reduce to an ordinary linear program. Solving everything together hides this structure. A monolithic formulation forces the optimizer to explore the combinatorial complexity of the master variables while simultaneously carrying every continuous recourse variable and every constraint that governs them. That can be impractical when the master is mixed-integer or when many scenarios create repeated copies of the same subproblem structure. Projecting away the recourse variables is a natural idea, but it replaces an explicit model with an implicit value function and an exponentially large projected feasible region. Manual elimination or generic cutting-plane methods either produce an unmanageable number of inequalities or ignore the structural link between the master decision and the recourse subproblem, so neither is fully satisfactory.

Benders decomposition is the method that exploits this structure without writing the projection explicitly. It separates the problem into a master problem over the complicating variables and a subproblem over the recourse variables. The master chooses a candidate master decision and maintains an auxiliary variable, often called eta, that represents the best achievable recourse cost. For the fixed master decision, the subproblem evaluates the true recourse cost and, crucially, returns a dual certificate rather than only a primal response. If the subproblem is feasible, an optimal dual solution defines an affine lower bound on the recourse value function that holds for every master decision; this bound is added to the master as an optimality cut. If the subproblem is infeasible, a dual ray or Farkas certificate shows that the current master decision cannot be extended to a feasible recourse and yields a feasibility cut that removes a region of master decisions. The master is then resolved with the new cut, and the process repeats until the estimate eta agrees with the actual recourse value within tolerance.

The power of the method is that duality turns one solved subproblem into reusable global evidence. A primal recourse solution only says how to respond to the current master decision, but a dual solution states that under the current shadow prices every master decision must pay at least a certain affine amount for recourse. The master therefore learns only the parts of the hidden value function that the search encounters, building a piecewise-linear outer approximation from supporting planes and infeasibility regions. This makes Benders much more than divide and conquer: it is an evidence-driven cutting-plane algorithm for the implicit projected problem. It is most compelling when the master variables are few or structurally difficult and the subproblem is large but tractable, such as facility location with transportation subproblems, network design with flow subproblems, unit commitment with dispatch subproblems, and two-stage stochastic programs where each scenario creates a separable recourse subproblem.

The standard derivation assumes a linear subproblem so that strong duality supplies the certificates. If the subproblem is integer or nonconvex, the clean LP dual cut is no longer available and one must resort to generalized Benders, logic-based Benders, or problem-specific inference. Other practical concerns include weak cuts that cause many iterations, degenerate subproblems that return unstable dual solutions, and the interaction between generated cuts and branch-and-bound when the master is mixed-integer. Despite these limitations, the central idea remains the canonical way to avoid carrying downstream variables in the master while still respecting their cost and feasibility consequences.

```python
import numpy as np
from scipy.optimize import linprog


def benders_solve(c, A, b, T, W, h, q, x_bounds, eta_lb=-1e9, max_iter=1000, tol=1e-6):
    """
    Solve via Benders decomposition:
        min c^T x + q^T y
        s.t. A x <= b, x >= 0
             T x + W y >= h, y >= 0

    The subproblem for fixed x is the LP:
        phi(x) = min q^T y  s.t.  W y >= h - T x, y >= 0
    Its dual provides optimality cuts; infeasibility provides feasibility cuts.
    """
    n_x = len(c)
    n_u = len(h)
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float) if A is not None else None
    b = np.asarray(b, dtype=float) if b is not None else None
    T = np.asarray(T, dtype=float)
    W = np.asarray(W, dtype=float)
    h = np.asarray(h, dtype=float)
    q = np.asarray(q, dtype=float)

    cuts_A = []      # left-hand side coefficients for x
    cuts_eta = []    # left-hand side coefficient for eta
    cuts_b = []      # right-hand side

    lower_bound = -np.inf
    upper_bound = np.inf
    best_x = None

    for iteration in range(max_iter):
        # Restricted master: min c^T x + eta subject to original and generated cuts.
        f_master = np.concatenate([c, [1.0]])
        A_ub, b_ub = [], []
        if A is not None:
            A_ub.append(np.hstack([A, np.zeros((A.shape[0], 1))]))
            b_ub.extend(b)
        for alpha, beta, gamma in zip(cuts_A, cuts_eta, cuts_b):
            A_ub.append(np.concatenate([alpha, [beta]]))
            b_ub.append(gamma)
        if A_ub:
            A_ub = np.vstack(A_ub)
            b_ub = np.asarray(b_ub, dtype=float)
        else:
            A_ub = np.zeros((1, n_x + 1))
            b_ub = np.zeros(1)
        bounds = list(x_bounds) + [(eta_lb, None)]

        master = linprog(f_master, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not master.success:
            raise RuntimeError(f"Master problem failed: {master.message}")

        x = master.x[:n_x]
        eta = master.x[n_x]
        lower_bound = max(lower_bound, master.fun)

        # Subproblem: max (h - T x)^T u s.t. W^T u <= q, u >= 0.
        rhs = h - T @ x
        sub = linprog(-rhs, A_ub=W.T, b_ub=q, bounds=[(0, None)] * n_u, method="highs")

        if sub.status == 2:  # dual infeasible -> primal subproblem infeasible
            # Farkas certificate: u >= 0, W^T u <= 0, (h - T x)^T u > 0.
            farkas = linprog(
                -rhs,
                A_ub=np.vstack([W.T, np.ones(n_u)]),
                b_ub=np.concatenate([np.zeros(W.shape[1]), [1.0]]),
                bounds=[(0, None)] * n_u,
                method="highs",
            )
            if not farkas.success or -farkas.fun <= tol:
                raise RuntimeError("Could not find Farkas certificate")
            u_ray = farkas.x
            # Feasibility cut: (h - T z)^T u_ray <= 0  <=>  -(T^T u_ray)^T z <= -h^T u_ray.
            cuts_A.append(-(T.T @ u_ray))
            cuts_eta.append(0.0)
            cuts_b.append(-h @ u_ray)
            continue

        if not sub.success:
            raise RuntimeError(f"Subproblem failed: {sub.message}")

        u = sub.x
        recourse = rhs @ u
        total_cost = c @ x + recourse
        if total_cost < upper_bound:
            upper_bound = total_cost
            best_x = x.copy()

        # Optimality cut: eta >= (h - T z)^T u for all z.
        # => -(T^T u)^T z - eta <= -h^T u.
        cuts_A.append(-(T.T @ u))
        cuts_eta.append(-1.0)
        cuts_b.append(-h @ u)

        if upper_bound - lower_bound <= tol or abs(eta - recourse) <= tol:
            print(f"Converged in {iteration + 1} iterations")
            return best_x, upper_bound, lower_bound

    print(f"Reached max iterations; best upper bound = {upper_bound:.6f}")
    return best_x, upper_bound, lower_bound


if __name__ == "__main__":
    # Simple capacity-planning example.
    c = np.array([3.0, 2.0])          # first-stage costs
    A = np.array([[1.0, 1.0]])        # first-stage constraint
    b = np.array([2.0])
    T = np.array([[1.0, 0.0], [0.0, 1.0]])
    W = np.array([[2.0, 1.0], [1.0, 2.0]])
    h = np.array([4.0, 5.0])
    q = np.array([5.0, 4.0])
    x_bounds = [(0, 2), (0, 2)]

    x, ub, lb = benders_solve(c, A, b, T, W, h, q, x_bounds, eta_lb=0.0)
    print(f"x = {x}")
    print(f"cost = {ub:.6f}, lower bound = {lb:.6f}")
```
