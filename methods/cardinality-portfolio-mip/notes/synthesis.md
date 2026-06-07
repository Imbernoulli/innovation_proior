# Synthesis — cardinality-constrained portfolio as MIQP

## Sources retrieved (this run)
- PRIMARY: Bertsimas & Shioda 2009, "Algorithm for cardinality-constrained quadratic optimization",
  Comput Optim Appl 43:1-22. Full PDF at refs/bertsimas_shioda.pdf. Exact CCQO (1), relaxation (2),
  portfolio formulation (8)/(9) read.
- PRIMARY (base): Markowitz 1952 mean-variance — read sibling methods/markowitz/results/.
- PRIMARY (Bienstock 1996, Math Prog 74:121-140): PAYWALLED (Springer), OCR-hostile/no free full text
  after real search. Grounded via Bertsimas-Shioda's explicit discussion of it ([4] in their refs):
  Bienstock's tailored B&B replaces |supp(x)|<=K with the surrogate sum_i (x_i/u_i) <= K, branches on
  x_i directly (add x_j<=0 down, x_j>=alpha up) rather than introducing binaries, uses a primal
  feasible QP solver (Newton/steepest-descent/Frank-Wolfe) with quadratic-penalty warm starts, and a
  branch-and-cut implementation. GAP FLAGGED: I did not read Bienstock's original text; its formulation
  is reconstructed from the secondary (Bertsimas-Shioda) + the standard CCMV literature (Chang).
- BACKGROUND: continuous MVO QP — Chang/Maringer/Beasley/Sharaiha-lineage local-search paper
  (arXiv cs/0104017) §2.1, refs/chang_localsearch.pdf. Branch-and-bound for MIQP — Bertsimas-Shioda §2.
  big-M indicator linking — Chang §2.2 eqs (4)-(6); Lobo-Fazel-Boyd fixed-cost §1.1.
- THIRD-PARTY: Chang et al. local-search §2.2 = canonical cardinality+quantity (buy-in) formulation
  with binaries z_i: sum z_i <= k, eps_i z_i <= x_i <= delta_i z_i, z_i in {0,1}. PyPortfolioOpt
  code/discrete_allocation.py lp_portfolio (integer share counts via MILP, L1 deviation, leftover cash)
  + code/objective_functions.py transaction_cost = k*||w - w_prev||_1 (L1 turnover).

## Pain point (the move beyond continuous MVO)
Continuous Markowitz gives a fractional weight vector w in the simplex (sum w=1, l<=w<=u), min
w'Sigma w - q mu'w. Real desks cannot deploy it verbatim:
1. CARDINALITY: an institutional/focused fund must hold at most K << N names (monitoring cost, "not an
   indexer", focused-fund prospectus, separate accounts with tiny budgets where per-name transaction
   cost dominates). Continuous MVO routinely returns many tiny nonzero weights.
2. BUY-IN / MINIMUM POSITION: a holding that is nonzero must be at least some floor alpha_i (a 0.02%
   position is operationally pointless and costs a ticket). So each weight is SEMI-CONTINUOUS:
   x_i = 0 OR x_i in [l_i, u_i].
3. ROUND LOTS / INTEGER SHARES: you trade whole shares (or whole lots of 100). The deployable object
   is an integer share count n_i, not a real weight; w_i = n_i p_i / V.
All three are non-convex / combinatorial: the feasible set is a union of faces of the simplex, not a
convex set. The convex QP machinery cannot express "x_i = 0 or x_i >= alpha_i".

## Key idea — indicator (binary) variables + linking, then branch-and-bound
Introduce y_i in {0,1}, y_i = 1 iff name i is held. Link with a big-M / fixed-charge pair:
    l_i y_i <= x_i <= u_i y_i      (l_i>0 is the buy-in floor; u_i is the cap, the "big M")
  - y_i=0 forces 0 <= x_i <= 0 => x_i=0 (name dropped).
  - y_i=1 forces l_i <= x_i <= u_i (buy-in respected).
Cardinality:  sum_i y_i <= K.
Objective stays the mean-variance quadratic:  min (1/2) x'Sigma x - q mu'x  (q = risk tolerance;
or max mu'x - (delta/2) x'Sigma x). Now it's a MIQP: convex quadratic objective, linear constraints,
some binary vars. Relax y in [0,1] -> convex QP lower bound; branch on a fractional y_i (y_i=0 vs
y_i=1) -> branch-and-bound / branch-and-cut. Each node is a convex QP (Bertsimas-Shioda use Lemke's
pivoting for warm-started node solves; a generic MIQP solver / cvxpy-with-integer-vars does B&B for us).

NOTE the equivalence Bienstock used: instead of binaries, the surrogate sum_i (x_i/u_i) <= K and
direct branching on x_i (x_j<=0 / x_j>=alpha_j). Same combinatorial structure; binaries are the clean
modern statement.

## Round lots (integer share counts)
Replace continuous x_i by integer n_i (number of round lots, lot size s_i shares at price p_i):
position value = n_i s_i p_i; weight x_i = n_i s_i p_i / V with V = total budget. n_i in Z>=0, n_i
integer. Cardinality y_i still links: n_i <= M_i y_i, n_i >= (l_i V/(s_i p_i)) y_i. The discrete
allocation step (PyPortfolioOpt lp_portfolio) takes target continuous weights w* and finds integer
share counts minimizing the L1 tracking error |w* V - x_i p_i| plus leftover cash r = V - sum p_i x_i,
x_i integer >= 0, r >= 0. That is the round-lot projection as a MILP.

## Rebalancing: turnover + transaction cost
Start from current holdings x0 (or w_prev). Trade vector = x - x0. Split into buys/sells:
    x_i - x0_i = b_i - s_i,  b_i,s_i >= 0   (b_i=buy amount, s_i=sell amount)
Turnover = sum_i |x_i - x0_i| = sum_i (b_i + s_i)  (the L1 of the trade). At optimum only one of b_i,
s_i is nonzero per name, so the split is exact. Proportional (linear) cost = sum_i (tc_buy*b_i +
tc_sell*s_i); symmetric case = tau * ||x - x0||_1. Fixed/ticket cost per traded name = sum_i
beta_i * t_i with a binary t_i and linking |x_i - x0_i| <= M t_i (Lobo-Fazel-Boyd fixed-plus-linear
cost). PyPortfolioOpt's transaction_cost objective is the symmetric L1 form k*||w - w_prev||_1.
Subtract the cost from return (or add to objective) and optionally bound turnover sum(b+s) <= T_max.

## Design decisions -> why
- binary y_i with l_i y_i <= x_i <= u_i y_i (not a smooth penalty / L1): L1 (lasso-style) only SHRINKS,
  it cannot enforce an exact count K nor a hard floor l_i; "x=0 OR x>=l" is genuinely disjunctive, so
  you need an integer variable to pick the branch. big-M = u_i is the natural tight cap (the weight
  cap), keeping the relaxation tight.
- sum y_i <= K not = K: at-most-K is the operational constraint; equality can be infeasible/forced and
  wastes a name. (<= lets the optimizer hold fewer if cheaper.)
- buy/sell split b-s for turnover instead of cp.abs directly: cvxpy handles abs, but the explicit split
  exposes asymmetric buy/sell costs and a turnover budget, and is the LP-standard linearization; at
  optimum complementary (b_i s_i = 0) because minimizing b_i+s_i with fixed b_i-s_i drives the smaller
  to 0 — no extra constraint needed.
- integer n_i for round lots vs rounding the continuous weights: naive rounding violates the budget and
  can break cardinality; solving the integer projection respects budget (r>=0) and minimizes tracking
  error exactly.
- branch on y (or on x_i directly a la Bienstock): the only non-convexity is the integrality; relaxing
  it gives a convex QP bound, and the simplex/Lemke warm-start makes node resolves cheap. branch-and-
  CUT (Bienstock) adds valid inequalities (e.g. cardinality cover cuts) to tighten the relaxation.

## Solver reality
cvxpy with integer=True / boolean=True dispatches to a MIQP-capable solver. HiGHS (HIGHS) handles MILP
but NOT a quadratic objective with integers; for MIQP need SCIP/GUROBI/CPLEX/MOSEK. Strategy for the
small demo: solve the cardinality+buy-in MIQP via cvxpy (boolean y) with a MIQP solver if available;
ALSO give the round-lot MILP (lp_portfolio-style) which HiGHS can solve, and a turnover/transaction-
cost rebalance. To stay runnable everywhere, I will (a) reformulate the MIQP objective so it can fall
back to a SOC/quad solver, and (b) if no MIQP solver, demonstrate the round-lot MILP (linear) which
HiGHS solves, and solve the QP-with-cardinality by enumerating the small instance / using SCIP if
present. Will TEST which solvers exist in the venv before finalizing code.

## context scaffold (pre-method) — pieces to hollow out
- mean (mu), covariance (Sigma), budget sum x=1, box l<=x<=u  (EXIST, continuous MVO)
- prices p, lot size s, budget V  (EXIST)
- current holdings x0  (EXIST)
- EMPTY SLOT: the rule that enforces "hold at most K names, each 0 or >= floor, in whole lots, while
  paying to trade" — the discrete feasibility + the solve. -> filled by binaries+linking+B&B.
