# Branch and Bound for (mixed-)integer programming

## Problem

Solve a linear program in which some variables must be integer:

```
maximize  c'x   subject to   Ax <= b,   l <= x <= u,   x_j integer for j in J
```

(`J` = all variables for a pure integer program, a subset for a mixed one; binary variables are the
case `[l_j, u_j] = [0, 1]`). The integer-feasible set is the lattice points inside the polytope —
**non-convex**, so the simplex method cannot be aimed at it directly, and **exponentially large**
(`2^n` for `n` binaries), so total enumeration is hopeless. Rounding the continuous optimum is
generally infeasible or suboptimal. The goal is a *provably* optimal integer point, found
automatically, without walking the whole lattice.

## Key idea

Drop integrality to get the **LP relaxation**. Because relaxing (enlarging the feasible set) can
only raise a maximum, the relaxation's value is a certified **upper bound** on the integer optimum.
Two outcomes at a subproblem (node):

- The relaxation optimum `x*` is already integer in all of `J` → it is feasible and achieves the
  upper bound, hence **optimal for that subtree**; record it.
- Some `x*_j` is fractional → **branch**. No integer lies in the open strip `(floor(x*_j),
  ceil(x*_j))`, so split into two children, `x_j <= floor(x*_j)` and `x_j >= ceil(x*_j)`, losing no
  integer point while cutting `x*` out of both. Each child's relaxation is over a smaller feasible
  region, so its bound cannot be higher than the parent's, though it can tie if another optimum
  remains.

Carry the best integer point found as the **incumbent** (value `z_inc`, a *lower* bound on the
optimum). **Fathom** (prune) a node without exploring its subtree when it is **infeasible**, when
its relaxation upper bound `<= z_inc` (the whole subtree cannot beat the incumbent), or when its
relaxation is **integral** (a leaf — update the incumbent). The bound, not enumeration, discards the
subtrees that cannot contain the optimum. The **optimality gap** `z_bar - z_inc`, where `z_bar` is
the largest upper bound over still-open nodes, traps the optimum `z_inc <= z* <= z_bar`; gap zero is
a proof of optimality, a small gap is a principled early stop.

For minimization, flip every direction: the relaxation gives a *lower* bound, the incumbent an
*upper* bound, and a node is fathomed when its bound `>= z_inc`.

## Algorithm

```
incumbent z_inc = -inf,  x_inc = none
open nodes = { root: bounds [l_j, u_j] }                # depth-first or best-first
while open nodes remain:
    pop a node (its per-variable bounds)
    solve the LP relaxation -> (x*, u)                  # u = node upper bound
    if infeasible:                       fathom (empty)
    elif u <= z_inc:                     fathom (bound: cannot beat incumbent)
    elif x* integral on J:                              # leaf candidate
        if u > z_inc: z_inc, x_inc = u, x*              # update incumbent
        fathom
    else:                                               # branch
        pick fractional j in J (most-fractional: argmax_j |x*_j - round(x*_j)|)
        push child with x_j <= floor(x*_j)
        push child with x_j >= ceil(x*_j)
return x_inc, z_inc                       # provably optimal when open set empties
```

**Node selection.** *Best-first* expands the open node with the best bound — fewest nodes, but a
large open frontier in memory. *Depth-first* dives to a leaf — tiny memory and an early incumbent
that makes bound-pruning fire sooner. Practical solvers mix the two.

**Cut strengthening.** When a node's relaxation bound is loose, an existing cutting-plane routine
can add a valid inequality — e.g. a Gomory cut from the simplex tableau — that is violated by `x*`
but satisfied by every integer point. Re-solving the tightened relaxation can lower the node's
ceiling before the split.

## Code

A single self-contained C++17 program. It reads one MILP from stdin (variables, constraints,
objective, then per-variable bounds and an integer flag) and prints the optimal value with its
integer coordinates and the number of nodes explored. The node solver is a small inline Big-M
simplex on the bounded-variable LP relaxation; branching simply tightens a variable's bound.

```cpp
// Branch and Bound for (mixed-)integer linear programming.
// Reads a MILP from stdin and prints the provably optimal integer point.
//
// Input (maximization):
//   n m                       # n variables, m inequality constraints (Ax <= b)
//   c_1 ... c_n               # objective row to MAXIMIZE  c'x
//   then m lines:  a_i1 ... a_in  b_i      # one constraint row + its rhs
//   then n lines:  l_j u_j integer_flag    # bound l_j<=x_j<=u_j (u_j may be "inf"), 1 if x_j integer
// Output:
//   first line:  the optimal objective value, then the n integer coordinates
//   second line: number of nodes explored
//   ("INFEASIBLE" if no integer-feasible point exists; "UNBOUNDED" if the relaxation is unbounded)

#include <bits/stdc++.h>
using namespace std;

static const double INF = 1e30;
static const double EPS = 1e-9;
static const double TOL = 1e-6;   // integrality / pruning tolerance

struct LP {
    int n, m;
    vector<vector<double>> A;   // m x n
    vector<double> b;           // m
    vector<double> c;           // n  (maximize)
};

// ---------------------------------------------------------------------------
// Bounded-variable LP relaxation solver: maximize c'x s.t. Ax<=b, lo<=x<=up.
// Shift y_j = x_j - lo[j] >= 0, add an explicit row y_j <= up[j]-lo[j] for each
// finite upper bound, then run a Big-M primal simplex on
//      max c'y  s.t.  A'y (<= or >=) b',  y >= 0.
// Rows whose shifted rhs is negative are sign-flipped to ">=" and given an
// artificial variable with a large penalty. Returns true if feasible/bounded
// (filling x and value); false if infeasible; sets `unbounded` when unbounded.
// ---------------------------------------------------------------------------
static bool solve_lp(const LP& lp, const vector<double>& lo, const vector<double>& up,
                     vector<double>& x, double& value, bool& unbounded) {
    unbounded = false;
    int n = lp.n;

    vector<vector<double>> R;   // constraint rows in shifted variable y
    vector<double> rhs;
    vector<int> ge;             // 1 if row is ">=" (needs an artificial)

    for (int i = 0; i < lp.m; ++i) {
        vector<double> row(n);
        double bb = lp.b[i];
        for (int j = 0; j < n; ++j) { row[j] = lp.A[i][j]; bb -= lp.A[i][j] * lo[j]; }
        if (bb < 0) { for (double& v : row) v = -v; bb = -bb; ge.push_back(1); }
        else ge.push_back(0);
        R.push_back(row); rhs.push_back(bb);
    }
    for (int j = 0; j < n; ++j) {
        if (up[j] < INF / 2) {
            double cap = up[j] - lo[j];
            if (cap < -TOL) return false;       // lo_j > up_j : node infeasible
            vector<double> row(n, 0.0); row[j] = 1.0;
            R.push_back(row); rhs.push_back(max(cap, 0.0)); ge.push_back(0);
        }
    }

    int M = (int)R.size();
    int total = n + M + M;       // structural | slack/surplus | artificial
    int artStart = n + M;

    vector<vector<double>> T(M, vector<double>(total + 1, 0.0)); // last col = rhs
    vector<int> basis(M);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < n; ++j) T[i][j] = R[i][j];
        T[i][n + i] = (ge[i] ? -1.0 : 1.0);     // slack (+1) or surplus (-1)
        if (ge[i]) { T[i][artStart + i] = 1.0; basis[i] = artStart + i; }
        else        basis[i] = n + i;
        T[i][total] = rhs[i];
    }

    double BIGM = 0;
    for (int j = 0; j < n; ++j) BIGM = max(BIGM, fabs(lp.c[j]));
    for (double v : rhs)        BIGM = max(BIGM, fabs(v));
    BIGM = (BIGM + 1.0) * 1e7;

    vector<double> obj(total, 0.0);     // objective coefficient of each column
    for (int j = 0; j < n; ++j) obj[j] = lp.c[j];
    for (int i = 0; i < M; ++i) if (ge[i]) obj[artStart + i] = -BIGM;

    int guard = 0, maxIter = 20000 + 50 * (M + total);
    vector<double> red(total, 0.0);
    while (true) {
        if (++guard > maxIter) break;            // safety; effectively never hit
        // reduced cost of column j = (obj of basis . column j) - obj[j]
        int piv = -1; double best = -EPS;
        for (int j = 0; j < total; ++j) {
            double s = 0;
            for (int i = 0; i < M; ++i) s += obj[basis[i]] * T[i][j];
            double r = s - obj[j];
            if (r < best) { best = r; piv = j; }
        }
        if (piv < 0) break;                      // optimal
        int leave = -1; double bestRatio = 0;
        for (int i = 0; i < M; ++i) {
            if (T[i][piv] > EPS) {
                double ratio = T[i][total] / T[i][piv];
                if (leave < 0 || ratio < bestRatio - EPS) { bestRatio = ratio; leave = i; }
            }
        }
        if (leave < 0) { unbounded = true; return false; }
        double pv = T[leave][piv];
        for (int j = 0; j <= total; ++j) T[leave][j] /= pv;
        for (int i = 0; i < M; ++i) {
            if (i == leave) continue;
            double f = T[i][piv];
            if (fabs(f) < EPS) continue;
            for (int j = 0; j <= total; ++j) T[i][j] -= f * T[leave][j];
        }
        basis[leave] = piv;
    }

    // any artificial still basic at a positive level => infeasible node
    for (int i = 0; i < M; ++i)
        if (basis[i] >= artStart && T[i][total] > 1e-5) return false;

    vector<double> y(n, 0.0);
    for (int i = 0; i < M; ++i)
        if (basis[i] < n) y[basis[i]] = T[i][total];
    x.assign(n, 0.0);
    double val = 0;
    for (int j = 0; j < n; ++j) { x[j] = y[j] + lo[j]; val += lp.c[j] * x[j]; }
    value = val;
    return true;
}

// ---------------------------------------------------------------------------
// Branch and bound.  Open nodes = per-variable bound vectors on a depth-first
// stack. At each node: solve the relaxation (bound); fathom if infeasible, if
// its ceiling cannot beat the incumbent, or if it is integral (a leaf); else
// branch on the most-fractional integer variable into x_j<=floor and x_j>=ceil.
// ---------------------------------------------------------------------------
struct Node { vector<double> lo, up; };

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    LP lp; lp.n = n; lp.m = m;
    lp.c.resize(n);
    for (int j = 0; j < n; ++j) cin >> lp.c[j];
    lp.A.assign(m, vector<double>(n));
    lp.b.resize(m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) cin >> lp.A[i][j];
        cin >> lp.b[i];
    }
    vector<double> lo0(n), up0(n);
    vector<int> isInt(n, 0);
    for (int j = 0; j < n; ++j) {
        string ls, us; int f;
        cin >> ls >> us >> f;
        lo0[j] = (ls == "inf") ? INF : (ls == "-inf") ? -INF : stod(ls);
        up0[j] = (us == "inf") ? INF : (us == "-inf") ? -INF : stod(us);
        isInt[j] = f;
    }

    double bestVal = -INF;            // incumbent value = LOWER bound on the optimum
    vector<double> bestX;             // incumbent integer-feasible point
    long long nodes = 0;

    vector<Node> stack;
    stack.push_back({lo0, up0});
    while (!stack.empty()) {
        Node nd = stack.back(); stack.pop_back();
        ++nodes;
        vector<double> x; double ub; bool unbounded = false;
        bool feas = solve_lp(lp, nd.lo, nd.up, x, ub, unbounded);
        if (unbounded) { cout << "UNBOUNDED\n"; return 0; }
        if (!feas) continue;                       // FATHOM: infeasible
        if (ub <= bestVal + TOL) continue;         // FATHOM by bound: ceiling <= incumbent floor

        double frac = -1; int jbr = -1;            // most-fractional integer variable
        for (int j = 0; j < n; ++j) {
            if (!isInt[j]) continue;
            double f = fabs(x[j] - round(x[j]));
            if (f > frac) { frac = f; jbr = j; }
        }
        if (jbr < 0 || frac <= TOL) {              // relaxation integral -> leaf candidate
            if (ub > bestVal + TOL) { bestVal = ub; bestX = x; }   // update incumbent
            continue;
        }
        double fl = floor(x[jbr]), ce = ceil(x[jbr]);
        if (ce <= nd.up[jbr] + TOL) {              // child x_j >= ceil(x*_j)
            Node up = nd; up.lo[jbr] = ce; stack.push_back(up);
        }
        if (fl >= nd.lo[jbr] - TOL) {              // child x_j <= floor(x*_j)
            Node dn = nd; dn.up[jbr] = fl; stack.push_back(dn);
        }
    }

    if (bestX.empty()) { cout << "INFEASIBLE\n"; return 0; }
    cout << (long long)llround(bestVal);
    for (int j = 0; j < n; ++j) cout << ' ' << (long long)llround(bestX[j]);
    cout << '\n' << nodes << '\n';
    return 0;
}
```

On the 0/1 knapsack (`vals=[8,11,6,4,7,3]`, `wts=[5,7,4,3,5,2]`, `cap=14`, all binary), the program
reads the instance and prints `22 1 1 0 0 0 1` over 3 nodes; on the two-variable integer LP
(`c=[4,-1]`, `A=[[7,-2],[0,1],[2,-2]]`, `b=[14,3,3]`) it prints `7 2 1` over 7 nodes — both the
brute-force optima. The solver returns a candidate only after every live node has been fathomed by
infeasibility, by the incumbent bound, or by an integral relaxation. Divide the discrete space,
conquer each piece with a convex LP bound, and let the bound prune everything that cannot hold the
optimum.
