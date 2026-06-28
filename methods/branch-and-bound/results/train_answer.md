Linear programming is solved: the simplex method walks the vertices of a convex polytope and returns a globally optimal continuous solution. But many real problems demand whole-number answers—number of flights, machine setups, or binary yes/no decisions. The moment integrality is required, the feasible set becomes a scattered cloud of lattice points inside the polytope, no longer convex, so simplex has no vertex to walk to. Rounding the continuous optimum often violates constraints or produces a suboptimal point, and total enumeration of all integer points grows exponentially with the number of variables. What is needed is a general, automatic procedure that respects integrality exactly and returns a provably optimal integer point without inspecting the whole lattice.

The key insight is that dropping the integrality requirement gives the LP relaxation, whose feasible set contains every integer-feasible point and possibly more. Because enlarging the feasible set can only raise a maximum, the relaxation's optimum value is a certified upper bound on the true integer optimum. When the relaxation itself happens to be integral, it is immediately optimal. When it is fractional in some marked variable, say x_j = 2.6, no integer point lies in the open strip between 2 and 3, so the problem can be split into two subproblems: x_j <= 2 and x_j >= 3. This loses no integer point and removes the fractional optimum from both children. Repeating this recursively creates a tree of LP relaxations whose leaves are pinned-down integer candidates.

The method is called Branch and Bound. It combines three operations: bound, branch, and fathom. At each node of the tree, solve the LP relaxation to obtain an upper bound on what that subtree can achieve. If the relaxation is fractional, branch on the most-fractional marked variable by tightening its bounds to the nearest integers on either side. Track the best integer-feasible solution found anywhere, called the incumbent; its value is a lower bound on the optimum. A node is fathomed—discarded without further exploration—when its relaxation is infeasible, when its relaxation upper bound is no better than the incumbent, or when its relaxation is already integral. The bound, not enumeration, prunes entire subtrees that cannot contain the optimum.

The algorithm is correct because branching partitions the integer-feasible set exactly and fathoming only discards subtrees whose upper bound proves they cannot beat the incumbent. As the search proceeds, the incumbent rises and the outstanding ceilings fall, squeezing the optimum from both sides. The optimality gap, defined as the largest remaining upper bound minus the incumbent value, gives a live certificate of progress: when it reaches zero, optimality is proved. For a maximization, the relaxation provides the ceiling and the incumbent provides the floor; for minimization the roles reverse. Node selection is a practical choice: depth-first dives for an early incumbent on small memory, while best-first always expands the most promising node to minimize nodes explored. Branching on the variable closest to half-integer tends to make the most decisive split.

Concretely this lands as a single self-contained C++17 program. It reads one MILP from stdin — `n m`,
the objective row to maximize, then `m` constraint rows each followed by their right-hand side, then
`n` lines giving each variable's lower bound, upper bound (`inf` allowed), and a 0/1 integer flag —
and prints the optimal objective value with its coordinates on the first line and the number
of nodes explored on the second. The node solver is a small inline Big-M primal simplex on the
bounded-variable LP relaxation; a branch is nothing more than tightening one variable's bound, so a
child differs from its parent by a single number.

```cpp
// Branch and Bound for (mixed-)integer linear programming.
// Reads a MILP from stdin and prints the provably optimal point.
//
// Input (maximization):
//   n m                       # n variables, m inequality constraints (Ax <= b)
//   c_1 ... c_n               # objective row to MAXIMIZE  c'x
//   then m lines:  a_i1 ... a_in  b_i      # one constraint row + its rhs
//   then n lines:  l_j u_j integer_flag    # bound l_j<=x_j<=u_j (inf/-inf allowed), 1 if x_j integer
// Output:
//   first line:  the optimal objective value, then the n coordinates
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

static bool finite_bound(double v) {
    return fabs(v) < INF / 2;
}

static string format_number(double v) {
    if (fabs(v) < 5e-10) v = 0.0;
    double r = round(v);
    if (fabs(v - r) <= 1e-7) return to_string((long long)llround(r));
    ostringstream out;
    out << setprecision(10) << v;
    return out.str();
}

struct ExpandedVar {
    int original;
    double scale;
};

// ---------------------------------------------------------------------------
// Bounded/free-variable LP relaxation solver: maximize c'x s.t. Ax<=b,
// lo<=x<=up. Each original variable is represented with nonnegative simplex
// variables: x=lo+y for finite lower bounds, x=up-y for upper-only bounds,
// and x=y+ - y- for free variables. Finite upper bounds become explicit rows.
// A Big-M primal simplex then solves the resulting nonnegative LP.
// ---------------------------------------------------------------------------
static bool solve_lp(const LP& lp, const vector<double>& lo, const vector<double>& up,
                     vector<double>& x, double& value, bool& unbounded) {
    unbounded = false;
    int n = lp.n;

    vector<double> constant(n, 0.0);
    vector<vector<pair<int, double>>> expr(n);
    vector<ExpandedVar> expanded;
    vector<double> obj;
    vector<vector<double>> extraRows;
    vector<double> extraRhs;

    auto add_expanded = [&](int original, double scale, double coeff) {
        int id = (int)obj.size();
        obj.push_back(coeff);
        expanded.push_back({original, scale});
        expr[original].push_back({id, scale});
        return id;
    };

    for (int j = 0; j < n; ++j) {
        bool hasLo = finite_bound(lo[j]);
        bool hasUp = finite_bound(up[j]);
        if (lo[j] >= INF / 2 || up[j] <= -INF / 2) return false;
        if (hasLo && hasUp && lo[j] > up[j] + TOL) return false;

        if (hasLo) {
            constant[j] = lo[j];
            int y = add_expanded(j, 1.0, lp.c[j]);
            if (hasUp) {
                double cap = up[j] - lo[j];
                if (cap < -TOL) return false;
                vector<double> row(obj.size(), 0.0);
                row[y] = 1.0;
                extraRows.push_back(row);
                extraRhs.push_back(max(0.0, cap));
            }
        } else if (hasUp) {
            constant[j] = up[j];
            add_expanded(j, -1.0, -lp.c[j]);
        } else {
            add_expanded(j, 1.0, lp.c[j]);
            add_expanded(j, -1.0, -lp.c[j]);
        }
    }

    int N = (int)obj.size();
    vector<vector<double>> R;   // constraint rows in expanded variable y
    vector<double> rhs;
    vector<int> ge;             // 1 if row is ">=" (needs an artificial)

    auto add_row = [&](vector<double> row, double bb) {
        row.resize(N, 0.0);
        if (bb < -EPS) {
            for (double& v : row) v = -v;
            R.push_back(row);
            rhs.push_back(-bb);
            ge.push_back(1);
        } else {
            R.push_back(row);
            rhs.push_back(max(0.0, bb));
            ge.push_back(0);
        }
    };

    for (int i = 0; i < lp.m; ++i) {
        vector<double> row(N, 0.0);
        double bb = lp.b[i];
        for (int j = 0; j < n; ++j) {
            bb -= lp.A[i][j] * constant[j];
            for (auto [id, scale] : expr[j]) row[id] += lp.A[i][j] * scale;
        }
        add_row(row, bb);
    }
    for (int k = 0; k < (int)extraRows.size(); ++k) add_row(extraRows[k], extraRhs[k]);

    int M = (int)R.size();
    int total = N + M + M;       // structural | slack/surplus | artificial
    int artStart = N + M;

    vector<vector<double>> T(M, vector<double>(total + 1, 0.0)); // last col = rhs
    vector<int> basis(M);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) T[i][j] = R[i][j];
        T[i][N + i] = (ge[i] ? -1.0 : 1.0);     // slack (+1) or surplus (-1)
        if (ge[i]) { T[i][artStart + i] = 1.0; basis[i] = artStart + i; }
        else        basis[i] = N + i;
        T[i][total] = rhs[i];
    }

    double BIGM = 0;
    for (double v : obj) BIGM = max(BIGM, fabs(v));
    for (double v : rhs) BIGM = max(BIGM, fabs(v));
    BIGM = (BIGM + 1.0) * 1e7;

    vector<double> simplexObj(total, 0.0);
    for (int j = 0; j < N; ++j) simplexObj[j] = obj[j];
    for (int i = 0; i < M; ++i) if (ge[i]) simplexObj[artStart + i] = -BIGM;

    int guard = 0, maxIter = 20000 + 50 * (M + total);
    while (true) {
        if (++guard > maxIter) break;            // safety; effectively never hit
        int piv = -1;
        double best = -EPS;
        for (int j = 0; j < total; ++j) {
            double s = 0.0;
            for (int i = 0; i < M; ++i) s += simplexObj[basis[i]] * T[i][j];
            double reduced = s - simplexObj[j];
            if (reduced < best) { best = reduced; piv = j; }
        }
        if (piv < 0) break;                      // optimal
        int leave = -1;
        double bestRatio = 0.0;
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

    for (int i = 0; i < M; ++i)
        if (basis[i] >= artStart && T[i][total] > 1e-5) return false;

    vector<double> y(N, 0.0);
    for (int i = 0; i < M; ++i)
        if (basis[i] < N) y[basis[i]] = T[i][total];

    x = constant;
    for (int id = 0; id < N; ++id) x[expanded[id].original] += expanded[id].scale * y[id];

    value = 0.0;
    for (int j = 0; j < n; ++j) value += lp.c[j] * x[j];
    return true;
}

// ---------------------------------------------------------------------------
// Branch and bound. Open nodes = per-variable bound vectors on a depth-first
// stack. At each node: solve the relaxation (bound); fathom if infeasible, if
// its ceiling cannot beat the incumbent, or if it is integral in the marked
// variables; otherwise branch on the most-fractional integer variable.
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
        vector<double> x; double ub = 0.0; bool unbounded = false;
        bool feas = solve_lp(lp, nd.lo, nd.up, x, ub, unbounded);
        if (unbounded) { cout << "UNBOUNDED\n"; return 0; }
        if (!feas) continue;                       // FATHOM: infeasible
        if (ub <= bestVal + TOL) continue;         // FATHOM by bound: ceiling <= incumbent floor

        double frac = -1.0;
        int jbr = -1;
        for (int j = 0; j < n; ++j) {
            if (!isInt[j]) continue;
            double f = fabs(x[j] - round(x[j]));
            if (f > frac) { frac = f; jbr = j; }
        }
        if (jbr < 0 || frac <= TOL) {              // relaxation integral on marked variables
            if (ub > bestVal + TOL) { bestVal = ub; bestX = x; }
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
    cout << format_number(bestVal);
    for (int j = 0; j < n; ++j) cout << ' ' << format_number(bestX[j]);
    cout << '\n' << nodes << '\n';
    return 0;
}
```

On the 0/1 knapsack (`vals=[8,11,6,4,7,3]`, `wts=[5,7,4,3,5,2]`, `cap=14`, all binary) the program
prints `22 1 1 0 0 0 1` over 3 nodes — one split on the lone fractional item pins down the optimum
and the bound discards the dominated sibling. On the two-variable integer LP (`c=[4,-1]`,
`A=[[7,-2],[0,1],[2,-2]]`, `b=[14,3,3]`) it prints `7 2 1` over 7 nodes, exercising
fathoming-by-infeasibility and two incumbent updates. Both match exhaustive enumeration over the
(small) box. Divide the discrete space, conquer each piece with a convex LP bound, and let the bound
prune everything that cannot hold the optimum.
