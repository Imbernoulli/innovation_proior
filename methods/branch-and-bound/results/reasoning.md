Let me start from the thing that actually blocks me. I have a linear program, `max c'x` subject to `Ax <= b`, `x >= 0`, and the simplex method solves it cold — vertex to vertex, pivot by pivot, global optimum, done. But the problem in front of me has a catch: some of these variables have to come out *whole*. The number of tankers I dispatch, the number of machine setups I run, a yes/no switch on whether to build a depot. Three-and-seven-tenths tankers is not a thing. So I want `max c'x` subject to `Ax <= b`, `x >= 0`, and `x_j` integer for the marked `j`. And the moment I write "integer," simplex is the wrong tool, because the feasible set just stopped being convex.

Let me make sure I actually believe that, because everything downstream hangs on it. The continuous feasible set is the polytope `{Ax <= b, x >= 0}` — convex, simplex's home turf. The integer feasible set is the *lattice points inside that polytope*. Take two of them, say `(2,3)` and `(4,5)`; their midpoint `(3,4)` happens to be integer here, but take `(2,3)` and `(3,3)` — midpoint `(2.5, 3)`, not integer, not in the set. So the integer feasible set is generically non-convex: a cloud of isolated points. Simplex assumes one convex region and walks to its optimal vertex. There is no vertex to walk to here; there's a scatter of points. So I can't just point simplex at this. And I'd be wary of any local-search patch — hop between nearby lattice points keeping the best — because the discrete set has no convex structure tying a local best to the global one, so I'd have no certificate that I'd actually found the optimum rather than a good-looking trap. Whatever I build, I want it to come with a *proof* it's optimal, not just a point that resists small perturbations.

The brute-force escape is to enumerate. List every integer point in the box, check feasibility, keep the best. It's exact. It's also dead on arrival: `n` binary variables give `2^n` points, and general integers give far more. Forty binaries and I'm enumerating a trillion points. Worse, it's *stupid* enumeration — most of the lattice is obviously terrible, yet I'd look at all of it. I need to not look at the parts that can't possibly win. Which means I need, for chunks of the lattice at a time, a cheap way to prove "nothing in here beats what I already have." A bound. The question becomes: where do I get cheap bounds on a discrete maximization?

The integer feasible set is a *subset* of the continuous polytope — it's exactly the polytope's lattice points, nothing outside. And there's a fact I trust completely: enlarging the feasible set can only raise a maximum. If `F ⊆ F'`, then `max_{F'} c'x >= max_F c'x`, because every competitor in the smaller problem is still available in the larger one, plus possibly better ones. So if I *drop* the integrality requirement — solve the LP over the whole polytope, integers be damned — I get a value that is `>=` the true integer optimum. The continuous relaxation *over-estimates* the integer answer. For a maximization, the LP value is an **upper bound** on the integer optimum. (If I were minimizing, the same enlargement *lowers* the minimum, so the LP value would be a *lower* bound. I'll keep maximization straight and flip carefully if I ever need min.)

That's the whole seed. The relaxation is cheap (simplex), convex, and gives me a *certified ceiling* on what any integer point in this region can achieve. A ceiling is exactly the proof-of-impossibility I wanted for pruning.

So step zero: solve the LP relaxation, ignore integrality. Call the optimal point `x*` and value `z_LP`. Two cases. If `x*` happens to already be integer in all the marked variables — wonderful. It's feasible for the integer problem (it satisfies every constraint *and* the integrality), and it achieves the upper bound `z_LP`, which no integer point can exceed. So it's *optimal*, immediately, no search. I think this can really happen and isn't just a degenerate edge case — totally-unimodular problems like transportation come out integral on their own, and even when the root relaxation is fractional its ceiling can already equal the integer optimum, so a single branch suffices to certify it — but I'll hold that as a claim to *test* once I have code, not assume. But in general some marked `x*_j` is fractional, say `x*_j = 2.6`. Now what?

The fractional value is the crack I pry on. The true integer optimum has *some* integer value for `x_j`. Whatever it is, it's either `<= 2` or `>= 3` — because there is no integer strictly between 2 and 3. The open strip `2 < x_j < 3` contains no integer point at all. So if I split the problem into two subproblems,

one with the extra constraint `x_j <= 2`, the other with `x_j >= 3`,

I have lost *nothing*: every integer-feasible point of the original lands in exactly one of the two children (or, if exactly integer on the boundary, in one of them), and the only thing I threw away is the open strip, which held no integer point to begin with. The two children's integer-feasible sets union back up to the original's. This is the *branch*. In general: pick a fractional marked variable `x*_j`, and form the two children `x_j <= floor(x*_j)` and `x_j >= ceil(x*_j)`.

And notice what the split bought me beyond completeness. The parent's LP optimum `x*` had `x_j = 2.6`. In the `x_j <= 2` child, `x*` is *infeasible* — it's been cut out. In the `x_j >= 3` child, also infeasible. So when I re-solve each child's relaxation, neither child can use that fractional vertex again. Because each child has a smaller feasible region than the parent, its relaxation value is at most the parent's relaxation value; it may tie if another equally good vertex survives, but it can never be higher. The branching makes genuine progress — I'm not going to loop forever re-finding the same 2.6 point.

Each child is, structurally, the same kind of object I started with: a linear program with marked-integer variables, just with one variable's bounds tightened. (And in a bounded-variable LP, "`x_j <= 2`" isn't even a new row — it's just lowering `x_j`'s upper bound. The child differs from the parent by *one bound*, which is exactly the situation the dual simplex warm-starts from cheaply.) So I recurse. Solve the child's relaxation for its own upper bound; if still fractional, branch again. A tree of LP relaxations, each node a subproblem, each edge a tightened integer bound. The root is the bare relaxation; the leaves, if I let them grow, are subproblems pinned down to integer points.

Now I have to make the tree *finite and cheap*, which is where the bound earns its keep — otherwise this is enumeration wearing a tree costume. I need to keep track of the best *integer-feasible* solution I've actually found anywhere in the tree so far. Call it the **incumbent**, with value `z_inc`. Every time a node's relaxation comes back integral, that's a real feasible integer point — a candidate. If its value beats `z_inc`, it becomes the new incumbent. The incumbent is a *lower* bound on the true optimum `z*` (I have an integer point achieving it, so `z* >= z_inc`), while any open node's relaxation value is an *upper* bound on what that node's subtree can achieve. Those two bounds, squeezing from opposite sides, are the engine.

Because here is the prune. Take any node, solve its relaxation, get upper bound `u`. If `u <= z_inc` — if the very *best* that this whole subtree could conceivably achieve is no better than an integer solution I already hold — then there is no reason to explore the subtree at all. Every integer point under this node is bounded above by `u`, and `u` doesn't beat the incumbent, so none of them can be the optimum. *Fathom* the node: discard it, unexplored, with a proof that nothing was lost. The relaxation bound did the pruning — it killed an entire subtree by solving one LP. That is the difference between this and enumeration: I never open the parts of the lattice that the bound certifies can't win.

So at each node I have three ways to stop without branching — three *fathoming* conditions. First, **infeasible**: the relaxation has no solution (the tightened bounds contradict the constraints), so the subtree is empty; drop it. Second, **bound**: the relaxation value `u <= z_inc`; the subtree can't beat the incumbent; drop it. Third, **integral**: the relaxation optimum is already integer in all marked variables; it's a leaf — update the incumbent if it's better, then stop, because a relaxation that's already integral *is* the best the subtree offers, there's nothing finer to find below. Only if none of these fire — feasible, bound strictly above incumbent, and fractional — do I branch.

Let me make sure the logic is airtight, because it's easy to fool myself on the bound direction. Maximization. Relaxation value = upper bound (enlarged feasible set raises the max). Incumbent = lower bound (an actual achieved integer value). Prune when `upper_bound_of_node <= incumbent`. Yes — for a *max*, you discard a node when its ceiling is at or below your floor. (Flipped for min: relaxation gives a *floor*, incumbent gives a *ceiling*, prune when node's floor `>=` incumbent.) I'll commit to max and feed `-c` to a minimizing LP solver, negating the value on the way back, so the bookkeeping stays in one convention.

There's a subtlety I should pin down: completeness. Does this tree provably reach the optimum, or could it skip it? At every branch the two children's integer-feasible sets union to the parent's, and the only thing excluded is an integer-free strip — so no integer point ever falls out of the tree. Every integer-feasible point lives in some leaf. I only ever fathom a node when (a) it's empty, (b) its *upper bound* — which dominates every integer point beneath it — is `<= z_inc`, meaning those points can't beat what I hold, or (c) it's already an integral leaf I've evaluated. So I never discard a node that could contain a *strictly better* integer point than my incumbent. When the search ends — every node fathomed — the incumbent has been compared, directly or by bound, against all of them. So I expect it to be optimal. (Assuming the variables are bounded so the tree is finite; integer variables in a box give a finite lattice, so depth is bounded.) That argument feels right, but it's exactly the kind of thing I can fool myself on — a sign error in the prune direction, an off-by-one in floor/ceil, an integer point that quietly slips through the cracks of a branch. I won't trust it until I've watched the tree run on an instance whose answer I can get independently. That comes after the code.

While the search is *running*, I also get a live certificate of how close I am — the **optimality gap**. The incumbent `z_inc` is the best integer value found; the largest upper bound over all *still-open* nodes, call it `z_bar`, is the best anything unexplored could possibly do. So the true optimum is trapped: `z_inc <= z* <= z_bar`. The absolute gap is `z_bar - z_inc`; a scale-free version can divide by something like `max(1, |z_inc|)` once an incumbent exists. When the absolute gap hits zero, I've *proved* optimality — every open node's ceiling has dropped to the incumbent's floor. And the gap gives me a principled early stop: if I only need a solution within a stated tolerance, I halt when the certified gap is below that tolerance, long before the tree is exhausted. That's huge in practice — the last fraction of the gap is often the most expensive to close.

Now, *which* open node do I expand next, and *which* fractional variable do I branch on? These are choices, and they trade off differently. Take node selection first. One option, **best-first**: always expand the open node with the largest upper bound — the most promising ceiling. This tends to minimize the number of nodes explored, because you never waste effort on a node once a better incumbent has dropped every competitor's ceiling below it; you go straight at whatever could still hold the optimum. The cost: you may keep a huge *frontier* of open nodes alive in memory, each with its stored bounds and basis. The other option, **depth-first**: dive — always expand a child of the node you just made, plunging to a leaf. The memory is tiny, just the bounds along one root-to-leaf path. And diving has a second virtue: it reaches integer leaves *fast*, so you get an incumbent *early*, and an early incumbent means the bound-prune starts firing sooner on everything else. The cost: without a good incumbent guiding you, you might dive into and explore subtrees a best-first search would have pruned outright. In practice you mix them — dive to get an incumbent, then let the bounds steer — but the two pure strategies are the poles. (Doing this by hand, with paper for storage, you'd naturally pursue a branch until its bound was no longer the best and then set it aside — a best-first flavour; on a computer with finite memory, depth-first's small footprint usually wins, and you sprinkle in diving.)

Branching variable: there are several fractional marked variables to choose from; which one splits the problem most usefully? A clean default is **most-fractional** — pick the `x*_j` whose fractional part is closest to `0.5`, i.e. maximize `|x*_j - round(x*_j)|`. The intuition: a variable sitting at, say, `2.5` is maximally *undecided* — the relaxation has no opinion which way it should go — so forcing it either way (`<= 2` or `>= 3`) is a big, genuine commitment that should move the child relaxations substantially and tighten both bounds hard. A variable at `2.95` is almost decided already; branching on it barely changes anything in one child and is nearly redundant. (There are sharper rules — estimate, per candidate, how much the objective would *drop* in each child and pick the variable promising the biggest drop, i.e. pseudocost/strong branching — but those need either history or extra LP probes; most-fractional is the simple, no-bookkeeping choice and it's a fine default.)

So the method has assembled itself out of three moves I can name: *bound* — solve the LP relaxation at each node for a certified ceiling; *branch* — when the relaxation is fractional in a marked variable, split on `x_j <= floor` vs `x_j >= ceil`, losing no integer points; and *prune/fathom* — kill any node that's infeasible, or whose ceiling can't beat the incumbent, or that's already an integral leaf. The incumbent rises, the open bounds fall, the gap closes, and when it shuts the incumbent is provably optimal. Divide the discrete space, conquer each piece with a cheap convex bound, and let the bound throw away everything that can't matter.

The relaxation bound is only as good as the relaxation is *tight*. If the continuous polytope bulges far past the integer hull near a fractional vertex, the ceiling `u` is loose, it rarely drops below the incumbent, and I branch a lot. Gomory's cuts already give one way to sharpen such a node: derive a valid linear inequality from the simplex tableau, one that every integer-feasible point satisfies but the current fractional `x*` violates; add it, re-solve, and the ceiling can drop with no integer point lost. But the tree search itself does not need cut generation to be correct. I can keep cuts as a bound-strengthening option and build the exact solver from relaxation, incumbent, split, and fathom.

Let me write it as a single self-contained program. The pieces map one-to-one: a routine that solves a node's relaxation (a bounded-variable LP solver — since there's no library to lean on I'll carry a small Big-M simplex inline, fed `c` to maximize), a routine that picks the most-fractional marked variable, a stack of open nodes where each node is just the per-variable bound vector, the incumbent and its value carried across the search, and the three fathoming tests. The disposition is a competition single-file C++ program that reads one MILP from stdin — `n m`, the objective row, the `m` constraint rows with their right-hand sides, then per-variable `l u integer_flag` lines — and prints the optimal value with its integer coordinates and the node count. Then I'll run it on tiny instances where exhaustive enumeration is still affordable, and *watch the tree*, so the completeness argument and the prune directions get checked rather than assumed.

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

Now I run it and look at what actually happens, because the prose proof is only as good as the code that's supposed to embody it.

The 0/1 knapsack first — `vals = [8,11,6,4,7,3]`, `wts = [5,7,4,3,5,2]`, `cap = 14`. The relaxation maximizes value packing fractional items into capacity 14. The root relaxation comes back at `x = [1,1,0.5,0,0,0]`, ub `22` — fractional in item 2, so branching fires; forcing `x2 <= 0` immediately yields the integer point `x = [1,1,0,0,0,1]`, value `22`, and the `x2 >= 1` sibling caps out at `21.857 < 22` and is fathomed by the bound. Three nodes total. Let me sanity-check that's a real optimum, not a bug masquerading as a clean answer. Items 0,1,5 weigh `5+7+2 = 14`, exactly the capacity, value `8+11+3 = 22`. A brute-force loop over all `2^6 = 64` subsets returns the same `[1,1,0,0,0,1]`, value `22`. So the relaxation's ceiling of `22` was already integral-achievable, and a single split on the one fractional item pins it down — and notice this is also the first place I see *condition (b)* actually fire on a feasible node: the `x2 >= 1` child is feasible, but its ceiling `21.857` can't beat the incumbent `22`, so the bound discards it without any further branching. (The relaxation's ceiling `22` happening to equal the integer optimum is the lucky structure here — change the capacity to 13 and the relaxation slices an item and the ceiling drops below the best integer value — but the mechanism is real.)

Now the two-variable integer LP, where I *want* branching to fire so I can watch the tree. `c = [4,-1]`, constraints `7x0 - 2x1 <= 14`, `x1 <= 3`, `2x0 - 2x1 <= 3`, both variables integer. I print every node as it's popped:

```
node 1: lo=[0,0]   up=[inf,inf]  x=[2.857, 3.0]  ub=8.4286  -> branch on x0: down x0<=2, up x0>=3
node 2: lo=[0,0]   up=[2,inf]    x=[2.0, 0.5]    ub=7.5     -> branch on x1: down x1<=0, up x1>=1
node 3: lo=[0,0]   up=[2,0]      x=[1.5, 0.0]    ub=6.0     -> branch on x0: down x0<=1, up x0>=2
node 4: lo=[0,0]   up=[1,0]      x=[1.0, 0.0]    ub=4.0     -> integral leaf, NEW incumbent 4.0
node 5: lo=[2,0]   up=[2,0]      infeasible                 -> fathom (empty)
node 6: lo=[0,1]   up=[2,inf]    x=[2.0, 1.0]    ub=7.0     -> integral leaf, NEW incumbent 7.0
node 7: lo=[3,0]   up=[inf,inf]  infeasible                 -> fathom (empty)
output: 7 2 1   (value 7 at x=[2,1])   nodes 7
```

Let me walk it and check the pieces against what I argued they'd do. Root: `x0 = 2.857` is the fractional one (`x1 = 3` is already integer), and the upper bound is `8.43` — that's my ceiling on the whole problem; no integer point can beat it. Branch on `x0`: down `x0<=2`, up `x0>=3`. The up-child is node 7, and it's *infeasible* — with `x0>=3`, constraint `7x0 - 2x1 <= 14` forces `21 - 2x1 <= 14`, so `x1 >= 3.5`, but `x1 <= 3`; contradiction, empty, fathomed. Good: that's the third fathoming condition doing exactly what it should, and it matches my by-hand check. The down-side, node 2, has bound `7.5 <= 8.43` — strictly *below* the parent's ceiling, as I claimed each child must be (smaller region, no higher max). It's still fractional (`x1 = 0.5`), so it branches again, and so on. Two incumbents appear in order: `4.0` at node 4, then `7.0` at node 6 replaces it. After node 6 sets the incumbent to 7, are there open nodes that *should* have been pruned by the bound but weren't? Node 7 is the only thing left, and it's infeasible anyway, so there was no bound-prune to exercise on this tiny tree — worth noting the example doesn't actually fire condition (b). The final answer `[2,1]`, value `7`: brute force over the box `x0 in 0..10, x1 in 0..3` returns `[2,1]`, value `7`, assert passes.

So both directions of the engine are now checked on worked instances: a 3-node knapsack tree where one split on the lone fractional item pins down the optimum and the bound discards the dominated sibling, and a 7-node integer-LP tree with fathoming-by-infeasibility and two incumbent updates — both landing on the brute-force optimum. The ceilings descend down each branch (`8.43 -> 7.5 -> 6.0 -> 4.0`; `8.43 -> 7.0`) exactly as the relaxation-monotonicity argument required, and no integer-feasible point went missing. All three fathoming conditions have now fired on real nodes: infeasibility (integer-LP nodes 5 and 7), integral leaf (the incumbent updates), and the bound-prune of a *feasible* dominated subtree (the knapsack's `x2 >= 1` child, ceiling `21.857 <= 22`). That's enough to trust the construction.

The causal chain, end to end: the integer feasible set is the lattice inside a polytope — non-convex, so simplex can't touch it directly, and exponentially large, so enumeration is hopeless; but dropping integrality gives the LP relaxation, whose optimum *over-estimates* the integer maximum and so hands me a *certified upper bound* cheaply at any subproblem. If that relaxation is already integer I'm done; otherwise some `x*_j` is fractional, and since no integer lies in `(floor, ceil)` I split into `x_j <= floor(x*_j)` and `x_j >= ceil(x*_j)`, losing no integer point while removing the current fractional vertex from both children. I carry the best integer point found — the incumbent, a *lower* bound — and fathom any node that's infeasible, or whose relaxation ceiling can't beat the incumbent, or that's already integral; the bound, not enumeration, throws away the subtrees that can't contain the optimum. The incumbent rises, the best remaining ceiling falls or gets discarded, and the gap `z_bar - z_inc` between them certifies how close I am — zero gap is a proof of optimality, a small gap is a principled early stop. Best-first expansion minimizes nodes at the cost of memory; depth-first dives for a quick incumbent on a tiny footprint; most-fractional branching commits the most-undecided variable; and a valid cut can tighten a loose relaxation before I branch, without changing the correctness of the tree search. Divide the discrete space, conquer each piece with a convex bound, and let the bound prune.
