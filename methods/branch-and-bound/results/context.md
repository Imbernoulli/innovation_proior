# Context: solving an optimization where the variables must come out whole

## Research question

Linear programming is settled. Given a profit vector `c`, a constraint matrix `A`, a right-hand
side `b`, the problem

```
maximize  c'x   subject to   Ax <= b,   x >= 0
```

is solved to global optimality by the simplex method: the feasible region is a convex polyhedron,
the optimum sits at one of its vertices, and simplex walks vertex to vertex by pivots until no
improving move remains. Mature, automatic, fast in practice.

But a great many real problems carry an extra demand: **some or all of the variables must take
whole-number values.** One schedules whole flights, runs whole machine setups, dispatches whole
tankers; and binary variables encode yes/no decisions ("build this depot or not", "assign job `i`
to machine `j` or not"), taking only the values 0 or 1. The continuous theory honours none of this:
its optimum is in general fractional.

So the question is: **how does one solve a linear program in which some variables are required to be
integers, automatically, by a procedure a computer can run?** The integer feasible set is a *subset*
of the continuous polytope — exactly the lattice points inside it. At the time, what existed were
procedures for *verifying* a conjectured integer solution and a cutting-plane idea for pure integer
programs.

## Background

The field rests on a small number of load-bearing facts.

- **Linear programming and the simplex method (Dantzig, 1947; published 1951).** A linear program
  `max c'x s.t. Ax <= b, x >= 0` has its feasible set a convex polyhedron; if it has an optimum, at
  least one optimal point is a vertex (extreme point) of that polyhedron. Simplex moves from vertex
  to vertex — algebraically, swapping one column of `A` in and one out of a *basis* by a *pivot* —
  always improving the objective, halting when no pivot improves. Dantzig's stated motivation was to
  *automatize* optimization so a computer could run it. By the mid-1950s implementations solved
  hundreds of constraints and a thousand variables.

- **LP duality gives certified bounds.** Every linear program has a dual, and weak duality says any
  dual-feasible solution's value bounds the primal optimum. So the simplex tableau does not just
  *find* the optimum; it *certifies* a bound on it. A bound is a proof of impossibility ("no
  feasible point can beat this value").

- **Relaxation: enlarging the feasible set can only raise a maximum.** A standard convex-analysis
  fact. If a problem's feasible set `F` is replaced by a superset `F' ⊇ F`, then for a maximization
  `max_{x in F'} c'x >= max_{x in F} c'x`. Dropping a constraint relaxes the problem and its optimum
  *over-estimates* the harder problem's optimum (under-estimates, for a minimization). The integer
  feasible set is a *subset* of the continuous polytope — exactly the lattice points inside it — so
  the continuous optimum over-estimates the integer optimum. The continuous LP is a relaxation of
  the integer problem, and its value is therefore a bound.

- **The integer feasible set is not convex.** The lattice points inside a polytope do not form a
  convex set: the midpoint of two integer points is generally not integer. So no single convex
  program has exactly that feasible set, and simplex — which assumes one convex region and walks
  to its vertex optimum — operates over the continuous polytope rather than the lattice.

- **The convex hull of the integer points is itself a polytope.** The integer points inside a
  polytope have a convex hull, itself a polytope, whose *vertices are exactly integer points*. With
  that hull's inequality description, integer optimization would be an LP over it. Writing the hull
  down explicitly takes, in general, exponentially many inequalities.

- **Cutting planes (Gomory, 1958).** One way to push the continuous polytope toward that hull *as
  needed*: when the LP optimum is fractional, derive from the simplex tableau a valid linear
  inequality — a *cut* — that is satisfied by every integer feasible point but *violated by the
  current fractional optimum*. Add it, re-solve, repeat. Each cut shaves a sliver containing the
  fractional vertex without removing any integer point. Gomory proved this converges for pure integer
  programs. For some structured problems, like the symmetric travelling-salesman problem, cuts are
  very effective.

- **A subset of a discrete problem is itself a discrete problem.** If a discrete feasible set is
  split into pieces, the optimum over the whole is the best of the optima over the pieces —
  `z = max_k z_k` for `S = S_1 ∪ ... ∪ S_K`. And a *bound* on each piece bounds the whole: the
  largest upper bound over the pieces is an upper bound on `z`.

The two objects in play are a relaxation that is cheap and convex but answers a fractional optimum,
and a feasible set that is exactly the lattice but non-convex and exponentially large to enumerate.

## Baselines

- **Round the LP solution.** Solve the continuous relaxation, then round each fractional variable to
  a nearby integer. Cheap, and applicable when the variables are large enough that the rounding
  error is small relative to their magnitude.

- **Total enumeration.** List every integer point in the box and keep the best feasible one. Exact:
  it returns the true integer optimum. The list has `2^n` points for `n` binary variables, more for
  general integers.

- **Cutting planes (Gomory, 1958).** Iteratively add tableau-derived valid inequalities until the LP
  optimum becomes integer. Exact and convergent for pure integer programs, and strong on special
  structures.

- **Ad hoc / problem-specific routines.** For particular combinatorial problems (assignment,
  transportation, specific schedules) there were tailored procedures, some yielding integer solutions
  directly (e.g. transportation problems whose LP optima are already integral by total
  unimodularity). Each addresses its own problem class.

## Evaluation settings

The natural inputs and yardsticks that exist before any discrete-solving routine:

- **Inputs.** A linear program `max c'x s.t. Ax <= b, x >= 0` together with a marked subset of
  variables required to be integer (all of them, for a pure integer program; some, for a mixed one),
  and per-variable lower/upper bounds `[l_j, u_j]` (binary variables are the special case
  `[0, 1]`). Canonical small instances: the 0/1 knapsack (`max value'x s.t. weight'x <= capacity`,
  `x in {0,1}`), the assignment problem, and small two- or three-variable integer LPs whose lattice
  can be drawn.
- **Feasible object.** The lattice points of the polytope: `{x : Ax <= b, l <= x <= u, x_j integer
  for marked j}` — a non-convex finite (or, with general integers, large) set.
- **Reportable quantities.** The objective value `c'x` at the returned point; whether it is provably
  optimal; the *optimality gap* between the best integer value found and the best outstanding bound;
  the number of subproblems (relaxations) solved as a measure of work; and, for verification, an
  exhaustive brute-force optimum on instances small enough to enumerate.
- **Illustrative scale.** Tiny instances (a handful of binary variables, a two-variable integer LP)
  where the lattice can be inspected directly and the answer cross-checked by full enumeration, and
  larger sizes where enumeration is impractical.

## Code framework

The deliverable is a single self-contained C++17 program. It reads one mixed-integer linear program
from `stdin` and writes the result to `stdout`.

Input is `n m`, then the `n` objective coefficients to maximize, then `m` constraint rows of
`n` coefficients followed by the right-hand side, then `n` bound rows giving lower bound, upper
bound, and a `0`/`1` flag for whether the variable must be integer. Bounds are read as strings so
`inf` and `-inf` can be accepted.

The program prints `INFEASIBLE` if no integer-feasible point exists and `UNBOUNDED` if the instance
has no finite optimum. Otherwise it prints the optimal objective value followed by the `n`
coordinates on the first line, and the number of explored nodes on the second line.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<double> c(n);
    for (int j = 0; j < n; ++j) cin >> c[j];

    vector<vector<double>> A(m, vector<double>(n));
    vector<double> b(m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) cin >> A[i][j];
        cin >> b[i];
    }

    const double INF = 1e30;
    auto parse_bound = [&](const string& s) -> double {
        if (s == "inf") return INF;
        if (s == "-inf") return -INF;
        return stod(s);
    };

    vector<double> lower(n), upper(n);
    vector<int> integer_flag(n);
    for (int j = 0; j < n; ++j) {
        string lo, hi;
        cin >> lo >> hi >> integer_flag[j];
        lower[j] = parse_bound(lo);
        upper[j] = parse_bound(hi);
    }

    bool feasible = false;
    bool unbounded = false;
    long long optimal_value = 0;
    vector<long long> coordinates(n, 0);
    long long explored_nodes = 0;

    // TODO: compute the required output.

    if (unbounded) {
        cout << "UNBOUNDED\n";
        return 0;
    }
    if (!feasible) {
        cout << "INFEASIBLE\n";
        return 0;
    }

    cout << optimal_value;
    for (long long x : coordinates) cout << ' ' << x;
    cout << '\n' << explored_nodes << '\n';
    return 0;
}
```
