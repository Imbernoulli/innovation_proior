# Context: Faster Approximate Maximum s-t Flow in Undirected Graphs

## Research question

Given an undirected graph `G = (V, E)` with `n` vertices, `m` edges, a source `s`, a
sink `t`, and a positive integer capacity `u_e` on each edge, the maximum `s-t` flow
problem asks for an assignment `f : E → ℝ` obeying conservation at every vertex other
than `s, t` and respecting `|f(e)| ≤ u_e`, of maximum value `|f|` (the net flow out of
`s`). Its dual, the minimum `s-t` cut problem, asks for the partition `(S, V∖S)` with
`s ∈ S, t ∉ S` of minimum cut capacity; by the Max-Flow Min-Cut theorem the two optima
are equal.

The figure of merit is asymptotic running time. For undirected, unit-capacity graphs with
`m = O(n)` edges, the fastest known exact / `(1−ε)`-approximate algorithm runs in
`O(n^{3/2}) = O(m^{3/2})` time. For general capacitated undirected graphs the best running
time for a `(1−ε)`-approximate flow is `Õ(m√n · ε^{-1})`, and for a `(1+ε)`-approximate cut
value it is `Õ(m + n^{3/2} ε^{-3})`. The question is how fast one can compute a `(1−ε)`-
approximate `s-t` flow and a `(1+ε)`-approximate `s-t` cut on undirected capacitated graphs.
A faster routine would speed up every algorithm that calls approximate `s-t` flow as a
subroutine (sparsest cut, for instance).

The deliverable is a single self-contained C++17 program that reads the graph instance from
stdin and writes the resulting certificate or flow data to stdout.

(`Õ(g)` hides `polylog(g)` factors; throughout, `ε` is treated as a small constant and the
results are interesting whenever they beat the `O(m^{3/2})` of exact methods.)

## Background

**Flows as optimization.** A flow is a vector `f ∈ ℝ^m`. Fix an arbitrary orientation of
each edge and let `B` be the `n × m` edge-vertex incidence matrix (`B_{v,e} = +1` if `e`
points into `v`, `−1` if out, `0` otherwise). Conservation with one unit routed from `s`
to `t` is the linear constraint `Bᵀf = χ_{s,t}`, where `χ_{s,t}` has `+1` at `s`, `−1` at
`t`, `0` elsewhere (for value `F`, the right side is `F·χ_{s,t}`). Define the congestion of
an edge as `cong(f, e) = |f(e)| / u_e`; a flow is feasible iff `cong(f, e) ≤ 1` for all
`e`. The maximum-flow value problem is therefore: route as much flow as possible while
keeping `max_e cong(f, e) ≤ 1`. This is an `ℓ_∞`-type constraint on the congestion vector,
and the whole problem is a linear program — solvable by simplex/ellipsoid/interior point.

**Combinatorial flow algorithms.** The classical line — augmenting paths, blocking flows
(Dinic, Even-Tarjan), push-relabel — augments flow along `s-t` paths or along blocking flows
in a layered graph. Goldberg and Rao (J. ACM 1998), "Beyond the Flow Decomposition Barrier,"
gave the fastest such algorithm: exact max flow in
`O(m·min(n^{2/3}, m^{1/2})·log(n²/m)·log U)` for integer capacities in `[1, U]`, by assigning
each arc a *binary* length based on residual capacity and computing blocking flows on the
resulting acyclic graph. They name the `Ω(mn)` *flow-decomposition barrier*: any algorithm
that produces an explicit path/flow decomposition and augments one arc at a time faces a
worst-case total path length of `Θ(mn)`. Removing `log U` (via `ε`-scaling) and applying
Karger's smoothing gives an `Õ(m√n ε^{-1})`-time `(1−ε)`-approximate flow; applying their
algorithm to a sparsifier gives an `Õ(m + n^{3/2} ε^{-3})`-time `(1+ε)`-approximate cut.

**Electrical networks: the `ℓ_2` flow that linear algebra solves.** View each edge as a
resistor of resistance `r_e > 0`, collected into `r ∈ ℝ^m`; the conductance is `c_e = 1/r_e`.
The energy of a flow is `E_r(f) = Σ_e r_e f(e)²`. Among all `s-t` flows of a fixed value `F`,
the unique energy minimizer is the *electrical flow* — physically, the current that flows
when one attaches a current source across `s, t`. It need not respect capacities. Unlike the
`ℓ_∞` max-flow problem, the electrical flow is computed by *solving a linear system*. With
`C = diag(c_e)`, the (weighted) graph Laplacian is `L = B C Bᵀ`, an `n × n` symmetric,
diagonally dominant matrix. An electrical flow is a *potential flow*: there is a voltage
vector `φ` with `f(u,v) = (φ_v − φ_u)/r_{uv}`, i.e. `f = C Bᵀφ`. Substituting into
`B f = χ_{s,t}` gives `L φ = χ_{s,t}`, so `φ = L⁺ χ_{s,t}` (Moore-Penrose pseudoinverse) and
`f = C Bᵀ L⁺ χ_{s,t}`. The energy of the unit electrical flow has the closed form
`E_r(f) = χᵀ L⁺ χ = φᵀ L φ`.

**Effective resistance and its monotonicity.** The effective `s-t` resistance is
`R_eff(r) = φ(s) − φ(t)` for the unit electrical flow, equivalently `R_eff(r) = χᵀ L⁺ χ =`
the energy of the unit electrical flow; the effective conductance is `C_eff = 1/R_eff`.
Thomson's / Dirichlet's principle gives the dual variational form
`C_eff(r) = min_{φ: φ_s=1, φ_t=0} Σ_{(u,v)} (φ_u − φ_v)² / r_{uv}` (Bollobás, *Modern Graph
Theory*, IX.2 Cor. 5), with the minimizer being the electrical potentials. An immediate
consequence is **Rayleigh monotonicity**: raising any resistance can only raise the effective
resistance. The energy interpretation `R_eff = energy of the unit flow` and `R f² = v²/R =`
`w v²` (Spielman, *Spectral Graph Theory*, Lecture 14) are standard facts about resistor
networks.

**Nearly-linear-time Laplacian / SDD solvers.** Spielman and Teng (2006) gave the first
nearly-linear-time solver for symmetric diagonally dominant (SDD) systems; Koutis, Miller,
and Peng (2010), "Approaching optimality for solving SDD systems," gave a simpler solver:
for an SDD matrix `A` with `m` nonzeros and any `ε > 0`, in time `O(m log²n log log n ·`
`log(1/ε))` it returns `x̃` with `‖x̃ − A⁺b‖_A ≤ ε‖A⁺b‖_A`, where `‖y‖_A = √(yᵀ A y)`. Because
`L` is SDD, an electrical flow can thus be *approximately* computed in `Õ(m)` time — error
measured in the Laplacian (energy) norm, not per-coordinate. Daitch and Spielman (2008)
showed such solvers could be the inner step of interior-point methods for flow, reaching
`Õ(m^{3/2} log U)`.

**Multiplicative weights.** The multiplicative-weights update method (Plotkin-Shmoys-Tardos
1995 for fractional packing/covering; Arora-Hazan-Kale survey) maintains a weight `w_e ≥ 1`
per constraint, repeatedly calls an oracle that satisfies a *single weighted-average*
constraint, and multiplicatively increases the weight of violated constraints. Convergence is
controlled by an exponential potential `Σ_e w_e` and by the oracle's *width* `ρ` — the worst
single-constraint violation the oracle can produce — with the number of iterations growing
*linearly* in `ρ`. A recurring sub-theme in this literature is *width reduction*: shrinking
`ρ` (e.g. Garg-Könemann viewed as width reduction for Plotkin-Shmoys-Tardos) to cut the
iteration count.

**Smoothing and sparsification.** Karger (1998) "graph smoothing" lets random sampling speed
up an exact or approximate flow algorithm: a `(1−ε)`-flow routine with running time
`T(m, n, ε)` yields a `(1−ε)`-flow in time `Õ(ε² m/n · T(Õ(n ε^{-2}), n, Ω(ε)))`. Benczúr and
Karger (1996) build, in `Õ(m)` time, a cut sparsifier `G'` with `O(n log n / ε²)` edges whose
every cut is within `(1 ± ε)` of its value in `G`. Running a cut algorithm on `G'` transfers
back to `G`.

## Baselines

- **Even-Tarjan blocking flow (1975).** Layered-graph blocking flows; `O(m√n)` for
  unit-capacity (`O(n^{3/2})` when `m = O(n)`). It is path/layer-based: each phase pushes a
  blocking flow, and the number of phases is `O(√n)`.

- **Goldberg-Rao (1998).** Binary length function on arcs (lengths `0` or `1` by residual
  capacity) plus blocking-flow computations on the resulting DAG; exact flow in
  `O(m·min(n^{2/3}, m^{1/2})·log(n²/m)·log U)`, `(1−ε)`-approx flow `Õ(m√n ε^{-1})`, `(1+ε)`-
  approx cut `Õ(m + n^{3/2} ε^{-3})`. It is a path-augmentation method operating on flow
  decompositions one arc at a time.

- **Daitch-Spielman interior point + Laplacian solves (2008).** An interior-point method for
  max flow / min-cost flow whose Newton step is a Laplacian solve done in `Õ(m)` time, giving
  `Õ(m^{3/2} log U)` (≈ `√m` interior-point iterations, each a Laplacian solve).

- **Plotkin-Shmoys-Tardos / multiplicative weights for flows.** A general framework that
  reduces a feasibility/packing flow problem to repeated calls of a single-constraint oracle.
  The iteration count is proportional to the oracle's *width* `ρ`.

## Evaluation settings

The natural yardstick is asymptotic running time as a function of `n`, `m`, and `1/ε`, for
`(1−ε)`-approximate `s-t` flow and `(1+ε)`-approximate `s-t` cut on undirected capacitated
graphs, with the reference points being `O(n^{3/2})` (unit capacity, `m = O(n)`),
`Õ(m√n ε^{-1})` (approx flow), and `Õ(m + n^{3/2} ε^{-3})` (approx cut). One first reduces to
the case where the ratio of largest to smallest capacity is polynomially bounded (compute the
max-bottleneck `s-t` path in `O(m + n log n)` time to bracket `F*` within a factor `m`, then
drop tiny-capacity edges), so capacities may be taken to be integers in `[1, poly(m/ε)]`. A
crude bound `B ≤ F* ≤ mB` from the bottleneck path lets a value be pinned down by binary
search with only an `O(log)` overhead. Correctness is measured against the exact optimum via
the Max-Flow Min-Cut theorem, which lets any cut's capacity upper-bound any feasible flow's
value.

## Input-output contract

The program is a single self-contained C++17 source file. It reads from stdin:

```text
n m s t F eps
a_0 b_0 u_0
a_1 b_1 u_1
...
a_{m-1} b_{m-1} u_{m-1}
```

Vertices are 0-indexed. Each edge line gives an undirected edge `a b` with positive capacity
`u`. The program writes to stdout either `FAIL` on one line, or:

```text
value <V>
<f_0>
<f_1>
...
<f_{m-1}>
maxcong <c>
```

where the `m` middle lines are the per-edge flow values in input order and `maxcong` is the
maximum absolute congestion over all edges.

## Code framework

The scaffold below is intentionally pre-method: it fixes only the C++17 stdin/stdout contract
and leaves the implementation body blank.

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Edge {
    int a, b;
    double u;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, s, t;
    long long m_ll;
    double F, eps;
    if (!(cin >> n >> m_ll >> s >> t >> F >> eps)) return 0;
    int m = static_cast<int>(m_ll);

    vector<Edge> edges(m);
    for (int e = 0; e < m; ++e) {
        cin >> edges[e].a >> edges[e].b >> edges[e].u;
    }

    bool failed = false;
    double value = 0.0;
    vector<double> flow(m, 0.0);
    double maxcong = 0.0;

    // TODO: implement the computation here.

    cout.setf(ios::fixed);
    cout << setprecision(6);

    if (failed) {
        cout << "FAIL\n";
        return 0;
    }

    cout << "value " << value << "\n";
    for (int e = 0; e < m; ++e) {
        cout << flow[e] << "\n";
    }
    cout << "maxcong " << maxcong << "\n";
    return 0;
}
```
