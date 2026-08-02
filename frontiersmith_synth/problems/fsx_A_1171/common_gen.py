"""
Shared instance builder for fsx_A_1171 (circuit-fault-locate).
Imported by BOTH gen.py (prints the PUBLIC instance) and verify.py (recomputes the SAME
instance, including the secret true resistances and the held-out excitation patterns, from
testId alone). Everything here is deterministic given testId -- no wall clock, no OS entropy.

Topology (fixed shape; only numeric values / filler size vary with testId):

  Terminals (indices 0..n_terminal-1):
    0 = T0  (ground / voltage reference, always 0V)
    1 = P0  (shown excitation A)
    2 = Q0  (shown excitation B)
    3 = P1  (held-out excitation A)
    4 = Q1  (held-out excitation B)
    5 = P1b (held-out excitation A, duplicate access point)

  Interior (hidden, unobservable) nodes:
    N1, N2, N3  -- a 3-node hidden chain  T0 --R1-- N1 --R2-- N2 --R3-- N3
    FR          -- root of an unrelated filler sub-network (adds size/noise, never faulted)
    filler...   -- extra filler interior nodes (count grows with testId)

  Core edges (FIXED indices 0..7, so every reference solution can rely on the order):
    0  R1   (T0 - N1)     always nominal -- the "obvious" but WRONG single-component decoy
    1  R2   (N1 - N2)     drift candidate #1
    2  R3   (N2 - N3)     drift candidate #2
    3  lead_P0  (P0 - N3) always nominal, private access resistor for P0
    4  lead_Q0  (Q0 - N1) always nominal, private access resistor for Q0
    5  lead_P1  (P1 - N3) always nominal, private access resistor for P1 (held-out)
    6  lead_Q1  (Q1 - N1) always nominal, private access resistor for Q1 (held-out)
    7  lead_P1b (P1b- N3) always nominal, private access resistor for P1b (held-out)
    8+ filler edges (random, never touch N1/N2/N3/P0/Q0/P1/Q1/P1b)

  Excitation pattern = (source terminal s, sink terminal g=0, current magnitude Q>0):
  +Q is injected at s, -Q extracted at g=0 (ground); every other node has zero net current.
  T0 is fixed at 0V by definition (the physical reference), for every pattern.

Why this shape produces the intended trap (see AGENT_BRIEF_INNOVATION_ADDENDUM):
  - A SINGLE excitation through P0 only exposes the SUM R1+R2+R3 (one equation). Attributing
    the whole discrepancy to R1 alone (edge 0, always checked first, always solvable in
    closed form since it enters the sum additively) fits that ONE pattern exactly, but R1
    never actually drifts, so this explanation is wrong.
  - A SECOND, structurally different excitation through Q0 isolates R1 ALONE (a different,
    non-redundant equation). Any explanation that requires R1 to have moved is exposed by Q0
    -- but only if the solver actually checks Q0, not just P0.
  - Held-out patterns P1/P1b (repeat the "sum" reading from fresh, unseen leads) and Q1
    (repeats the "R1 alone" reading from a fresh, unseen lead) grade whether the CLAIMED
    fault set generalizes, not just whether it curve-fits the shown data.
"""
import random
from fractions import Fraction

N_TERMINAL = 6          # T0, P0, Q0, P1, Q1, P1b
N1, N2, N3, FR = 6, 7, 8, 9
CORE_EDGES = 8           # indices 0..7 are the fixed core edges described above
SHOWN_PAIRS = [(1, 0), (2, 0)]           # P0 (sum R1+R2+R3), Q0 (R1 alone)
HELDOUT_PAIRS = [(3, 0), (4, 0), (5, 0)]  # P1 (sum), Q1 (R1 alone), P1b (sum)
FAULT_FACTORS = [Fraction(1, 4), Fraction(3, 10), Fraction(19, 50),
                  Fraction(11, 5), Fraction(27, 10), Fraction(16, 5), Fraction(37, 10)]


def build_instance(test_id: int):
    """Deterministic full instance (public + secret) for this testId."""
    rng = random.Random(90210 + 104729 * test_id)

    n_filler_interior = 1 + (test_id % 4)          # 1..4 -> difficulty ladder
    n = 10 + n_filler_interior
    filler = list(range(10, n))

    edges = []  # each entry [u, v, R_nominal(int)]

    def add_edge(u, v, R):
        edges.append([u, v, R])
        return len(edges) - 1

    def randR(lo=15, hi=80):
        return rng.randint(lo, hi)

    i_R1 = add_edge(0, N1, randR())
    i_R2 = add_edge(N1, N2, randR())
    i_R3 = add_edge(N2, N3, randR())
    i_lp0 = add_edge(1, N3, randR())
    i_lq0 = add_edge(2, N1, randR())
    i_lp1 = add_edge(3, N3, randR())
    i_lq1 = add_edge(4, N1, randR())
    i_lp1b = add_edge(5, N3, randR())
    assert (i_R1, i_R2, i_R3, i_lp0, i_lq0, i_lp1, i_lq1, i_lp1b) == tuple(range(CORE_EDGES))

    # filler sub-network: never touches N1, N2, N3 or any P*/Q* terminal.
    connected = {0, FR}
    pool = list(filler)
    rng.shuffle(pool)
    for node in pool:
        k_links = 1 if rng.random() < 0.5 else 2
        cands = list(connected)
        rng.shuffle(cands)
        for c in cands[:k_links]:
            add_edge(node, c, randR())
        connected.add(node)
    extra = 1 + (test_id % 3)
    safe_nodes = [0, FR] + filler
    for _ in range(extra):
        u, v = rng.choice(safe_nodes), rng.choice(safe_nodes)
        if u != v:
            add_edge(u, v, randR())
    if not any(e[0] == FR or e[1] == FR for e in edges):
        add_edge(FR, 0, randR())

    n_nodes = n

    # ---- plant the fault(s): R2 and/or R3 drift; R1 NEVER drifts ----
    two_faults = (test_id % 2 == 0)
    true_R = [e[2] for e in edges]
    faults = {}
    f1 = rng.choice(FAULT_FACTORS)
    v1 = max(3, round(true_R[i_R2] * f1))
    faults[i_R2] = v1
    true_R[i_R2] = v1
    if two_faults:
        f2 = rng.choice(FAULT_FACTORS)
        v2 = max(3, round(true_R[i_R3] * f2))
        faults[i_R3] = v2
        true_R[i_R3] = v2

    Q_shown = [rng.randint(9, 17) for _ in SHOWN_PAIRS]
    Q_held = [rng.randint(9, 17) for _ in HELDOUT_PAIRS]

    return dict(n=n_nodes, n_terminal=N_TERMINAL, n_edges=len(edges), edges=edges,
                true_R=true_R, shown_pairs=SHOWN_PAIRS, Q_shown=Q_shown,
                heldout_pairs=HELDOUT_PAIRS, Q_held=Q_held, faults=faults)


# ---------------------------- nodal-analysis solvers ----------------------------
def solve_nodal_fraction(n, edges, R_list, s, g, Q):
    """Exact rational nodal solve (node 0 = ground, V[0]=0). Used for the TRUE circuit
    so gen.py's published measurements and verify.py's ground truth are bit-exact."""
    G = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for (u, v, _), R in zip(edges, R_list):
        cond = Fraction(1, R)
        G[u][u] += cond; G[v][v] += cond
        G[u][v] -= cond; G[v][u] -= cond
    I = [Fraction(0)] * n
    if s != g:
        I[s] += Q
        I[g] -= Q
    idxs = [i for i in range(n) if i != 0]
    m = len(idxs)
    A = [[G[idxs[i]][idxs[j]] for j in range(m)] for i in range(m)]
    b = [I[idxs[i]] for i in range(m)]
    for col in range(m):
        piv = None
        for r in range(col, m):
            if A[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        pv = A[col][col]
        for r in range(m):
            if r != col and A[r][col] != 0:
                factor = A[r][col] / pv
                for c2 in range(col, m):
                    A[r][c2] -= factor * A[col][c2]
                b[r] -= factor * b[col]
    x = [Fraction(0)] * m
    for i in range(m):
        if A[i][i] != 0:
            x[i] = b[i] / A[i][i]
    V = [Fraction(0)] * n
    for i, idx in enumerate(idxs):
        V[idx] = x[i]
    return V


def solve_nodal_float(n, edges, R_list, s, g, Q):
    """Float nodal solve (node 0 = ground). Used by verify.py to forward-simulate a
    participant's CLAIMED circuit (arbitrary real-valued claimed resistances)."""
    G = [[0.0] * n for _ in range(n)]
    for (u, v, _), R in zip(edges, R_list):
        cond = 1.0 / R
        G[u][u] += cond; G[v][v] += cond
        G[u][v] -= cond; G[v][u] -= cond
    I = [0.0] * n
    if s != g:
        I[s] += Q
        I[g] -= Q
    idxs = [i for i in range(n) if i != 0]
    m = len(idxs)
    A = [[G[idxs[i]][idxs[j]] for j in range(m)] for i in range(m)]
    b = [I[idxs[i]] for i in range(m)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-12:
            continue
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        pv = A[col][col]
        for r in range(m):
            if r != col and A[r][col] != 0:
                factor = A[r][col] / pv
                for c2 in range(col, m):
                    A[r][c2] -= factor * A[col][c2]
                b[r] -= factor * b[col]
    x = [0.0] * m
    for i in range(m):
        if abs(A[i][i]) > 1e-12:
            x[i] = b[i] / A[i][i]
    V = [0.0] * n
    for i, idx in enumerate(idxs):
        V[idx] = x[i]
    return V


def make_readings(inst):
    """(s, g, Q, V[s] as float, V[g] as float) for a list of (pair, Q) using the TRUE circuit."""
    def _mk(pairs, Qs):
        out = []
        for (s, g), Q in zip(pairs, Qs):
            V = solve_nodal_fraction(inst['n'], inst['edges'], inst['true_R'], s, g, Q)
            out.append((s, g, Q, float(V[s]), float(V[g])))
        return out
    shown = _mk(inst['shown_pairs'], inst['Q_shown'])
    held = _mk(inst['heldout_pairs'], inst['Q_held'])
    return shown, held
