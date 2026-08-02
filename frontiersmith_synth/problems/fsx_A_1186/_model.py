"""Shared deterministic instance model for fsx_A_1186 (matrix-completion-structured).
Imported ONLY by gen.py and verify.py (both trusted, run outside the solution sandbox).
Never shipped to / readable by a submitted solution.

Ground-truth model (a "clustered additive" matrix -- low rank in the algebraic sense,
built FORWARD from row/column group membership, not recovered):
    true[i][j] = base[rowgroup[i]][colgroup[j]] + row_bias[i] + col_bias[j] + noise[i][j]

Two observation-pattern regimes (mechanism: observation-pattern-bias):
  - 'random'  : each cell independently observed w.p. p_obs.
  - 'block'   : background random observation PLUS one or more (row-group, col-group)
                pairs where a *subset* of that row-group's rows crossed with a *subset*
                of that col-group's columns is entirely unobserved (a structural, not
                random, missing rectangle). Some rows/cols of the same two groups stay
                visible outside the rectangle, so the (group,group) pair is still
                represented elsewhere in the data -- side information (mechanism:
                side-information-graph) can transfer it across the empty block; a
                method that only trusts the observed-entry pattern (nuclear-norm /
                low-rank completion) has no reason to treat that rectangle differently
                from ordinary missing-at-random cells and is systematically fooled.
"""
import random

def params(test_id):
    table = {
        1:  dict(n=12, m=12, K=3, L=3, mode='random', p_obs=0.62, n_blocks=0),
        2:  dict(n=16, m=16, K=3, L=3, mode='random', p_obs=0.58, n_blocks=0),
        3:  dict(n=20, m=20, K=4, L=4, mode='random', p_obs=0.55, n_blocks=0),
        4:  dict(n=18, m=18, K=3, L=3, mode='block',  p_obs=0.62, n_blocks=1),
        5:  dict(n=20, m=20, K=4, L=4, mode='block',  p_obs=0.60, n_blocks=1),
        6:  dict(n=22, m=22, K=4, L=4, mode='block',  p_obs=0.58, n_blocks=1),
        7:  dict(n=24, m=24, K=4, L=4, mode='block',  p_obs=0.58, n_blocks=2),
        8:  dict(n=26, m=26, K=5, L=5, mode='block',  p_obs=0.55, n_blocks=2),
        9:  dict(n=28, m=28, K=5, L=5, mode='block',  p_obs=0.55, n_blocks=2),
        10: dict(n=30, m=30, K=5, L=5, mode='block',  p_obs=0.52, n_blocks=3),
    }
    return table[test_id]


def build(test_id):
    p = params(test_id)
    n, m, K, L = p['n'], p['m'], p['K'], p['L']
    seed = 20260 + 97 * test_id
    rng = random.Random(seed)

    row_group = [i % K for i in range(n)]
    rng.shuffle(row_group)
    col_group = [j % L for j in range(m)]
    rng.shuffle(col_group)

    base = [[rng.randint(20, 80) for _ in range(L)] for _ in range(K)]
    row_bias = [round(rng.uniform(-5.0, 5.0), 2) for _ in range(n)]
    col_bias = [round(rng.uniform(-5.0, 5.0), 2) for _ in range(m)]
    noise = [[round(rng.uniform(-3.5, 3.5), 2) for _ in range(m)] for _ in range(n)]

    true = [[base[row_group[i]][col_group[j]] + row_bias[i] + col_bias[j] + noise[i][j]
             for j in range(m)] for i in range(n)]

    mask = [[rng.random() < p['p_obs'] for _ in range(m)] for _ in range(n)]

    if p['mode'] == 'block':
        rows_by_group = [[i for i in range(n) if row_group[i] == g] for g in range(K)]
        cols_by_group = [[j for j in range(m) if col_group[j] == g] for g in range(L)]
        # deterministic choice of distinct (rowgroup,colgroup) pairs with enough members
        candidates = [(A, B) for A in range(K) for B in range(L)
                      if len(rows_by_group[A]) >= 4 and len(cols_by_group[B]) >= 4]
        rng.shuffle(candidates)
        chosen = candidates[:p['n_blocks']]
        for (A, B) in chosen:
            rA, cB = rows_by_group[A][:], cols_by_group[B][:]
            rng.shuffle(rA)
            rng.shuffle(cB)
            keep_r = max(2, len(rA) - max(2, int(0.6 * len(rA))))
            keep_c = max(2, len(cB) - max(2, int(0.6 * len(cB))))
            missing_rows = set(rA[:len(rA) - keep_r])
            missing_cols = set(cB[:len(cB) - keep_c])
            for i in missing_rows:
                for j in missing_cols:
                    mask[i][j] = False

    # guarantee every row / column keeps >=1 observed cell
    for i in range(n):
        if not any(mask[i]):
            mask[i][rng.randrange(m)] = True
    for j in range(m):
        if not any(mask[i][j] for i in range(n)):
            mask[rng.randrange(n)][j] = True

    observed = [(i, j, true[i][j]) for i in range(n) for j in range(m) if mask[i][j]]
    query = [(i, j) for i in range(n) for j in range(m) if not mask[i][j]]

    def spanning_edges(members):
        members = sorted(members)
        return [(members[k], members[k + 1]) for k in range(len(members) - 1)]

    row_edges = []
    for g in range(K):
        row_edges.extend(spanning_edges([i for i in range(n) if row_group[i] == g]))
    col_edges = []
    for g in range(L):
        col_edges.extend(spanning_edges([j for j in range(m) if col_group[j] == g]))

    return dict(n=n, m=m, K=K, L=L, seed=seed, true=true, mask=mask,
                observed=observed, query=query, row_edges=row_edges, col_edges=col_edges,
                row_group=row_group, col_group=col_group)
