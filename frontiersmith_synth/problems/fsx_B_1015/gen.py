import sys, random

# -----------------------------------------------------------------------------
# self-trapping-reward-tour  (format C, maximize reward collected)
#   `python3 gen.py <testId>`  prints ONE instance to stdout.  Deterministic in
#   testId only (seeded RNG).
#
# WORLD MODEL
#   A set of grid cells (r,c) connected by an explicit list of orthogonal
#   corridor edges (NOT every grid-adjacent pair is connected -- only the
#   listed edges are legal moves).  Each cell carries a reward (collected once,
#   on first visit).  Some cells are KEY cells (grant a key id) and some are
#   GATE cells (only enterable if the matching key was already collected
#   earlier in the walk).  The solver's artifact is a self-avoiding walk
#   (sequence of distinct cells, consecutive cells joined by a declared edge,
#   gates respected) starting at the given start cell.
#
# PLANTED STRUCTURE (the innovation hook)
#   From the start, a long "spine" corridor runs across the map.  At several
#   points a short DECOY spur branches off the spine; its very first cell
#   carries a big, tempting reward, but the spur is a strict dead end (self-
#   avoiding-trap): stepping in for that fat immediate reward permanently
#   forfeits the rest of the (much larger) spine.  Further along, the spine
#   forks into two parallel branches that later reconverge (a small diamond):
#   the "wrong" branch is short and slightly richer per-step; the "right"
#   branch is longer, slightly poorer per-step, but it alone carries the KEY.
#   Past the reconvergence sits a locked GATE that only the key opens; beyond
#   the gate lies a long, richly-rewarded VAULT tail.  Skipping the key
#   branch -- or detouring into any decoy -- permanently strands the vault.
#   A one-step-lookahead ("take the richest neighbour") walker always loses
#   both bets: it dead-ends in the first decoy, or -- lacking a decoy -- takes
#   the richer-looking wrong fork and starves at the locked gate.
# -----------------------------------------------------------------------------


class World:
    def __init__(self):
        self.reward = {}      # (r,c) -> int reward
        self.kind = {}        # (r,c) -> ('key', id) | ('gate', id)   (absent = normal)
        self.adj = {}         # (r,c) -> set of neighbour (r,c) actually connected

    def add(self, cell, reward, kind=None):
        self.reward[cell] = reward
        if kind is not None:
            self.kind[cell] = kind
        self.adj.setdefault(cell, set())

    def link(self, a, b):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, (a, b)
        self.adj.setdefault(a, set()).add(b)
        self.adj.setdefault(b, set()).add(a)


def build(t):
    rng = random.Random(424242 + 97 * t)
    W = World()

    # ---- scale with testId: bigger, longer instances at higher ids ----
    n_decoys = 3 + (t - 1) // 2                 # 3..7
    fa_len = 5 + (t - 1) // 3                    # wrong-fork filler length: 5..8
    vault_len = 26 + 5 * (t - 1)                 # 26..71
    gap = 4 + (t % 3)                            # filler cells between decoy junctions

    SPINE_R = 2                                  # plain spine filler cell reward
    DECOY_FIRST_R = 140 + rng.randint(0, 20)     # tempting first cell of a decoy
    DECOY_REST_R = 16
    DECOY_LEN = 3                                # cells per decoy spur (incl. first)
    JUNCTION_CONT_R = 3                          # spine "continue" cell at a decoy junction
    FORK_WRONG_FIRST_R = 6                       # forkA (no key) first-cell reward
    FORK_WRONG_REST_R = 3
    FORK_KEY_FIRST_R = 4                         # forkB (key) first-cell reward (< wrong's)
    FORK_KEY_REST_R = 3
    KEY_R = 3
    GATE_R = 0
    VAULT_R = 8 + rng.randint(0, 2)

    r, c = 0, 0
    start = (0, 0)
    W.add(start, 0)

    col = 0
    # ---- spine with decoy junctions ----
    for d in range(n_decoys):
        # filler run up to the junction
        for _ in range(gap):
            col += 1
            cell = (0, col)
            W.add(cell, SPINE_R)
            W.link((0, col - 1), cell)
        # junction cell already placed as the last filler cell (0,col); attach a
        # decoy spur DOWN from it, and a "continue" cell to its right.
        junc = (0, col)
        spur_prev = junc
        for k in range(DECOY_LEN):
            rr = k + 1
            cell = (rr, col)
            rew = DECOY_FIRST_R if k == 0 else DECOY_REST_R
            W.add(cell, rew)
            W.link(spur_prev, cell)
            spur_prev = cell
        # continue the spine
        col += 1
        cont = (0, col)
        W.add(cont, JUNCTION_CONT_R)
        W.link(junc, cont)

    # a short additional filler run before the fork
    for _ in range(gap):
        col += 1
        cell = (0, col)
        W.add(cell, SPINE_R)
        W.link((0, col - 1), cell)

    # ---- the fork (diamond): wrong branch (row 0, no key) vs key branch (row 1) ----
    jf = col
    fork_start = (0, jf)
    merge_col = jf + fa_len
    merge = (0, merge_col)

    # wrong branch: continues along row 0
    prev = fork_start
    for k in range(fa_len):
        cell = (0, jf + 1 + k)
        rew = FORK_WRONG_FIRST_R if k == 0 else FORK_WRONG_REST_R
        if cell == merge:
            # last wrong-branch cell IS merge; give merge a small filler reward
            rew = FORK_WRONG_REST_R
        W.add(cell, rew) if cell not in W.reward else None
        W.link(prev, cell)
        prev = cell

    # key branch: drop to row 1, run right (fa_len+1 cells), rise back to merge
    # (row 1, columns jf..jf+fa_len -> up to (0, jf+fa_len) == merge)
    prev = fork_start
    for k in range(fa_len + 1):
        cell = (1, jf + k)
        rew = FORK_KEY_FIRST_R if k == 0 else FORK_KEY_REST_R
        kind = ('key', 1) if k == 0 else None
        if cell not in W.reward:
            W.add(cell, rew, kind)
        W.link(prev, cell)
        prev = cell
    W.link(prev, merge)   # rise back onto the spine at the merge column

    # ---- gate + vault tail ----
    col = merge_col
    col += 1
    gate = (0, col)
    W.add(gate, GATE_R, ('gate', 1))
    W.link(merge, gate)

    prev = gate
    for k in range(vault_len):
        col += 1
        cell = (0, col)
        W.add(cell, VAULT_R + (k % 5))
        W.link(prev, cell)
        prev = cell

    return W, start


def emit(t):
    W, start = build(t)
    cells = sorted(W.reward.keys())
    idx = {cell: i for i, cell in enumerate(cells)}
    N = max(max(r, c) for (r, c) in cells) + 1
    V = len(cells)
    edges = set()
    for a, nbrs in W.adj.items():
        for b in nbrs:
            ia, ib = idx[a], idx[b]
            edges.add((min(ia, ib), max(ia, ib)))
    edges = sorted(edges)
    S = idx[start]

    out = []
    out.append("%d %d %d %d" % (N, V, len(edges), S))
    for cell in cells:
        r, c = cell
        rew = W.reward[cell]
        kind = W.kind.get(cell)
        if kind is None:
            k, kid = 0, 0
        elif kind[0] == 'key':
            k, kid = 1, kind[1]
        else:
            k, kid = 2, kind[1]
        out.append("%d %d %d %d %d" % (r, c, rew, k, kid))
    for (u, v) in edges:
        out.append("%d %d" % (u, v))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    t = int(sys.argv[1])
    t = max(1, min(10, t))
    emit(t)


if __name__ == "__main__":
    main()
