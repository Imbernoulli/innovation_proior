# TIER: strong
# Insight: throughput is capped by how much rock interface the carved network exposes,
# not by how wide any single passage is -- a solid block wastes most of its budget on
# interior cells that never touch rock at all, AND a single-file trunk that tries to
# carry too much collected flow through one 1-wide pipe starts to choke on its own
# transport resistance. The right shape is a branching COLLECTOR: a 1-wide trunk down
# to the cistern with several thin tributary STUBS poking sideways off it, SPACED OUT
# along its length (so the branch-merge cost lands on isolated junction cells, never a
# whole corridor). Rather than committing to one guess, this solution locally simulates
# a handful of candidate shapes (a plain trunk, several solid-block widths, and several
# comb spacings/stub-lengths) with its own copy of the checker's physics and keeps
# whichever candidate its own simulation ranks highest -- a small internal search over
# the qualitatively different strategies, not just "greedy plus more iterations".
import sys
from collections import deque


def neighbors4(r, c, R, C):
    if r > 0: yield r - 1, c
    if r < R - 1: yield r + 1, c
    if c > 0: yield r, c - 1
    if c < C - 1: yield r, c + 1


def sim_flow(R, C, r_out, c_out, P_IN, P_OUT, C_TUNNEL, iters, perm, carved):
    """Local (cheaper) copy of the checker's physics, used only to RANK candidates."""
    dist = {(r_out, c_out): 0}
    dq = deque([(r_out, c_out)])
    while dq:
        cell = dq.popleft()
        r, c = cell
        for nr, nc in neighbors4(r, c, R, C):
            if (nr, nc) in carved and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[cell] + 1
                dq.append((nr, nc))
    upcount = {}
    for (r, c) in carved:
        d = dist.get((r, c))
        if d is None:
            continue
        cnt = 0
        for nr, nc in neighbors4(r, c, R, C):
            if (nr, nc) in carved and dist.get((nr, nc)) == d + 1:
                cnt += 1
        upcount[(r, c)] = cnt

    def cond(u, v):
        cu, cv = u in carved, v in carved
        if cu and cv:
            du, dv = dist.get(u), dist.get(v)
            if du is not None and dv is not None and abs(du - dv) == 1:
                vdown = u if du < dv else v
                uc = upcount.get(vdown, 1)
                if uc >= 2:
                    return C_TUNNEL // uc
            return C_TUNNEL
        pu = C_TUNNEL if cu else perm[u[0]][u[1]]
        pv = C_TUNNEL if cv else perm[v[0]][v[1]]
        return pu if pu < pv else pv

    nbcond = [[None] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            lst = []
            for nr, nc in neighbors4(r, c, R, C):
                lst.append((nr, nc, cond((r, c), (nr, nc))))
            nbcond[r][c] = lst

    P = [[0] * C for _ in range(R)]
    for c in range(C):
        P[0][c] = P_IN
    P[r_out][c_out] = P_OUT
    fixed = [[False] * C for _ in range(R)]
    for c in range(C):
        fixed[0][c] = True
    fixed[r_out][c_out] = True

    for _ in range(iters):
        newP = [row[:] for row in P]
        for r in range(1, R):
            frow = fixed[r]
            prow = newP[r]
            for c in range(C):
                if frow[c]:
                    continue
                sc = 0
                sf = 0
                for nr, nc, cd in nbcond[r][c]:
                    sc += cd
                    sf += cd * P[nr][nc]
                if sc > 0:
                    prow[c] = sf // sc
        P = newP

    flow = 0
    for nr, nc, cd in nbcond[r_out][c_out]:
        flow += cd * (P[nr][nc] - P[r_out][c_out])
    return flow if flow > 0 else 0


def build_trunk(R, C, B, r_out, c_out):
    rows = list(range(1, r_out))
    if len(rows) > B:
        rows = rows[-B:]
    return set((r, c_out) for r in rows)


def build_block(R, C, B, r_out, c_out, w):
    depth = max(1, r_out - 1)
    w = max(1, min(C, w))
    lo_c0 = max(0, c_out - w + 1)
    hi_c0 = min(C - w, c_out)
    if hi_c0 < lo_c0:
        lo_c0, hi_c0, w = 0, 0, min(C, w)
    c0 = max(lo_c0, min(hi_c0, c_out - w // 2))
    cells = set()
    used = 0
    for c in range(c0, c0 + w):
        for r in range(1, r_out):
            if used >= B:
                break
            cells.add((r, c))
            used += 1
    return cells


def build_comb(R, C, B, r_out, c_out, spacing, max_stub_len, perm):
    """Trunk + tributary stubs at rows spaced 'spacing' apart. Grows stubs breadth-first
    (every selected row gets +1 cell on each side before any row gets a 2nd), so the
    available budget is spent spreading across many rows rather than a few deep arms,
    up to max_stub_len; any further leftover then falls back to denser row spacing."""
    trunk = build_trunk(R, C, min(B, r_out - 1), r_out, c_out)
    cells = set(trunk)
    remaining = B - len(trunk)

    def grow(rows):
        nonlocal remaining
        depth = 1
        while remaining > 0 and depth <= max_stub_len:
            progressed = False
            for r in rows:
                for direction in (1, -1):
                    if remaining <= 0:
                        break
                    c = c_out + direction * depth
                    if c < 0 or c >= C:
                        continue
                    if (r, c) in cells:
                        continue
                    cells.add((r, c))
                    remaining -= 1
                    progressed = True
            depth += 1
            if not progressed and depth > max_stub_len:
                break

    all_rows = list(range(1, r_out))
    all_rows.sort(key=lambda r: -(perm[r][max(0, c_out - 1)] + perm[r][min(C - 1, c_out + 1)]))
    rows = [r for r in all_rows if (r - 1) % max(1, spacing) == 0] or all_rows
    grow(rows)
    if remaining > 0:
        grow(all_rows)  # denser fallback to use any leftover budget
    return cells


def main():
    toks = sys.stdin.read().split()
    idx = 0
    R, C, B, r_out, c_out = (int(toks[idx + k]) for k in range(5)); idx += 5
    P_IN, P_OUT, C_TUNNEL, ITERS = (int(toks[idx + k]) for k in range(4)); idx += 4
    perm = []
    for r in range(R):
        row = [int(toks[idx + k]) for k in range(C)]
        idx += C
        perm.append(row)

    # use the checker's own iteration count so candidate ranking matches the real score
    # (an under-converged probe misjudges which shape actually wins)
    probe_iters = ITERS

    candidates = []
    candidates.append(build_trunk(R, C, B, r_out, c_out))
    for w in (2, 3, 4, 6, 8):
        candidates.append(build_block(R, C, B, r_out, c_out, w))
    for spacing in (1, 2, 3, 4):
        for stub_len in (2, 3, 4, 6):
            candidates.append(build_comb(R, C, B, r_out, c_out, spacing, stub_len, perm))

    best_cells, best_score = None, -1
    for cells in candidates:
        if not cells or len(cells) > B:
            continue
        score = sim_flow(R, C, r_out, c_out, P_IN, P_OUT, C_TUNNEL, probe_iters, perm, cells)
        if score > best_score:
            best_score, best_cells = score, cells

    if best_cells is None:
        best_cells = build_trunk(R, C, B, r_out, c_out)

    cells = sorted(best_cells)
    out = [str(len(cells))]
    for (r, c) in cells:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
