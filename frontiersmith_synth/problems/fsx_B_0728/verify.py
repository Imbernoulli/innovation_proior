import sys
from collections import defaultdict


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def walk(creases, N):
    pos = [0] * N
    d = 1
    for i in range(N - 1):
        if creases[i] == 'V':
            pos[i + 1] = pos[i] + d
        else:
            pos[i + 1] = pos[i] - d
            d = -d
    return pos


def thickness(pos, W):
    agg = defaultdict(int)
    for i, c in enumerate(pos):
        agg[c] += W[i]
    return max(agg.values())


def baseline_creases(N, groups):
    """Checker's own trivial-but-feasible construction: M-first / low-index
    tie-break per vertex group, 'V' default for anything left over."""
    cr = ['?'] * (N - 1)
    for idxs, target in groups:
        vals = [cr[i] for i in idxs]
        fixed_sum = sum(1 if v == 'M' else -1 for v in vals if v != '?')
        free = [i for i in idxs if cr[i] == '?']
        nfree = len(free)
        if nfree == 0:
            continue
        need = target - fixed_sum
        m_needed = (need + nfree) // 2
        m_needed = max(0, min(nfree, m_needed))
        chosen = set(sorted(free)[:m_needed])
        for i in free:
            cr[i] = 'M' if i in chosen else 'V'
    for i in range(N - 1):
        if cr[i] == '?':
            cr[i] = 'V'
    return cr


def parse_instance(text):
    it = iter(text.split())
    N = int(next(it)); K = int(next(it)); G = int(next(it))
    W = [int(next(it)) for _ in range(N)]
    creases_in = [next(it) for _ in range(N - 1)]
    hinges = []
    for _ in range(K):
        p = int(next(it)); q = int(next(it)); lab = next(it)
        hinges.append((p, q, lab))
    groups = []
    for _ in range(G):
        idxs = [int(next(it)) for _ in range(4)]
        t = int(next(it))
        groups.append((idxs, t))
    return N, K, G, W, creases_in, hinges, groups


def main():
    inp_text = open(sys.argv[1]).read()
    try:
        N, K, G, W, creases_in, hinges, groups = parse_instance(inp_text)
    except Exception:
        fail("bad input")

    try:
        out_toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    it = iter(out_toks)
    try:
        step_out = [next(it) for _ in range(N - 1)]
        hinge_out = [next(it) for _ in range(K)]
        height_toks = [next(it) for _ in range(N)]
    except StopIteration:
        fail("truncated output")

    # ---- token validity ----
    for t in step_out:
        if t not in ('M', 'V'):
            fail("bad step label %r" % t)
    for t in hinge_out:
        if t not in ('M', 'V'):
            fail("bad hinge label %r" % t)

    heights = []
    for t in height_toks:
        try:
            v = int(t)
        except Exception:
            fail("non-integer height %r" % t)
        if v != v or abs(v) == float('inf'):
            fail("non-finite height")
        if abs(v) > 10 ** 15:
            fail("height out of range")
        heights.append(v)
    if len(set(heights)) != N:
        fail("heights not distinct")

    # ---- pin consistency (step creases carry no pins in this family; hinges do) ----
    for i in range(N - 1):
        if creases_in[i] != '?' and creases_in[i] != step_out[i]:
            fail("step crease %d violates pin" % i)
    for k, (p, q, lab) in enumerate(hinges):
        if lab != '?' and lab != hinge_out[k]:
            fail("hinge %d violates pin" % k)

    # ---- vertex-group Maekawa-style parity floor ----
    for idxs, target in groups:
        s = sum(1 if step_out[i] == 'M' else -1 for i in idxs)
        if s != target:
            fail("vertex group %s parity %d != target %d" % (idxs, s, target))

    # ---- routing walk determined by the resolved step creases ----
    pos = walk(step_out, N)

    # ---- hinge coincidence + order + noncrossing ----
    active = []  # (p, q, lab, col)
    for k, (p, q, lab) in enumerate(hinges):
        pin_lab, out_lab = lab, hinge_out[k]
        if pos[p] != pos[q]:
            continue
        c = pos[p]
        if out_lab == 'V':
            if not (heights[p] < heights[q]):
                fail("hinge (%d,%d) valley order violated" % (p, q))
        else:
            if not (heights[p] > heights[q]):
                fail("hinge (%d,%d) mountain order violated" % (p, q))
        active.append((p, q, c))

    by_col = defaultdict(list)
    for p, q, c in active:
        by_col[c].append((p, q))
    for c, lst in by_col.items():
        for i in range(len(lst)):
            p1, q1 = lst[i]
            for j2 in range(i + 1, len(lst)):
                p2, q2 = lst[j2]
                if (p1 < p2 < q1 < q2) or (p2 < p1 < q2 < q1):
                    fail("crossing hinges (%d,%d) x (%d,%d) in column %d" % (p1, q1, p2, q2, c))

    # ---- objective: peak weighted stack thickness (minimize) ----
    F = thickness(pos, W)

    # ---- internal baseline B: checker's own naive-but-feasible construction ----
    b_cr = baseline_creases(N, groups)
    b_pos = walk(b_cr, N)
    B = thickness(b_pos, W)
    B = max(1, B)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
