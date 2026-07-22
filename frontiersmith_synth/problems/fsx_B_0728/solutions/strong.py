# TIER: strong
# Insight: feasibility (the Maekawa-style parity floor) rarely pins every crease --
# most vertex groups have residual freedom (several M/V arrangements hit the same
# target sum). The greedy recipe burns that freedom on a position-blind tie-break and
# never asks what it does to the footprint. Here we run a beam search over the
# sequence of vertex-group decisions: every state tracks the routing walk's
# (column, heading), the running per-column weight histogram, AND which hinge links
# have gone active (so a candidate that would cross two active hinges in the same
# column is pruned on the spot, not discovered after the fact) -- so a choice that
# looks fine locally but boxes the walk into a future pileup, or into an illegal
# crossing, gets dropped before it can hurt the objective, instead of being locked in
# by a single fixed tie-break.
import sys, itertools
from collections import defaultdict, deque

BEAM = 96


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


def compute_heights(N, pos, hinge_resolved):
    adj = defaultdict(list)
    indeg = [0] * N
    for p, q, lab in hinge_resolved:
        if pos[p] != pos[q]:
            continue
        u, v = (p, q) if lab == 'V' else (q, p)
        adj[u].append(v)
        indeg[v] += 1
    dq = deque(sorted(i for i in range(N) if indeg[i] == 0))
    order = []
    indeg2 = indeg[:]
    seen = set()
    while dq:
        u = dq.popleft()
        if u in seen:
            continue
        seen.add(u)
        order.append(u)
        for v in sorted(adj[u]):
            indeg2[v] -= 1
            if indeg2[v] == 0:
                dq.append(v)
    for i in range(N):
        if i not in seen:
            order.append(i)
    height = {node: rank for rank, node in enumerate(order)}
    return [height[i] for i in range(N)]


def resolve_baseline(N, groups, creases_in):
    """Same construction as the checker's own internal baseline -- always
    crossing-free and feasible; used both as a beam seed and as a safety fallback."""
    cr = ['?'] * (N - 1)
    for idxs, target in groups:
        vals = [cr[i] for i in idxs]
        fixed_sum = sum(1 if v == 'M' else -1 for v in vals if v != '?')
        free = [i for i in idxs if cr[i] == '?']
        nfree = len(free)
        if nfree == 0:
            continue
        need = target - fixed_sum
        m_needed = max(0, min(nfree, (need + nfree) // 2))
        chosen = set(sorted(free)[:m_needed])
        for i in free:
            cr[i] = 'M' if i in chosen else 'V'
    for i in range(N - 1):
        if cr[i] == '?':
            cr[i] = 'V'
    return cr


def main():
    text = sys.stdin.read()
    N, K, G, W, creases_in, hinges, groups = parse_instance(text)

    hinges_by_q = defaultdict(list)
    for p, q, lab in hinges:
        hinges_by_q[q].append(p)

    resolved0 = [None if creases_in[i] == '?' else creases_in[i] for i in range(N - 1)]

    covered = set()
    for idxs, _ in groups:
        covered.update(idxs)
    blocks = [(sorted(idxs), target) for idxs, target in
               sorted(groups, key=lambda g: min(g[0]))]
    for i in range(N - 1):
        if i not in covered:
            blocks.append(([i], None))

    # beam state: (peak, resolved_array, pos_list, dir, hist, active_by_col)
    beam = [(W[0], list(resolved0), [0], 1, {0: W[0]}, {})]

    for idxs, target in blocks:
        nxt = []
        for peak0, resolved, pos_list, dir_state, hist, active in beam:
            free = sorted(i for i in idxs if resolved[i] is None)
            nfree = len(free)
            if nfree == 0:
                nxt.append((peak0, resolved, pos_list, dir_state, hist, active))
                continue
            if target is None:
                combo_sizes = [0, 1] if nfree == 1 else range(nfree + 1)
            else:
                fixed_sum = sum(1 if resolved[i] == 'M' else -1 for i in idxs if resolved[i] is not None)
                need = target - fixed_sum
                m_needed = max(0, min(nfree, (need + nfree) // 2))
                combo_sizes = [m_needed]
            for m_needed in combo_sizes:
                for chosen_tuple in itertools.combinations(free, m_needed):
                    chosen = set(chosen_tuple)
                    sp, sd = pos_list[-1], dir_state
                    lp = list(pos_list)
                    lh = dict(hist)
                    la = active
                    ok = True
                    for ci in free:
                        if ci in chosen:
                            sp = sp - sd
                            sd = -sd
                        else:
                            sp = sp + sd
                        new_flap = ci + 1
                        lp.append(sp)
                        lh[sp] = lh.get(sp, 0) + W[new_flap]
                        for p in hinges_by_q.get(new_flap, []):
                            if lp[p] == sp:
                                lst = la.get(sp, [])
                                bad = False
                                for (p2, q2) in lst:
                                    if (p < p2 < new_flap < q2) or (p2 < p < q2 < new_flap):
                                        bad = True
                                        break
                                if bad:
                                    ok = False
                                    break
                                if la is active:
                                    la = dict(active)
                                la[sp] = lst + [(p, new_flap)]
                        if not ok:
                            break
                    if not ok:
                        continue
                    new_resolved = resolved[:]
                    for i in free:
                        new_resolved[i] = 'M' if i in chosen else 'V'
                    peak = max(lh.values())
                    nxt.append((peak, new_resolved, lp, sd, lh, la))
        nxt.sort(key=lambda s: s[0])
        beam = nxt[:BEAM]
        if not beam:
            break

    if beam:
        beam.sort(key=lambda s: s[0])
        resolved = beam[0][1]
    else:
        # safety net: every branch got pruned by a crossing constraint somewhere --
        # fall back to the checker's own always-feasible baseline construction.
        resolved = resolve_baseline(N, groups, creases_in)

    hinge_lab = []
    for p, q, lab in hinges:
        hinge_lab.append(lab if lab != '?' else 'V')

    pos = walk(resolved, N)
    heights = compute_heights(N, pos, [(hinges[k][0], hinges[k][1], hinge_lab[k]) for k in range(K)])

    out = []
    out.append(" ".join(resolved))
    out.append(" ".join(hinge_lab))
    out.append(" ".join(map(str, heights)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
