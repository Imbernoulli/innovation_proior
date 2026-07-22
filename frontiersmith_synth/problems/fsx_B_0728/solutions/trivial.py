# TIER: trivial
import sys
from collections import defaultdict, deque


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


def resolve_groups_lowfirst(creases, groups):
    cr = creases[:]
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
    return cr


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


def main():
    text = sys.stdin.read()
    N, K, G, W, creases_in, hinges, groups = parse_instance(text)

    cr = resolve_groups_lowfirst(creases_in, groups)
    for i in range(N - 1):
        if cr[i] == '?':
            cr[i] = 'V'

    hinge_lab = []
    for p, q, lab in hinges:
        hinge_lab.append(lab if lab != '?' else 'V')

    pos = walk(cr, N)
    heights = compute_heights(N, pos, [(hinges[k][0], hinges[k][1], hinge_lab[k]) for k in range(K)])

    out = []
    out.append(" ".join(cr))
    out.append(" ".join(hinge_lab))
    out.append(" ".join(map(str, heights)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
