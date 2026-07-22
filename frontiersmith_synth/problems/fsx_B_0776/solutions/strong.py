# TIER: strong
"""Exact marginal-value greedy (a water-filling exchange argument on the TRUE
objective, not a closed-form proxy). Repeatedly buys whichever single diameter
upgrade yields the best (true unmet-demand reduction) / (marginal cost), where the
reduction is evaluated EXACTLY: bumping edge e's diameter lowers head loss on e by
dHL, which raises the head available at EVERY node in e's subtree by that same dHL
(loss compounds down the whole path), so the benefit is summed over the whole
downstream subtree's demand, not just the edge's own flow.

This is what actually exploits the head-loss convexity (~Q^2/D^5): an edge with
large downstream flow Q_e, when under-sized, both loses a lot per diameter step
(the Q^2 term) AND that loss is inherited by every one of its many descendants (the
subtree-wide impact) -- so its true marginal value per dollar can be enormous
compared to a leaf edge, far more than the LINEAR weighting 'greedy' uses. The
search below discovers this by direct evaluation, not by assuming it a priori."""
import sys


def read_instance():
    data = sys.stdin.read().split('\n')
    data = [t for t in data if t.strip() != ""]
    n = int(data[0].split()[0])
    line2 = data[1].split()
    ndiam = int(line2[0])
    diams = [int(x) for x in line2[1:1 + ndiam]]
    C = int(data[2].split()[0])
    parent = [-1] * n
    demand = [0] * n
    length = [0] * n
    unit_cost = [0] * n
    K = [0.0] * n
    for i in range(1, n):
        row = data[2 + i].split()
        parent[i] = int(row[0]) - 1
        demand[i] = int(row[1])
        length[i] = int(row[2])
        unit_cost[i] = int(row[3])
        K[i] = float(row[4])
    return n, diams, C, parent, demand, length, unit_cost, K


def main():
    n, diams, C, parent, demand, length, unit_cost, K = read_instance()
    children = [[] for _ in range(n)]
    for v in range(1, n):
        children[parent[v]].append(v)

    order = []
    st = [(0, False)]
    while st:
        node, processed = st.pop()
        if processed:
            order.append(node); continue
        st.append((node, True))
        for c in children[node]:
            st.append((c, False))
    S = [0] * n
    for v in order:
        S[v] = demand[v] + sum(S[c] for c in children[v])

    def cost(v, D):
        return unit_cost[v] * length[v] * D * D

    def hl(v, D):
        Q = S[v]
        return K[v] * length[v] * Q * Q / (D ** 5)

    def path_loss(D_const_idx_fn):
        L = [0.0] * n
        stt = [0]
        while stt:
            v = stt.pop()
            for c in children[v]:
                L[c] = L[v] + hl(c, D_const_idx_fn(c))
                stt.append(c)
        return L

    bestL = path_loss(lambda c: diams[-1])
    worstL = path_loss(lambda c: diams[0])
    denom = [max(1e-9, worstL[v] - bestL[v]) for v in range(n)]

    sub_nodes = [None] * n

    def collect(v):
        acc = [v]
        for c in children[v]:
            acc.extend(collect(c))
        sub_nodes[v] = acc
        return acc

    collect(0)

    idx = [0] * n
    L = path_loss(lambda c: diams[idx[c]])
    remaining = C - sum(cost(v, diams[0]) for v in range(1, n))

    def f_of(v, Lv):
        x = (worstL[v] - Lv) / denom[v]
        return 0.0 if x < 0 else (1.0 if x > 1 else x)

    while remaining > 1e-9:
        best_v = None; best_score = -1.0; best_cost = None; best_dHL = None
        for v in range(1, n):
            if idx[v] + 1 >= len(diams):
                continue
            D0, D1 = diams[idx[v]], diams[idx[v] + 1]
            HL0 = hl(v, D0)
            HL1 = hl(v, D1)
            dHL = HL0 - HL1
            c_next = cost(v, D1) - cost(v, D0)
            if c_next > remaining or c_next <= 0:
                continue
            benefit = 0.0
            for u in sub_nodes[v]:
                if u == 0:
                    continue
                benefit += (f_of(u, L[u] - dHL) - f_of(u, L[u])) * demand[u]
            score = benefit / c_next
            if score > best_score:
                best_score = score; best_v = v; best_cost = c_next; best_dHL = dHL
        if best_v is None or best_score <= 1e-12:
            break
        idx[best_v] += 1
        remaining -= best_cost
        for u in sub_nodes[best_v]:
            L[u] -= best_dHL

    print(" ".join(str(idx[v]) for v in range(1, n)))


if __name__ == "__main__":
    main()
