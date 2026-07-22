# TIER: greedy
"""'Linear demand-share' recipe: compute each edge's downstream flow Q_e (the total
demand of its subtree -- the flow it actually carries under the tree's conservation
law), then repeatedly water-fill whatever budget remains PROPORTIONALLY to Q_e,
buying each edge's next diameter step whenever its running share can afford it.

This is the natural first idea a strong coder reaches for: "pipes that carry more
flow should get proportionally more budget." It correctly notices that flow matters,
but treats the trade-off as LINEAR in Q_e. Since head loss actually scales as
Q_e^2 / D^5, the correct exchange rate is far more convex than linear -- so this
recipe systematically under-funds the few pipes that dominate total head loss (the
trunk) and over-funds the rest, exactly the trap the family is built around."""
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
    for i in range(1, n):
        row = data[2 + i].split()
        parent[i] = int(row[0]) - 1
        demand[i] = int(row[1])
        length[i] = int(row[2])
        unit_cost[i] = int(row[3])
    return n, diams, C, parent, demand, length, unit_cost


def main():
    n, diams, C, parent, demand, length, unit_cost = read_instance()
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

    idx = [0] * n
    remaining = C - sum(cost(v, diams[0]) for v in range(1, n))
    active = [v for v in range(1, n) if idx[v] + 1 < len(diams)]
    while remaining > 1e-9 and active:
        total_q = sum(S[v] for v in active) or 1
        bought_any = False
        still_active = []
        for v in active:
            share = remaining * S[v] / total_q
            c_next = cost(v, diams[idx[v] + 1]) - cost(v, diams[idx[v]])
            if c_next <= share + 1e-9 and c_next <= remaining:
                idx[v] += 1
                remaining -= c_next
                bought_any = True
            if idx[v] + 1 < len(diams):
                still_active.append(v)
        active = still_active
        if not bought_any:
            break

    print(" ".join(str(idx[v]) for v in range(1, n)))


if __name__ == "__main__":
    main()
