# TIER: strong
# Interference-graph block scheduling with a maximin-sized closing refresher.
#
# INSIGHT: the maximin objective plus multiplicative interference means a
# skill's final proficiency is set by (a) its post-drill peak the last time it
# was drilled, and (b) how much every drill AFTER that last drill decayed it.
# So the montage is a graph-partitioning problem in TIME: skills whose mutual
# interference is mild (a "reinforcement clique") can be blocked together
# cheaply, while antagonist pairs should not be interleaved once either has
# locked in its peak. Concretely:
#   1. Build a compatibility graph from the antagonism weight
#      w[i][j] = (1 - interfere[i][j]) + (1 - interfere[j][i]); union-find merge
#      any pair below a fixed threshold into the same reinforcement clique.
#   2. Order cliques by ASCENDING vulnerability = average decay a clique's
#      members suffer from skills OUTSIDE it. Cliques that other skills barely
#      touch go first (they can safely absorb being decayed by everything that
#      follows); the clique most exposed to outside interference goes last, so
#      its members see the least decay after their own drills.
#   3. Reserve the final third of the montage. In the remaining "body" budget,
#      give each skill just enough drills to approach its diminishing-returns
#      saturation point (over-drilling a saturated skill wastes body budget),
#      water-filled down if the body budget is tight, interleaved within each
#      clique block in clique order.
#   4. In the reserved tail, repeatedly drill whichever skill is CURRENTLY
#      weakest under the running simulation. This is the maximin-sized closing
#      refresher: it targets exactly the skill the final score depends on, and
#      because it happens late, little or nothing decays it afterward.
import sys, json


def simulate_step(p, K, j, gain, interfere):
    gj = gain[j]
    row = interfere[j]
    for i in range(K):
        if i == j:
            p[i] = p[i] + gj * (1.0 - p[i])
        else:
            p[i] = p[i] * row[i]


def main():
    inst = json.load(sys.stdin)
    K = inst["K"]; T = inst["T"]
    p0 = inst["p0"]; gain = inst["gain"]; interfere = inst["interfere"]

    # 1. reinforcement cliques via union-find on a fixed antagonism threshold
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            w = (1 - interfere[i][j]) + (1 - interfere[j][i])
            pairs.append((w, i, j))
    THRESH = 0.05
    for w, i, j in pairs:
        if w <= THRESH:
            union(i, j)
    clusters = {}
    for i in range(K):
        clusters.setdefault(find(i), []).append(i)
    cl_list = list(clusters.values())

    # 2. order clusters by ascending vulnerability to OUTSIDE interference
    def vuln(c):
        cs = set(c)
        tot = 0.0; cnt = 0
        for i in c:
            for j in range(K):
                if j not in cs:
                    tot += (1 - interfere[j][i]); cnt += 1
        return tot / cnt if cnt else 0.0

    cl_list.sort(key=vuln)

    # 3. body: per-skill saturation-sized practice counts, water-filled to budget
    reserve_tail = max(2 * K, T // 3)
    body_budget = max(0, T - reserve_tail)

    def sat_n(i):
        n = 0; p = p0[i]
        while n < 40 and p < 0.99:
            p = p + gain[i] * (1.0 - p); n += 1
        return max(1, n)

    raw_n = [sat_n(i) for i in range(K)]
    total_raw = sum(raw_n)
    if total_raw <= body_budget:
        n_i = raw_n[:]
    else:
        scaled = [max(1, (raw_n[i] * body_budget) // total_raw) for i in range(K)]
        while sum(scaled) > body_budget:
            mi = max(range(K), key=lambda i: scaled[i])
            if scaled[mi] > 1:
                scaled[mi] -= 1
            else:
                break
        n_i = scaled

    body = []
    for c in cl_list:
        remaining = {i: n_i[i] for i in c}
        while any(v > 0 for v in remaining.values()):
            for i in c:
                if remaining[i] > 0:
                    body.append(i)
                    remaining[i] -= 1

    p = list(p0)
    for j in body:
        simulate_step(p, K, j, gain, interfere)

    # 4. tail: myopic refresher of the current worst skill, closing late
    tail = []
    tail_len = T - len(body)
    for _ in range(tail_len):
        worst = min(range(K), key=lambda i: p[i])
        tail.append(worst)
        simulate_step(p, K, worst, gain, interfere)

    seq = body + tail
    # safety pad/trim (should already equal T by construction)
    while len(seq) < T:
        worst = min(range(K), key=lambda i: p[i])
        seq.append(worst)
        simulate_step(p, K, worst, gain, interfere)
    seq = seq[:T]

    print(json.dumps({"sequence": seq}))


if __name__ == "__main__":
    main()
