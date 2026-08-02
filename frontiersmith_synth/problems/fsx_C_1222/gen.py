#!/usr/bin/env python3
"""
gen.py <testId>  -- generator for "consensus-log-reconcile" (fsx_C_1222).

Builds R replicas' write histories on K keys, using real (per-replica-counter)
vector clocks -- no gossip is simulated between conflicting writers, which is
exactly what makes two different-replica writes on the same key provably
CONCURRENT (neither vector clock dominates the other) unless we explicitly
chain them on one replica (same-replica program order is always causal).

Three key "shapes" are emitted (the checker never sees these labels -- it only
ever looks at vector-clock dominance):
  * chain  -- one replica issues 2-3 sequential writes on the key. Causally
              totally ordered (frontier size 1 always); timestamps are never
              corrupted here, so every strategy that respects causality (or
              even naive last-timestamp order) recovers the right answer.
              These keys exist purely to require genuine per-key frontier
              detection (mechanism: causal-ordering) rather than "if >1
              writer, treat everyone as concurrent".
  * fair burst      -- 2..R distinct replicas write the SAME key with nobody
              gossiping first => pairwise concurrent (frontier = everybody).
              Weights vary; timestamps are independent of weight.
  * adversarial burst -- same as fair burst, but the LOWEST-weight writer is
              deliberately given the LARGEST timestamp. Last-writer-wins by
              wall-clock timestamp is guaranteed to pick the worst option.

Deterministic: all randomness seeded from testId only.
"""
import sys
import random

LADDER = {
    #           R  n_chain  fair_sizes   adv_sizes
    1:  dict(R=2, n_chain=1, fair=[2],        adv=[]),
    2:  dict(R=3, n_chain=1, fair=[2, 3],     adv=[]),
    3:  dict(R=3, n_chain=1, fair=[2],        adv=[2]),
    4:  dict(R=3, n_chain=2, fair=[3, 2],     adv=[]),
    5:  dict(R=3, n_chain=1, fair=[2],        adv=[3]),
    6:  dict(R=4, n_chain=2, fair=[3, 3],     adv=[2]),
    7:  dict(R=4, n_chain=1, fair=[2, 3],     adv=[4]),
    8:  dict(R=4, n_chain=2, fair=[3, 4, 2],  adv=[3]),
    9:  dict(R=5, n_chain=2, fair=[3, 2],     adv=[5, 3]),
    10: dict(R=5, n_chain=2, fair=[4, 3, 2],  adv=[5, 4]),
}


def gen_instance(test_id: int):
    if test_id not in LADDER:
        test_id = max(1, min(10, test_id))
    cfg = LADDER[test_id]
    R = cfg["R"]
    rnd = random.Random(2654435761 * test_id + 12345)

    knowledge = [[0] * R for _ in range(R)]  # knowledge[r] = replica r's vector clock

    def issue(r):
        knowledge[r][r] += 1
        return list(knowledge[r])

    ops = []          # dicts: replica,key,value,weight,ts,vc
    key_specs = []     # (mtype, mcost) per key, mtype: 0=NONE,1=SUM,2=MAX
    ts_cursor = [1000]

    def next_ts_base():
        ts_cursor[0] += rnd.randint(80, 200)
        return ts_cursor[0]

    key_id = [0]

    def new_key():
        k = key_id[0]
        key_id[0] += 1
        return k

    # ---- 1) chain keys: one replica, 2-3 sequential (uncorrupted) writes ----
    # Deliberately LOW weight scale: chains exist to force genuine per-key
    # frontier detection (nobody gets credit for free by assuming every
    # multi-writer key is "all concurrent"), but must not dominate the
    # score budget -- the real signal lives in the bursts below.
    for _ in range(cfg["n_chain"]):
        k = new_key()
        r = rnd.randrange(R)
        length = rnd.randint(2, 3)
        base = next_ts_base()
        for i in range(length):
            vc = issue(r)
            val = rnd.randint(1, 100)
            w = rnd.randint(4, 9) if i == length - 1 else rnd.randint(1, 9)
            ts = base + i * 10
            ops.append(dict(replica=r, key=k, value=val, weight=w, ts=ts, vc=vc))
        key_specs.append((0, rnd.randint(5, 15)))

    def merge_type():
        return rnd.choices([1, 2, 0], weights=[50, 25, 25])[0]

    # ---- 2) fair concurrent bursts ----
    # Weight and timestamp are POSITIVELY correlated with noise: in the
    # common case, "most recently written" tends to roughly track "most
    # important update" (a believable real-world tendency), so a
    # naive wall-clock LWW usually -- but not via any causal reasoning --
    # lands on a decent member. This gives 'greedy' a real, reliable edge
    # over the checker's arbitrary-worst-pick baseline on ordinary
    # concurrent keys, while the ADVERSARIAL bursts below explicitly invert
    # this correlation as the planted trap.
    for size in cfg["fair"]:
        k = new_key()
        n = min(size, R)
        replicas = rnd.sample(range(R), n)
        weights = [rnd.randint(6, 34) for _ in range(n)]
        vals = [rnd.randint(1, 100) for _ in range(n)]
        base = next_ts_base()
        rank = sorted(range(n), key=lambda i: weights[i])  # ascending weight
        tss = [0] * n
        for pos, i in enumerate(rank):
            tss[i] = base + pos * 40 + rnd.randint(0, 25)
        mtype = merge_type()
        mcost = rnd.randint(6, 16)
        for idx, r in enumerate(replicas):
            vc = issue(r)
            ops.append(dict(replica=r, key=k, value=vals[idx], weight=weights[idx],
                             ts=tss[idx], vc=vc))
        key_specs.append((mtype, mcost))

    # ---- 3) adversarial concurrent bursts (the trap) ----
    for size in cfg["adv"]:
        k = new_key()
        n = min(size, R)
        replicas = rnd.sample(range(R), n)
        weights = [rnd.randint(6, 34) for _ in range(n)]
        vals = [rnd.randint(1, 100) for _ in range(n)]
        base = next_ts_base()
        tss = [base + rnd.randint(0, 150) for _ in range(n)]
        min_idx = min(range(n), key=lambda i: weights[i])
        tss[min_idx] = base + 1000  # force the WORST writer to look "latest"
        mtype = merge_type()
        mcost = rnd.randint(6, 16)
        for idx, r in enumerate(replicas):
            vc = issue(r)
            ops.append(dict(replica=r, key=k, value=vals[idx], weight=weights[idx],
                             ts=tss[idx], vc=vc))
        key_specs.append((mtype, mcost))

    K = key_id[0]
    N = len(ops)
    total_merge_cost = sum(c for (t, c) in key_specs if t != 0)
    frac = rnd.uniform(0.45, 0.65)
    BUDGET = max(1, int(round(total_merge_cost * frac))) if total_merge_cost > 0 else rnd.randint(5, 10)

    return R, K, N, BUDGET, key_specs, ops


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    R, K, N, BUDGET, key_specs, ops = gen_instance(test_id)

    out = [f"{R} {K} {N} {BUDGET}"]
    out.append(" ".join(str(t) for (t, c) in key_specs))
    out.append(" ".join(str(c) for (t, c) in key_specs))
    for op in ops:
        vc_str = " ".join(str(x) for x in op["vc"])
        out.append(f"{op['replica']} {op['key']} {op['value']} {op['weight']} {op['ts']} {vc_str}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
