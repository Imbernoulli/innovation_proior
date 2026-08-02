#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE instance of "Undocumented Logs" (log-schema-inference).

A fixed-width log corpus: N records, each exactly W whitespace tokens.  The
corpus is a mixture of K hidden TEMPLATE FAMILIES (unknown to the solver)
plus a small fraction of pure NOISE lines.  Each family fixes, per token
position, either a CONSTANT literal (same string in every line from that
family) or a VARIABLE slot drawn from a domain (int / hex / enum / ipv4-like).
Position 0 is always a family's constant "tag" -- the natural first signal a
simple parser keys off.

Planted trap (>=3 of the 10 tests, set TRAP below): on trap tests, two (or
three, on the largest tests) DISTINCT families are made to share the exact
same tag at position 0, while differing almost everywhere else (different
constant literals AND different variable-slot layouts from position 1 on).
A parser that buckets purely by "first token" merges these families into one
bucket and then must wildcard nearly every remaining position to keep the
merged group feasible -- the "treat every varying token as variable" failure
mode the checker's objective is built to punish.  Non-trap tests give every
family a distinct tag, so first-token bucketing happens to work fine there
(so a merely-decent recipe still posts a positive score on average).

Determinism: all randomness comes from random.Random(testId * 1000003 + 1231).
"""
import sys
import random

#              1    2    3    4    5    6    7    8    9    10
N_LIST     = [ 40,  55,  75, 100, 140, 190, 260, 360, 500, 700]
W_LIST     = [  6,   6,   7,   7,   8,   8,   9,   9,  10,  10]
K_LIST     = [  3,   4,   4,   5,   5,   6,   6,   7,   7,   8]
# trap tests: two (three on the last two) hidden families deliberately share
# their position-0 tag literal so naive first-token bucketing merges them.
TRAP = {2: 1, 4: 1, 6: 1, 8: 1, 9: 2, 10: 2}   # testId -> #collisions to plant

WORDS = ["svc", "auth", "net", "db", "cache", "queue", "worker", "route",
         "node", "proc", "sess", "conn", "gate", "edge", "pool", "disk",
         "mem", "cpu", "io", "sync", "async", "retry", "flush", "commit",
         "abort", "spawn", "kill", "bind", "listen", "accept", "close",
         "open", "seek", "lock", "unlock", "hashx", "index", "merge",
         "split", "scan", "probe", "ping", "pong", "ack", "nack", "drop",
         "queued", "ready", "stale", "fresh", "valid", "dirty", "clean",
         "pending", "active", "idle", "relay", "shard", "zone", "cell",
         "epoch", "batch", "chunk", "frame", "layer", "vault", "token"]

HEXCH = "0123456789abcdef"


def gen_const_pool(rng, n):
    """n distinct literal words, none numeric-looking, none '*'."""
    pool = list(WORDS)
    rng.shuffle(pool)
    out = []
    i = 0
    while len(out) < n:
        if i < len(pool):
            out.append(pool[i])
        else:
            out.append(pool[i % len(pool)] + str(i))
        i += 1
    return out


def domain_value(rng, kind, slot_state):
    if kind == "INT":
        return str(rng.randint(0, 999999))
    if kind == "HEX":
        return "".join(rng.choice(HEXCH) for _ in range(8))
    if kind == "IPV4":
        return "%d.%d.%d.%d" % (rng.randint(0, 255), rng.randint(0, 255),
                                 rng.randint(0, 255), rng.randint(0, 255))
    if kind == "ENUM":
        return rng.choice(slot_state)
    raise ValueError(kind)


def build_family(rng, W, tag, const_pool_iter):
    """Return a list of length W, each entry either ('CONST', literal) or
    ('VAR', domain_kind, enum_pool_or_None). Position 0 is always ('CONST', tag)."""
    kinds = [None] * W
    kinds[0] = ("CONST", tag)
    n_var = rng.randint(1, max(1, W // 3))
    var_positions = set(rng.sample(range(1, W), min(n_var, W - 1)))
    for p in range(1, W):
        if p in var_positions:
            kind = rng.choice(["INT", "HEX", "IPV4", "ENUM"])
            slot_state = None
            if kind == "ENUM":
                slot_state = [next(const_pool_iter) for _ in range(rng.randint(3, 5))]
            kinds[p] = ("VAR", kind, slot_state)
        else:
            kinds[p] = ("CONST", next(const_pool_iter))
    return kinds


def instantiate(rng, kinds, W):
    line = [None] * W
    for p in range(W):
        k = kinds[p]
        if k[0] == "CONST":
            line[p] = k[1]
        else:
            _, dkind, slot_state = k
            line[p] = domain_value(rng, dkind, slot_state)
    return line


def gen(test_id):
    rng = random.Random(test_id * 1000003 + 1231)
    idx = (test_id - 1) % 10
    N, W, K = N_LIST[idx], W_LIST[idx], K_LIST[idx]
    ncollide = TRAP.get(test_id, 0)

    const_pool = gen_const_pool(rng, 4000)
    pool_iter = iter(const_pool)

    tags = []
    used_tags = set()
    while len(tags) < K:
        t = next(pool_iter) + "_" + str(len(tags))
        if t not in used_tags:
            used_tags.add(t)
            tags.append(t)

    # plant collisions: force `ncollide` extra families to reuse an earlier tag
    if ncollide > 0 and K >= 2:
        base_families = list(range(K))
        rng.shuffle(base_families)
        victims = base_families[:min(ncollide, K - 1)]
        donor = base_families[-1]
        for v in victims:
            if v != donor:
                tags[v] = tags[donor]

    families = []
    for f in range(K):
        families.append(build_family(rng, W, tags[f], pool_iter))

    noise_n = max(1, round(0.05 * N))
    body_n = N - noise_n
    weights = [rng.uniform(0.4, 1.6) for _ in range(K)]
    tot_w = sum(weights)
    counts = [max(3, int(round(body_n * w / tot_w))) for w in weights]
    diff = body_n - sum(counts)
    j = 0
    while diff != 0:
        counts[j % K] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        j += 1
    counts = [max(1, c) for c in counts]

    lines = []
    for f in range(K):
        for _ in range(counts[f]):
            lines.append(instantiate(rng, families[f], W))
    while len(lines) < body_n:
        f = rng.randrange(K)
        lines.append(instantiate(rng, families[f], W))
    lines = lines[:body_n]

    for _ in range(N - len(lines)):
        row = []
        for p in range(W):
            kind = rng.choice(["INT", "HEX", "IPV4", "WORD"])
            if kind == "WORD":
                row.append(rng.choice(WORDS) + str(rng.randint(0, 99)))
            else:
                row.append(domain_value(rng, kind, None))
        lines.append(row)

    rng.shuffle(lines)
    return N, W, lines


def main():
    test_id = int(sys.argv[1])
    N, W, lines = gen(test_id)
    out = [f"{N} {W}"]
    for row in lines:
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
