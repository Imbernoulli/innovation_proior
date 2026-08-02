#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE cache-coherence trace instance to stdout.
Deterministic: seeded only by testId. Difficulty ladder small->large/adversarial.
Plants >=3 (in fact >=6) "mixed sharing pattern" test cases where a single GLOBAL
policy (all-INV or all-UPD) is forced to be badly wrong on at least one subset of
lines, while a per-line (optionally regime-adaptive) policy handles every subset well.
"""
import sys, random


def gen_write_storm(rng, cores, storm_len):
    """Line kind: a couple of cores read once (become sharers), then ONE core
    writes many times in a row with nobody else reading in between.
    INV pays the invalidate messages once; UPD re-pays a broadcast on every
    single write forever (write-heavy, rarely-read -> UPD floods)."""
    writer = cores[0]
    nshare = 2 if len(cores) >= 3 else 1
    readers = cores[1:1 + nshare]
    ev = [(r, "R") for r in readers]
    ev += [(writer, "W") for _ in range(storm_len)]
    return ev


def gen_read_heavy(rng, cores, cycles, k):
    """Line kind: one core writes, then several distinct cores read after
    EVERY write, repeated over many cycles.
    INV re-pays a miss for each reader after every write (thrashing);
    UPD pays one broadcast per write and the readers stay valid (cheap)."""
    writer = cores[0]
    readers = cores[1:1 + max(1, min(k, len(cores) - 1))]
    ev = []
    for _ in range(cycles):
        ev.append((writer, "W"))
        for r in readers:
            ev.append((r, "R"))
    return ev


def gen_regime_shift(rng, cores, storm_len, cycles, k):
    """Line kind: the SAME line behaves like a write-storm for a while, then
    the access pattern flips to read-heavy-shared. No static per-line choice
    (INV or UPD) is good in both halves; only a reactive/adaptive rule is."""
    part1 = gen_write_storm(rng, cores, storm_len)
    part2 = gen_read_heavy(rng, list(reversed(cores)), cycles, k)
    return part1 + part2


def gen_uniform(rng, cores, length):
    """Line kind: plain noisy interleave, no strong planted structure."""
    p_write = rng.uniform(0.3, 0.6)
    ev = []
    for _ in range(length):
        c = rng.choice(cores)
        op = "W" if rng.random() < p_write else "R"
        ev.append((c, op))
    return ev


def gen_private(rng, cores, length):
    """Line kind: only one core ever touches it -- INV and UPD tie (no sharers)."""
    c = cores[0]
    ev = []
    for _ in range(length):
        op = "W" if rng.random() < 0.5 else "R"
        ev.append((c, op))
    return ev


def random_merge(rng, streams):
    """Deterministically interleave several per-line event streams into one
    global chronological trace while preserving each stream's own order."""
    streams = [list(s) for s in streams]
    out = []
    while True:
        idxs = [i for i, s in enumerate(streams) if s]
        if not idxs:
            break
        i = rng.choice(idxs)
        out.append(streams[i].pop(0))
    return out


PLAN = {
    1: (2, ["uniform", "uniform"]),
    2: (3, ["uniform", "private", "uniform"]),
    3: (3, ["uniform", "uniform", "private", "uniform"]),
    4: (4, ["write_storm", "read_heavy", "write_storm", "read_heavy"]),
    5: (5, ["write_storm", "write_storm", "read_heavy", "read_heavy", "regime"]),
    6: (5, ["regime", "write_storm", "read_heavy", "read_heavy", "write_storm", "regime"]),
    7: (6, ["regime", "regime", "write_storm", "read_heavy", "read_heavy", "write_storm"]),
    8: (6, ["write_storm", "write_storm", "read_heavy", "read_heavy", "read_heavy", "regime", "regime", "write_storm"]),
    9: (7, ["regime", "regime", "write_storm", "write_storm", "read_heavy", "read_heavy", "read_heavy", "regime"]),
    10: (8, ["regime", "regime", "regime", "write_storm", "write_storm", "read_heavy", "read_heavy", "read_heavy", "write_storm", "regime"]),
}


def build(tid):
    rng = random.Random(20000 + 97 * tid)
    C, kinds = PLAN[tid]
    L = len(kinds)
    MISS = rng.randint(6, 10)
    INVC = rng.randint(1, 3)
    UPDC = rng.randint(1, 3)

    storm_len = 15 + 4 * tid
    cycles = 4 + tid
    k = max(1, min(C - 1, 3 + tid // 3))

    core_ids = list(range(C))
    per_line = []
    for li, kind in enumerate(kinds):
        rng.shuffle(core_ids)
        cores = list(core_ids)  # a fresh rotation per line, deterministic
        if kind == "uniform":
            length = 6 + 2 * tid
            ev = gen_uniform(rng, cores, length)
        elif kind == "private":
            length = 4 + tid
            ev = gen_private(rng, cores, length)
        elif kind == "write_storm":
            ev = gen_write_storm(rng, cores, storm_len)
        elif kind == "read_heavy":
            ev = gen_read_heavy(rng, cores, cycles, k)
        elif kind == "regime":
            ev = gen_regime_shift(rng, cores, max(4, storm_len // 2), max(2, cycles // 2), k)
        else:
            raise ValueError(kind)
        if not ev:
            ev = [(cores[0], "R")]
        per_line.append([(c, li, op) for (c, op) in ev])

    merged = random_merge(rng, per_line)
    T = len(merged)

    out = [f"{T} {C} {L}", f"{MISS} {INVC} {UPDC}"]
    for c, l, op in merged:
        out.append(f"{c} {l} {op}")
    return "\n".join(out) + "\n"


def main():
    tid = int(sys.argv[1])
    sys.stdout.write(build(tid))


if __name__ == "__main__":
    main()
