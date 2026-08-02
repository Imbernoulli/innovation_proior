#!/usr/bin/env python3
"""gen.py <testId> -- prints one rate-limiter-design instance to stdout.
Deterministic: all randomness seeded from testId only.

Format:
  line 1: T M A B P
  line 2: R
  next R lines: t key label      (t in [1,T], label in {0,1}: 1=legit 0=abusive)

Requests are emitted in nondecreasing time order (ties broken by emission order);
the checker replays them in exactly this file order.
"""
import sys, random

KEYMAX = 4_000_000_000
HASH_A = 2654435761
HASH_B = 40503
HASH_P = 2147483647  # 2^31 - 1 (Mersenne prime)


def make_case(testId):
    rnd = random.Random(900001 + 97 * testId)

    # ---- per-test scale/shape (1-4: warm-up/feasible, 5-10: cardinality-explosion attacks)
    #   sustained=True -> abuse heavy hitter is a huge-volume decoy spread over the whole
    #   horizon (burst-vs-sustained trap: a single scalar bucket can't admit a good burst
    #   AND choke a sustained flood at the same time); sustained=False -> abuse heavy
    #   hitter is a modest, low-volume nuisance that shouldn't dominate the trace.
    cfg = {
        1:  dict(T=250,  M=30, good_hh=3, abuse_hh=1, sustained=False, flood=0,    fl_abuse=0.0),
        2:  dict(T=300,  M=34, good_hh=4, abuse_hh=2, sustained=False, flood=0,    fl_abuse=0.0),
        3:  dict(T=400,  M=28, good_hh=2, abuse_hh=1, sustained=True,  flood=0,    fl_abuse=0.0),
        4:  dict(T=450,  M=34, good_hh=3, abuse_hh=2, sustained=True,  flood=0,    fl_abuse=0.0),
        5:  dict(T=500,  M=30, good_hh=3, abuse_hh=1, sustained=False, flood=0,    fl_abuse=0.0),
        6:  dict(T=600,  M=34, good_hh=4, abuse_hh=2, sustained=False, flood=0,    fl_abuse=0.0),
        7:  dict(T=700,  M=34, good_hh=3, abuse_hh=1, sustained=True,  flood=4200, fl_abuse=0.80),
        8:  dict(T=800,  M=38, good_hh=2, abuse_hh=2, sustained=True,  flood=5500, fl_abuse=0.85),
        9:  dict(T=1000, M=38, good_hh=3, abuse_hh=2, sustained=True,  flood=7000, fl_abuse=0.88),
        10: dict(T=1200, M=40, good_hh=4, abuse_hh=3, sustained=True,  flood=9000, fl_abuse=0.88),
    }[max(1, min(10, testId))]

    T, M = cfg["T"], cfg["M"]
    n_good_hh, n_abuse_hh = cfg["good_hh"], cfg["abuse_hh"]
    flood, fl_abuse = cfg["flood"], cfg["fl_abuse"]

    base = 10_000_000 * testId
    events = []  # (t, key, label)

    key_ctr = 0

    def next_key():
        nonlocal key_ctr
        key_ctr += 1
        return base + key_ctr

    # ---- good heavy hitters: bursty, concentrated in a short window ----
    for _ in range(n_good_hh):
        k = next_key()
        w = max(8, T // 25)
        t0 = rnd.randint(1, max(1, T - w))
        cnt = rnd.randint(180, 260)
        for _ in range(cnt):
            t = rnd.randint(t0, t0 + w - 1)
            events.append((t, k, 1))

    # ---- abusive heavy hitters ----
    for _ in range(n_abuse_hh):
        k = next_key()
        if cfg["sustained"]:
            # decoy: huge-volume, spread across the ENTIRE horizon (sustained flood) --
            # a naive volume-proportional bucket admits nearly all of it.
            cnt = rnd.randint(2200, 2800)
            for _ in range(cnt):
                t = rnd.randint(1, T)
                events.append((t, k, 0))
        else:
            # modest nuisance -- low volume, should not dominate the trace
            w = max(20, T // 6)
            t0 = rnd.randint(1, max(1, T - w))
            cnt = rnd.randint(1, 5)
            for _ in range(cnt):
                t = rnd.randint(t0, t0 + w - 1)
                events.append((t, k, 0))

    # ---- generic light background traffic (always present) ----
    for _ in range(20):
        k = next_key()
        cnt = rnd.randint(1, 5)
        label = 1 if rnd.random() < 0.9 else 0
        for _ in range(cnt):
            t = rnd.randint(1, T)
            events.append((t, k, label))

    # ---- key-cardinality-explosion: thousands of one/two/three-shot keys ----
    for _ in range(flood):
        k = next_key()
        cnt = rnd.randint(1, 3)
        label = 0 if rnd.random() < fl_abuse else 1
        for _ in range(cnt):
            t = rnd.randint(1, T)
            events.append((t, k, label))

    events.sort(key=lambda e: e[0])  # stable sort: ties keep insertion order

    out = []
    out.append(f"{T} {M} {HASH_A} {HASH_B} {HASH_P}")
    out.append(str(len(events)))
    for t, k, lab in events:
        out.append(f"{t} {k} {lab}")
    return "\n".join(out) + "\n"


def main():
    testId = int(sys.argv[1])
    sys.stdout.write(make_case(testId))


if __name__ == "__main__":
    main()
