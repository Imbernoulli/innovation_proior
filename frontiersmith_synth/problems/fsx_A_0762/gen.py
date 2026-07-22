#!/usr/bin/env python3
"""gen.py <testId> -> prints one tug-escort-chaining instance to stdout.

Deterministic: seeded purely by testId. Cases 3,5,7,9 are hand-engineered "trap" cases
(a persistent, currently-idle pair of tugs can chain through a corridor of same-k jobs on
time, but a decoy tug that LOOKS closest -- because it is committed to finish an unrelated
long tow right at the corridor's doorstep -- lures a naive nearest-tug-by-raw-distance
dispatcher into picking it, and its late free time blows the tight tide window). The other
six cases are generic randomized instances (still with a guaranteed-feasible window per
job) that grow the ladder from tiny to large/adversarial.
"""
import random
import sys


def mk_job(a, b, k, release, weight, windows):
    assert a != b
    return {"a": a, "b": b, "k": k, "release": release, "weight": weight,
            "windows": windows, "dur": abs(b - a)}


def print_instance(T, N, L, pos, coeff, pen, jobs):
    assert len(pos) == T and len(jobs) == N
    out = [f"{T} {N} {L}", " ".join(map(str, pos)), f"{coeff} {pen}"]
    for j in jobs:
        out.append(f"{j['a']} {j['b']} {j['k']} {j['release']} {j['weight']} {len(j['windows'])}")
        out.append(" ".join(f"{o} {c}" for o, c in j["windows"]))
    sys.stdout.write("\n".join(out) + "\n")


def random_job(rng, L, T, max_k, release_hint):
    max_k = max(1, min(max_k, T))
    k = rng.randint(1, max_k)
    a = rng.randint(0, L - 10)
    d = rng.randint(3, min(20, L - a) if L - a >= 4 else 3)
    if a + d > L:
        d = max(3, L - a)
    b = a + d if rng.random() < 0.5 or a + d <= L else a - d
    if b == a:
        b = a + 3 if a + 3 <= L else a - 3
    dur = abs(b - a)
    release = max(0, release_hint + rng.randint(-3, 6))
    weight = rng.randint(1, 5)
    # one guaranteed-feasible generous window, plus (sometimes) a tighter decoy window before it
    open1 = release + rng.randint(0, 5)
    close1 = open1 + dur + rng.randint(10, 40)
    windows = [(open1, close1)]
    if rng.random() < 0.4:
        # an earlier, impossible-to-reach sliver window, just to exercise window search
        o0 = max(0, release - rng.randint(5, 15))
        c0 = o0 + max(1, dur // 3)
        if c0 < open1:
            windows = [(o0, c0), (open1, close1)]
    windows.sort()
    return mk_job(a, b, k, release, weight, windows)


def generic_case(seed, T, N, L, max_k=3, coeff=1, pen=250):
    rng = random.Random(seed)
    pos = [rng.randint(0, L) for _ in range(T)]
    jobs = []
    rel = 0
    for i in range(N):
        j = random_job(rng, L, T, max_k, rel)
        jobs.append(j)
        rel += rng.randint(0, 6)
    return T, N, L, pos, coeff, pen, jobs


def trap_case(n_chain, k, decoy_gap, T_extra_idle, T_extra_filler, coeff=1, pen=300,
              step=None, base=10):
    """One decoy job (k=1) whose tug starts right next to the corridor entrance but is
    committed to a long unrelated tow that ends exactly at the corridor entrance (so it
    LOOKS like distance 0 once assigned) -- and n_chain corridor jobs of size k, spaced
    `step` apart, each with a tight tide window that only a team already synced and idle
    can reach. `T_extra_idle` extra idle tugs sit near the corridor start (the real team
    candidates); `T_extra_filler` extra tugs sit far away and are irrelevant noise.
    """
    if step is None:
        step = 7 + decoy_gap % 3
    pos = []
    jobs = []

    # decoy job A: k=1, long tow ending exactly at the first corridor job's rendezvous
    a0 = base
    decoy_far = a0 + 150 + decoy_gap
    tugX_pos = decoy_far + 1
    pos.append(tugX_pos)  # tug 0 = decoy-bait
    jobs.append(mk_job(decoy_far, a0, 1, 0, 1, [(0, 500 + decoy_gap)]))

    # the real (idle) team candidates: k tugs, staggered close to a0
    idle_positions = [a0 + 5 + 3 * i for i in range(max(k, T_extra_idle))]
    idle_ids = list(range(1, 1 + len(idle_positions)))
    pos.extend(idle_positions)

    # filler tugs, far away, irrelevant
    filler_positions = [decoy_far + 200 + 5 * i for i in range(T_extra_filler)]
    pos.extend(filler_positions)

    T = len(pos)

    # corridor jobs: chain of n_chain jobs of size k, each a short tow, windows tight
    # enough that only an already-synced idle team (arriving together, no straggler) fits.
    cur = a0
    release = 0
    for i in range(n_chain):
        a = cur
        b = a + 5
        dur = 5
        # earliest an idle same-position-cluster team of size k could plausibly rendezvous:
        # enough slack for the idle-staggered team (max spread 3*k+2 ticks) to just make
        # it, but far too tight for a team stuck waiting on the decoy's late free time.
        slack = 3 * k + 3
        o = release
        c = o + dur + slack
        jobs.append(mk_job(a, b, k, release, 5, [(o, c)]))
        cur = b + (step - 5 if step > 5 else 2)
        release = o + slack  # next job becomes reachable right as this one would finish

    L = max(pos + [j["a"] for j in jobs] + [j["b"] for j in jobs]) + 20
    return T, len(jobs), L, pos, coeff, pen, jobs


def build(test_id):
    if test_id == 1:
        return generic_case(seed=101, T=4, N=3, L=100, max_k=2, coeff=1, pen=150)
    if test_id == 2:
        return generic_case(seed=202, T=5, N=4, L=130, max_k=2, coeff=1, pen=170)
    if test_id == 3:
        return trap_case(n_chain=2, k=2, decoy_gap=0, T_extra_idle=2, T_extra_filler=1)
    if test_id == 4:
        return generic_case(seed=404, T=6, N=5, L=160, max_k=3, coeff=1, pen=200)
    if test_id == 5:
        return trap_case(n_chain=3, k=2, decoy_gap=11, T_extra_idle=2, T_extra_filler=1, base=15)
    if test_id == 6:
        return generic_case(seed=606, T=7, N=6, L=180, max_k=3, coeff=1, pen=230)
    if test_id == 7:
        return trap_case(n_chain=3, k=3, decoy_gap=23, T_extra_idle=3, T_extra_filler=2, base=20)
    if test_id == 8:
        return generic_case(seed=808, T=9, N=8, L=240, max_k=4, coeff=1, pen=270)
    if test_id == 9:
        return trap_case(n_chain=4, k=2, decoy_gap=37, T_extra_idle=2, T_extra_filler=3, base=25)
    if test_id == 10:
        return generic_case(seed=1010, T=12, N=10, L=300, max_k=4, coeff=1, pen=300)
    # fallback for any extra ids the harness might request
    return generic_case(seed=1000 + test_id, T=6, N=5, L=150, max_k=3, coeff=1, pen=200)


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    T, N, L, pos, coeff, pen, jobs = build(test_id)
    print_instance(T, N, L, pos, coeff, pen, jobs)


if __name__ == "__main__":
    main()
