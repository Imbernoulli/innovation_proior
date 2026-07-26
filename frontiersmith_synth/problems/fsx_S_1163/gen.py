#!/usr/bin/env python3
"""
gen.py <testId> -- emits one 'beacon-story-consistency-probes' instance to stdout.

Deterministic: all randomness seeded ONLY from testId.

INSTANCE MODEL
--------------
A telescope has D observation NIGHTS (1..D). Each night offers several bookable
POINTINGS (slots); you may book AT MOST ONE pointing per night, and AT MOST B
nights total (a schedule). Every pointing has a quality weight and a list of sky
sectors it observes.

There are K rival STORY FAMILIES. Each family holds M_f candidate narratives; a
narrative is pinned to one sector and carries a signature value. If a booked
pointing observes a narrative's sector, that pointing's READING for the narrative
is the signature value; otherwise the reading is 0 (quiet). A booked schedule
DISTINGUISHES two narratives of the same family iff at least one booked pointing
produces a different reading for the two.

Sector layout (fixed, deterministic, disjoint pools):
  sector 1                  (= AS) -> anchor sector: EVERY ordinary pointing sees it,
                                       so any reasonable schedule keeps every family's
                                       floor positive.
  sectors 2..1+FS_SIZE       (= FS) -> "wide-field" sectors: seen ONLY by ordinary
                                       pointings whose field of view is wide (their
                                       coverage size is large); the narrowest ordinary
                                       pointing on each night (the checker's own
                                       trivial baseline always uses exactly that one)
                                       never sees them.
  sectors after that, up to generic_pool -> the shared generic pool (other families).
  sectors generic_pool+1..N -> "reserved" sectors, one per family chosen HARD this test.

Weight is correlated with field-of-view size, so quality-greedy legitimately does well
on ordinary families (it ends up preferring WIDE pointings, which is exactly what is
needed to read the FS sectors). Each hard family's single reserved sector is observed
by exactly ONE low-weight, single-purpose pointing, parked on its OWN dedicated
"trap night" (no competing option that night) so it can never be crowded out by an
unrelated booking -- a quality/coverage scan simply never has a reason to visit it.
"""
import random
import sys

AS_SECTOR = 1     # anchor sector, present in every ordinary pointing's coverage
FS_SIZE = 5        # size of the "wide-field only" sector set
WIDE_MIN_SIZE = 4  # an ordinary pointing must draw >= this many random sectors to
                   # additionally see the wide-field set FS


def build(test_id: int):
    rng = random.Random(20000 + 7 * test_id)

    K = 4                                   # number of story families (fixed)
    hard_count = 1 if test_id < 7 else 2    # more adversarial on larger tests
    generic_extra = 60 + 6 * test_id        # extra shared-pool sectors (other families)
    H = hard_count

    fs_lo, fs_hi = 2, 1 + FS_SIZE
    generic_lo, generic_hi = fs_hi + 1, fs_hi + generic_extra
    N = generic_hi + H

    D = 9 + test_id                         # nights
    B = 4 + test_id // 2                    # booking budget (< D always)
    V = 5                                   # signature value range 1..V

    # family K-1 always has the LARGEST M_f (see below) and is deliberately never made
    # hard, so it stays a big, genuinely easy target a coverage-seeking strategy can win
    # on -- while a construction that mechanically always takes the narrowest pointing
    # (the checker's own trivial baseline) structurally never reaches its dedicated
    # wide-field sectors FS.
    base = (test_id - 1) % (K - 1)
    hard_families = {base} if hard_count == 1 else {base, (base + 1) % (K - 1)}

    # ---- reserve trap nights, one per hard family, each holding ONLY that family's
    # low-weight reserved-sector pointing (no competing ordinary option that night) ----
    trap_nights = list(range(D - H + 1, D + 1))
    ordinary_nights = D - H

    # ---- build ordinary slots (pointings), grouped by night ----
    night_slots = [[] for _ in range(D + 1)]
    for d in range(1, ordinary_nights + 1):
        # the night's narrowest pointing: size 1, low weight, no FS access -- this is
        # what the trivial/baseline construction always uses.
        w0 = 15 * 1 + rng.randint(5, 20)
        night_slots[d].append((w0, [AS_SECTOR]))
        n_opts = rng.randint(3, 5)
        for _ in range(n_opts):
            sz = rng.randint(2, 8)
            w = 15 * sz + rng.randint(5, 20)
            secs = set(rng.randint(generic_lo, generic_hi) for _ in range(sz))
            secs.add(AS_SECTOR)
            if sz >= WIDE_MIN_SIZE:
                secs.update(range(fs_lo, fs_hi + 1))
            night_slots[d].append((w, sorted(secs)))
        # guarantee at least one genuinely wide (FS-visible) option every night, so a
        # weight-driven scan always has abundant supply to prefer.
        sz = rng.randint(6, 8)
        w = 15 * sz + rng.randint(5, 20)
        secs = set(rng.randint(generic_lo, generic_hi) for _ in range(sz))
        secs.add(AS_SECTOR)
        secs.update(range(fs_lo, fs_hi + 1))
        night_slots[d].append((w, sorted(secs)))

    hard_reserved_sector = {}
    for j, f in enumerate(sorted(hard_families)):
        sec = generic_hi + 1 + j
        hard_reserved_sector[f] = sec
        d = trap_nights[j]
        w = rng.randint(4, 12)
        night_slots[d].append((w, [sec]))

    # ---- flatten into a global slot list: slot_id 1..T, night-ascending / emission order ----
    slots = []          # (slot_id, night, weight, sectors)
    sid = 0
    for d in range(1, D + 1):
        for (w, secs) in night_slots[d]:
            sid += 1
            slots.append((sid, d, w, secs))
    T = sid

    # ---- build narratives per family ----
    # every family: narratives 0,1 are the "anchor pair" (sector = AS_SECTOR, values
    # 1 and 2) -- reachable by ANY ordinary pointing, guaranteeing a positive floor.
    # remaining M_f - 2 narratives:
    #   family K-1 (always easy) -> drawn from the small wide-field set FS, so a
    #                  wide-pointing-seeking schedule resolves almost all of it, while
    #                  the narrow-only baseline resolves none of it.
    #   other easy families -> a random sector from the big shared generic pool,
    #                  random value -- ordinary, redundantly-covered, no special case.
    #   hard family -> the family's single reserved sector, split into two equal-size
    #                  value groups (value 1 / value 2) so within-group pairs are
    #                  PERMANENTLY confusable (no pointing can ever split them) while
    #                  cross-group pairs are split ONLY by the one reserved pointing.
    families = []
    for f in range(K):
        # family K-1 is ALWAYS given strictly more narratives than any other family's
        # cap (11) can reach, so its anchor-only floor value is STRICTLY lower than any
        # hard family's -- no coincidental ties at the M_f cap.
        Mf = (12 + test_id // 2) if f == K - 1 else min(11, 6 + f + test_id // 3)
        narrs = [(AS_SECTOR, 1), (AS_SECTOR, 2)]
        rest = Mf - 2
        if f in hard_families:
            sec = hard_reserved_sector[f]
            half = rest // 2
            for i in range(rest):
                val = 1 if i < half else 2
                narrs.append((sec, val))
        elif f == K - 1:
            for i in range(rest):
                sec = fs_lo + (i % FS_SIZE)
                val = 1 + (i % V)
                narrs.append((sec, val))
        else:
            for _ in range(rest):
                sec = rng.randint(generic_lo, generic_hi)
                val = rng.randint(1, V)
                narrs.append((sec, val))
        families.append(narrs)

    return N, D, K, B, T, slots, families


def emit(test_id: int):
    N, D, K, B, T, slots, families = build(test_id)
    out = [f"{N} {D} {K} {B} {T}"]
    for (sid, night, w, secs) in slots:
        out.append(f"{sid} {night} {w} {len(secs)} " + " ".join(map(str, secs)))
    for f in range(K):
        narrs = families[f]
        out.append(str(len(narrs)))
        for (sec, val) in narrs:
            out.append(f"{sec} {val}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    emit(tid)
