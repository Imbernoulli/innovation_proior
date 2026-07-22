#!/usr/bin/env python3
"""
gen.py <testId> -- Archivist's Compaction Cadence (fsx_A_0741)

Emits ONE instance to stdout:

  N M
  s_1 lo_1 hi_1        (box i: letter count, catalogue-key range)
  ...
  s_N lo_N hi_N
  T
  ev_1
  ...
  ev_T                 each line is either "I" (next box, in order, arrives)
                        or "L q" (a reading-room request for catalogue key q)

Exactly N "I" lines and M "L" lines appear among the T = N+M event lines, in
chronological order. Deterministic: all randomness seeded from testId only.

Every case shares one structural bias, because it is what makes both
mechanisms (write-read amplification duality, lookup-hotness windows) bite at
once: most boxes' catalogue ranges deliberately overlap a shared "popular"
band, and most reading-room requests target that band, so an UNMERGED
collection genuinely costs many probes per request, while a merge genuinely
costs real bytes. Timing (windows vs quiet write-only stretches) then decides
who wins.

testId 1..10 is a difficulty ladder. Cases 3, 6, 9 are TRAP cases: a fixed
small seed of boxes overlapping the popular band arrives, THEN a long
write-only stretch of unrelated narrow boxes (box count grows past any
plausible fixed threshold with zero read payoff) is interleaved with three
separated bursts of requests on the popular band. A box-count-triggered
policy either fires early (wasting merges on the quiet stretch) or fires late
(missing the first burst); it cannot express "merge now because a burst is
imminent, then hold."
"""
import sys
import random


def emit(N, M, boxes, events):
    out = []
    out.append(f"{N} {M}")
    for (s, lo, hi) in boxes:
        out.append(f"{s} {lo} {hi}")
    out.append(str(len(events)))
    for e in events:
        out.append(e)
    sys.stdout.write("\n".join(out) + "\n")


def make_popular_band(rng, MAXKEY):
    band_w = int(MAXKEY * rng.uniform(0.18, 0.26))
    band_c = rng.randint(MAXKEY // 3, 2 * MAXKEY // 3)
    blo = max(1, band_c - band_w // 2)
    bhi = min(MAXKEY, blo + band_w)
    blo = max(1, bhi - band_w)
    return blo, bhi


def hot_box(rng, MAXKEY, blo, bhi):
    """A box whose range FULLY covers the popular band (plus random jitter
    past its edges), so it matches every lookup that lands in the band --
    this is what makes overlapping unmerged boxes genuinely expensive to
    probe, and what a merge genuinely consolidates."""
    extra_lo = int((bhi - blo) * rng.uniform(0.03, 0.30))
    extra_hi = int((bhi - blo) * rng.uniform(0.03, 0.30))
    lo = max(1, blo - rng.randint(0, extra_lo + 1))
    hi = min(MAXKEY, bhi + rng.randint(0, extra_hi + 1))
    hi = max(hi, lo + 1)
    return (rng.randint(6, 14), lo, hi)


def cold_box(rng, MAXKEY, blo, bhi):
    """A narrow box that avoids the popular band entirely."""
    width = rng.randint(max(2, int(MAXKEY * 0.015)), max(3, int(MAXKEY * 0.035)))
    for _ in range(60):
        center = rng.randint(1, MAXKEY)
        lo = max(1, center - width // 2)
        hi = min(MAXKEY, lo + width)
        lo = max(1, hi - width)
        if hi < blo - 1 or lo > bhi + 1:
            return (rng.randint(2, 6), lo, hi)
    # fallback: clamp to just below the band
    hi = max(1, blo - 2)
    lo = max(1, hi - width)
    return (rng.randint(2, 6), lo, hi)


def build_case(rng, N, M, MAXKEY, n_hot, n_bursts, seed_cold, scatter_frac):
    """Shared construction for every test: a FIXED small seed of boxes
    overlapping a popular catalogue band arrives, then write-only stretches
    of narrow, unrelated boxes are interleaved with `n_bursts` separated
    bursts of requests on the band. `n_hot`/`seed_cold` (independent of N)
    control how early real overlap exists relative to any plausible fixed
    box-count threshold; `scatter_frac` controls how much of the filler
    traffic is genuinely off-band noise (keeping instances honest) versus
    still-band background load. Larger n_bursts / smaller seed_cold / lower
    scatter_frac make the timing decision matter more (harder cases)."""
    blo, bhi = make_popular_band(rng, MAXKEY)

    n_hot = min(n_hot, max(2, N - 2))
    n_cold = N - n_hot

    box_specs = [None] * N
    for k in range(n_hot):
        box_specs[k] = hot_box(rng, MAXKEY, blo, bhi)
    for k in range(n_hot, N):
        box_specs[k] = cold_box(rng, MAXKEY, blo, bhi)
    boxes = box_specs

    burst_len = max(5, int(M * 0.72) // n_bursts)
    used_lookups = n_bursts * burst_len
    filler = max(0, M - used_lookups)

    events = []
    seed_cold = min(n_cold, seed_cold)
    ptr_cold = 0
    for _ in range(n_hot):
        events.append("I")
    for _ in range(seed_cold):
        events.append("I")
        ptr_cold += 1

    for b in range(n_bursts):
        for _ in range(burst_len):
            events.append(f"L {rng.randint(blo, bhi)}")
        if b < n_bursts - 1:
            remaining_bursts = n_bursts - 1 - b
            batch = min(n_cold - ptr_cold,
                        max(1, (n_cold - ptr_cold) // max(1, remaining_bursts)))
            for _ in range(batch):
                events.append("I")
                ptr_cold += 1
    while ptr_cold < n_cold:
        events.append("I")
        ptr_cold += 1
    for _ in range(filler):
        # background noise: a controllable mix of still-band load (keeps
        # a lingering merge earning its keep) and genuinely off-band
        # queries (keeps the instance honest -- not everything is hot).
        if rng.random() < scatter_frac:
            events.append(f"L {rng.randint(1, MAXKEY)}")
        else:
            events.append(f"L {rng.randint(blo, bhi)}")

    assert sum(1 for e in events if e == "I") == N
    return boxes, events


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    testId = int(sys.argv[1])
    rng = random.Random(741000 + 97 * testId)

    N = 8 + testId * 2          # 10 .. 28
    M = 30 + testId * 16        # 46 .. 190
    MAXKEY = 240 + testId * 24

    if testId in (3, 6, 9):
        # TRAP: tiny fixed hot seed, many bursts, almost no filler slack --
        # a count-threshold policy cannot time its one merge correctly.
        n_hot, n_bursts, seed_cold, scatter_frac = 4, 4, 1, 0.15
    else:
        # milder but still real timing signal: a small hot seed, several
        # bursts, modest slack, some off-band noise.
        n_hot = 5
        n_bursts, seed_cold, scatter_frac = 3, 2, 0.30

    boxes, events = build_case(rng, N, M, MAXKEY, n_hot, n_bursts, seed_cold, scatter_frac)
    emit(N, M, boxes, events)


if __name__ == "__main__":
    main()
