#!/usr/bin/env python3
"""
verify.py <in> <out> <ans> -- deterministic checker for beacon-story-consistency-probes.

Reads the instance, validates the participant's booked schedule strictly, computes the
worst-family narrative-pair separation fraction F, normalizes against the checker's own
internal baseline construction B (nights 1..B ascending, each night's FIRST-listed / common
pointing), and prints the final score as "Ratio: <float in [0,1]>".
"""
import sys


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    it = iter(toks)

    def nxt():
        return next(it)

    N = int(nxt()); D = int(nxt()); K = int(nxt()); B = int(nxt()); T = int(nxt())
    slots = {}          # slot_id -> (night, weight, set(sectors))
    night_first = {}     # night -> first-encountered slot_id (common pointing)
    for _ in range(T):
        sid = int(nxt()); night = int(nxt()); w = int(nxt()); ns = int(nxt())
        secs = set(int(nxt()) for _ in range(ns))
        slots[sid] = (night, w, secs)
        if night not in night_first:
            night_first[night] = sid
    families = []
    for _ in range(K):
        Mf = int(nxt())
        narrs = []
        for _ in range(Mf):
            sec = int(nxt()); val = int(nxt())
            narrs.append((sec, val))
        families.append(narrs)
    return {"N": N, "D": D, "K": K, "B": B, "T": T, "slots": slots,
            "night_first": night_first, "families": families}


def reading(secs, sector, val):
    return val if sector in secs else 0


def worst_family_frac(inst, chosen_ids):
    chosen_secs = [inst["slots"][sid][2] for sid in chosen_ids]
    worst = None
    for narrs in inst["families"]:
        m = len(narrs)
        total = m * (m - 1) // 2
        if total == 0:
            continue
        sep = 0
        for i in range(m):
            si, vi = narrs[i]
            for j in range(i + 1, m):
                sj, vj = narrs[j]
                distinguished = False
                for secs in chosen_secs:
                    if reading(secs, si, vi) != reading(secs, sj, vj):
                        distinguished = True
                        break
                if distinguished:
                    sep += 1
        frac = sep / total
        if worst is None or frac < worst:
            worst = frac
    return 0.0 if worst is None else worst


def baseline_schedule(inst):
    B = inst["B"]; D = inst["D"]
    nights = list(range(1, min(B, D) + 1))
    return [inst["night_first"][d] for d in nights if d in inst["night_first"]]


def fail(msg):
    print("INVALID: %s Ratio: 0.0" % msg)
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]
    inst = read_instance(in_path)
    T = inst["T"]; B = inst["B"]

    with open(out_path, "r") as f:
        raw = f.read()
    toks = raw.split()
    if not toks:
        fail("empty output")

    try:
        m = int(toks[0])
    except ValueError:
        fail("first token not an integer")
    if m < 0 or m > B:
        fail("booked count out of [0,B]")
    if len(toks) != 1 + m:
        fail("token count mismatch (expected %d, got %d)" % (1 + m, len(toks)))

    chosen = []
    for tk in toks[1:]:
        try:
            v = int(tk)
        except ValueError:
            fail("non-integer slot id %r" % tk)
        chosen.append(v)

    if len(set(chosen)) != len(chosen):
        fail("duplicate slot id")
    for sid in chosen:
        if sid < 1 or sid > T:
            fail("slot id %d out of range [1,%d]" % (sid, T))

    nights_used = [inst["slots"][sid][0] for sid in chosen]
    if len(set(nights_used)) != len(nights_used):
        fail("two booked slots share the same night")

    F = worst_family_frac(inst, chosen)

    base_ids = baseline_schedule(inst)
    Fbase = worst_family_frac(inst, base_ids)

    sc = min(1000.0, 100.0 * F / max(1e-9, Fbase))
    print("F=%.6f Fbase=%.6f booked=%d Ratio: %.6f" % (F, Fbase, m, sc / 1000.0))


if __name__ == "__main__":
    main()
