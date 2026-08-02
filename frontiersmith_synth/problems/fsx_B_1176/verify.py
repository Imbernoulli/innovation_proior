#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for fsx_B_1176.
Family: load-disaggregate-meter (format C, maximize).

Re-derives the SAME hidden per-appliance TRUE state sequences that gen.py
built (identical deterministic construction, keyed only by the appliance
library + trace length read from <in> -- gen.py never prints the hidden
sequences, so this duplication is the only channel; nothing importable is
shipped to solvers).
"""
import sys
import math
import random

MAX_TOKENS = 20000


# ---- identical deterministic construction to gen.py (kept in sync by hand) ----
def simulate_hidden(T, archs, test_id_hint):
    """Rebuild the hidden truth from (T, archs) using the SAME seed-retry rule
    as gen.py. test_id_hint is recovered from the instance itself (T, A, the
    per-appliance params, and wT/wA together uniquely identify the testId in
    this fixed 10-case table), so no separate channel is needed."""
    attempt = 0
    while True:
        seed = 20000 + 137 * test_id_hint + 991 * attempt
        rng = random.Random(seed)
        cand = []
        for (P, mon, mxon, moff, mxoff) in archs:
            seq = []
            state = 0
            while len(seq) < T:
                d = rng.randint(moff, mxoff) if state == 0 else rng.randint(mon, mxon)
                d = min(d, T - len(seq))
                seq.extend([state] * d)
                state = 1 - state
            cand.append(seq[:T])
        seen = set()
        collide = False
        for seq in cand:
            for t in range(1, T):
                if seq[t] != seq[t - 1]:
                    if t in seen:
                        collide = True
                        break
                    seen.add(t)
            if collide:
                break
        if not collide:
            return cand
        attempt += 1
        if attempt > 300:
            return cand


ARCH = {
    "FRIDGE":    (150,  5, 8,   8, 14),
    "KETTLE":    (2000, 2, 4,  10, 18),
    "WASHER":    (500,  7, 11, 12, 20),
    "DRYER":     (700,  6, 9,   7, 11),
    "MICROWAVE": (1200, 2, 4,  16, 24),
    "HEATER":    (1200, 11, 15, 3, 5),
    "POOLPUMP":  (900,  2, 4,  10, 15),
    "ACCOMP":    (900,  10, 14, 3, 5),
}
TABLE = {
    1:  (36, ["FRIDGE", "MICROWAVE", "HEATER"], 2.0, 1.0),
    2:  (40, ["FRIDGE", "POOLPUMP", "ACCOMP"], 1.8, 1.2),
    3:  (42, ["FRIDGE", "MICROWAVE", "HEATER", "WASHER"], 2.0, 1.3),
    4:  (44, ["FRIDGE", "MICROWAVE", "HEATER", "KETTLE"], 1.8, 1.4),
    5:  (46, ["MICROWAVE", "HEATER", "POOLPUMP", "ACCOMP", "FRIDGE"], 2.0, 1.5),
    6:  (48, ["MICROWAVE", "HEATER", "POOLPUMP", "ACCOMP", "KETTLE"], 1.7, 1.6),
    7:  (50, ["MICROWAVE", "HEATER", "POOLPUMP", "ACCOMP", "WASHER"], 1.6, 1.7),
    8:  (54, ["MICROWAVE", "HEATER", "DRYER", "POOLPUMP", "ACCOMP", "FRIDGE"], 1.8, 1.8),
    9:  (58, ["MICROWAVE", "HEATER", "DRYER", "POOLPUMP", "ACCOMP", "KETTLE"], 1.6, 2.0),
    10: (62, ["MICROWAVE", "HEATER", "DRYER", "POOLPUMP", "ACCOMP", "WASHER"], 1.5, 2.2),
}


def find_test_id(T, archs, wT, wA):
    """The instance (T, per-appliance params, wT, wA) uniquely determines which
    of the 10 fixed table rows produced it -- recover testId by exact match."""
    for tid, (Tt, names, wTt, wAt) in TABLE.items():
        if Tt != T or abs(wTt - wT) > 1e-6 or abs(wAt - wA) > 1e-6:
            continue
        a2 = [ARCH[nm] for nm in names]
        if a2 == archs:
            return tid
    return None


def read_instance(path):
    toks = open(path).read().split()
    ptr = 0
    T = int(toks[ptr]); ptr += 1
    A = int(toks[ptr]); ptr += 1
    wT = float(toks[ptr]); ptr += 1
    wA = float(toks[ptr]); ptr += 1
    archs = []
    for _ in range(A):
        P = int(toks[ptr]); mon = int(toks[ptr + 1]); mxon = int(toks[ptr + 2])
        moff = int(toks[ptr + 3]); mxoff = int(toks[ptr + 4]); ptr += 5
        archs.append((P, mon, mxon, moff, mxoff))
    aggregate = [int(toks[ptr + i]) for i in range(T)]
    ptr += T
    return T, A, wT, wA, archs, aggregate


def parse_output(text, T, A):
    toks = text.split()
    if len(toks) == 0:
        return None, "empty output"
    if len(toks) > MAX_TOKENS:
        return None, "too many tokens"
    try:
        vals = [int(t) for t in toks]
    except ValueError:
        return None, "non-integer token (nan/inf/garbage)"
    if len(vals) != 1 + A * T:
        return None, f"expected {1 + A * T} tokens, got {len(vals)}"
    if vals[0] != A:
        return None, "A mismatch"
    seqs = []
    ptr = 1
    for a in range(A):
        seq = vals[ptr:ptr + T]; ptr += T
        for v in seq:
            if v not in (0, 1):
                return None, f"appliance {a}: state token {v} not in {{0,1}}"
        seqs.append(seq)
    return seqs, "ok"


def check_legal(seq, mon, mxon, moff, mxoff):
    n = len(seq)
    runs = []
    i = 0
    while i < n:
        j = i
        while j < n and seq[j] == seq[i]:
            j += 1
        runs.append((seq[i], j - i))
        i = j
    for idx, (st, dur) in enumerate(runs):
        lo, hi = (moff, mxoff) if st == 0 else (mon, mxon)
        if dur > hi:
            return False, f"run{idx} state{st} dur{dur}>max{hi}"
        is_boundary = (idx == 0) or (idx == len(runs) - 1)
        if not is_boundary and dur < lo:
            return False, f"run{idx} state{st} dur{dur}<min{lo}"
    return True, "ok"


def evaluate(seqs, archs, hidden, aggregate, T, wT, wA):
    recon = [0] * T
    for (P, *_rest), seq in zip(archs, seqs):
        for t in range(T):
            if seq[t] == 1:
                recon[t] += P
    resid = sum(abs(a - b) for a, b in zip(aggregate, recon))
    denom = max(1, sum(aggregate))
    trace_fit_raw = max(0.0, 1.0 - resid / denom)
    accs = []
    for htrue, sseq in zip(hidden, seqs):
        m = sum(1 for a, b in zip(htrue, sseq) if a == b)
        accs.append(m / T)
    acc_raw = sum(accs) / len(accs)
    # Both raw fractions have an inherent floor well above 0 (an OFF/ON
    # sequence with ~40-60% duty cycle overlaps a phase-blind construction on
    # roughly half its samples "by luck", and small residual errors already
    # cancel a lot of the L1 gap) -- a linear score would compress every
    # construction into a narrow high band near that floor, leaving no room
    # to separate a mediocre recipe from a genuinely correct reconstruction.
    # Squaring/rescaling super-linearly punishes that floor while still
    # letting a true match (trace_fit_raw, acc_raw -> 1) saturate at 1.
    trace_fit = trace_fit_raw * trace_fit_raw
    mean_acc = max(0.0, 2.0 * acc_raw - 1.0)
    F = wT * trace_fit + wA * mean_acc
    return F, trace_fit, mean_acc


def baseline_seqs(archs, T):
    """Checker's own naive reference: each appliance cycles at its own MINIMUM
    legal dwell (min_off then min_on, repeating), ignoring the observed trace
    entirely. Always legal (every interior run == the min bound)."""
    out = []
    for (P, mon, mxon, moff, mxoff) in archs:
        seq = []
        state = 0
        while len(seq) < T:
            d = moff if state == 0 else mon
            d = min(d, T - len(seq))
            seq.extend([state] * d)
            state = 1 - state
        out.append(seq[:T])
    return out


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    inf, outf = sys.argv[1], sys.argv[2]
    T, A, wT, wA, archs, aggregate = read_instance(inf)

    text = open(outf).read()
    seqs, reason = parse_output(text, T, A)
    if seqs is None:
        print(f"infeasible: {reason}")
        print("Ratio: 0.0")
        return 0

    for a, (P, mon, mxon, moff, mxoff) in enumerate(archs):
        ok, why = check_legal(seqs[a], mon, mxon, moff, mxoff)
        if not ok:
            print(f"infeasible: appliance {a} {why}")
            print("Ratio: 0.0")
            return 0

    test_id = find_test_id(T, archs, wT, wA)
    if test_id is None:
        print("infeasible: instance not recognized")
        print("Ratio: 0.0")
        return 0
    hidden = simulate_hidden(T, archs, test_id)

    F, trace_fit, mean_acc = evaluate(seqs, archs, hidden, aggregate, T, wT, wA)
    if not (math.isfinite(F) and math.isfinite(trace_fit) and math.isfinite(mean_acc)):
        print("non-finite objective")
        print("Ratio: 0.0")
        return 0

    base = baseline_seqs(archs, T)
    B, Bt, Ba = evaluate(base, archs, hidden, aggregate, T, wT, wA)
    B = max(B, 1e-6)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print(f"F={F:.4f} traceFit={trace_fit:.4f} meanAcc={mean_acc:.4f} baseline={B:.4f}")
    print("Ratio: %.6f" % (sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
