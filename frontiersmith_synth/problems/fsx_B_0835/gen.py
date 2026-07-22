#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE instance of "triage-reagent-sharing" to stdout.
Deterministic: all structure is a pure function of testId (no wall-clock, no OS entropy).
"""
import sys


def eval_dag(instrs, F):
    """instrs: list of (op,a,b) token-triples. F: list of raw feature ints.
    Returns list of computed integer values, one per instruction, in order."""
    vals = []
    def resolve(tok):
        if tok[0] == 'F':
            return F[int(tok[1:])]
        if tok[0] == 'I':
            return vals[int(tok[1:])]
        if tok[0] == 'C':
            return int(tok[1:])
        raise ValueError(tok)
    for (op, a, b) in instrs:
        va, vb = resolve(a), resolve(b)
        if op == 'ADD':
            v = va + vb
        elif op == 'SUB':
            v = va - vb
        elif op == 'MUL':
            v = va * vb
        else:
            raise ValueError(op)
        vals.append(v)
    return vals


def shift(instrs, offset):
    """Rewrite local I-refs to absolute (global-table) indices by adding offset."""
    out = []
    for (op, a, b) in instrs:
        if a[0] == 'I':
            a = 'I%d' % (int(a[1:]) + offset)
        if b[0] == 'I':
            b = 'I%d' % (int(b[1:]) + offset)
        out.append((op, a, b))
    return out


def build_group_A(PA):
    """Two required tests t0,t1 that share a PA-instruction prep prefix built from
    raw features F0,F1. Returns instrs of length PA+2 (last two = t0-extra, t1-extra).
    Uses only feature-refs and constant-scalar MUL so the whole chain stays affine
    in (F0,F1), which keeps the generator's target-search below well-behaved."""
    instrs = []
    instrs.append(('MUL', 'F0', 'C3'))       # I0 = 3*F0
    instrs.append(('ADD', 'I0', 'F1'))       # I1 = 3*F0 + F1
    k = 2
    pad_ops = ['ADD', 'SUB', 'MUL']
    pad_consts = [2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 2, 1]
    ci = 0
    while k < PA:
        op = pad_ops[k % 3]
        c = pad_consts[ci % len(pad_consts)]
        ci += 1
        if op == 'MUL':
            c = 2 if c > 2 else max(c, 2)  # keep multiplicative growth mild
        instrs.append((op, 'I%d' % (k - 1), 'C%d' % c))
        k += 1
    assert len(instrs) == PA
    instrs.append(('ADD', 'I%d' % (PA - 1), 'F0'))  # index PA    -> t0 extra
    instrs.append(('ADD', 'I%d' % (PA - 1), 'F1'))  # index PA+1  -> t1 extra
    return instrs


def build_feature_chain(L, feat_tok):
    """Standalone L-instruction chain, monotonic increasing in the single feature
    feat_tok (e.g. 'F2'), never sharing anything with group A."""
    instrs = [('ADD', feat_tok, 'C1')]
    pad_ops = ['ADD', 'MUL', 'SUB', 'ADD']
    pad_consts = [3, 2, 1, 4, 2, 1, 3, 2, 1, 4, 2, 1, 3, 2, 1, 4, 2, 1, 3, 2]
    ci = 0
    while len(instrs) < L:
        op = pad_ops[len(instrs) % len(pad_ops)]
        c = pad_consts[ci % len(pad_consts)]
        ci += 1
        if op == 'MUL':
            c = 2  # keep growth mild but strictly positive => stays monotonic increasing
        instrs.append((op, 'I%d' % (len(instrs) - 1), 'C%d' % c))
    return instrs[:L]


def build_constant_chain(L, base):
    """Standalone L-instruction chain that touches NO raw feature at all -> its
    value (and hence its test outcome) is the SAME for every patient. Zero
    information gain, but a full-scan baseline still pays for it."""
    instrs = [('ADD', 'C%d' % base, 'C%d' % (base + 1))]
    pad_ops = ['ADD', 'SUB', 'MUL', 'ADD']
    pad_consts = [5, 3, 2, 7, 4, 2, 5, 3, 2, 7, 4, 2, 5, 3, 2, 7, 4, 2]
    ci = 0
    while len(instrs) < L:
        op = pad_ops[len(instrs) % len(pad_ops)]
        c = pad_consts[ci % len(pad_consts)]
        ci += 1
        if op == 'MUL':
            c = 1  # keep it bounded; still a genuine extra instruction
        instrs.append((op, 'I%d' % (len(instrs) - 1), 'C%d' % c))
    return instrs[:L]


def search_group_A_features(instrs_full, PA, rng_range):
    """For each of the 4 (b0,b1) target combos, find integer (F0,F1) that realizes it,
    reading off outcomes from instruction indices PA (t0) and PA+1 (t1), threshold 0."""
    found = {}
    for f0 in range(-rng_range, rng_range + 1):
        for f1 in range(-rng_range, rng_range + 1):
            vals = eval_dag(instrs_full, [f0, f1, 0])
            b0 = 1 if vals[PA] >= 0 else 0
            b1 = 1 if vals[PA + 1] >= 0 else 0
            key = (b0, b1)
            if key not in found:
                found[key] = (f0, f1)
        if len(found) == 4:
            break
    assert len(found) == 4, "group A search failed to realize all 4 combos"
    return found


def search_decoy2_feature(instrs_full, decoy2_final_idx, target_bit, rng_range, f0f1):
    for f2 in range(-rng_range, rng_range + 1):
        F = [f0f1[0], f0f1[1], f2]
        vals = eval_dag(instrs_full, F)
        bit = 1 if vals[decoy2_final_idx] >= 0 else 0
        if bit == target_bit:
            return f2
    raise AssertionError("decoy2 search failed")


# ---- per-testId ladder parameters ----
def params_for(tid):
    PA = 6 + (tid - 1) // 2                 # 6,6,7,7,8,8,9,9,10,10
    L2 = PA + 10 + (tid - 1)                # decoy2 grows faster -> stays "expensive"
    L3 = 10 + tid                            # constant decoy #1
    L4 = 9 + tid                             # constant decoy #2
    repeats = 1 + (tid - 1) // 2             # 1,1,2,2,3,3,4,4,5,5
    wscale = tid                             # scales total population
    return PA, L2, L3, L4, repeats, wscale


def main():
    tid = int(sys.argv[1])
    PA, L2, L3, L4, repeats, wscale = params_for(tid)

    groupA = build_group_A(PA)
    off2 = PA + 2
    decoy2_local = build_feature_chain(L2, 'F2')
    decoy2 = shift(decoy2_local, off2)
    off3 = off2 + L2
    decoy3_local = build_constant_chain(L3, 11 + tid)
    decoy3 = shift(decoy3_local, off3)
    off4 = off3 + L3
    decoy4_local = build_constant_chain(L4, 31 + tid)
    decoy4 = shift(decoy4_local, off4)

    instrs = groupA + decoy2 + decoy3 + decoy4
    M = len(instrs)
    K = 3  # F0, F1, F2

    t0_final, t1_final = PA, PA + 1
    t2_final = off2 + L2 - 1
    t3_final = off3 + L3 - 1
    t4_final = off4 + L4 - 1

    # search group-A (F0,F1) realizations for the 4 combos
    combo_feats = search_group_A_features(instrs, PA, rng_range=60)

    # decoy2 outcome per combo MUST equal XOR(b0,b1): fully redundant given t0,t1
    decoy2_f2 = {}
    for (b0, b1), (f0, f1) in combo_feats.items():
        target = b0 ^ b1
        decoy2_f2[(b0, b1)] = search_decoy2_feature(instrs, t2_final, target, rng_range=80, f0f1=(f0, f1))

    # constant decoys: fix threshold so outcome is always 0, for ANY features
    val3 = eval_dag(instrs, [0, 0, 0])[t3_final]
    thr3 = val3 + 1000003
    val4 = eval_dag(instrs, [0, 0, 0])[t4_final]
    thr4 = val4 + 1000007

    thr0, thr1, thr2 = 0, 0, 0

    tests = [
        (t0_final, thr0),
        (t1_final, thr1),
        (t2_final, thr2),
        (t3_final, thr3),
        (t4_final, thr4),
    ]
    T = len(tests)

    label_of = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

    # weight recipe: (1,15,15,69) tuned so info-gain(decoy2/XOR) > info-gain(t0) ~= info-gain(t1) > 0,
    # while gains of the two constant decoys are exactly 0. Split each combo's total weight across
    # `repeats` duplicate (same-feature) patients so N grows with the ladder.
    base_w = {(0, 0): 1, (0, 1): 15, (1, 0): 15, (1, 1): 69}

    patients = []  # (F0,F1,F2, weight, label)
    for combo, bw in base_w.items():
        f0, f1 = combo_feats[combo]
        f2 = decoy2_f2[combo]
        total_w = bw * wscale
        # split total_w into `repeats` positive integer parts
        base = total_w // repeats
        rem = total_w % repeats
        for i in range(repeats):
            w = base + (1 if i < rem else 0)
            if w <= 0:
                w = 1
            patients.append((f0, f1, f2, w, label_of[combo]))
    N = len(patients)

    out = []
    out.append("%d %d %d %d" % (K, M, T, N))
    for (op, a, b) in instrs:
        out.append("%s %s %s" % (op, a, b))
    for (final_idx, thr) in tests:
        out.append("%d %d" % (final_idx, thr))
    for (f0, f1, f2, w, lab) in patients:
        out.append("%d %d %d %d %d" % (f0, f1, f2, w, lab))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
