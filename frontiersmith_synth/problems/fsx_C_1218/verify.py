#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -> prints 'Ratio: <x in [0,1]>' (last Ratio: line authoritative).

Deterministic exact scorer for the wire-protocol-transcode shim.

Instance (see gen.py) gives:
  - NEW engine's M_NEW message-type affine transforms on a K_NEW = K_OLD+K_EXTRA vector
    (first K_OLD coords observable, last K_EXTRA internal scratch, reset to 0 at the
    start of translating each single OLD message).
  - OLD protocol's M_OLD message-type affine transforms on the K_OLD observable vector.
  - A set of client SESSIONS: an initial observable state + an ordered sequence of OLD
    message-type ids.

Participant output = a TRANSLATION TABLE: for each of the M_OLD old types, an emulation
sequence of 0..L_MAX NEW message-type ids to emit whenever that old type occurs.

Fidelity is judged on the OBSERVABLE trajectory only, replayed STATEFULLY: the shim's
carried-forward observable vector (not the "true" old-engine vector) feeds every
subsequent step, so one wrong emulation early in a session can cascade -- this is what
makes the OBJECTIVE genuinely session-stateful, not a set of independent snapshots.

Objective (per instance, summed over all sessions/checkpoints):
    F = (# checkpoints whose emulated observable vector exactly equals the true old-engine
         observable vector at that point)  -  CPX * (total extra NEW messages emitted beyond
         one-per-call, i.e. sum over invocations of max(0, L_used - 1))
Baseline B = same F computed for the trivial "always emit nothing" table (a legal, if inert,
translation). B is deterministic and > 0 because every session is forced to open with the
identity/handshake old type, which "emit nothing" always reproduces correctly.

Maximization normalization: sc = min(1000, 100*max(0,F)/max(1e-9,B)); Ratio = sc/1000.
Any feasibility / schema violation prints 'Ratio: 0.0' and exits 0.
"""
import sys
from fractions import Fraction

MAX_OUT_BYTES = 200_000


def fail(reason):
    sys.stdout.write("reason: %s\nRatio: 0.0\n" % reason)
    sys.exit(0)


def matvec_step(A, b, v, mod):
    n = len(v)
    r = len(A)
    out = [0] * r
    for i in range(r):
        Ai = A[i]
        s = 0
        for j in range(n):
            s += Ai[j] * v[j]
        out[i] = (s + b[i]) % mod
    return out


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    K_OLD = int(nxt()); K_NEW = int(nxt()); M_OLD = int(nxt()); M_NEW = int(nxt())
    P = int(nxt()); L_MAX = int(nxt()); CPX_NUM = int(nxt()); CPX_DEN = int(nxt())
    assert nxt() == "NEWTYPES"
    A_new, b_new = [], []
    for _ in range(M_NEW):
        A = [[int(nxt()) for _ in range(K_NEW)] for _ in range(K_NEW)]
        b = [int(nxt()) for _ in range(K_NEW)]
        A_new.append(A); b_new.append(b)
    assert nxt() == "OLDTYPES"
    A_old, b_old = [], []
    for _ in range(M_OLD):
        A = [[int(nxt()) for _ in range(K_OLD)] for _ in range(K_OLD)]
        b = [int(nxt()) for _ in range(K_OLD)]
        A_old.append(A); b_old.append(b)
    assert nxt() == "SESSIONS"
    num_sessions = int(nxt())
    sessions = []
    for _ in range(num_sessions):
        init = [int(nxt()) for _ in range(K_OLD)]
        length = int(nxt())
        seq = [int(nxt()) for _ in range(length)]
        sessions.append((init, seq))

    return dict(K_OLD=K_OLD, K_NEW=K_NEW, K_EXTRA=K_NEW - K_OLD, M_OLD=M_OLD, M_NEW=M_NEW,
                P=P, L_MAX=L_MAX, CPX=Fraction(CPX_NUM, CPX_DEN),
                A_new=A_new, b_new=b_new, A_old=A_old, b_old=b_old, sessions=sessions)


def parse_table(path, inst):
    import os
    try:
        sz = os.path.getsize(path)
    except Exception:
        fail("cannot stat output")
    if sz > MAX_OUT_BYTES:
        fail("output too large (%d bytes)" % sz)
    try:
        with open(path) as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")
    toks = raw.split()
    p = 0

    def nxt():
        nonlocal p
        if p >= len(toks):
            fail("truncated output")
        v = toks[p]
        p += 1
        return v

    def nxt_int():
        v = nxt()
        try:
            return int(v)
        except ValueError:
            fail("non-integer token: %r" % v)

    m_old_claim = nxt_int()
    if m_old_claim != inst["M_OLD"]:
        fail("declared M_OLD=%d != instance M_OLD=%d" % (m_old_claim, inst["M_OLD"]))

    table = []
    for t in range(inst["M_OLD"]):
        L = nxt_int()
        if L < 0 or L > inst["L_MAX"]:
            fail("type %d: emulation length %d out of [0,%d]" % (t, L, inst["L_MAX"]))
        seq = []
        for _ in range(L):
            s = nxt_int()
            if s < 0 or s >= inst["M_NEW"]:
                fail("type %d: new-type id %d out of range" % (t, s))
            seq.append(s)
        table.append(seq)

    if p != len(toks):
        fail("trailing garbage after translation table (%d extra tokens)" % (len(toks) - p))

    return table


def replay(inst, table):
    """Returns (raw_matches:int, extra_msgs:int, total_checks:int)."""
    P = inst["P"]; K_OLD = inst["K_OLD"]; K_EXTRA = inst["K_EXTRA"]
    A_new, b_new = inst["A_new"], inst["b_new"]
    A_old, b_old = inst["A_old"], inst["b_old"]

    raw_matches = 0
    extra_msgs = 0
    total_checks = 0

    for init, seq in inst["sessions"]:
        true_state = list(init)
        part_obs = list(init)
        for old_t in seq:
            true_state = matvec_step(A_old[old_t], b_old[old_t], true_state, P)

            emu = table[old_t]
            v = part_obs + [0] * K_EXTRA
            for s in emu:
                v = matvec_step(A_new[s], b_new[s], v, P)
            pred_obs = v[:K_OLD]

            total_checks += 1
            if pred_obs == true_state:
                raw_matches += 1
            if len(emu) > 1:
                extra_msgs += len(emu) - 1

            part_obs = pred_obs

    return raw_matches, extra_msgs, total_checks


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    inst = read_instance(inp)
    table = parse_table(outp, inst)

    matches, extra, total = replay(inst, table)
    F = Fraction(matches) - inst["CPX"] * extra
    if F < 0:
        F = Fraction(0)

    baseline_table = [[] for _ in range(inst["M_OLD"])]
    b_matches, b_extra, _ = replay(inst, baseline_table)
    B = Fraction(b_matches) - inst["CPX"] * b_extra  # b_extra is always 0

    sc = min(Fraction(1000), Fraction(100) * F / max(Fraction(1, 10**9), B))
    ratio = float(sc) / 1000.0
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0

    sys.stdout.write(
        "checkpoints=%d matches=%d extra_msgs=%d F=%s B=%s\nRatio: %.6f\n"
        % (total, matches, extra, F, B, ratio)
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
