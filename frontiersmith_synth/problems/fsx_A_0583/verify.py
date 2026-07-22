#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the temple bell-ledger recovery task.

- Reads the test id t from the instance header, then REGENERATES the whole hidden
  ledger from t: the modulus m, the per-residue-class short recurrences, the clean
  latent counts a(1..N+K), and which lines were corrupted.  The hidden law lives
  ONLY here (and in gen.py); the sandboxed solver cannot read it.
- The participant outputs K integers: predictions for the held-out ledger lines
  a(N+1..N+K).  The held-out lines CONTINUE to be corrupted at the same rate, so a
  handful of them are unpredictable no matter how perfectly the law is recovered
  -- that irreducible slack keeps the score ceiling open.
- Per-line loss is the clamped relative error against the OBSERVED held-out ledger,
  normalised by the clean count magnitude:
        loss_i = min(1, |pred_i - obs(N+i)| / (1 + |clean(N+i)|))
  Objective  F = sum_i loss_i  (minimise).  Internal baseline  B = the loss of the
  trivial "predict 0" ledger.  Score = min(1000, 100*B/F)/1000, so predict-0 ~ 0.1
  and a perfect law-recovery is capped below 1.0 by the corrupted held-out lines.
"""
import sys, random
from fractions import Fraction

MAX_OUT_BYTES = 400000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------------- hidden ledger (identical to gen.py) ----------------
def hidden_build(t):
    rng = random.Random(918273117 + t * 2654435761)

    if t <= 3:
        m = 2
        N = 120
        train_corr = 3
    else:
        m = 3
        N = 168
        train_corr = 3 + (t % 3)
    K = 50
    c_out = 6

    while True:
        laws = []
        used_ab = set()
        ok = True
        for r in range(m):
            A = rng.choice([1, 1, 2])
            B = rng.choice([-1, 0, 1])
            if (A, B) in used_ab:
                ok = False
                break
            used_ab.add((A, B))
            C = rng.choice([x for x in range(-6, 7) if x != 0])
            laws.append((A, B, C))
        if not ok:
            continue
        a1 = rng.randint(3, 40)
        a2 = rng.randint(3, 40)
        latent = [0] * (N + K + 1)
        latent[1], latent[2] = a1, a2
        cap_ok = True
        for n in range(3, N + K + 1):
            A, B, C = laws[n % m]
            v = A * latent[n - 1] + B * latent[n - 2] + C
            latent[n] = v
            if abs(v) > 10 ** 9:
                cap_ok = False
                break
        if not cap_ok:
            continue
        mx = max(abs(latent[n]) for n in range(1, N + K + 1))
        if mx < 2000:
            continue
        break

    observed = latent[:]
    pos_train = sorted(rng.sample(range(3, N + 1), train_corr))
    pos_out = sorted(rng.sample(range(N + 1, N + K + 1), c_out))
    for p in pos_train + pos_out:
        base = 1 + abs(latent[p])
        offset = base + rng.randint(0, 3 + abs(latent[p]))
        sign = rng.choice([-1, 1])
        observed[p] = latent[p] + sign * offset

    return N, K, m, laws, latent, observed


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[2])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000:
        fail("bad test id")

    N, K, m, laws, latent, observed = hidden_build(t)

    # ---- read participant predictions: exactly K finite integers ----
    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    toks = raw.decode("utf-8", "replace").split()
    if len(toks) != K:
        fail("expected %d integers, got %d" % (K, len(toks)))
    preds = []
    for tk in toks:
        try:
            v = int(tk)               # strict integer (rejects nan/inf/floats)
        except ValueError:
            fail("non-integer token %r" % tk[:16])
        preds.append(v)

    # ---- scoring: clamped relative error vs the observed held-out ledger ----
    F = 0.0
    B = 0.0
    for i in range(K):
        n = N + 1 + i
        target = observed[n]
        scale = 1 + abs(latent[n])
        F += min(1.0, abs(preds[i] - target) / scale)
        B += min(1.0, abs(0 - target) / scale)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f m=%d N=%d K=%d  Ratio: %.6f"
          % (F, B, m, N, K, sc / 1000.0))


if __name__ == "__main__":
    main()
