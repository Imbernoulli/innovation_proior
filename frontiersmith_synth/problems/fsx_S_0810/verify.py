#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for "Infer the substitution behind an observed word".

- Reads (t, n_train) from <in>'s header, then regenerates the hidden
  substitution sigma (and its incidence matrix M) entirely from t via the
  SAME make_morphism() as gen.py -- the hidden law lives only here + gen.py.
- Parses the participant's guessed morphism: exactly K=3 non-empty lines,
  line i = the guessed image of letter i, each a nonempty string over
  {0,1,2} of length <= LMAX_OUT. Any violation -> Ratio: 0.0.
- Scores by EXACT extrapolation: at three held-out horizons well beyond
  n_train (n_train+10, +18, +26; never told to the training side), the
  grader compares the true letter-count vector c_true = M^n[axiom] against
  the guessed c_guess = M_guess^n[axiom], computed by exact integer matrix
  exponentiation (fast doubling) -- no string is ever materialised at the
  held-out horizon, so this is exact and O(log n).
  For each horizon:
      F_len  = exp(-|log|w_guess| - log|w_true|| / SCALE_LEN)   (growth rate)
      F_freq = max(0, 1 - L1(freq_guess, freq_true) / 2)         (letter mix)
      F_h    = 0.5*F_len + 0.5*F_freq
  F = mean over the 3 horizons.
  Ratio = min(1000, 100*F/BASELINE_B) / 1000, with a FIXED calibration
  constant BASELINE_B (an "assume no further growth" guess reproduces
  ~0.1..0.2; an exact reconstruction of sigma tops out at ~0.85, leaving
  headroom below 1.0).
"""
import sys, math, random

K = 3
ALPHABET = "012"
LMAX_TRUE = 4
LMAX_OUT = 12
BASE_SEED = 913461
MAX_TRIES = 500
OFFSETS = (10, 18, 26)
SCALE_LEN = 6.0
BASELINE_B = 0.117647058823529  # = 0.1 / 0.85
MAX_OUT_BYTES = 200000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden substitution (identical to gen.py) ----------
def n_train(t):
    return 5 + (t - 1) % 4


def _bool_matmul(A, B):
    C = [[0] * K for _ in range(K)]
    for i in range(K):
        for k in range(K):
            if A[i][k]:
                for j in range(K):
                    if B[k][j]:
                        C[i][j] = 1
    return C


def _primitive(M):
    Bm = [[1 if M[i][j] > 0 else 0 for j in range(K)] for i in range(K)]
    P = [row[:] for row in Bm]
    for _ in range(K * K + 2):
        if all(all(x > 0 for x in row) for row in P):
            return True
        P = _bool_matmul(P, Bm)
    return all(all(x > 0 for x in row) for row in P)


def _spectral_radius(M, iters=400):
    v = [1.0 / K] * K
    for _ in range(iters):
        w = [sum(M[i][j] * v[j] for j in range(K)) for i in range(K)]
        s = sum(w)
        if s <= 0:
            return 0.0
        v = [x / s for x in w]
    w = [sum(M[i][j] * v[j] for j in range(K)) for i in range(K)]
    return sum(w)


def apply_morphism(images, s):
    return ''.join(images[int(c)] for c in s)


def build_levels(images, axiom, n):
    levels = [str(axiom)]
    for _ in range(n):
        levels.append(apply_morphism(images, levels[-1]))
    return levels


def make_morphism(t):
    nt = n_train(t)
    tries = 0
    while True:
        tries += 1
        if tries > MAX_TRIES:
            raise RuntimeError("no valid morphism for t=%d" % t)
        rng = random.Random(BASE_SEED + t * 7919 + tries * 104729)
        images = []
        for i in range(K):
            L = rng.randint(1, LMAX_TRUE)
            images.append(''.join(rng.choice(ALPHABET) for _ in range(L)))
        M = [[images[i].count(str(j)) for j in range(K)] for i in range(K)]
        if not _primitive(M):
            continue
        lam = _spectral_radius(M)
        if not (1.2 <= lam <= 3.6):
            continue
        levels = build_levels(images, 0, nt)
        seen = set(''.join(levels[:-1]))
        if len(seen) < K:
            continue
        return images, M, lam, nt, levels


# ---------- exact integer matrix power ----------
def mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(K)) for j in range(K)] for i in range(K)]


def mat_pow(M, n):
    R = [[1 if i == j else 0 for j in range(K)] for i in range(K)]
    base = [row[:] for row in M]
    while n > 0:
        if n & 1:
            R = mat_mul(R, base)
        base = mat_mul(base, base)
        n >>= 1
    return R


def counts_at(M, axiom, n):
    Mn = mat_pow(M, n)
    return Mn[axiom]


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[0])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) != K:
        fail("expected exactly %d non-empty lines (one image per letter), got %d" % (K, len(lines)))
    guess_images = []
    for ln in lines:
        if len(ln) < 1 or len(ln) > LMAX_OUT:
            fail("image length out of range [1,%d]" % LMAX_OUT)
        if any(ch not in ALPHABET for ch in ln):
            fail("image contains a character outside {0,1,2}")
        guess_images.append(ln)

    M_guess = [[guess_images[i].count(str(j)) for j in range(K)] for i in range(K)]

    # regenerate hidden sigma + its incidence matrix from t alone
    images, M_true, lam, nt, levels = make_morphism(t)

    Fs = []
    for off in OFFSETS:
        nq = nt + off
        ct = counts_at(M_true, 0, nq)
        cs = counts_at(M_guess, 0, nq)
        total_true = sum(ct)
        total_guess = sum(cs)
        if total_true <= 0:
            fail("internal: degenerate truth")
        if total_guess <= 0:
            # every image is >=1 char so this cannot happen for a valid guess,
            # but guard defensively rather than crash
            Fs.append(0.0)
            continue
        log_true = math.log(total_true)
        log_guess = math.log(total_guess)
        err = abs(log_guess - log_true)
        f_len = math.exp(-err / SCALE_LEN)
        freq_true = [c / total_true for c in ct]
        freq_guess = [c / total_guess for c in cs]
        l1 = sum(abs(a - b) for a, b in zip(freq_true, freq_guess))
        f_freq = max(0.0, 1.0 - l1 / 2.0)
        Fs.append(0.5 * f_len + 0.5 * f_freq)

    F = sum(Fs) / len(Fs)
    if F != F or F in (float("inf"), float("-inf")):
        fail("non-finite objective")

    sc = min(1000.0, 100.0 * F / max(1e-9, BASELINE_B))
    print("F=%.6f nt=%d horizons=%s  Ratio: %.6f" % (F, nt, OFFSETS, sc / 1000.0))


if __name__ == "__main__":
    main()
