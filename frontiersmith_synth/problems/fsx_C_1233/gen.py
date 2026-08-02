#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE chunking-format instance to stdout.

Format:
  N K
  A_1 ... A_N
  INS_POS INS_LEN
  V_1 ... V_INS_LEN
  WC WI WS WE
  Q
  q_1 ... q_Q

The corpus is built from a small library of D distinct "unit" byte patterns
(each longer than the chunker's 8-byte lookback window) concatenated with
REPEATS, so identical substrings genuinely recur at different absolute
offsets -- the structural precondition a content-defined chunker can exploit
for block-level dedup, and a fixed-period chunker (whose boundaries are pure
functions of absolute position) essentially never can. testId 1..10 is a
hand-authored difficulty/trap ladder: small corpora first, then increasingly
large corpora with more repeats and an earlier splice point (INS_POS), which
maximizes how much of the file lies downstream of the edit -- the regime
where position-based chunking's boundaries fully decouple from content while
window-based chunking's do not.

Deterministic: a self-contained 64-bit LCG seeded only from testId (no
stdlib `random`, no wall-clock, no external entropy).
"""
import sys


class LCG:
    def __init__(self, seed):
        self.s = (seed & 0xFFFFFFFFFFFFFFFF) or 1

    def nxt(self):
        self.s = (6364136223846793005 * self.s + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.s

    def randint(self, a, b):
        n = b - a + 1
        return a + (self.nxt() >> 33) % n


# N, K, D(#distinct units), unit length range, insertion fraction, insertion
# length, (WC,WI,WS,WE), #queries
CASES = {
    1:  dict(N=150,  K=16, D=3, ulo=25, uhi=35, insfrac=0.60, inslen=5,  W=(3, 3, 2, 2), Q=8),
    2:  dict(N=220,  K=16, D=3, ulo=25, uhi=35, insfrac=0.50, inslen=6,  W=(3, 2, 3, 2), Q=10),
    3:  dict(N=350,  K=24, D=4, ulo=30, uhi=45, insfrac=0.40, inslen=8,  W=(4, 2, 2, 3), Q=12),
    4:  dict(N=600,  K=24, D=4, ulo=35, uhi=50, insfrac=0.30, inslen=10, W=(2, 2, 2, 4), Q=14),
    5:  dict(N=800,  K=24, D=5, ulo=35, uhi=55, insfrac=0.15, inslen=12, W=(2, 2, 2, 4), Q=16),
    6:  dict(N=1100, K=32, D=5, ulo=40, uhi=60, insfrac=0.50, inslen=10, W=(3, 3, 2, 2), Q=14),
    7:  dict(N=1400, K=32, D=5, ulo=40, uhi=65, insfrac=0.35, inslen=14, W=(4, 2, 2, 3), Q=16),
    8:  dict(N=1800, K=32, D=6, ulo=40, uhi=65, insfrac=0.20, inslen=16, W=(3, 2, 2, 4), Q=18),
    9:  dict(N=2300, K=32, D=6, ulo=45, uhi=70, insfrac=0.25, inslen=18, W=(4, 2, 2, 4), Q=20),
    10: dict(N=3000, K=32, D=7, ulo=45, uhi=75, insfrac=0.10, inslen=20, W=(4, 3, 2, 4), Q=22),
}


def build_instance(tid):
    c = CASES[tid]
    rng = LCG(1000 + tid * 7919)
    N, K, D = c['N'], c['K'], c['D']
    units = []
    for _ in range(D):
        ulen = rng.randint(c['ulo'], c['uhi'])
        units.append([rng.randint(0, K - 1) for _ in range(ulen)])
    corpus = []
    while len(corpus) < N:
        uidx = rng.randint(0, D - 1)
        corpus.extend(units[uidx])
    corpus = corpus[:N]
    ins_pos = max(1, min(N - 1, int(N * c['insfrac'])))
    ins_len = c['inslen']
    ins_vals = [rng.randint(0, K - 1) for _ in range(ins_len)]
    WC, WI, WS, WE = c['W']
    Q = c['Q']
    queries = sorted(rng.randint(0, N - 1) for _ in range(Q)) if N > 0 else []
    return dict(N=N, K=K, corpus=corpus, ins_pos=ins_pos, ins_vals=ins_vals,
                W=(WC, WI, WS, WE), queries=queries)


def main():
    tid = int(sys.argv[1])
    tid = ((tid - 1) % 10) + 1
    inst = build_instance(tid)
    out = []
    out.append(f"{inst['N']} {inst['K']}")
    out.append(" ".join(map(str, inst['corpus'])))
    out.append(f"{inst['ins_pos']} {len(inst['ins_vals'])}")
    out.append(" ".join(map(str, inst['ins_vals'])) if inst['ins_vals'] else "")
    out.append(" ".join(map(str, inst['W'])))
    out.append(str(len(inst['queries'])))
    out.append(" ".join(map(str, inst['queries'])) if inst['queries'] else "")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
