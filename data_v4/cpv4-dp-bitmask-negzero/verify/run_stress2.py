import subprocess, sys
from itertools import product
import random

SOL = "/tmp/cpv4-dp-bitmask-negzero_sol"

def gen(seed):
    rng = random.Random(seed + 100000)
    n = rng.randint(0, 6)
    m = rng.randint(1, 5)
    bias = rng.randint(0, 3)
    rows = []
    for _ in range(n):
        row = []
        for _ in range(m):
            if bias == 0:
                v = rng.randint(-12, 12)
            elif bias == 1:
                v = rng.randint(-12, 0)   # negatives and zeros only
            elif bias == 2:
                v = rng.choice([-12, 0, 0, 0, 7])
            else:
                v = rng.randint(0, 12)
            row.append(v)
        rows.append(row)
    out = [f"{n} {m}"]
    for row in rows:
        out.append(" ".join(map(str, row)))
    return "\n".join(out) + "\n", n, m, rows

def brute(n, m, p):
    best = 0
    for assign in product(*[range(-1, m) for _ in range(n)]):
        used = set(); ok = True; total = 0
        for i, j in enumerate(assign):
            if j == -1: continue
            if j in used: ok = False; break
            used.add(j); total += p[i][j]
        if ok: best = max(best, total)
    return best

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
mism = 0
for s in range(1, N + 1):
    inp, n, m, rows = gen(s)
    exp = brute(n, m, rows)
    r = subprocess.run([SOL], input=inp, capture_output=True, text=True)
    got = r.stdout.strip()
    if got != str(exp):
        mism += 1
        if mism <= 10:
            print(f"MISMATCH seed={s} sol={got!r} brute={exp}\ninput:\n{inp}")
print(f"CASES={N} MISMATCHES={mism}")
