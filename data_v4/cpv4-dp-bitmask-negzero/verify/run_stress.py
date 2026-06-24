import subprocess, sys
from itertools import product
import random

SOL = "/tmp/cpv4-dp-bitmask-negzero_sol"

def gen(seed):
    rng = random.Random(seed)
    mode = rng.randint(0, 4)
    n = rng.randint(0, 5)
    m = rng.randint(1, 4)
    rows = []
    for _ in range(n):
        row = []
        for _ in range(m):
            if mode == 0:
                v = rng.randint(-9, 9)
            elif mode == 1:
                v = rng.randint(-9, -1)
            elif mode == 2:
                v = 0
            elif mode == 3:
                v = rng.randint(1, 9)
            else:
                v = rng.choice([-9, -5, 0, 0, 3, 9])
            row.append(v)
        rows.append(row)
    out = [f"{n} {m}"]
    for row in rows:
        out.append(" ".join(map(str, row)))
    return "\n".join(out) + "\n", n, m, rows

def brute(n, m, p):
    best = 0
    choices = [list(range(-1, m)) for _ in range(n)]
    for assign in product(*choices):
        used = set()
        ok = True
        total = 0
        for i, j in enumerate(assign):
            if j == -1:
                continue
            if j in used:
                ok = False
                break
            used.add(j)
            total += p[i][j]
        if ok:
            best = max(best, total)
    return best

def main():
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
                print(f"MISMATCH seed={s} sol={got!r} brute={exp}")
                print("input:\n" + inp)
    print(f"CASES={N} MISMATCHES={mism}")

main()
