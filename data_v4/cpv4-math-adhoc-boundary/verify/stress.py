import sys, subprocess, random

SOL = "/tmp/cpv4-math-adhoc-boundary_sol"

def brute_solve(n, a, b):
    if n <= 0 or a > b:
        return 0
    c = 0
    for x in range(1, n + 1):
        q = n // x
        if a <= q <= b:
            c += 1
    return c

def gen_case(rng, big):
    if big:
        n = rng.randint(1, 300)
        a = rng.randint(-1, n + 2)
        b = rng.randint(-1, n + 2)
    else:
        n = rng.randint(1, 60)
        mode = rng.randint(0, 3)
        if mode == 0:
            a = rng.randint(0, n + 2); b = rng.randint(0, n + 2)
        elif mode == 1:
            x = rng.randint(1, n); v = n // x; a = v; b = v
        elif mode == 2:
            a = rng.randint(0, n + 1); b = a
        else:
            a = rng.randint(0, n + 3); b = rng.randint(0, n + 3)
    if a > b:
        a, b = b, a
    return n, a, b

def run(num, big, base_seed):
    rng = random.Random(base_seed)
    mism = 0
    for t in range(num):
        q = rng.randint(1, 8)
        cases = [gen_case(rng, big) for _ in range(q)]
        inp = str(q) + "\n" + "\n".join(f"{n} {a} {b}" for (n, a, b) in cases) + "\n"
        got = subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout
        exp = "\n".join(str(brute_solve(n, a, b)) for (n, a, b) in cases) + "\n"
        gnorm = [l for l in got.split() ]
        enorm = [l for l in exp.split() ]
        if gnorm != enorm:
            mism += 1
            if mism <= 3:
                print("MISMATCH (big=%s) input:\n%s\nSOL : %s\nBRUTE: %s" % (big, inp, gnorm, enorm))
    return mism

if __name__ == "__main__":
    total = 0
    m1 = run(400, False, 12345); total += 400
    m2 = run(400, True, 67890); total += 400
    print("CASES=%d MISMATCHES=%d" % (total, m1 + m2))
