#!/usr/bin/env python3
import subprocess, sys
import brute, gen  # noqa

SOL = "/tmp/fcs-nt-03_x"

def sol_run(n):
    p = subprocess.run([SOL], input=f"{n}\n", capture_output=True, text=True)
    return p.stdout.strip()

def brute_run(n):
    return str(brute.solve(n))

def main():
    total = 0
    mismatch = 0
    # explicit edge cases
    edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 25, 36, 49, 50,
             99, 100, 101, 999, 1000, 1001, 9999, 10000, 99991]
    import random
    cases = list(edges)
    for seed in range(1, 1001):
        rng = random.Random(seed)
        r = rng.random()
        if r < 0.15:
            n = rng.randint(0, 5)
        elif r < 0.30:
            n = rng.randint(0, 50)
        else:
            n = rng.randint(1, 20000)
        cases.append(n)
    for n in cases:
        a = sol_run(n)
        b = brute_run(n)
        total += 1
        if a != b:
            mismatch += 1
            print(f"MISMATCH n={n} sol={a} brute={b}")
    print(f"TOTAL={total} MISMATCH={mismatch}")

if __name__ == "__main__":
    main()
