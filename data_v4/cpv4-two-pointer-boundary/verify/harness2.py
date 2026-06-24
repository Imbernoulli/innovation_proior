import random
import subprocess
import sys

SOL = "/tmp/cpv4-two-pointer-boundary_sol"

def gen(seed):
    random.seed(seed * 7919 + 13)
    n = random.randint(0, 40)
    vmax = random.choice([1, 2, 3, 5, 10, 50])
    D = random.choice([0, 1, 2, 3, 5, 10, 100])
    vals = [random.randint(-vmax, vmax) for _ in range(n)]
    return f"{n} {D}\n" + " ".join(str(v) for v in vals) + "\n"

def brute(text):
    data = text.split()
    n = int(data[0]); D = int(data[1])
    a = [int(data[2 + i]) for i in range(n)]
    count = 0
    for l in range(n):
        cmax = cmin = a[l] if n else 0
        for r in range(l + 1, n):
            if a[r] > cmax: cmax = a[r]
            if a[r] < cmin: cmin = a[r]
            if cmax - cmin <= D:
                count += 1
    return str(count)

def sol(text):
    p = subprocess.run([SOL], input=text, capture_output=True, text=True)
    return p.stdout.strip()

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    mismatch = 0
    shown = 0
    for s in range(1, N + 1):
        text = gen(s)
        bs = brute(text); ss = sol(text)
        if bs != ss:
            mismatch += 1
            if shown < 8:
                shown += 1
                print(f"MISMATCH seed={s}\n{text!r}\nsol={ss} bru={bs}\n---")
    print(f"TOTAL={N} MISMATCHES={mismatch}")

main()
