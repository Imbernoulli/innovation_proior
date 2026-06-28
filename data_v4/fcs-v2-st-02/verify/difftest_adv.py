#!/usr/bin/env python3
# Adversarial differential test: binary alphabet + heavy wildcards (max
# transitivity-trap density), plus near-period corruption.
import sys, subprocess, random
sys.path.insert(0, "/srv/home/bohanlyu/innovation_proior/data_v4/fcs-v2-st-02/verify")
from brute import smallest_period
EXE = "/tmp/fcs-v2-st-02_x"

def run_exe(s):
    r = subprocess.run([EXE], input=s + "\n", capture_output=True, text=True)
    return r.stdout.strip()

def gen(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 60)
    sigma = rng.choice([1, 2, 2, 2, 3])
    letters = "abc"[:sigma]
    qrate = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    return ''.join('?' if rng.random() < qrate else rng.choice(letters) for _ in range(n))

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    mism = 0
    for seed in range(1, N + 1):
        s = gen(seed * 7919 + 13)
        e = run_exe(s); b = str(smallest_period(s))
        if e != b:
            print(f"MISMATCH seed={seed} s={s!r} sol={e} brute={b}")
            mism += 1
            if mism >= 15: break
    print(f"adversarial done: random={N} mismatches={mism}")

if __name__ == "__main__":
    main()
