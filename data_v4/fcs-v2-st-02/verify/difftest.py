#!/usr/bin/env python3
import sys, subprocess, random
sys.path.insert(0, "/srv/home/bohanlyu/innovation_proior/data_v4/fcs-v2-st-02/verify")
from brute import smallest_period

EXE = "/tmp/fcs-v2-st-02_x"

def gen_case(seed):
    rng = random.Random(seed)
    mode = rng.randint(0, 5)
    if mode == 0: n = rng.randint(1, 3)
    elif mode == 1: n = rng.randint(1, 8)
    elif mode == 2: n = rng.randint(1, 20)
    elif mode == 3: n = rng.randint(1, 40)
    else: n = rng.randint(1, 80)
    sigma = rng.choice([1, 2, 2, 3, 3, 4, 6])
    letters = "abcdefghijklmnopqrstuvwxyz"[:sigma]
    qrate = rng.choice([0.0, 0.1, 0.2, 0.3, 0.5, 0.7])
    chars = []
    for _ in range(n):
        chars.append('?' if rng.random() < qrate else rng.choice(letters))
    if rng.random() < 0.35 and n >= 4:
        p = rng.randint(1, n)
        base = [rng.choice(letters) for _ in range(p)]
        chars = [base[i % p] for i in range(n)]
        for i in range(n):
            if rng.random() < qrate: chars[i] = '?'
        if rng.random() < 0.6:
            chars[rng.randrange(n)] = rng.choice(letters)
    return ''.join(chars)

def run_exe(s):
    r = subprocess.run([EXE], input=s + "\n", capture_output=True, text=True)
    return r.stdout.strip()

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    mism = 0
    # explicit edge cases first
    edges = ["a", "b?a", "a?b", "??????", "ab??aa", "abcabc", "aba", "abab",
             "?", "z", "aa", "ab", "a?", "?a", "aaa", "a?a", "?b?", "ab?ab",
             "aabbaabb", "a?b?a?b?", "????a????", "abababab", "xyzxyzxy",
             "qqq?qqq", "p?q?p?q?p", "mn?mn?mn"]
    for s in edges:
        e = run_exe(s); b = str(smallest_period(s))
        if e != b:
            print(f"EDGE MISMATCH s={s!r} sol={e} brute={b}")
            mism += 1
    for seed in range(1, N + 1):
        s = gen_case(seed)
        e = run_exe(s); b = str(smallest_period(s))
        if e != b:
            print(f"MISMATCH seed={seed} s={s!r} sol={e} brute={b}")
            mism += 1
            if mism >= 15:
                break
    print(f"done: edges={len(edges)} random={N} mismatches={mism}")

if __name__ == "__main__":
    main()
