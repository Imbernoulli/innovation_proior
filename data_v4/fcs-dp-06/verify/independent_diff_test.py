#!/usr/bin/env python3
import random
import subprocess
import sys
from pathlib import Path


FULL = (1 << 20) - 1


def oracle(values):
    ans = 0
    for i in range(len(values)):
        ai = values[i]
        for j in range(i + 1, len(values)):
            if (ai & values[j]) == 0:
                ans += 1
    return ans


def run_case(exe, values):
    data = str(len(values)) + "\n"
    if values:
        data += " ".join(map(str, values)) + "\n"
    proc = subprocess.run(
        [str(exe)],
        input=data,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solver exited {proc.returncode}: {proc.stderr}")
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError("solver produced empty output")
    return int(out)


def adversarial_cases():
    cases = [
        [],
        [0],
        [1],
        [FULL],
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 2],
        [FULL, 0],
        [FULL, FULL],
        [0, 0, 0],
        [1, 2, 3, 0],
        [FULL] * 12,
        [0] * 12,
        [0] * 20 + [FULL] * 20,
        [1] * 10 + [2] * 10 + [3] * 10,
        [1 << b for b in range(20)],
        [FULL ^ (1 << b) for b in range(20)],
        [0] + [1 << b for b in range(20)] + [FULL],
        [0, FULL, 1, FULL ^ 1, 2, FULL ^ 2, 3, FULL ^ 3],
    ]
    for b in range(20):
        cases.append([0, 1 << b, FULL ^ (1 << b), FULL])
        cases.append([1 << b] * 5 + [FULL ^ (1 << b)] * 5)
    return cases


def random_cases(count, seed=20260628):
    rng = random.Random(seed)
    cases = []
    for t in range(count):
        mode = t % 10
        if mode == 0:
            n = rng.randint(0, 8)
            vals = [rng.randrange(1 << rng.randint(0, 6)) for _ in range(n)]
        elif mode == 1:
            n = rng.randint(1, 60)
            vals = [0 if rng.random() < 0.65 else rng.randrange(FULL + 1) for _ in range(n)]
        elif mode == 2:
            n = rng.randint(1, 60)
            vals = [FULL if rng.random() < 0.65 else rng.randrange(FULL + 1) for _ in range(n)]
        elif mode == 3:
            n = rng.randint(1, 60)
            pool = [0, 1, 2, 3, 7, FULL, FULL ^ 1, FULL ^ 2]
            vals = [rng.choice(pool) for _ in range(n)]
        elif mode == 4:
            n = rng.randint(1, 50)
            vals = []
            for _ in range(n):
                k = rng.randint(0, 3)
                bits = rng.sample(range(20), k)
                x = 0
                for b in bits:
                    x |= 1 << b
                vals.append(x)
        elif mode == 5:
            n = rng.randint(1, 50)
            vals = []
            for _ in range(n):
                k = rng.randint(17, 20)
                unset = rng.sample(range(20), 20 - k)
                x = FULL
                for b in unset:
                    x ^= 1 << b
                vals.append(x)
        elif mode == 6:
            n = rng.randint(2, 40)
            vals = []
            for _ in range(n // 2):
                x = rng.randrange(FULL + 1)
                vals.extend([x, FULL ^ x])
            if len(vals) < n:
                vals.append(rng.randrange(FULL + 1))
            rng.shuffle(vals)
        elif mode == 7:
            n = rng.randint(1, 70)
            vals = [rng.choice([0, 1, 2, 4, 8, 16, 31, 63, 127, FULL]) for _ in range(n)]
        elif mode == 8:
            n = rng.randint(1, 70)
            vals = [rng.randrange(FULL + 1) for _ in range(n)]
        else:
            n = rng.randint(1, 80)
            base = rng.randrange(FULL + 1)
            vals = [base if rng.random() < 0.5 else rng.randrange(FULL + 1) for _ in range(n)]
        cases.append(vals)
    return cases


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: independent_diff_test.py /path/to/solver")
    exe = Path(sys.argv[1])
    cases = adversarial_cases() + random_cases(360)
    for idx, values in enumerate(cases):
        got = run_case(exe, values)
        want = oracle(values)
        if got != want:
            print(f"Mismatch on case {idx}", file=sys.stderr)
            print(f"n={len(values)}", file=sys.stderr)
            print("values=" + " ".join(map(str, values)), file=sys.stderr)
            print(f"solver={got} oracle={want}", file=sys.stderr)
            return 1
    print(f"PASS {len(cases)} cases ({len(adversarial_cases())} adversarial, 360 random)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
