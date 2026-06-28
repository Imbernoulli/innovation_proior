#!/usr/bin/env python3
from fractions import Fraction
import os
import random
import subprocess
import sys


ROOT = os.path.dirname(__file__)
SOL = os.path.join(ROOT, "sol")


def oracle(constraints):
    """Exact Fourier-Motzkin feasibility oracle for a*x + b*y <= c."""
    upper_x = []
    lower_x = []
    y_ineq = []

    for a, b, c in constraints:
        if a > 0:
            upper_x.append((a, b, c))
        elif a < 0:
            lower_x.append((a, b, c))
        else:
            y_ineq.append((Fraction(b), Fraction(c)))

    for al, bl, cl in lower_x:
        # x >= (cl - bl*y) / al, where al < 0
        pl = Fraction(-bl, al)
        ql = Fraction(cl, al)
        for au, bu, cu in upper_x:
            # x <= (cu - bu*y) / au, where au > 0
            pu = Fraction(-bu, au)
            qu = Fraction(cu, au)
            # lower <= upper: (pl - pu) * y <= qu - ql
            y_ineq.append((pl - pu, qu - ql))

    lo = None
    hi = None
    for alpha, beta in y_ineq:
        if alpha == 0:
            if beta < 0:
                return False
        elif alpha > 0:
            bound = beta / alpha
            hi = bound if hi is None else min(hi, bound)
        else:
            bound = beta / alpha
            lo = bound if lo is None else max(lo, bound)

    return lo is None or hi is None or lo <= hi


def run_sol(constraints):
    payload = [str(len(constraints))]
    payload.extend(f"{a} {b} {c}" for a, b, c in constraints)
    data = "\n".join(payload) + "\n"
    got = subprocess.check_output([SOL], input=data.encode(), timeout=2).decode().strip()
    if got not in ("YES", "NO"):
        raise RuntimeError(f"bad solver output {got!r}")
    return got == "YES"


def rand_constraint(rng, coeff=8):
    while True:
        a = rng.randint(-coeff, coeff)
        b = rng.randint(-coeff, coeff)
        if a or b:
            c = rng.randint(-coeff, coeff)
            return (a, b, c)


def random_cases(rng, count):
    for _ in range(count):
        n = rng.randint(0, 9)
        yield [rand_constraint(rng) for _ in range(n)]


def adversarial_cases():
    cases = [
        [],
        [(1, 0, 1)],
        [(1, 0, 1), (-1, 0, -2)],
        [(1, 0, 1), (-1, 0, 0)],
        [(1, 0, 0), (-1, 0, 0)],
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
        [(1, 0, 0), (-1, 0, 0), (0, 1, -1)],
        [(1, 0, 1), (0, 1, 1), (-1, -1, -3)],
        [(1, 0, 1), (0, 1, 1), (-1, -1, 0)],
        [(2, 2, 2), (1, 1, 0), (-1, -1, 0)],
        [(1, 1, 0), (2, 2, -1)],
        [(-1, -3, 0), (0, -5, 0), (2, 6, -1), (-2, -3, 0)],
        [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1)],
        [(1, 0, 0), (-1, 0, 0), (1, 1, 0), (-1, -1, 0)],
        [(1000000, 999999, 1000000), (999999, 999998, -1000000)],
        [
            (1000000, 999999, 1000000),
            (-1000000, -999999, -1000000),
            (999999, 999998, -1000000),
            (-999999, -999998, 1000000),
        ],
        [(1000000, 999999, 999999), (-1000000, -999999, -1000000)],
    ]

    normals = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 3), (-3, 2)]
    for a, b in normals:
        cases.append([(a, b, 0), (-a, -b, 0)])
        cases.append([(a, b, 0), (-a, -b, -1)])
        cases.append([(a, b, 3), (2 * a, 2 * b, 4), (-a, -b, 1)])

    return cases


def main():
    rng = random.Random(82736491)
    cases = list(adversarial_cases())
    cases.extend(random_cases(rng, 1000))

    for idx, constraints in enumerate(cases, 1):
        want = oracle(constraints)
        got = run_sol(constraints)
        if got != want:
            print(f"Mismatch on case {idx}", file=sys.stderr)
            print(f"solver={'YES' if got else 'NO'} oracle={'YES' if want else 'NO'}", file=sys.stderr)
            print(len(constraints), file=sys.stderr)
            for row in constraints:
                print(*row, file=sys.stderr)
            return 1

    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
