#!/usr/bin/env python3
import itertools
import random
import subprocess
import sys


SOLVER = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fcs_sol"


def brute_sat(n, clauses):
    for values in itertools.product((0, 1), repeat=n):
        ok = True
        for i, a, j, b in clauses:
            if values[i] != a and values[j] != b:
                ok = False
                break
        if ok:
            return True, values
    return False, None


def render_case(n, clauses):
    lines = [f"{n} {len(clauses)}"]
    lines += [f"{i} {a} {j} {b}" for i, a, j, b in clauses]
    return "\n".join(lines) + "\n"


def parse_and_validate_output(n, clauses, text):
    lines = text.splitlines()
    if not lines:
        return False, None, "empty output"

    head = lines[0].strip()
    if head == "NO":
        if any(line.strip() for line in lines[1:]):
            return False, None, "NO followed by extra nonempty output"
        return True, False, None

    if head != "YES":
        return False, None, f"first line is not YES/NO: {head!r}"
    if len(lines) < 2:
        return False, None, "YES without assignment line"

    tokens = lines[1].split()
    if len(tokens) != n:
        return False, None, f"assignment length {len(tokens)} != n {n}"
    try:
        vals = tuple(int(x) for x in tokens)
    except ValueError:
        return False, None, "assignment has non-integer token"
    if any(x not in (0, 1) for x in vals):
        return False, None, "assignment has value outside {0,1}"

    for idx, (i, a, j, b) in enumerate(clauses):
        if vals[i] != a and vals[j] != b:
            return False, None, f"assignment fails clause {idx}: {(i, a, j, b)}"
    return True, True, vals


def run_solver(n, clauses):
    proc = subprocess.run(
        [SOLVER],
        input=render_case(n, clauses),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3,
        check=False,
    )
    if proc.returncode != 0:
        return False, None, f"solver exited {proc.returncode}, stderr={proc.stderr!r}"
    return parse_and_validate_output(n, clauses, proc.stdout)


def unit(var, val):
    return (var, val, var, val)


def implies(i, a, j, b):
    # (x_i == a) -> (x_j == b) is (~(x_i == a) OR (x_j == b)).
    return (i, 1 - a, j, b)


def adversarial_cases():
    cases = []
    cases.append((0, []))
    cases.append((1, []))
    cases.append((1, [unit(0, 1)]))
    cases.append((1, [unit(0, 0)]))
    cases.append((1, [unit(0, 1), unit(0, 0)]))
    cases.append((1, [(0, 0, 0, 1)]))
    cases.append((1, [(0, 0, 0, 0), (0, 1, 0, 1)]))
    cases.append((2, [(0, 1, 1, 1), (0, 0, 1, 0), (0, 1, 1, 0), (0, 0, 1, 1)]))

    for n in range(2, 13):
        chain = [implies(i, 1, i + 1, 1) for i in range(n - 1)]
        cases.append((n, [unit(0, 1)] + chain + [unit(n - 1, 1)]))
        cases.append((n, [unit(0, 1)] + chain + [unit(n - 1, 0)]))
        cases.append((n, [unit(0, 0)] + [implies(i, 0, i + 1, 0) for i in range(n - 1)]))

    for n in range(1, 8):
        all_clauses = []
        for i in range(n):
            for j in range(n):
                for a in (0, 1):
                    for b in (0, 1):
                        all_clauses.append((i, a, j, b))
        cases.append((n, all_clauses))
        cases.append((n, all_clauses[:]))

    return cases


def random_cases(count, seed=90210):
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        n = rng.randint(0, 11)
        if n == 0:
            cases.append((n, []))
            continue
        mode = rng.randrange(5)
        if mode == 0:
            m = rng.randint(0, n + 3)
        elif mode == 1:
            m = rng.randint(0, 4 * n + 10)
        elif mode == 2:
            m = rng.randint(4 * n, 12 * n + 20)
        else:
            m = rng.randint(0, 30)
        clauses = [
            (
                rng.randrange(n),
                rng.randrange(2),
                rng.randrange(n),
                rng.randrange(2),
            )
            for _ in range(m)
        ]
        if mode == 4 and n:
            v = rng.randrange(n)
            clauses += [unit(v, 0), unit(v, 1)]
        cases.append((n, clauses))
    return cases


def main():
    cases = adversarial_cases() + random_cases(1200)
    yes = no = 0
    for case_no, (n, clauses) in enumerate(cases, 1):
        expected_sat, _ = brute_sat(n, clauses)
        valid, solver_sat, detail = run_solver(n, clauses)
        if not valid:
            print(f"OUTPUT CONTRACT FAILURE on case {case_no}: {detail}")
            print(render_case(n, clauses), end="")
            return 1
        if solver_sat != expected_sat:
            print(f"DECISION MISMATCH on case {case_no}: solver={solver_sat}, oracle={expected_sat}")
            print(render_case(n, clauses), end="")
            return 1
        if expected_sat:
            yes += 1
        else:
            no += 1
    print(f"PASS cases={len(cases)} random=1200 adversarial={len(adversarial_cases())} sat={yes} unsat={no}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
