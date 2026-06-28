#!/usr/bin/env python3
import random
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOL = Path("/tmp/fcs_sol")


def oracle(ops):
    versions = [[]]
    out = []
    for op in ops:
        if op[0] == 1:
            _, v, p, x = op
            base = versions[v]
            versions.append(base[:p] + [x] + base[p:])
        elif op[0] == 2:
            _, v, l, r = op
            arr = versions[v][:]
            arr[l : r + 1] = reversed(arr[l : r + 1])
            versions.append(arr)
        else:
            _, v, l, r = op
            out.append(str(sum(versions[v][l : r + 1])))
    return "\n".join(out) + ("\n" if out else "")


def run_case(ops):
    data = [str(len(ops))]
    data.extend(" ".join(map(str, op)) for op in ops)
    inp = "\n".join(data) + "\n"
    got = subprocess.run(
        [str(SOL)],
        input=inp,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    expected = oracle(ops)
    if got.returncode != 0 or got.stdout != expected:
        print("MISMATCH", file=sys.stderr)
        print("returncode:", got.returncode, file=sys.stderr)
        print("stderr:", got.stderr, file=sys.stderr)
        print("input:", file=sys.stderr)
        print(inp, file=sys.stderr)
        print("expected:", repr(expected), file=sys.stderr)
        print("got:", repr(got.stdout), file=sys.stderr)
        sys.exit(1)


def current_lengths(versions):
    return [len(v) for v in versions]


def append_op(ops, versions, op):
    ops.append(op)
    if op[0] == 1:
        _, v, p, x = op
        base = versions[v]
        versions.append(base[:p] + [x] + base[p:])
    elif op[0] == 2:
        _, v, l, r = op
        arr = versions[v][:]
        arr[l : r + 1] = reversed(arr[l : r + 1])
        versions.append(arr)


def random_case(rng, q, max_len):
    ops = []
    versions = [[]]
    for _ in range(q):
        lengths = current_lengths(versions)
        nonempty = [i for i, n in enumerate(lengths) if n]
        can_insert = any(n < max_len for n in lengths)
        if not nonempty or not can_insert:
            typ = 1 if can_insert else 3
        else:
            typ = rng.choices([1, 2, 3], weights=[5, 3, 6])[0]
            if typ == 1 and not can_insert:
                typ = rng.choice([2, 3])

        if typ == 1:
            candidates = [i for i, n in enumerate(lengths) if n < max_len]
            v = rng.choice(candidates)
            p = rng.randrange(lengths[v] + 1)
            x = rng.choice(
                [
                    rng.randint(-20, 20),
                    -10**9,
                    10**9,
                    0,
                    rng.randint(-10**9, 10**9),
                ]
            )
            append_op(ops, versions, (1, v, p, x))
        else:
            v = rng.choice(nonempty)
            n = lengths[v]
            if typ == 2:
                l = rng.randrange(n)
                r = rng.randrange(l, n)
                append_op(ops, versions, (2, v, l, r))
            else:
                l = rng.randrange(n)
                r = rng.randrange(l, n)
                append_op(ops, versions, (3, v, l, r))
    return ops


def adversarial_cases():
    cases = []
    cases.append([])

    ops = [
        (1, 0, 0, 5),
        (1, 1, 1, 1),
        (1, 2, 2, 2),
        (1, 3, 3, 4),
        (1, 4, 4, 3),
        (3, 5, 1, 3),
        (2, 5, 1, 3),
        (3, 6, 1, 1),
        (3, 5, 1, 1),
    ]
    cases.append(ops)

    ops = []
    versions = [[]]
    for x in [10**9, -10**9, 7, -3, 0, 10**9]:
        append_op(ops, versions, (1, len(versions) - 1, len(versions[-1]), x))
    append_op(ops, versions, (3, 6, 0, 5))
    append_op(ops, versions, (2, 6, 0, 5))
    append_op(ops, versions, (3, 7, 0, 5))
    append_op(ops, versions, (2, 7, 1, 4))
    append_op(ops, versions, (3, 8, 1, 4))
    append_op(ops, versions, (3, 6, 0, 5))
    cases.append(ops)

    ops = []
    versions = [[]]
    for i in range(9):
        base = len(versions) - 1
        pos = 0 if i % 2 else len(versions[-1])
        append_op(ops, versions, (1, base, pos, i - 4))
    for _ in range(4):
        base = len(versions) - 1
        append_op(ops, versions, (2, base, 0, len(versions[base]) - 1))
    append_op(ops, versions, (3, len(versions) - 1, 0, len(versions[-1]) - 1))
    append_op(ops, versions, (3, 9, 0, len(versions[9]) - 1))
    cases.append(ops)

    ops = []
    versions = [[]]
    for x in range(1, 8):
        append_op(ops, versions, (1, len(versions) - 1, len(versions[-1]), x))
    append_op(ops, versions, (2, 7, 1, 5))
    append_op(ops, versions, (1, 8, 3, 99))
    append_op(ops, versions, (2, 8, 0, 6))
    append_op(ops, versions, (3, 9, 0, 7))
    append_op(ops, versions, (3, 10, 0, 6))
    append_op(ops, versions, (3, 8, 1, 5))
    append_op(ops, versions, (3, 7, 1, 5))
    cases.append(ops)

    ops = []
    versions = [[]]
    for i in range(12):
        base = len(versions) - 1
        append_op(ops, versions, (1, base, i, (-1) ** i * i))
        append_op(ops, versions, (3, len(versions) - 1, i, i))
        append_op(ops, versions, (2, len(versions) - 1, i, i))
        append_op(ops, versions, (3, len(versions) - 1, i, i))
    cases.append(ops)

    return cases


def main():
    if not SOL.exists():
        print(f"compiled solution not found: {SOL}", file=sys.stderr)
        return 2

    for ops in adversarial_cases():
        run_case(ops)

    rng = random.Random(0xF00DCAFE)
    for _ in range(500):
        q = rng.randint(1, 90)
        max_len = rng.randint(1, 24)
        run_case(random_case(rng, q, max_len))

    large_ops = random_case(random.Random(123456789), 1000, 80)
    run_case(large_ops)
    print("PASS: 500 random small cases, 6 adversarial cases, and 1 larger mixed case")


if __name__ == "__main__":
    raise SystemExit(main())
