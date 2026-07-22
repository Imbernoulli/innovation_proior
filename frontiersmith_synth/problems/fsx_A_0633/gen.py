#!/usr/bin/env python3
"""gen.py <testId> -- print one change-ringing-touch instance to stdout.

Deterministic: all randomness is seeded from testId only.
"""
import itertools
import random
import sys


def apply_call(row, call):
    row = list(row)
    for j in call:
        row[j - 1], row[j] = row[j], row[j - 1]
    return tuple(row)


def identity(n):
    return tuple(range(1, n + 1))


def reverse_perm(n):
    return tuple(range(n, 0, -1))


def valid_call(call, n):
    idx = sorted(call)
    if not idx:
        return False
    for a, b in zip(idx, idx[1:]):
        if b - a < 2:
            return False
    return all(1 <= x <= n - 1 for x in idx)


def all_valid_calls(n):
    idxs = list(range(1, n))
    res = []
    for r in range(1, len(idxs) + 1):
        for comb in itertools.combinations(idxs, r):
            if valid_call(comb, n):
                res.append(comb)
    return res


def odd_even_phase1(n):
    """Canonical minimal path rounds -> reverse-row, valid (all rows distinct,
    reaches the reverse row) for odd n."""
    R0 = reverse_perm(n)
    row = identity(n)
    rows = [row]
    calls = []
    for phase in range(0, 3 * n):
        odd = tuple(x for x in range(1, n, 2) if 1 <= x <= n - 1)
        even = tuple(x for x in range(2, n, 2) if 1 <= x <= n - 1)
        call = odd if phase % 2 == 0 else even
        if not call:
            continue
        newrow = apply_call(rows[-1], call)
        rows.append(newrow)
        calls.append(call)
        if newrow == R0:
            break
    return rows, calls


def mirror_row(row, n):
    return tuple(n + 1 - row[n - 1 - i] for i in range(n))


# testId -> (n, Kmax, B, mode)   mode in {'normal','trap'}; n always odd.
CASE_TABLE = {
    1:  (5, 11, 3, 'normal'),
    2:  (5, 11, 4, 'normal'),
    3:  (5, 12, 4, 'normal'),
    4:  (5, 12, 7, 'trap'),
    5:  (7, 15, 3, 'normal'),
    6:  (7, 16, 4, 'normal'),
    7:  (7, 16, 4, 'trap'),
    8:  (7, 16, 7, 'trap'),
    9:  (7, 17, 7, 'trap'),
    10: (7, 17, 8, 'trap'),
}

PAL_BONUS = 0.3
PAIR_WEIGHT = 1.6
DISTRACTOR_WEIGHT = 1.2


def build_musical(n, seed, mode, canon_rows, B):
    rng = random.Random(seed)
    calls_all = all_valid_calls(n)
    musical = []
    used = {identity(n), reverse_perm(n)}

    # A couple of "cheap" rows near rounds on the canonical path, PLUS their
    # reflection-mirror partners -- an insightful solver that routes through
    # the cheap row automatically also visits the mirror partner for free.
    pair_js = [1, 2] if len(canon_rows) - 1 >= 3 else [1]
    for j in pair_js:
        if j >= len(canon_rows):
            continue
        r = canon_rows[j]
        mr = mirror_row(r, n)
        if r not in used:
            musical.append((PAIR_WEIGHT, r))
            used.add(r)
        if mr not in used:
            musical.append((PAIR_WEIGHT, mr))
            used.add(mr)

    def walk(steps):
        row = identity(n)
        for _ in range(steps):
            row = apply_call(row, rng.choice(calls_all))
        return row

    tries = 0
    while len(musical) < B and tries < 500:
        tries += 1
        r = walk(n - 1) if mode == 'trap' else walk(2)
        if r in used:
            continue
        used.add(r)
        musical.append((DISTRACTOR_WEIGHT, r))
    return musical


def main():
    testId = int(sys.argv[1])
    n, Kmax, B, mode = CASE_TABLE[((testId - 1) % 10) + 1]
    canon_rows, _ = odd_even_phase1(n)
    musical = build_musical(n, testId, mode, canon_rows, B)

    out = []
    out.append(f"{n} {Kmax} {PAL_BONUS}")
    out.append(f"{len(musical)}")
    for w, row in musical:
        out.append(f"{w} " + " ".join(map(str, row)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
