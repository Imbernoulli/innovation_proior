# TIER: strong
"""The insight: palindromic call symmetry lets you design only HALF a touch.
Route from rounds to the reverse row R0=(n,n-1,...,1) -- the only row whose
conjugation preserves the adjacent-swap structure, i.e. the only usable
"involution pivot" -- then mirror: replay the same calls in reverse order
with swap-index j turned into n-j.  This closure is automatic (algebraic
identity), doubles up any musical row hit near the pivot for free (its
reflection partner shows up in the second half at no extra cost), and always
earns the full palindrome bonus.  A beam search over first-half routes (that
still land exactly on R0) additionally chases musical rows, valuing a row
that is EITHER itself musical OR whose mirror image is musical -- since the
second half will visit that mirror image automatically."""
import itertools
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


def mirror_row(row, n):
    return tuple(n + 1 - row[n - 1 - i] for i in range(n))


def mirror_call(call, n):
    return tuple(sorted(n - j for j in call))


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


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    n, Kmax = int(header[0]), int(header[1])
    B = int(data[1].strip())
    musical_targets = {}
    for i in range(B):
        toks = data[2 + i].split()
        w = float(toks[0])
        row = tuple(int(x) for x in toks[1:1 + n])
        musical_targets[row] = musical_targets.get(row, 0.0) + w

    calls_all = all_valid_calls(n)
    R0 = reverse_perm(n)
    canon_rows, canon_calls = odd_even_phase1(n)
    m0 = len(canon_calls)
    m_max = Kmax // 2

    best = None
    for m in range(m0, min(m_max, m0 + 6) + 1):
        beam = [(0.0, [identity(n)], [], {identity(n)})]
        for step in range(m):
            newbeam = []
            for score, rows, calls, visited in beam:
                for c in calls_all:
                    nr = apply_call(rows[-1], c)
                    if nr in visited:
                        continue
                    if step == m - 1 and nr != R0:
                        continue
                    bonus = musical_targets.get(nr, 0.0) + musical_targets.get(mirror_row(nr, n), 0.0)
                    newbeam.append((score + bonus, rows + [nr], calls + [c], visited | {nr}))
            if not newbeam:
                beam = []
                break
            newbeam.sort(key=lambda x: -x[0])
            beam = newbeam[:400]
        for score, rows, calls, visited in beam:
            if rows[-1] == R0 and (best is None or score > best[0]):
                best = (score, rows, calls)

    if best is None:
        rows, calls = canon_rows, canon_calls
    else:
        _, rows, calls = best

    m = len(calls)
    full_calls = list(calls)
    for t in range(1, m + 1):
        src = calls[m - t]
        full_calls.append(mirror_call(src, n))

    print(len(full_calls))
    for c in full_calls:
        print(" ".join(map(str, c)))


if __name__ == "__main__":
    main()
