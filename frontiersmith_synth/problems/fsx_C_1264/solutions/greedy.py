# TIER: greedy
"""
The obvious recipe: rank each call site by its OWN removed-overhead density
(freq_i * CALL_OVERHEAD) / inline_size_i, sort descending, and greedily accept while the
code image still fits under ICACHE_CAP. This values every call site in complete
isolation -- it never looks at parent/bonus, so it happily inlines a showy high-frequency
site that a chain-aware view would skip, and it happily skips a low-frequency chain root
whose own direct saving looks tiny, even when that root is the only thing standing
between the budget and a much bigger downstream bonus. It also never considers
overshooting the icache cap on purpose.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    m = int(next(it))
    S_base = int(next(it))
    ICACHE_CAP = int(next(it))
    CALL_OVERHEAD = int(next(it))
    _PENALTY_COEF = int(next(it))

    freq = [0] * (m + 1)
    inline_size = [0] * (m + 1)
    for i in range(1, m + 1):
        freq[i] = int(next(it))
        _base_cost = int(next(it))
        inline_size[i] = int(next(it))
        _parent = int(next(it))
        _bonus = int(next(it))

    order = sorted(range(1, m + 1),
                    key=lambda i: (-(freq[i] * CALL_OVERHEAD) / inline_size[i], i))
    chosen = []
    S = S_base
    for i in order:
        if S + inline_size[i] <= ICACHE_CAP:
            chosen.append(i)
            S += inline_size[i]

    print(len(chosen))
    print(" ".join(map(str, chosen)))


if __name__ == "__main__":
    main()
