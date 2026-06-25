import sys


def is_sidon(seq):
    """True iff all pairwise sums seq[i]+seq[j] (i<=j) are distinct."""
    sums = set()
    for i in range(len(seq)):
        for j in range(i, len(seq)):
            t = seq[i] + seq[j]
            if t in sums:
                return False
            sums.add(t)
    return True


def lex_min_sidon(n):
    """
    The lexicographically smallest size-n Sidon set (values >= 0), found by a direct
    backtracking search over increasing sequences. We try the smallest possible value at
    each position; the FIRST complete sequence produced in this order is, by definition,
    the lexicographically smallest one. This is an independent brute force: it never
    references the greedy construction, it just searches.
    """
    if n == 0:
        return []
    result = []

    # a soft cap on the search range; the lex-min set is known to be small for tiny n.
    cap = max(4 * n * n, 8)

    def rec(seq):
        if len(seq) == n:
            result.extend(seq)
            return True
        start = (seq[-1] + 1) if seq else 0
        for v in range(start, cap + 1):
            if is_sidon(seq + [v]):
                if rec(seq + [v]):
                    return True
        return False

    ok = rec([])
    assert ok, "no Sidon set found within cap; increase cap"
    return result


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    ans = lex_min_sidon(n)
    if n == 0:
        sys.stdout.write("\n")
    else:
        sys.stdout.write(" ".join(str(v) for v in ans) + "\n")


if __name__ == "__main__":
    main()
