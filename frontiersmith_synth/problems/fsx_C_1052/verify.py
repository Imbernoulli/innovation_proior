import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def clean_prefix_length(word, p):
    """word: list[int]. Returns F = length of the longest prefix of `word`
    that contains no palindromic factor of length > p (F = len(word) if the
    whole thing is clean).

    Equivalently: F = min right-endpoint (0-indexed) over every palindromic
    factor of `word` whose length exceeds p, or len(word) if none exists.
    A palindromic factor with right endpoint r is fully contained in every
    prefix of length >= r+1, so the prefix first becomes "dirty" at length
    r+1, i.e. the clean prefix length contributed by that factor is r.

    Implemented via classic expand-around-center (both parities), O(n^2)
    worst case, no substring slicing/hashing -> exact, no collision risk.
    """
    n = len(word)
    if n == 0:
        return 0
    best_end = None  # smallest right-endpoint (0-indexed) of any violation

    # odd-length centers
    for c in range(n):
        l, r = c, c
        while l - 1 >= 0 and r + 1 < n and word[l - 1] == word[r + 1]:
            l -= 1
            r += 1
        maxlen = r - l + 1  # longest odd-length palindrome centered at c
        smallest_violating = (p + 1) if (p % 2 == 0) else (p + 2)
        if maxlen >= smallest_violating:
            rad = (smallest_violating - 1) // 2
            right = c + rad
            if best_end is None or right < best_end:
                best_end = right

    # even-length centers between c and c+1
    for c in range(n - 1):
        if word[c] != word[c + 1]:
            continue
        l, r = c, c + 1
        while l - 1 >= 0 and r + 1 < n and word[l - 1] == word[r + 1]:
            l -= 1
            r += 1
        maxlen = r - l + 1  # longest even-length palindrome centered here
        smallest_violating = (p + 1) if (p % 2 == 1) else (p + 2)
        if maxlen >= smallest_violating:
            rad = smallest_violating // 2
            right = c + rad
            if best_end is None or right < best_end:
                best_end = right

    return n if best_end is None else best_end


def main():
    inp_tokens = open(sys.argv[1]).read().split()
    try:
        a, p, L = int(inp_tokens[0]), int(inp_tokens[1]), int(inp_tokens[2])
    except Exception:
        fail("bad input")
    if a < 2 or p < 1 or L < 1:
        fail("bad input ranges")

    out_tokens = open(sys.argv[2]).read().split()
    if len(out_tokens) != 1:
        fail("expected exactly one output token, got %d" % len(out_tokens))
    token = out_tokens[0]

    if len(token) < 1 or len(token) > L:
        fail("output length %d not in [1,%d]" % (len(token), L))

    ASCII_DIGITS = "0123456789"
    word = []
    for ch in token:
        if ch not in ASCII_DIGITS:
            fail("non-digit character %r" % ch)
        v = ASCII_DIGITS.index(ch)
        if v >= a:
            fail("digit %d out of alphabet range [0,%d]" % (v, a - 1))
        word.append(v)

    F = clean_prefix_length(word, p)

    # ---- internal baseline B: naive cyclic word 0,1,...,a-1,0,1,...,a-1,...
    #      truncated to length L, scored by the exact same rule ----
    baseline = [i % a for i in range(L)]
    B = clean_prefix_length(baseline, p)
    B = max(1, B)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
