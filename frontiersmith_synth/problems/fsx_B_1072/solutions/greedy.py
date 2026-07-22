# TIER: greedy
import sys


def main():
    d = sys.stdin.read().split()
    a = int(d[0]); k = int(d[1]); L = int(d[2])

    # The "obvious" textbook recipe: a de-Bruijn-style greedy walk that maximizes the
    # number of DISTINCT SUBSTRINGS (k-length windows), one linear step at a time --
    # exactly what most coders would write for "cover as many k-mers as possible".
    # It has no notion that a window and its cyclic rotation are the SAME necklace, so it
    # happily spends length re-visiting rotations of a necklace it has already covered.
    s = [0] * min(k - 1, L)
    seen_substrings = set()

    while len(s) < L:
        placed = False
        for c in range(a):
            if len(s) + 1 >= k:
                window = tuple(s[-(k - 1):] + [c]) if k > 1 else (c,)
            else:
                window = None
            if window is None or window not in seen_substrings:
                s.append(c)
                if window is not None:
                    seen_substrings.add(window)
                placed = True
                break
        if not placed:
            # every extension repeats an already-seen substring: fall back to filler
            c = 0
            if len(s) + 1 >= k:
                window = tuple(s[-(k - 1):] + [c]) if k > 1 else (c,)
                seen_substrings.add(window)
            s.append(c)

    print("".join(str(v) for v in s))


if __name__ == "__main__":
    main()
