# TIER: strong
# The insight: w_1 is literally sigma(axiom) with no ambiguity (the axiom is
# a single letter), and every later level w_{i+1} is the concatenation of
# sigma over the letters of w_i in order. The statement TELLS us every image
# has length 1..4, so there are only 4^3=64 possible (len0,len1,len2)
# assignments. For each candidate length assignment, greedily segment every
# consecutive pair of training levels into blocks of the hypothesised
# per-letter lengths and demand: (a) the blocks are perfectly consistent
# (the same letter always yields the same block, everywhere it occurs across
# ALL training transitions) and (b) each transition's block lengths sum
# exactly to the next level's length. With >=4 independent transitions and
# only 3 letters, a spurious candidate essentially never survives -- the
# unique surviving assignment IS the hidden substitution, reconstructed
# exactly, which reproduces both the true growth eigenvalue and eigenvector.
import sys, itertools

K = 3
LMAX_TRUE = 4


def try_lengths(levels, lens):
    images = {}
    for i in range(len(levels) - 1):
        src, dst = levels[i], levels[i + 1]
        pos = 0
        for c in src:
            Lc = lens[int(c)]
            if pos + Lc > len(dst):
                return None
            blk = dst[pos:pos + Lc]
            pos += Lc
            ci = int(c)
            if ci in images:
                if images[ci] != blk:
                    return None
            else:
                images[ci] = blk
        if pos != len(dst):
            return None
    if len(images) != K:
        return None
    return [images[i] for i in range(K)]


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0"); print("1"); print("2"); return
    t = int(data[0]); nt = int(data[1])
    levels = data[2:2 + nt + 1]
    if len(levels) < 2:
        print("0"); print("1"); print("2"); return

    found = None
    for lens in itertools.product(range(1, LMAX_TRUE + 1), repeat=K):
        cand = try_lengths(levels, lens)
        if cand is not None:
            found = cand
            break

    if found is None:
        # fallback: shouldn't happen given the problem's guarantees, but stay
        # safe rather than crash -- degrade to the greedy scalar-rate guess.
        import math
        ys = [math.log(len(w)) for w in levels]
        xs = list(range(len(ys)))
        n = len(xs)
        mx = sum(xs) / n; my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        b = num / den if den > 1e-12 else 0.0
        Lhat = max(1, min(12, round(pow(2.71828182845904523536, b))))
        found = ["0" * Lhat, "1", "2"]

    for img in found:
        print(img)


if __name__ == "__main__":
    main()
