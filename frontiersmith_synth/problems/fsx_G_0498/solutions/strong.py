# TIER: strong
# Gallager hard-decision bit-flipping decoder. Iteratively flips the coordinate that
# participates in the most currently-unsatisfied parity checks, until the syndrome
# vanishes (a codeword) or an iteration cap is hit. This corrects many multi-bit error
# patterns that the weight-1 greedy decoder cannot, but it is only a heuristic: it misses
# high-weight bursts and can mis-converge, so it stays well short of optimal decoding.
import sys


def parity(x):
    return bin(x).count("1") & 1


def bits_to_int(s, n):
    v = 0
    for j, ch in enumerate(s):
        if ch == "1":
            v |= (1 << j)
    return v


def to_bits(v, n):
    return "".join("1" if (v >> j) & 1 else "0" for j in range(n))


def main():
    data = sys.stdin.read().split("\n")
    idx = 0
    while data[idx].strip() == "":
        idx += 1
    n, r, m, T = map(int, data[idx].split())
    idx += 1
    H = []
    for _ in range(r):
        while data[idx].strip() == "":
            idx += 1
        H.append(bits_to_int(data[idx].strip(), n))
        idx += 1

    # column -> list of check rows it belongs to
    col_rows = [[] for _ in range(n)]
    for t, row in enumerate(H):
        v = row
        while v:
            j = (v & -v).bit_length() - 1
            col_rows[j].append(t)
            v &= v - 1

    MAXIT = 60
    out = []
    for _ in range(m):
        while data[idx].strip() == "":
            idx += 1
        y = bits_to_int(data[idx].strip(), n)
        idx += 1

        x = y
        # per-row syndrome bits
        s = [1 if parity(row & x) else 0 for row in H]
        unsat = sum(s)
        it = 0
        while unsat > 0 and it < MAXIT:
            best_j = -1
            best = 0
            for j in range(n):
                cnt = 0
                for t in col_rows[j]:
                    cnt += s[t]
                if cnt > best:
                    best = cnt
                    best_j = j
            if best_j < 0:
                break
            x ^= (1 << best_j)
            for t in col_rows[best_j]:
                if s[t]:
                    s[t] = 0
                    unsat -= 1
                else:
                    s[t] = 1
                    unsat += 1
            it += 1

        if unsat == 0:
            out.append(to_bits(x, n))     # converged to a codeword
        else:
            out.append(to_bits(y, n))     # give up: emit received word (not a codeword)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
