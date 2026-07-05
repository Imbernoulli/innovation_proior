# TIER: greedy
# Greedy Golomb ruler (Mian-Chowla): repeatedly add the smallest next position
# that keeps every pairwise separation distinct. Much shorter than the reference,
# but far from optimal.
import sys

def main():
    n = int(sys.stdin.read().split()[0])
    seq = [0]
    diffs = set()
    while len(seq) < n:
        c = seq[-1] + 1
        while True:
            newd = [c - s for s in seq]
            ok = len(set(newd)) == len(newd) and not (diffs & set(newd))
            if ok:
                for d in newd:
                    diffs.add(d)
                seq.append(c)
                break
            c += 1
    print(" ".join(map(str, seq)))

if __name__ == "__main__":
    main()
