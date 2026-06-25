import sys

def solve(n, a):
    # Brute force: choose a contiguous non-empty window [l..r], then optionally
    # delete at most one element strictly inside the kept window (deleting one
    # element is allowed only if at least one element remains, i.e. window length >= 2).
    # Maximize the sum of the kept elements. The kept set must be non-empty.
    best = None
    for l in range(n):
        for r in range(l, n):
            seg = a[l:r+1]
            total = sum(seg)
            # delete zero elements
            cand = total
            if best is None or cand > best:
                best = cand
            # delete exactly one element (only if length >= 2 so something remains)
            if len(seg) >= 2:
                for k in range(len(seg)):
                    cand2 = total - seg[k]
                    if best is None or cand2 > best:
                        best = cand2
    return best

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    a = [int(data[idx+i]) for i in range(n)]
    print(solve(n, a))

if __name__ == "__main__":
    main()
