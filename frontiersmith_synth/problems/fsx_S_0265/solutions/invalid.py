# TIER: invalid
# Deliberately INFEASIBLE: delete every link incident to the source relay s. Since s is
# a corner node (degree 2 <= k), this isolates s and severs the s-t path -> the checker's
# connectivity gate must reject this (score 0).
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); k = int(next(it))
    s = int(next(it)); t = int(next(it))
    inc = []
    for e in range(m):
        u = int(next(it)); v = int(next(it)); w = int(next(it))
        if u == s or v == s:
            inc.append(e)
    inc = inc[:k]  # stay within budget; corner degree (2) <= k
    out = [str(len(inc))]
    out.extend(str(e) for e in inc)
    sys.stdout.write(" ".join(out) + "\n")

if __name__ == "__main__":
    main()
