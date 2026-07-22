# TIER: trivial
# Home-order first-fit: moor each boat (processed by home berth) at the earliest
# free berth in its reach window.  This is exactly the checker's baseline.
import sys

def main():
    toks = open(sys.argv[1]).read().split() if len(sys.argv) > 1 else sys.stdin.read().split()
    it = iter(toks)
    m = int(next(it)); n = int(next(it)); R = int(next(it))
    home = [0]*n; fp = [0]*n
    for i in range(n):
        home[i] = int(next(it)); fp[i] = int(next(it))
    occ = [False]*m
    berth = [0]*n
    order = sorted(range(n), key=lambda i: (home[i], i))
    for i in order:
        h = home[i]
        for d in range(R+1):
            s = (h+d) % m
            if not occ[s]:
                occ[s] = True; berth[i] = s; break
    sys.stdout.write(" ".join(str(berth[i]) for i in range(n)) + "\n")

if __name__ == "__main__":
    main()
