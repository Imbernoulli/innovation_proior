# TIER: greedy
# The obvious approach: greedy set-cover of the target differences DIRECTLY in Z_n.
# Repeatedly add the residue (best of a random candidate batch) that creates the most
# new target-hitting differences with the current set. Ignores the CRT product structure;
# in a huge ring the random batch rarely lands aligned residues, so coverage plateaus.
import sys, random

def main():
    tk = sys.stdin.buffer.read().split()
    it = iter(tk)
    t = int(next(it)); ms = [int(next(it)) for _ in range(t)]; k = int(next(it))
    A = []
    for i in range(t):
        s = int(next(it)); A.append(set(int(next(it)) % ms[i] for _ in range(s)))
    n = 1
    for m in ms:
        n *= m

    def res(x):
        return tuple(x % m for m in ms)

    def hit(dr):
        for a, d in zip(A, dr):
            if d not in a:
                return False
        return True

    rng = random.Random(1)
    B = [0]
    Bres = [res(0)]
    SAMPLE = 500
    while len(B) < k:
        bx = None; bn = -1
        for _ in range(SAMPLE):
            x = rng.randrange(n)
            xr = res(x)
            cnt = 0
            for br in Bres:
                d1 = tuple((xr[i] - br[i]) % ms[i] for i in range(t))
                if hit(d1):
                    cnt += 1
                d2 = tuple((br[i] - xr[i]) % ms[i] for i in range(t))
                if d2 != d1 and hit(d2):
                    cnt += 1
            if cnt > bn:
                bn = cnt; bx = x
        if bx is None or bx in B:
            bx = rng.randrange(n)
        B.append(bx); Bres.append(res(bx))
    print(len(B))
    sys.stdout.write("\n".join(map(str, B)) + "\n")

if __name__ == "__main__":
    main()
