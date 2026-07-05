# TIER: strong
# Strong construction: use best-known short non-redundant arrays (from extensive
# offline search), fall back to greedy Mian-Chowla, and additionally try a
# seeded multi-restart local search; emit the SHORTEST VALID candidate found.
import sys, random

BEST = {
 7:[0,1,4,10,18,23,25],
 8:[0,1,4,9,15,22,32,34],
 9:[0,1,5,12,25,27,35,41,44],
 10:[0,1,6,10,23,26,34,41,53,55],
 11:[0,1,4,13,28,33,47,54,64,70,72],
 12:[0,2,6,24,29,40,43,55,68,75,76,85],
 13:[0,2,5,25,37,43,59,70,85,89,98,99,106],
 14:[0,4,6,20,35,52,59,77,78,86,89,99,122,127],
 15:[0,4,20,30,57,59,62,76,100,111,123,136,144,145,151],
 16:[0,1,4,11,26,32,56,68,76,115,117,134,150,163,168,177],
 17:[0,5,7,17,52,56,67,80,81,100,122,138,159,165,168,191,199],
 18:[0,2,10,22,53,56,82,83,89,98,130,148,153,167,188,192,205,216],
}

def is_golomb(marks):
    if len(set(marks)) != len(marks):
        return False
    diffs = set()
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            d = abs(marks[i] - marks[j])
            if d == 0 or d in diffs:
                return False
            diffs.add(d)
    return True

def mian_chowla(n):
    seq = [0]
    diffs = set()
    while len(seq) < n:
        c = seq[-1] + 1
        while True:
            newd = [c - s for s in seq]
            if len(set(newd)) == len(newd) and not (diffs & set(newd)):
                for d in newd:
                    diffs.add(d)
                seq.append(c)
                break
            c += 1
    return seq

def restart_search(n, seed, tries=40):
    # randomized greedy: at each step pick among the smallest few valid extensions
    rng = random.Random(seed)
    best = None
    for _ in range(tries):
        seq = [0]
        diffs = set()
        ok = True
        while len(seq) < n:
            c = seq[-1] + 1
            cands = []
            probe = c
            while len(cands) < 3 and probe < seq[-1] + 400:
                nd = [probe - s for s in seq]
                if len(set(nd)) == len(nd) and not (diffs & set(nd)):
                    cands.append(probe)
                probe += 1
            if not cands:
                ok = False
                break
            pick = cands[rng.randrange(len(cands))]
            for d in [pick - s for s in seq]:
                diffs.add(d)
            seq.append(pick)
        if ok and (best is None or seq[-1] < best[-1]):
            best = seq
    return best

def main():
    n = int(sys.stdin.read().split()[0])
    candidates = []
    if n in BEST and is_golomb(BEST[n]):
        candidates.append(BEST[n])
    mc = mian_chowla(n)
    if is_golomb(mc):
        candidates.append(mc)
    rs = restart_search(n, seed=12345)
    if rs is not None and is_golomb(rs):
        candidates.append(rs)
    best = min(candidates, key=lambda m: m[-1] - m[0])
    print(" ".join(map(str, best)))

if __name__ == "__main__":
    main()
