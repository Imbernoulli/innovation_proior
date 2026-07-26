# TIER: trivial
# Nearest-consonant-tone construction: at every position pick the legal pitch
# closest to the cantus firmus (breaking ties toward the smaller pitch). No
# attempt is made to steer toward contrary motion, to diversify melodic
# interval sizes, or to place the climax anywhere deliberate. This is the
# simplest thing that reads the rule table and honors the hard constraints
# it can see one note at a time.
import sys, json


def solve(inst):
    cantus = inst["cantus"]
    L = len(cantus)
    lo, hi = inst["cp_range"]
    rules = inst["rules"]
    cc = set(rules["consonant_classes"])
    pc = set(rules["perfect_classes"])
    bc = set(rules["boundary_classes"])
    max_leap = rules["max_leap"]

    cp = []
    prev = None
    prev_class = None
    for i in range(L):
        def gather(relax_parallel, relax_leap):
            out = []
            for p in range(lo, hi + 1):
                vc = (p - cantus[i]) % 7
                if vc not in cc:
                    continue
                if (i == 0 or i == L - 1) and vc not in bc:
                    continue
                if prev is not None and not relax_leap and abs(p - prev) > max_leap:
                    continue
                if (not relax_parallel and prev_class is not None
                        and vc in pc and prev_class in pc and vc == prev_class):
                    continue
                out.append(p)
            return out

        cands = gather(False, False) or gather(True, False) or gather(True, True)
        if not cands:
            cands = [cantus[i]]
        best = min(cands, key=lambda p: (abs(p - cantus[i]), p))
        cp.append(best)
        prev = best
        prev_class = (best - cantus[i]) % 7
    return {"cp": cp}


def main():
    inst = json.load(sys.stdin)
    print(json.dumps(solve(inst)))


main()
