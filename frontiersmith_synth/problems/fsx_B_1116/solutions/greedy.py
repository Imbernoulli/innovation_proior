# TIER: greedy
# The obvious first move once you notice the contrary-motion rule: at every
# position, among the legal candidates, always prefer one that moves opposite
# to the cantus firmus's current step (falling back to the nearest legal tone
# only when no contrary candidate exists). Still a single left-to-right pass:
# no lookahead, no backtracking, no notion of "have I already used my highest
# note", and no awareness that this is one of ten pieces being judged
# together.
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

        if prev is not None and i > 0:
            dcf = cantus[i] - cantus[i - 1]

            def keyfn(p):
                dcp = p - prev
                is_c = dcp != 0 and dcf != 0 and (dcp > 0) != (dcf > 0)
                return (0 if is_c else 1, abs(p - cantus[i]), p)
            best = min(cands, key=keyfn)
        else:
            best = min(cands, key=lambda p: (abs(p - cantus[i]), p))
        cp.append(best)
        prev = best
        prev_class = (best - cantus[i]) % 7
    return {"cp": cp}


def main():
    inst = json.load(sys.stdin)
    print(json.dumps(solve(inst)))


main()
