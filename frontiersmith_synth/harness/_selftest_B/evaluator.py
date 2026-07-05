import sys, json, math, random, isorun
def make_instances():
    out=[]
    for s in range(8):
        rng=random.Random(300+s); n=25+6*s
        out.append({"public":{"points":[[rng.random(),rng.random()] for _ in range(n)]}, "hidden":{}})
    return out
def _len(pts, order):
    return sum(math.dist(pts[order[i]], pts[order[(i+1)%len(order)]]) for i in range(len(order)))
def baseline(inst):
    pts=inst["public"]["points"]; return _len(pts, list(range(len(pts))))
def score(inst, ans):
    pts=inst["public"]["points"]; n=len(pts)
    if not isinstance(ans, dict) or "tour" not in ans: return False, 0
    tour=ans["tour"]
    if not isinstance(tour, list) or sorted(tour)!=list(range(n)): return False, 0
    L=_len(pts, tour)
    if not (L==L and L<math.inf): return False, 0
    return True, L
def main():
    cand=sys.argv[1]; insts=make_instances(); vec=[]
    for inst in insts:
        ans,st=isorun.run_candidate(cand, inst["public"], timeout=20)
        if st!="OK": vec.append(0.0); continue
        ok,L=score(inst,ans)
        if not ok: vec.append(0.0); continue
        vec.append(min(1.0, 0.1*baseline(inst)/max(L,1e-12)))
    print("Ratio: %.6f"%(sum(vec)/len(vec)))
    print("Vector: "+json.dumps([round(x,6) for x in vec]))
main()
