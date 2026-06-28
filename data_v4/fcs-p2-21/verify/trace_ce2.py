def oriented_of(boxes_in):
    oriented=[]
    for (x,y,z) in boxes_in:
        d=[x,y,z]
        for k in range(3):
            h=d[k]; a=d[(k+1)%3]; b=d[(k+2)%3]
            if a>b: a,b=b,a
            oriented.append((a,b,h))
    return oriented

def opt_of(oriented):
    m=len(oriented); import sys; sys.setrecursionlimit(10000); memo={}
    def best(i):
        if i in memo: return memo[i]
        wi,di,hi=oriented[i]; r=hi
        for j in range(m):
            wj,dj,hj=oriented[j]
            if wi>wj and di>dj: r=max(r,hi+best(j))
        memo[i]=r; return r
    return max(best(i) for i in range(m))

# Heuristic mistake #1: "use each box exactly once, in its tallest orientation, then stack."
# i.e. forbid using a box type in multiple orientations and pick the max-height face down.
def greedy_one_orientation_tallest(boxes_in):
    chosen=[]
    for (x,y,z) in boxes_in:
        d=sorted([x,y,z])  # base two smallest, height largest
        chosen.append((d[0],d[1],d[2]))  # base (d0,d1) sorted, height d2 (tallest)
    # now longest decreasing chain by base among these single orientations
    return opt_of(chosen)

# Heuristic mistake #2: greedy "tallest face down per box, area-sort, chain"
for boxes_in in [
    [(3,8,8),(2,7,9),(1,6,10)],
    [(6,6,10),(5,9,9),(4,8,8)],
    [(2,2,9),(3,3,4),(2,4,4)],
]:
    o=oriented_of(boxes_in)
    print(boxes_in, "OPT=",opt_of(o), "greedy1orient=",greedy_one_orientation_tallest(boxes_in))
