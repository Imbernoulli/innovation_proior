def analyze(boxes_in, label):
    oriented=[]
    for (x,y,z) in boxes_in:
        d=[x,y,z]
        for k in range(3):
            h=d[k]; a=d[(k+1)%3]; b=d[(k+2)%3]
            if a>b: a,b=b,a
            oriented.append((a,b,h))
    m=len(oriented)
    import sys; sys.setrecursionlimit(10000)
    memo={}
    def best(i):
        if i in memo: return memo[i]
        wi,di,hi=oriented[i]; r=hi
        for j in range(m):
            wj,dj,hj=oriented[j]
            if wi>wj and di>dj: r=max(r,hi+best(j))
        memo[i]=r; return r
    opt=max(best(i) for i in range(m))
    # greedy by height: tallest orientation as bottom, then keep adding tallest that fits
    order=sorted(range(m), key=lambda i:(-oriented[i][2], -oriented[i][0]*oriented[i][1]))
    used=None; tot=0; picks=[]
    for i in order:
        w,d,h=oriented[i]
        if used is None or (w<used[0] and d<used[1]):
            tot+=h; used=(w,d); picks.append((w,d,h))
    print(f"{label}: OPT={opt}  GREEDY_height={tot}  greedy_picks={picks}")
    return opt, tot

# Tempting: one fat tall box (big base, big height) that nothing can sit on,
# versus a stack of medium boxes that together are taller.
analyze([(9,9,12),(8,8,8),(6,6,8),(4,4,8)], "A")
analyze([(9,9,11),(8,8,7),(6,6,7),(4,4,7)], "B")
analyze([(10,10,20),(9,9,9),(7,7,9),(5,5,9),(3,3,9)], "C")
