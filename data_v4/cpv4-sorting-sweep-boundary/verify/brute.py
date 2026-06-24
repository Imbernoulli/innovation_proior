import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    iv = []
    for _ in range(n):
        s = int(data[idx]); e = int(data[idx+1]); idx += 2
        if s < e:
            iv.append((s, e))

    if not iv:
        print(0)
        return

    # Independent brute force: the maximum number of half-open intervals [s,e)
    # covering a common point is attained at some start coordinate s_i (a point
    # where coverage can only go up). For each candidate instant t = s_i, count
    # how many intervals contain t, i.e. s <= t < e. We test every start point.
    candidates = sorted(set(s for (s, e) in iv))
    best = 0
    for t in candidates:
        c = 0
        for (s, e) in iv:
            if s <= t < e:
                c += 1
        if c > best:
            best = c
    print(best)

main()
