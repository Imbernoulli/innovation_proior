import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    W = int(data[idx]); idx += 1
    wt = []
    val = []
    for i in range(n):
        wt.append(int(data[idx])); idx += 1
        val.append(int(data[idx])); idx += 1

    # Exhaustive over all 2^n subsets. Obviously correct for tiny n.
    best = 0
    for mask in range(1 << n):
        tw = 0
        tv = 0
        for i in range(n):
            if mask & (1 << i):
                tw += wt[i]
                tv += val[i]
        if tw <= W and tv > best:
            best = tv
    print(best)

main()
