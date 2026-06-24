import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    bal = [0] * (n + 1)
    for i in range(1, n + 1):
        bal[i] = int(data[idx]); idx += 1

    # comp[i] = component label of player i (1..n). Start each in its own.
    comp = list(range(n + 1))

    out = []
    for _ in range(q):
        t = int(data[idx]); idx += 1
        if t == 1:
            u = int(data[idx]); idx += 1
            v = int(data[idx]); idx += 1
            cu, cv = comp[u], comp[v]
            if cu != cv:
                # relabel everyone in cv to cu (O(n) brute relabel)
                for i in range(1, n + 1):
                    if comp[i] == cv:
                        comp[i] = cu
        else:
            # group players by component, compute size and sum for each
            sums = {}
            sizes = {}
            for i in range(1, n + 1):
                c = comp[i]
                sums[c] = sums.get(c, 0) + bal[i]
                sizes[c] = sizes.get(c, 0) + 1
            best = None
            for c, s in sizes.items():
                if s >= 2:
                    val = sums[c]
                    if best is None or val > best:
                        best = val
            if best is None or best <= 0:
                out.append("0")
            else:
                out.append(str(best))

    sys.stdout.write("\n".join(out))
    if out:
        sys.stdout.write("\n")

main()
