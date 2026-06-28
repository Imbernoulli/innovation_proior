import sys

def main():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    par = [0] * (n + 1)
    for v in range(1, n + 1):
        par[v] = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        v = int(data[idx]); idx += 1
        k = int(data[idx]); idx += 1
        cur = v
        ok = True
        for _step in range(k):
            cur = par[cur]
            if cur == 0:
                ok = False
                break
        out.append(str(cur if ok else 0))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

if __name__ == "__main__":
    main()
