# TIER: trivial
"""Echo the template H itself as the answer (N=nH, M=mH).  H trivially
matches its own moments exactly, but its diameter is tiny -- this is the
'do nothing clever' baseline."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    idx = 0
    nH, k, n = map(int, data[idx].split()); idx += 1
    idx += 1  # eps
    idx += 1  # eps2
    idx += 1  # D_MAX M_MAX
    mH = int(data[idx]); idx += 1
    edges = []
    for _ in range(mH):
        u, v = map(int, data[idx].split()); idx += 1
        edges.append((u, v))

    out = [str(nH), str(mH)]
    for (u, v) in edges:
        out.append(f"{u} {v}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
