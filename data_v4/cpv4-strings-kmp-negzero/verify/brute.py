import sys

def print_result(hits):
    # Always two lines: count, then space-separated positions (empty if none).
    line2 = " ".join(str(x) for x in hits)
    sys.stdout.write(str(len(hits)) + "\n" + line2 + "\n")

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    t = [int(data[idx + j]) for j in range(n)]; idx += n
    m = int(data[idx]); idx += 1
    p = [int(data[idx + j]) for j in range(m)]; idx += m

    # Empty pattern: no window to align, answer 0.
    if m == 0:
        print_result([])
        return

    hits = []
    # For every start position i where a window of length m fits, check directly
    # whether there exists a constant c with t[i+j] = p[j] + c for all j.
    for i in range(0, n - m + 1):
        c = t[i] - p[0]
        ok = True
        for j in range(m):
            if t[i + j] != p[j] + c:
                ok = False
                break
        if ok:
            hits.append(i)
    print_result(hits)

if __name__ == "__main__":
    main()
