import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    q = int(data[idx]); idx += 1
    present = {}  # badge -> count present (always 0 or 1 given the guarantees, but use multiset-safe)
    out = []
    for _ in range(q):
        t = data[idx]; idx += 1
        if t == '+':
            b = int(data[idx]); idx += 1
            present[b] = present.get(b, 0) + 1
        elif t == '-':
            b = int(data[idx]); idx += 1
            present[b] = present.get(b, 0) - 1
            if present[b] == 0:
                del present[b]
        else:  # '?'
            lo = int(data[idx]); idx += 1
            hi = int(data[idx]); idx += 1
            cnt = 0
            for b, c in present.items():
                if lo <= b <= hi:
                    cnt += c
            out.append(str(cnt))
    sys.stdout.write('\n'.join(out) + ('\n' if out else ''))

main()
