import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    s = data[idx]; idx += 1
    q = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        l = int(data[idx]); r = int(data[idx + 1]); idx += 2
        L = l - 1
        R = r - 1
        t = s[L:R + 1]          # inclusive substring, Python slice end-exclusive
        length = len(t)
        ans = length            # p = length always valid (empty overlap)
        for p in range(1, length):
            # period p valid iff t[i] == t[i+p] for all 0 <= i < length - p
            ok = True
            for i in range(length - p):
                if t[i] != t[i + p]:
                    ok = False
                    break
            if ok:
                ans = p
                break
        out.append(str(ans))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

main()
