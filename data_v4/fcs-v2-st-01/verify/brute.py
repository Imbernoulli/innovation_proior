import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    s = next(it)
    q = int(next(it))
    out = []
    for _ in range(q):
        l1 = int(next(it)); len1 = int(next(it))
        l2 = int(next(it)); len2 = int(next(it))
        a = s[l1 - 1: l1 - 1 + len1]
        b = s[l2 - 1: l2 - 1 + len2]
        if a < b:
            out.append("-1")
        elif a > b:
            out.append("1")
        else:
            out.append("0")
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

main()
