import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    s = data[0]
    n = len(s)
    out = []
    for L in range(1, n + 1):
        pat = s[:L]
        # count overlapping occurrences of pat in s
        cnt = 0
        for i in range(0, n - L + 1):
            if s[i:i+L] == pat:
                cnt += 1
        out.append(str(cnt))
    print(' '.join(out))

main()
