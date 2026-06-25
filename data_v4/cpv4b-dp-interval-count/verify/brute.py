import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    A = int(next(it))
    B = int(next(it))
    M = int(next(it))

    # Brute force: enumerate every string of length n over k colors (0..k-1),
    # check that every MAXIMAL monochromatic run has length in [A, B].
    # Count valid strings mod M.
    # n == 0: the empty string. It has no runs, so vacuously valid -> 1 string.
    if n == 0:
        print(1 % M)
        return

    def valid(s):
        i = 0
        L = len(s)
        while i < L:
            j = i
            while j < L and s[j] == s[i]:
                j += 1
            run = j - i
            if run < A or run > B:
                return False
            i = j
        return True

    count = 0
    # iterate over all k^n strings
    total = k ** n
    for code in range(total):
        s = []
        c = code
        for _ in range(n):
            s.append(c % k)
            c //= k
        if valid(s):
            count += 1
    print(count % M)

main()
