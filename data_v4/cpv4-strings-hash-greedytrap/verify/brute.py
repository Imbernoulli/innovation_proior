import sys

def is_square(w):
    L = len(w)
    if L % 2 != 0:
        return False
    h = L // 2
    return w[:h] == w[h:]

def solve(s):
    n = len(s)
    # dp[i] = max total covered length using non-overlapping squares within s[0:i]
    # A "square" is a substring of even length >=2 of the form uu.
    NEG = -1
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        # option 1: position i-1 is not covered by the right end of any chosen square
        dp[i] = dp[i - 1]
        # option 2: choose a square ending exactly at i (i.e. s[j:i] is a square)
        # iterate over all start positions j with even length
        for j in range(i - 2, -1, -2):
            w = s[j:i]
            if is_square(w):
                cand = dp[j] + (i - j)
                if cand > dp[i]:
                    dp[i] = cand
    return dp[n]

def main():
    data = sys.stdin.read().split()
    # first token n, then the string token
    idx = 0
    n = int(data[idx]); idx += 1
    if n == 0:
        s = ""
    else:
        s = data[idx]; idx += 1
    print(solve(s))

if __name__ == "__main__":
    main()
