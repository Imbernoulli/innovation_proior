import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        # empty input -> score 0
        print(0)
        return
    s = data[0]
    n = len(s)

    score = 0
    # For each prefix length len (1..n), count occurrences of s[0:len] as a substring of s.
    for length in range(1, n + 1):
        p = s[:length]
        cnt = 0
        # all start positions where p fits
        for start in range(0, n - length + 1):
            if s[start:start + length] == p:
                cnt += 1
        score += cnt
    print(score)

main()
