import sys
sys.setrecursionlimit(100000)

# Brute oracle: given a target prefix-function array pi[0..n-1], decide whether
# SOME string over a small alphabet has EXACTLY this prefix function, and if so
# output one such string. Otherwise output -1.
#
# Method (obviously correct): backtracking. Build the string one character at a
# time. After appending character c at position i we can compute the true
# prefix-function value at i incrementally from the already-fixed earlier values
# (standard KMP advance using the partial pi we have built). We require it to
# equal the target pi[i]. We try characters from a fixed alphabet in order. The
# alphabet size is bounded: a valid string never needs more than n distinct
# letters, and in fact >=2 distinct letters never appear unless forced, so an
# alphabet of size n+1 is always enough. We use letters 0,1,2,... and print them
# as 'a','b','c',...
#
# Because pi[i] is computed by the genuine KMP recurrence on the string we are
# constructing, any string we accept provably has the requested prefix function.

def true_pi_step(s, pi, i):
    # standard KMP prefix-function advance: s and pi are filled for [0..i],
    # pi[0..i-1] already correct; compute pi[i].
    if i == 0:
        return 0
    k = pi[i-1]
    while k > 0 and s[i] != s[k]:
        k = pi[k-1]
    if s[i] == s[k]:
        k += 1
    return k

def solve(target):
    n = len(target)
    if n == 0:
        return ""  # empty string is valid, empty prefix function
    s = [0]*n
    pi = [0]*n
    ALPHA = n + 1  # always enough distinct letters

    # iterative backtracking with explicit stack of "next char to try" per pos
    nxt = [0]*n
    i = 0
    while i >= 0:
        placed = False
        while nxt[i] < ALPHA:
            c = nxt[i]
            nxt[i] += 1
            s[i] = c
            val = true_pi_step(s, pi, i)
            if val == target[i]:
                pi[i] = val
                placed = True
                break
        if placed:
            if i == n-1:
                return ''.join(chr(ord('a')+x) for x in s)
            i += 1
            nxt[i] = 0
        else:
            i -= 1
    return None

def main():
    data = sys.stdin.read().split()
    if not data:
        print(-1)
        return
    n = int(data[0])
    pi = list(map(int, data[1:1+n]))
    res = solve(pi)
    if res is None:
        print(-1)
    else:
        # n==0 prints an empty line for the string (consistent with sol)
        print(res)

main()
