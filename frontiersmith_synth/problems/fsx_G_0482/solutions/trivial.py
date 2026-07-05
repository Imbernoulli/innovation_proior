# TIER: trivial
# Reproduce the checker's Erdos-Turan reference array verbatim -> Ratio ~= 0.1.
import sys

def isprime(x):
    if x < 2: return False
    i = 2
    while i * i <= x:
        if x % i == 0: return False
        i += 1
    return True

def next_prime(n):
    p = n
    while not isprime(p): p += 1
    return p

def main():
    n = int(sys.stdin.read().split()[0])
    p = next_prime(n)
    marks = [2 * p * i + (i * i) % p for i in range(n)]
    print(" ".join(map(str, marks)))

if __name__ == "__main__":
    main()
