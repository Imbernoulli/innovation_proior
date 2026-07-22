# TIER: greedy
# The "obvious" textbook approach: Fibonacci (golden-ratio) multiplicative hashing with a
# single well-known fixed constant, then read off the top bits. This is what most
# competent engineers reach for first, and it never looks at the specific published
# sweep -- it just trusts that the constant is "generically good".
import sys

A0 = 0x9E3779B97F4A7C15

def main():
    sys.stdin.read()  # the sweep is available, but greedy never inspects it
    print("1 TOPBITS")
    print(f"MUL {A0} 0")

if __name__ == "__main__":
    main()
