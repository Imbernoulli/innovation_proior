# TIER: invalid
# Emits a short garbage word that does NOT synchronize the automaton -> must score 0.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    # a single symbol almost never collapses n>2 states to one -> not a reset word
    sys.stdout.write("0\n")

if __name__ == "__main__":
    main()
