# TIER: greedy
# The obvious recipe: sinks go where the heat is GENERATED.  Rank junctions by
# their heat generation g_i (ties by lower index) and sink the top-k.  It never
# evaluates the steady state, so on trapped networks it spends the whole budget
# on the loud, well-ventilated core while modest sources behind thin
# conductive necks -- the real hotspots once the core is cool -- keep burning.
import sys


def main():
    it = sys.stdin.read().split()
    p = 0
    n = int(it[p]); m = int(it[p + 1]); k = int(it[p + 2]); p += 3
    g = [float(it[p + i]) for i in range(n)]; p += n

    order = sorted(range(n), key=lambda i: (-g[i], i))
    sinks = sorted(order[:k])
    out = [str(k)] + [str(i) for i in sinks]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
