# TIER: trivial
"""
Directly externally activates every single node at time 0. Always feasible
(the external boost alone always exceeds every threshold), but uses N
events -- exactly the checker's own baseline construction, so this scores
~0.1.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0
    n = int(toks[p]); p += 1
    # rest of the instance is irrelevant to this construction
    out = [str(n)]
    for v in range(n):
        out.append(f"{v} 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
