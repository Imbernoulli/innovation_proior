# TIER: invalid
"""Deliberately infeasible artifact: a correct-looking header (matches k) but
every FIX target uses a token outside the type grammar. Guaranteed to fail the
checker's strict output-schema parse on ANY instance (unlike "FIX bool" for
every slot, which can accidentally be sound whenever an instance's pinned
truth happens to be `bool` -- this one is unconditionally rejected).
"""
import sys


def main():
    data = sys.stdin.read().split()
    k = int(data[0])
    out = [str(k)]
    for _ in range(k):
        out.append("FIX notatype")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
