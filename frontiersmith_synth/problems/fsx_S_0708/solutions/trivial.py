# TIER: trivial
import sys


def main():
    t = sys.stdin.readline().rstrip("\n")
    n = len(t)
    out = []
    for ch in t:
        out.append("T %s" % ch)
    if n >= 2:
        prev = 1  # rule id of the running concatenation so far (starts as char 1)
        for i in range(2, n + 1):
            new_id = n + (i - 1)
            out.append("C %d %d" % (prev, i))
            prev = new_id
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
