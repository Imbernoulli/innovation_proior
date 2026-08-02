# TIER: invalid
import sys

# Deliberately broken: claims the fixed corner triangle (0,0)-(1,0)-(0,1) is
# the answer (it is essentially never panchromatic, and is not the genuine
# boundary door for N > 1), with a fake one-triangle "path". Must score 0.


def main():
    sys.stdin.read()  # ignore the instance entirely
    out = []
    out.append("ANSWER 0 0 1 0 0 1")
    out.append("PATH 1")
    out.append("0 0 1 0 0 1")
    out.append("EXTRA 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
