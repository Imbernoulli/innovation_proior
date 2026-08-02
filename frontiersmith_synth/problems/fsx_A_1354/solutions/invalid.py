# TIER: invalid
"""Emits a syntactically well-formed but wrong artifact: a single rule expanding to just
"H" (length 1), which can never equal the true crossing-sequence length (>=5 on every
test case in this problem) -> the checker's length check fails -> Ratio 0.0 always."""
import sys


def main():
    sys.stdin.read()  # ignore the instance entirely
    out = ["1", "0 H 1 H 0", "ANSWER 0"]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
