#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE market-abuse-surveillance instance to stdout.
Deterministic: seeded only by testId (see labels.py). Never prints the hidden
manipulator labels -- only the raw event stream the solver is allowed to see."""
import sys
from labels import generate_instance


def main():
    test_id = int(sys.argv[1])
    inst = generate_instance(test_id)
    N, W, K = inst["N"], inst["W"], inst["K"]
    events = inst["events"]

    out = []
    out.append(str(test_id))
    out.append(f"{N} {W} {K}")
    out.append(str(len(events)))
    for (w, pid, t, side, action, size) in events:
        out.append(f"{w} {pid} {t} {side} {action} {size}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
