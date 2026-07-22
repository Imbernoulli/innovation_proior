# TIER: invalid
# Degenerate allocation: dump the entire budget on class 0 regardless of S/K, and
# emit a list shorter/longer than S half the time -- structurally infeasible or
# catastrophically bad on essentially every instance.
import sys, json


def main():
    inst = json.load(sys.stdin)
    K = inst["K"]
    alloc = [K, 0]  # wrong length on any instance with S != 2 -> rejected;
                     # even when S==2 this ignores the true skew/variance entirely
    print(json.dumps({"alloc": alloc}))


if __name__ == "__main__":
    main()
