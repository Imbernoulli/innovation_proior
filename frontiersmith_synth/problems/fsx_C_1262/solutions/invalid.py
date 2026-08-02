# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)

    def nxt():
        return int(next(it))

    N = nxt()
    # Deliberately infeasible: dump EVERY neuron onto core 0, slot 0. Any
    # instance with N > slot_cap violates the per-(core,slot) capacity (and
    # almost certainly the rate budget too), so this must score 0.
    out = [str(N)]
    for _ in range(N):
        out.append("0 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
