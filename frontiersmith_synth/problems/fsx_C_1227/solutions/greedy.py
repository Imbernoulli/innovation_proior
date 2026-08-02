# TIER: greedy
"""The obvious first attempt: measure the trace's aggregate behaviour ONCE and
apply a single global policy to every line -- simulate ALL-INV vs ALL-UPD over
the whole trace and pick whichever total is cheaper. This is the textbook
"pick the protocol that matches the workload's overall read/write mix"
heuristic. It has no notion that different lines want different treatment,
so it is badly wrong on any line whose own pattern disagrees with the
trace-wide average (write-heavy-but-shared lines when the trace is mostly
read-heavy, or vice versa, and any line whose regime changes mid-trace)."""
import sys


def simulate_line(events, MISS, INVC, UPDC, mode):
    valid = set()
    cost = 0.0
    for c, op in events:
        if op == "R":
            if c not in valid:
                cost += MISS
                valid.add(c)
        else:
            others = valid - {c}
            if mode == "INV":
                cost += INVC * len(others)
                valid = {c}
            else:
                cost += UPDC * len(others)
                valid.add(c)
    return cost


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    T = int(nxt())
    C = int(nxt())
    L = int(nxt())
    MISS = float(nxt())
    INVC = float(nxt())
    UPDC = float(nxt())
    events_by_line = [[] for _ in range(L)]
    for _ in range(T):
        c = int(nxt())
        l = int(nxt())
        op = nxt()
        events_by_line[l].append((c, op))

    total_inv = sum(simulate_line(ev, MISS, INVC, UPDC, "INV") for ev in events_by_line)
    total_upd = sum(simulate_line(ev, MISS, INVC, UPDC, "UPD") for ev in events_by_line)
    chosen = "INV" if total_inv <= total_upd else "UPD"

    print("\n".join(["%s 1 0.0" % chosen] * L))


if __name__ == "__main__":
    main()
