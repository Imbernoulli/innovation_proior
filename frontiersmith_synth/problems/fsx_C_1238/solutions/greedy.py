# TIER: greedy
"""The obvious first approach: each window, chase the highest-value flag
whose REQUIRES are currently satisfied. If it conflicts with something
already active, roll back the cheapest active blocker to make room.
This is optimal on independent-flag warm-ups, but on tightly coupled
conflict clusters it thrashes -- repeatedly evicting the flag it just
enabled to chase the next-highest value -- burning rollout windows on
rollbacks and permanently forfeiting the flags it rolls back."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = [0]

    def nxt():
        v = data[pos[0]]
        pos[0] += 1
        return v

    N = int(nxt())
    values = [int(nxt()) for _ in range(N)]
    R = int(nxt())
    req_of = {i: [] for i in range(1, N + 1)}
    for _ in range(R):
        c = int(nxt())
        p = int(nxt())
        req_of[c].append(p)
    C = int(nxt())
    conf_of = {i: set() for i in range(1, N + 1)}
    for _ in range(C):
        a = int(nxt())
        b = int(nxt())
        conf_of[a].add(b)
        conf_of[b].add(a)

    active = set()
    ever_enabled = set()
    ever_disabled = set()
    out = []
    for _ in range(N):
        ready = [
            f for f in range(1, N + 1)
            if f not in ever_enabled and f not in ever_disabled
            and all(p in active for p in req_of[f])
        ]
        if not ready:
            out.append("P")
            continue
        target = max(ready, key=lambda f: (values[f - 1], -f))
        blockers = [j for j in active if j in conf_of[target]]
        if blockers:
            j = min(blockers, key=lambda z: (values[z - 1], -z))
            active.discard(j)
            ever_disabled.add(j)
            out.append(f"R {j}")
        else:
            active.add(target)
            ever_enabled.add(target)
            out.append(f"E {target}")
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
