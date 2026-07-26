# TIER: strong
import sys


def main():
    data = sys.stdin.read().split("\n")
    N, C, TMAX = (int(x) for x in data[0].split())
    A = [0] * (N + 1)
    deadline = [0] * (N + 1)
    for i in range(1, N + 1):
        a, d = data[i].split()
        A[i] = int(a)
        deadline[i] = A[i] + int(d)

    crashes = []
    ptr = N + 1
    for j in range(C):
        c_tick, m = (int(x) for x in data[ptr].split())
        needed = [int(x) for x in data[ptr + 1].split()]
        crashes.append((c_tick, needed))
        ptr += 2

    # Same write-side batch schedule as the recipe (deadline-driven group
    # commit) -- the insight is not about WHEN to seal, it's about what goes
    # INSIDE the seal. Recovery always re-scans a page starting from its
    # first listed record; the log is a write structure but also an index
    # that recovery reads. Weight each record by how many known fire-drills
    # will need it, and place the most crash-relevant records first inside
    # each page, so replay resolves them within a few probes instead of
    # walking the whole page.
    weight = [0] * (N + 1)
    for c_tick, needed in crashes:
        for rid in needed:
            weight[rid] += 1

    pages = []
    pending = []
    for i in range(1, N + 1):
        pending.append(i)
        min_dl = min(deadline[j] for j in pending)
        if min_dl <= A[i]:
            pages.append((min_dl, list(pending)))
            pending = []
    if pending:
        pages.append((min(deadline[j] for j in pending), list(pending)))

    out = []
    for tick, order in pages:
        order_sorted = sorted(order, key=lambda x: (-weight[x], x))
        out.append(f"{tick} {len(order_sorted)} " + " ".join(str(x) for x in order_sorted))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
