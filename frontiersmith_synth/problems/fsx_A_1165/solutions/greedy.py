# TIER: greedy
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

    # Textbook deadline-driven group commit: batch pending records, flush
    # (seal) the whole pending queue the moment the earliest deadline among
    # them is reached. Records go into the page in plain arrival/id order --
    # the natural thing to do, with no thought given to how recovery will
    # later scan the page.
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
        out.append(f"{tick} {len(order)} " + " ".join(str(x) for x in order))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
