# TIER: greedy
import sys
import heapq


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    balance = [0] * (n + 1)
    for _ in range(m):
        d = int(next(it)); c = int(next(it)); a = int(next(it))
        balance[d] -= a
        balance[c] += a

    # Textbook "minimum cash flow" recipe: repeatedly match the current largest
    # net creditor against the current largest net debtor, transfer the smaller
    # magnitude, and repeat. This is the reflex an average strong coder writes
    # first. It never looks for exact zero-sum subgroups; it only chases the
    # single biggest imbalance each round.
    creditors = []  # max-heap via negation: (-balance, party)
    debtors = []    # max-heap via negation of |balance|: (-(-balance), party) i.e. (balance, party) since balance<0
    for i in range(1, n + 1):
        if balance[i] > 0:
            heapq.heappush(creditors, (-balance[i], i))
        elif balance[i] < 0:
            heapq.heappush(debtors, (balance[i], i))  # balance[i] negative -> smallest (most negative) pops first

    transfers = []
    while creditors and debtors:
        c_amt, c_id = heapq.heappop(creditors)
        d_amt, d_id = heapq.heappop(debtors)
        c_amt = -c_amt          # positive amount creditor is owed
        d_amt = -d_amt          # positive amount debtor owes
        pay = min(c_amt, d_amt)
        if pay > 0:
            transfers.append((d_id, c_id, pay))
        c_amt -= pay
        d_amt -= pay
        if c_amt > 0:
            heapq.heappush(creditors, (-c_amt, c_id))
        if d_amt > 0:
            heapq.heappush(debtors, (-d_amt, d_id))

    out = [str(len(transfers))]
    for u, v, a in transfers:
        out.append(f"{u} {v} {a}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
