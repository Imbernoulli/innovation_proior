# TIER: greedy
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); budget = int(next(it))
    projects = []
    for _ in range(n):
        row = [int(next(it)) for _ in range(9)]
        projects.append(row)

    # obvious first attempt: buy the cheapest verified tonnes -- sort by $/claimed-tonne
    # ascending and fill the budget. Ignores baseline gaming, additionality and permanence.
    order = sorted(range(n), key=lambda idx: (projects[idx][0], idx))
    spent = 0
    chosen = []
    for idx in order:
        c = projects[idx][0] * projects[idx][1]
        if spent + c <= budget:
            spent += c
            chosen.append(idx + 1)

    print(len(chosen))
    print(" ".join(map(str, chosen)))

main()
