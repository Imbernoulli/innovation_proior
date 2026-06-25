import sys
from functools import lru_cache

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    w = []
    for _ in range(n):
        w.append(int(data[idx])); idx += 1
    if n <= 1:
        print(0)
        return

    # Brute force: explicitly try every order of merging adjacent stacks.
    # State = tuple of current adjacent stack sizes. Total effort to collapse
    # the whole tuple into one stack, minimized over all merge orders.
    # This is exponential but fine for tiny n used in stress tests.
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def solve(state):
        m = len(state)
        if m == 1:
            return 0
        best = None
        for i in range(m - 1):
            cost = state[i] + state[i + 1]
            new_state = state[:i] + (state[i] + state[i + 1],) + state[i + 2:]
            total = cost + solve(new_state)
            if best is None or total < best:
                best = total
        return best

    print(solve(tuple(w)))

if __name__ == "__main__":
    main()
