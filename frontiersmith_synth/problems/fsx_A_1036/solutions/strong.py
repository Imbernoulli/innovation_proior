# TIER: strong
"""The insight: since no string can satisfy every DFA, the real target is a MAX-WEIGHT
SUBSET of DFAs realized by one string -- decided globally, not by priority order. Build
the weighted PRODUCT automaton implicitly: track the reachable set of joint states
(one component per DFA) as the string is extended, symbol by symbol. Because reward
(sum of weights of DFAs currently in an accepting state) never decreases as the string
grows -- each component only ever moves from "in progress" to a permanent accept or a
permanent dead sink -- the best achievable total is simply the best reward seen at any
reachable joint state up to length Lmax. This explores every combination of "which
DFAs to satisfy together" simultaneously (a longest-path search in the weighted
product automaton), which is exactly what a heaviest-first priority merge cannot do:
it never even considers abandoning the heaviest DFA in favor of a heavier COMBINATION."""
import sys


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    def nxt():
        return int(next(it))
    m = nxt(); A = nxt(); Lmax = nxt()
    dfas = []
    for _ in range(m):
        n = nxt(); start = nxt(); w = nxt()
        k = nxt()
        accept = set(nxt() for _ in range(k))
        trans = [[nxt() for _ in range(A)] for _ in range(n)]
        dfas.append({"n": n, "start": start, "w": w, "accept": accept, "trans": trans})
    return m, A, Lmax, dfas


def main():
    m, A, Lmax, dfas = read_instance()

    def reward(state):
        return sum(d["w"] for d, s in zip(dfas, state) if s in d["accept"])

    start_state = tuple(d["start"] for d in dfas)
    frontier = {start_state: []}

    best_r, best_depth, best_path = reward(start_state), 0, list(frontier[start_state])

    for depth in range(1, Lmax + 1):
        new_frontier = {}
        for state, path in frontier.items():
            for a in range(A):
                ns = tuple(d["trans"][s][a] for d, s in zip(dfas, state))
                if ns not in new_frontier:
                    new_frontier[ns] = path + [a]
        frontier = new_frontier
        for state, path in frontier.items():
            r = reward(state)
            if r > best_r + 1e-9:
                best_r, best_depth, best_path = r, depth, path

    print(len(best_path))
    print(" ".join(map(str, best_path)))


if __name__ == "__main__":
    main()
