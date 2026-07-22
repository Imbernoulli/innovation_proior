# TIER: greedy
# The "obvious" first idea: at every time step, aim at the column with the
# largest remaining nominal deficit (target - current height), tracking
# height with a naive internal model that assumes each shot lands exactly
# where aimed (h_model[aim] += 1) -- i.e. "deposit to shape" via the most
# natural priority rule, never simulating where a shot actually lands. When
# a doomed column (shadowed by an already-taller neighbor) keeps looking
# "most needed" under this model -- because the model insists it is filling
# up even though every shot is actually being stolen -- the recipe returns
# to it, and to its shadow-mates, again and again, driving the neighboring
# peak further and further past its own target while the doomed columns
# never move at all.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    L = int(next(it)); R = int(next(it)); M = int(next(it)); T = int(next(it))
    target = [int(next(it)) for _ in range(L)]

    h_model = [0] * L
    schedule = []
    for _ in range(T):
        best_i = 0
        best_def = target[0] - h_model[0]
        for i in range(1, L):
            d = target[i] - h_model[i]
            if d > best_def:
                best_def = d
                best_i = i
        schedule.append(best_i)
        h_model[best_i] += 1  # naive: assumes a direct hit, ignores shadowing

    print(" ".join(map(str, schedule)))


if __name__ == "__main__":
    main()
