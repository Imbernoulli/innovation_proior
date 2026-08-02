# TIER: greedy
# The obvious "textbook" allocator policy: never collect while there is still
# room, only collect (minor only, never major) the instant the next
# allocation would overflow the young generation. This minimizes the NUMBER
# of collections and total scan work on smooth traffic, but on a bursty trace
# it lets a whole burst pile up before being forced to sweep it all in one
# gigantic, budget-busting pause.
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    T = int(data[idx]); idx += 1
    Y = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    idx += 1  # Bpause (greedy ignores the pause budget entirely -- the trap)
    idx += 6  # c0_minor c0_major k_young k_old f_rate penalty
    alloc = [int(x) for x in data[idx:idx + T]]; idx += T
    lifetime = [int(x) for x in data[idx:idx + T]]; idx += T

    young = []  # [remaining, death_time, survive_count]
    actions = [0] * T
    for t in range(1, T + 1):
        pre_y = sum(c[0] for c in young)
        if pre_y + alloc[t - 1] > Y:
            actions[t - 1] = 1
            nxt = []
            for rem, death, surv in young:
                if t >= death:
                    continue
                surv += 1
                if surv < K:
                    nxt.append([rem, death, surv])
            young = nxt
        if alloc[t - 1] > 0:
            young.append([alloc[t - 1], t + lifetime[t - 1], 0])

    print(" ".join(map(str, actions)))


if __name__ == "__main__":
    main()
