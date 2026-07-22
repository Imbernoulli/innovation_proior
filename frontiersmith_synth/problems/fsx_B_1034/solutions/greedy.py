# TIER: greedy
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); D = int(next(it)); R = int(next(it)); cap = int(next(it))
    chain = [int(next(it)) for _ in range(D)]
    free = [int(next(it)) for _ in range(R)]

    cnt = [0] * N
    result = []

    # Chain: textbook maximally-even (Euclidean-rhythm) construction, canonical phase 0
    # for every layer. This is the natural "obvious" choice -- and it automatically
    # satisfies the nesting hierarchy, since phase 0 is common to every layer.
    for k in chain:
        step = N // k
        ons = [(t * step) % N for t in range(k)]
        for v in ons:
            cnt[v] += 1
        result.append(ons)

    # Free instruments: same textbook convention (phase 0 = "start on the downbeat"). Any
    # onset that blows the collision cap gets patched by a repair cursor that just scans
    # forward for the next open beat in absolute beat order (a plain "first fit" repair,
    # with no notion of the instrument's own modular structure or of staying anywhere near
    # the beat it's replacing). This fixes every violation it can reach, but it has no idea
    # that this instrument's onsets belonged to an arithmetic progression, so a repaired
    # onset can land anywhere in the cycle relative to its neighbors -- destroying the
    # evenness of exactly the instruments that needed a fix, and it can still run out of
    # room in a genuinely congested cycle.
    cursor = 0
    for f in free:
        step = N // f
        placed = []
        used_here = set()
        for t0 in range(f):
            t = (t0 * step) % N
            if cnt[t] < cap and t not in used_here:
                chosen = t
            else:
                chosen = None
                for _ in range(N):
                    cand = cursor % N
                    cursor += 1
                    if cnt[cand] < cap and cand not in used_here:
                        chosen = cand
                        break
                if chosen is None:
                    chosen = t  # give up: leave the collision (greedy just fails here)
            cnt[chosen] += 1
            used_here.add(chosen)
            placed.append(chosen)
        result.append(placed)

    out = [" ".join(str(x) for x in ons) for ons in result]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
