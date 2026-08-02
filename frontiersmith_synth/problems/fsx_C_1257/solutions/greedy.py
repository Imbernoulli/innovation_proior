# TIER: greedy
"""
The obvious "safe" recipe: a fixed sampling rate that spends (almost) the ENTIRE energy budget,
spread evenly over the whole horizon, never touching the cheap precursor channel. Under uniform
/ scattered arrivals this is genuinely close to the best any fixed rate can do -- it is a
perfectly sensible first idea and it never wastes energy. But every slot gets the SAME period
regardless of what the input says about precursor warnings or cluster structure, so on streams
where events arrive in short, tightly-packed bursts a period tuned to the whole horizon almost
never lands more than one or two hits inside any given burst.
"""
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[0])
    E = int(data[1])
    e_full = int(data[2])

    n_full = max(1, E // e_full)
    P0 = max(1, T // n_full)
    print(f"{P0} 0 {P0} 0")


if __name__ == "__main__":
    main()
