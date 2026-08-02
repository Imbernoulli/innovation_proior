# TIER: greedy
import sys


def main():
    toks = sys.stdin.read().split()
    T, C, Qmax, n_comp = int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3])
    # base_rtt_ego, init_cwnd, ALPHA, BETA, GAMMA, competitor lines: unused by
    # this classic AIMD-on-loss controller -- it only ever reacts to LOSS.
    fair = max(1, C // (n_comp + 1))
    inc = max(1, fair // 3)

    # Additive-increase / multiplicative-decrease, reacting only to r5=loss.
    # r4=cwnd_prev. Never looks at r6 (queue/delay signal) at all.
    print("DIV r10 r4 2")
    print("MAX r11 r10 1")
    print(f"ADD r12 r4 {inc}")
    print("SEL r13 r5 r11 r12")
    print("RESULT r13 r0 r1 r2 r3")


if __name__ == "__main__":
    main()
