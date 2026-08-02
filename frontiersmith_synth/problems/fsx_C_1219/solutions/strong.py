# TIER: strong
import sys


def main():
    toks = sys.stdin.read().split()
    T, C, Qmax, n_comp = int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3])
    fair = max(1, C // (n_comp + 1))
    inc = max(1, fair // 4)
    # Yield BEFORE loss: once the observed backlog (r6, the delay signal)
    # exceeds a modest fraction of the buffer, hold the window instead of
    # growing it. Loss (r5) is kept only as a backstop multiplicative cut.
    thresh = max(1, Qmax // 6)

    print(f"LT r10 r6 {thresh}")        # r10 = 1 if queue is still short (room)
    print(f"ADD r11 r4 {inc}")          # grow candidate
    print("MOV r12 r4")                  # hold candidate
    print("SEL r13 r10 r11 r12")         # delay-based decision
    print("DIV r14 r4 2")
    print("MAX r15 r14 1")
    print("SEL r16 r5 r15 r13")          # loss backstop overrides delay decision
    print("RESULT r16 r0 r1 r2 r3")


if __name__ == "__main__":
    main()
