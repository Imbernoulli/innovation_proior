# TIER: strong
"""Content-defined chunking: cut based on a residue of the last 9 bytes
scanned (current byte + an 8-byte trailing shift register carried in
memory), not on absolute position. Wherever the exact same 9-byte window
recurs -- including at a different absolute offset after the corpus has
been edited -- the SAME window produces the SAME cut/no-cut decision, so
chunk boundaries reproduce at every recurrence of a repeated stretch. That
buys three things at once: interior blocks are exact repeats of earlier
blocks (dedup, cheaper compressed size), blocks stay small (cheap seeks and
a bounded index), and boundaries downstream of an edit resynchronize with
the base layout as soon as the trailing window has scanned past the splice
-- instead of drifting for the rest of the file the way a fixed-period
counter's boundaries do.

M is fixed (small relative to the 9-term window-sum range so the residue is
close to uniformly distributed); the trigger threshold is tuned from N alone
to target a reasonable average chunk size."""
import sys

M = 61


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    target_size = max(20, N // 30)
    thresh = max(1, min(M - 1, round(M / target_size)))
    prog = [
        "ADD r12 r8 r0",
        "ADD r12 r12 r1",
        "ADD r12 r12 r2",
        "ADD r12 r12 r3",
        "ADD r12 r12 r4",
        "ADD r12 r12 r5",
        "ADD r12 r12 r6",
        "ADD r12 r12 r7",
        f"DIV r13 r12 {M}",
        f"MUL r14 r13 {M}",
        "SUB r15 r12 r14",
        f"LT r16 r15 {thresh}",
        "RESULT r16 r8 r0 r1 r2 r3 r4 r5 r6",
    ]
    print("\n".join(prog))


if __name__ == "__main__":
    main()
