import random
import sys


def build(test_id: int) -> str:
    """Deterministically build a target string T with a nested, palindrome-laced
    repeat hierarchy: B_0 is a longish random "hard" seed over a wide alphabet
    (so it and the level paddings carry little accidental redundancy of their
    own); each level wraps the previous block with fresh random padding, then
    appends a REVERSED copy of the block (planted mirror repeat).  Depth is
    kept small (2..4) and the hard/random content deliberately large relative
    to the mirrored core, so overall compressibility stays well short of the
    scoring cap for any approach -- only an algorithm that can recognize
    "this span equals the reverse of that span" can collapse the mirrored
    half for (near-)free; a purely forward, pair-frequency approach must pay
    for it again from scratch.  This composes hierarchical-factoring (B_k
    built from B_{k-1}) and repeat-detection (the reversed-repeat) into one
    instance, seeded only by test_id."""
    rng = random.Random(20260708 + 97 * test_id)
    alphabet = "abcdefgh"

    base_len = 40 + 8 * (test_id - 1)      # 40..112, grows with test_id
    pad_len = 10 + 2 * (test_id - 1)       # 10..28, grows with test_id
    depth = 2 + (test_id - 1) // 5         # 2 for test_id 1..5, 3 for 6..10

    block = "".join(rng.choice(alphabet) for _ in range(base_len))
    for _ in range(depth):
        p = "".join(rng.choice(alphabet) for _ in range(pad_len))
        q = "".join(rng.choice(alphabet) for _ in range(pad_len))
        block = p + block + block[::-1] + q
    return block


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    t = build(test_id)
    sys.stdout.write(t + "\n")


if __name__ == "__main__":
    main()
