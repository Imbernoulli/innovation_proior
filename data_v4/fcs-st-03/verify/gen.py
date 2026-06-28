import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small alphabet to force frequent overlaps between patterns and text.
    alpha_size = rng.choice([1, 2, 2, 3, 3, 4])
    alphabet = "abcdefghijklmnopqrstuvwxyz"[:alpha_size]

    m = rng.randint(0, 6)
    pats = []
    for _ in range(m):
        lp = rng.randint(1, 6)
        p = "".join(rng.choice(alphabet) for _ in range(lp))
        # weights may be negative, zero, or positive
        w = rng.randint(-5, 5)
        pats.append((p, w))

    # text: may be empty sometimes to test that corner
    if rng.random() < 0.1:
        text = ""
    else:
        lt = rng.randint(0, 12)
        text = "".join(rng.choice(alphabet) for _ in range(lt))

    out = []
    out.append(str(m))
    for p, w in pats:
        out.append(p + " " + str(w))
    # Always emit a text line. If empty, emit a single dot placeholder?  No --
    # both sol and brute must agree.  We emit a token that is guaranteed parseable.
    # To keep an explicit empty-text representable on a whitespace-split reader,
    # we never emit a truly empty text here when m==0 would make text the only
    # token; instead, when text=="" we still print an empty line, and both readers
    # treat "missing token" as empty.
    if text == "":
        out.append("")
    else:
        out.append(text)
    sys.stdout.write("\n".join(out) + "\n")

main()
