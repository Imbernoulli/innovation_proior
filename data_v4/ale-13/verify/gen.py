#!/usr/bin/env python3
"""Instance generator for "String Reassembly" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout:

    n s
    frag_0
    frag_1
    ...
    frag_{n-1}

where `n` is the number of fragments and `s` is the alphabet size (the fragments
use the first `s` lowercase letters 'a'..'a'+s-1). Each fragment is a non-empty
string. No fragment is a substring of another fragment (such fragments would be
redundant and are filtered out at generation time), and the multiset of fragments
is presented in a shuffled order.

Instance model (shotgun fragment assembly):
  We build a hidden "source" string by a low-entropy process (a small alphabet so
  that overlaps between fragments are plentiful), then sample many overlapping
  substrings of the source as the fragments. This is exactly the regime where a
  short common superstring exists and where greedy max-overlap merging followed by
  reordering of the merge sequence pays off. The source is NEVER revealed; the
  solver only sees the shuffled fragments and must reassemble a short superstring
  that contains every fragment as a (contiguous) substring.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0xA1E_0013 ^ (seed * 2654435761 & 0xFFFFFFFF))

    # Alphabet size: small alphabets -> rich overlaps -> harder/denser SCS.
    s = rng.randint(2, 4)
    alpha = [chr(ord('a') + i) for i in range(s)]

    # Hidden source length and number of fragments (deterministic from seed).
    src_len = rng.randint(1500, 4000)
    n_target = rng.randint(150, 400)

    # Build the hidden source. To create genuine long-range overlap structure
    # (not just i.i.d. noise), stitch the source from a handful of repeated
    # "motifs" interleaved with random filler. This yields a string with both
    # exact repeats (long overlaps possible) and unique regions.
    num_motifs = rng.randint(3, 7)
    motifs = []
    for _ in range(num_motifs):
        mlen = rng.randint(8, 24)
        motifs.append("".join(rng.choice(alpha) for _ in range(mlen)))

    src_chars = []
    while len(src_chars) < src_len:
        if rng.random() < 0.55 and motifs:
            src_chars.extend(rng.choice(motifs))
        else:
            run = rng.randint(3, 12)
            src_chars.extend(rng.choice(alpha) for _ in range(run))
    source = "".join(src_chars[:src_len])
    L = len(source)

    # Sample fragments as substrings of the source. Fragment lengths are drawn so
    # that consecutive samples overlap heavily on average (coverage > 1), which is
    # what makes a short superstring achievable.
    frags = []
    for _ in range(n_target):
        flen = rng.randint(12, 40)
        if flen >= L:
            flen = L
        start = rng.randint(0, L - flen)
        frags.append(source[start:start + flen])

    # Deduplicate and remove fragments that are substrings of another fragment
    # (they are redundant: covering the longer one covers them for free). Keep the
    # remaining ones; this also guarantees no fragment is a substring of another,
    # which keeps the scoring contract clean.
    uniq = list(dict.fromkeys(frags))  # stable de-dup
    uniq.sort(key=lambda t: (-len(t), t))  # longest first
    kept = []
    for f in uniq:
        if any(f in g for g in kept):
            continue
        kept.append(f)

    # If filtering shrank the set too much, top up with fresh random substrings
    # (still substrings of the source, still filtered for the no-substring rule).
    guard = 0
    while len(kept) < max(60, n_target // 2) and guard < 20 * n_target:
        guard += 1
        flen = rng.randint(12, 40)
        if flen >= L:
            flen = L
        start = rng.randint(0, L - flen)
        f = source[start:start + flen]
        if any(f in g for g in kept):
            continue
        kept = [g for g in kept if g not in f]  # drop any now-redundant shorter ones
        kept.append(f)

    rng.shuffle(kept)
    n = len(kept)
    s_used = len(set("".join(kept)))  # actual distinct symbols present
    # Report the alphabet size as the number of distinct symbols actually used so
    # the contract is exact; pad to at least 1.
    s_used = max(1, s_used)

    out = [f"{n} {s_used}"]
    out.extend(kept)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
