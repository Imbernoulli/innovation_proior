# TIER: trivial
# Reproduces the checker's own target-blind baseline: a square-free scaffold word
# (fixed point of 0->012,1->02,2->1) with letter 0 spliced in as a separator after
# every 3rd filler.  Realizes a flat 0.25/0.25/0.25/0.25 letter mix; ignores both the
# target frequency vector and the target transition preference entirely.
import sys


def _apply_morph(word):
    m = {0: (0, 1, 2), 1: (0, 2), 2: (1,)}
    out = []
    for c in word:
        out.extend(m[c])
    return out


def scaffold3(n):
    w = [0]
    while len(w) < n:
        w = _apply_morph(w)
    return w[:n]


def build_uniform_baseline(L):
    raw = scaffold3(L + 10)
    remap = {0: 1, 1: 2, 2: 3}
    out = []
    cnt = 0
    fi = 0
    while len(out) < L:
        if cnt % 3 == 0:
            out.append(0)
            if len(out) >= L:
                break
        out.append(remap[raw[fi]])
        fi += 1
        cnt += 1
    return out[:L]


def main():
    data = sys.stdin.read().split()
    L = int(data[0])
    word = build_uniform_baseline(L)
    print(len(word))
    print("".join(str(c) for c in word))


if __name__ == "__main__":
    main()
