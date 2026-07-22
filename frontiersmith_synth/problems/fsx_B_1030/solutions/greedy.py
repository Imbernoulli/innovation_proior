# TIER: greedy
# The obvious first idea: "pay more for harder fields."  Post a rate that
# scales linearly with each field's own difficulty -- a piece-rate MENU keyed
# only to the job's attributes.  It never looks at WHO is arriving or WHEN,
# so it has no way to reserve a hard field for the specialist who will show
# up later, nor to avoid tempting an early, poorly-matched worker who is
# happy to grab it anyway once the rate clears their (small) reservation
# wage.  This is the trap: a rate menu proportional to difficulty still gets
# cherry-picked by whoever happens to arrive first.
import sys


def main():
    it = sys.stdin.read().split()
    p = 0
    N = int(it[p]); M = int(it[p + 1]); C_UNIT = int(it[p + 2]); RMAX = int(it[p + 3]); p += 4
    fields = []
    for _ in range(N):
        v = int(it[p]); d = int(it[p + 1]); dl = int(it[p + 2]); p += 3
        fields.append((v, d, dl))
    # workers follow but are irrelevant to this recipe
    DMAX = 10
    out = []
    for (v, d, dl) in fields:
        frac = 0.30 + 0.35 * (d - 1) / (DMAX - 1)
        r = int(round(RMAX * frac))
        r = max(0, min(RMAX, r))
        out.append(str(r))
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
