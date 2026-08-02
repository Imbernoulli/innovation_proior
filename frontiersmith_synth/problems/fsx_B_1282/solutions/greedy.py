# TIER: greedy
"""The obvious upgrade over raw counting: flag "rapid order-cancel sequences"
by measuring peak cancel BURSTINESS (max cancels inside any short rolling
local-time window), still ignoring which side the cancels sit on and whether
an aggressive trade follows. This catches layering bursts a bit better than
plain counting -- but a market maker's rapid re-quote cycles are *also* bursty
on both sides, so the alert budget still floods with market makers. This is
the recipe the innovation_hook calls out: "flagging rapid order-cancel
sequences catches manipulation and also catches every market maker"."""
import sys

WINLEN = 3  # local-time window (ticks) used to measure burst rate


def burst_rate(times):
    """max number of cancel timestamps inside any window of length WINLEN."""
    if not times:
        return 0
    times = sorted(times)
    best = 0
    lo = 0
    for hi in range(len(times)):
        while times[hi] - times[lo] >= WINLEN:
            lo += 1
        best = max(best, hi - lo + 1)
    return best


def main():
    data = sys.stdin.read().split("\n")
    idx = 0
    test_id = int(data[idx].strip()); idx += 1
    N, W, K = map(int, data[idx].split()); idx += 1
    E = int(data[idx].strip()); idx += 1

    cancel_times = {}
    for i in range(E):
        toks = data[idx + i].split()
        w, pid, t, side, action, size = toks
        w = int(w); pid = int(pid); t = int(t)
        if action == "C":
            key = (w, pid)
            cancel_times.setdefault(key, []).append(t)
    idx += E

    all_pw = [(w, pid) for w in range(W) for pid in range(N)]
    score = {pw: burst_rate(cancel_times.get(pw, [])) for pw in all_pw}
    all_pw.sort(key=lambda pw: (-score[pw], pw[0], pw[1]))
    flagged = all_pw[:K]

    out = [str(len(flagged))]
    for (w, pid) in flagged:
        out.append(f"{w} {pid}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
