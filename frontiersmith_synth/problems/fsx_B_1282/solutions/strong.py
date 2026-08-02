# TIER: strong
"""The insight: manipulation has an ASYMMETRY legitimate two-sided quoting does
not. Instead of asking "how many/how fast were the cancels" (side- and
timing-blind, and therefore floods on market makers), for each participant-
window we measure:
  (1) SIDE ASYMMETRY of the cancel flow -- a layering burst dumps cancels on
      one side; a market maker cancels roughly symmetrically on both sides.
  (2) a SAME-SIDE aggressive trade shortly AFTER most of that side's cancels
      -- the causal signature of "clear the fake depth, then trade into the
      move you just faked." A market maker's rare trades are not temporally
      tied to a same-side cancel cluster.
Only windows with both a strong asymmetry AND a same-side causal follow-up
score highly, which lets the alert budget concentrate on genuine layering
instead of drowning in benign lookalikes."""
import sys

GAP_MAX = 8      # ticks: how "shortly after" a follow-up trade must land
MIN_ACT = 4       # ignore near-idle windows (too little signal either way)


def main():
    data = sys.stdin.read().split("\n")
    idx = 0
    test_id = int(data[idx].strip()); idx += 1
    N, W, K = map(int, data[idx].split()); idx += 1
    E = int(data[idx].strip()); idx += 1

    cancels = {}   # (w,pid) -> {'B':[t,...], 'S':[t,...]}
    trades = {}    # (w,pid) -> {'B':[t,...], 'S':[t,...]}
    for i in range(E):
        toks = data[idx + i].split()
        w, pid, t, side, action, size = toks
        w = int(w); pid = int(pid); t = int(t)
        key = (w, pid)
        if action == "C":
            cancels.setdefault(key, {"B": [], "S": []})[side].append(t)
        elif action == "T":
            trades.setdefault(key, {"B": [], "S": []})[side].append(t)
    idx += E

    def score_pw(pw):
        c = cancels.get(pw, {"B": [], "S": []})
        cb, cs = len(c["B"]), len(c["S"])
        total = cb + cs
        if total < MIN_ACT:
            return 0.0
        asym = abs(cb - cs) / total
        dom_side = "B" if cb >= cs else "S"
        dom_times = sorted(c[dom_side])
        if not dom_times:
            return 0.0
        # "most" of the dominant side's cancels have fired by this tick
        thresh_idx = max(0, (len(dom_times) * 3) // 5 - 1)
        thresh_time = dom_times[thresh_idx]
        tr = trades.get(pw, {"B": [], "S": []})[dom_side]
        followup = any(thresh_time < t <= thresh_time + GAP_MAX for t in tr)
        return total * (asym ** 2) * (1.0 if followup else 0.15)

    all_pw = [(w, pid) for w in range(W) for pid in range(N)]
    score = {pw: score_pw(pw) for pw in all_pw}
    all_pw.sort(key=lambda pw: (-score[pw], pw[0], pw[1]))
    flagged = all_pw[:K]

    out = [str(len(flagged))]
    for (w, pid) in flagged:
        out.append(f"{w} {pid}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
