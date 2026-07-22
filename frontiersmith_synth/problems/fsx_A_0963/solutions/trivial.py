# TIER: trivial
"""
Naive flat-rate guess: assume there is no bracket structure at all, and only bother
skimming the lower (more legible) half of the archived wage range. Fit a single
economy-wide marginal rate from the ratio of total hours to total wages across that
lower half (this is exactly the checker's own baseline construction), and submit a
flat tax T(z) = tau_bar * z. Reproduces the checker's baseline (~0.1).
"""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    N = int(data[idx]); idx += 1
    W_LO = float(data[idx]); idx += 1
    W_HI = float(data[idx]); idx += 1
    dT = float(data[idx]); idx += 1
    ws, hs = [], []
    for _ in range(N):
        w = float(data[idx]); idx += 1
        h = float(data[idx]); idx += 1
        ws.append(w); hs.append(h)
    order = sorted(range(N), key=lambda i: ws[i])
    bot = order[:max(N // 2, 1)]
    sum_w = sum(ws[i] for i in bot)
    sum_h = sum(hs[i] for i in bot)
    tau_bar = 1.0 - sum_h / sum_w if sum_w > 0 else 0.2
    tau_bar = min(max(tau_bar, 0.0), 0.95)
    print("%.6f * z" % tau_bar)


if __name__ == "__main__":
    main()
