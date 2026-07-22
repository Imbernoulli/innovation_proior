# TIER: trivial
# Do-nothing baseline: fit a plain constant-period linear ephemeris a+b*k by
# least squares to the training cycles and ignore everything else (no drift,
# no wobble). This reproduces the checker's own internal baseline construction
# exactly -> Ratio ~ 0.1.
import sys


def linfit(obs):
    ks = [k for k, _ in obs]
    ts = [v for _, v in obs]
    n = len(ks)
    sk = sum(ks); skk = sum(k * k for k in ks); st = sum(ts)
    skt = sum(k * v for k, v in obs)
    den = n * skk - sk * sk
    if abs(den) < 1e-9:
        b = 0.0
        a = st / n
    else:
        b = (n * skt - sk * st) / den
        a = (st - b * sk) / n
    return a, b


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    n = int(header[0])
    obs = []
    for ln in data[1:1 + n]:
        parts = ln.split()
        if len(parts) == 2:
            obs.append((int(parts[0]), float(parts[1])))
    if not obs:
        print("0.0")
        return
    a, b = linfit(obs)
    print("%.10g + %.10g*k" % (a, b))


if __name__ == "__main__":
    main()
