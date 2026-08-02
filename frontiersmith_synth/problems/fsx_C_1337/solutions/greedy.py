# TIER: greedy
# The "obvious" recipe: match the hottest available hot stream against the
# coldest available cold stream, repeatedly, filling as much duty as the
# minimum-approach rule and remaining capacity allow. This maximizes the LOCAL
# temperature driving force of every match it makes (cheapest area per unit
# duty it can find) -- but it commits a hot stream's duty to whichever cold
# stream happens to be coldest BEFORE checking whether some other, pickier
# stream has no other option. It never reconsiders once committed.
import sys


def main():
    tok = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = tok[p]
        p += 1
        return v

    NH = int(nxt()); NC = int(nxt())
    dtmin = float(nxt()); cH = float(nxt()); cC = float(nxt()); a = float(nxt())
    hots = []
    for _ in range(NH):
        hots.append((float(nxt()), float(nxt()), float(nxt())))
    colds = []
    for _ in range(NC):
        colds.append((float(nxt()), float(nxt()), float(nxt())))

    D = [cp * (ths - tht) for (ths, tht, cp) in hots]
    E = [cp * (tct - tcs) for (tcs, tct, cp) in colds]
    remH = D[:]
    remC = E[:]

    order_h = sorted(range(NH), key=lambda i: -hots[i][0])   # hottest supply first
    order_c = sorted(range(NC), key=lambda j: colds[j][0])   # coldest supply first

    matches = []
    for i in order_h:
        if remH[i] <= 1e-9:
            continue
        for j in order_c:
            if remH[i] <= 1e-9:
                break
            if remC[j] <= 1e-9:
                continue
            ths, tht, _ = hots[i]
            tcs, tct, _ = colds[j]
            d1 = ths - tct
            d2 = tht - tcs
            if d1 >= dtmin - 1e-9 and d2 >= dtmin - 1e-9:
                q = min(remH[i], remC[j])
                if q > 1e-9:
                    matches.append((i + 1, j + 1, q))
                    remH[i] -= q
                    remC[j] -= q

    out = [str(len(matches))]
    for (i, j, q) in matches:
        out.append(f"{i} {j} {q:.6f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
