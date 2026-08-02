# TIER: strong
# Genuine pinch-respecting design, in three steps:
#
# 1. PINCH LOCATION (problem-table / heat-cascade algorithm): shift every hot
#    stream's temperatures down by DTMIN, cascade the combined hot/cold heat
#    surplus-deficit from the top temperature down, and find the minimum
#    cumulative value. That deficit is the unavoidable minimum hot utility;
#    the breakpoint where the (utility-corrected) cascade first touches zero
#    is the pinch. This is the theoretically-grounded minimum-utility target
#    -- no rearrangement of matches can beat it.
#
# 2. NO HEAT ACROSS THE PINCH: each stream's duty is split into an "above
#    pinch" share and a "below pinch" share (a stream that straddles the pinch
#    contributes to both). Matches are only ever formed WITHIN a region: an
#    above-pinch chunk of a hot stream can only feed an above-pinch chunk of a
#    cold stream, and likewise below. This is exactly what the naive
#    hottest-to-coldest recipe violates -- it happily uses an above-pinch hot
#    stream's duty to satisfy a below-pinch cold demand because the local
#    driving force looks huge, which starves whoever actually needed that
#    duty on the correct side and forces MORE of both utilities network-wide.
#
# 3. AREA-VS-UTILITY: within a region, a candidate match is only worth making
#    if its area cost per unit duty (A/LMTD) is less than the utility it
#    displaces (CH+CC per unit duty, since one unit of matched duty removes
#    one unit of cold utility from the hot side AND one unit of hot utility
#    from the cold side simultaneously). Matches that fail this test are left
#    to utility on purpose.
import math
import sys


def lmtd(d1, d2):
    if abs(d1 - d2) < 1e-6:
        return d1
    return (d1 - d2) / math.log(d1 / d2)


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

    # ---- 1. problem-table cascade on shifted temperatures ----
    shot = [(ths - dtmin, tht - dtmin, cp) for (ths, tht, cp) in hots]
    bps = set()
    for (ts, tt, _) in shot:
        bps.add(ts); bps.add(tt)
    for (ts, tt, _) in colds:
        bps.add(ts); bps.add(tt)
    bps = sorted(bps, reverse=True)

    cum = [0.0]
    c = 0.0
    for k in range(len(bps) - 1):
        thi, tlo = bps[k], bps[k + 1]
        cph = sum(cp for (ts, tt, cp) in shot if tt <= tlo + 1e-9 and ts >= thi - 1e-9)
        cpc = sum(cp for (ts, tt, cp) in colds if ts <= tlo + 1e-9 and tt >= thi - 1e-9)
        c += (cph - cpc) * (thi - tlo)
        cum.append(c)
    m = min(cum) if cum else 0.0
    hu_min = -m if m < 0 else 0.0
    adj = [x + hu_min for x in cum]
    pk = 0
    for k, v in enumerate(adj):
        if abs(v) < 1e-6:
            pk = k
            break
    t_pinch_shifted = bps[pk] if bps else 0.0
    t_pinch_hot = t_pinch_shifted + dtmin
    t_pinch_cold = t_pinch_shifted

    # ---- 2. split every stream's duty at the pinch ----
    d_above, d_below = [], []
    for i, (ths, tht, cp) in enumerate(hots):
        da = cp * max(0.0, ths - max(tht, t_pinch_hot))
        d_above.append(da)
        d_below.append(D[i] - da)
    e_above, e_below = [], []
    for j, (tcs, tct, cp) in enumerate(colds):
        ea = cp * max(0.0, tct - max(tcs, t_pinch_cold))
        e_above.append(ea)
        e_below.append(E[j] - ea)

    thresh = cH + cC
    agg = {}
    for region in ("above", "below"):
        remH = d_above[:] if region == "above" else d_below[:]
        remC = e_above[:] if region == "above" else e_below[:]
        hidx = sorted((i for i in range(NH) if remH[i] > 1e-9), key=lambda i: hots[i][0])
        cidx = sorted((j for j in range(NC) if remC[j] > 1e-9), key=lambda j: colds[j][0])
        for i in hidx:
            if remH[i] <= 1e-9:
                continue
            for j in cidx:
                if remH[i] <= 1e-9:
                    break
                if remC[j] <= 1e-9:
                    continue
                ths, tht, _ = hots[i]
                tcs, tct, _ = colds[j]
                d1 = ths - tct
                d2 = tht - tcs
                if d1 < dtmin - 1e-9 or d2 < dtmin - 1e-9:
                    continue
                L = lmtd(d1, d2)
                if a > 0 and a / L >= thresh - 1e-12:
                    continue  # 3. area cost would exceed the utility it saves
                q = min(remH[i], remC[j])
                if q > 1e-9:
                    agg[(i, j)] = agg.get((i, j), 0.0) + q
                    remH[i] -= q
                    remC[j] -= q

    items = [(i, j, q) for (i, j), q in agg.items() if q > 1e-9]
    out = [str(len(items))]
    for (i, j, q) in items:
        out.append(f"{i + 1} {j + 1} {q:.6f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
