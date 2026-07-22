# TIER: strong
# Insight: reparametrize each ride by the EQUILIBRIUM number of buyers F, not by price.
# In equilibrium the marginal (F-th) buyer is indifferent, so the price that sustains
# exactly F buyers is p(F) = v_[F] * Savings(F), where Savings(F) collapses as F grows:
#   Savings(F) = (n-F)/s_reg - F/s_fast.
# Choosing F small keeps the fast lane SCARCE and Savings high -> revenue-optimal price
# sits ABOVE where any static demand curve points.  Enumerate F, score revenue+lam*CS
# at price p(F), keep the best.  (Per-ride; ignores the cross-ride budget coupling, so
# there is still headroom above this.)
import sys

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    M = int(next(it)); N = int(next(it)); LAM = int(next(it)); T = int(next(it))
    s_reg = [0]*M; s_fast = [0]*M
    for j in range(M):
        s_reg[j] = int(next(it)); s_fast[j] = int(next(it)); _r = int(next(it))
    vlist = [[] for _ in range(M)]
    for i in range(N):
        v = int(next(it)); K = int(next(it)); ks = int(next(it))
        for _ in range(ks):
            j = int(next(it)) - 1
            vlist[j].append(v)
    lam = LAM / 1000.0
    out = []
    for j in range(M):
        vs = vlist[j]
        if not vs:
            out.append("1")
            continue
        vs.sort(reverse=True)
        n_j = len(vs)
        inv_sr = 1.0 / s_reg[j]; inv_sf = 1.0 / s_fast[j]
        # prefix sum of top values
        pref = [0.0]*(n_j+1)
        for k in range(n_j):
            pref[k+1] = pref[k] + vs[k]
        best_obj = 0.0; best_p = None
        for F in range(1, n_j+1):
            sv = (n_j - F)*inv_sr - F*inv_sf      # savings with F buyers
            if sv <= 0.0:
                break                              # beyond here no positive savings
            p = vs[F-1] * sv                       # marginal indifference price
            revenue = p * F
            surplus = pref[F]*sv - p*F
            obj = revenue + lam*surplus
            if obj > best_obj:
                best_obj = obj; best_p = p
        if best_p is None:
            # no profitable scarce equilibrium -> price high enough to sell nothing
            best_p = vs[0] * ((n_j - 1)*inv_sr) + 1.0
        out.append("%.6f" % best_p)
    sys.stdout.write("\n".join(out) + "\n")

main()
