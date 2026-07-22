# TIER: strong
# Max-min budget reallocation. The 1% false-positive allowance is a single
# divisible resource shared by all K sensors, and (since each attack family
# only ever shows a signal on its own fingerprint sensor) each sensor's
# threshold determines detection for whichever family(ies) fingerprint it.
# Start from the "equal share" operating point (greedy's recipe: fp_cap/K per
# sensor, OR-fused) -- this is a floor we can never do worse than. Then
# repeatedly search, at several budget-transfer granularities (coarse to
# fine, so we don't get stuck in the tiny plateaus a single fixed step size
# would hit), for a transfer of allowance from one sensor to another --
# funded either from unused slack or by tightening a donor sensor -- that
# raises the CURRENT WORST family's detection, re-verifying the exact fused
# false-positive rate (not an additive estimate) after every trial. This
# concentrates the shared allowance on whichever sensors actually need it
# for the hardest-to-separate families, instead of spreading it thin.
import sys, json


def eval_config(K, F, L, n, benign, attacks, theta, w, tau):
    fp = 0
    for row in benign:
        s = 0.0
        for c in range(K):
            if row[c] > theta[c]:
                s += w[c]
        if s >= tau:
            fp += 1
    fp_rate = fp / n
    qfs = []
    for f in range(F):
        num = 0.0
        den = 0.0
        for l in range(L):
            samples = attacks[f][l]
            cnt = 0
            for vec in samples:
                s = 0.0
                for c in range(K):
                    if vec[c] > theta[c]:
                        s += w[c]
                if s >= tau:
                    cnt += 1
            det = cnt / len(samples)
            weight = l + 1
            num += weight * det
            den += weight
        qfs.append(num / den if den > 0 else 0.0)
    qmin = min(qfs) if qfs else 0.0
    return fp_rate, qmin, qfs


def solve(inst):
    K = inst["channels"]; F = inst["families"]; L = inst["levels"]
    n = inst["n_benign"]; fp_cap = inst["fp_cap"]
    benign = inst["benign"]; attacks = inst["attacks"]

    sorted_col = [sorted(row[c] for row in benign) for c in range(K)]

    def theta_at_k(c, k):
        col = sorted_col[c]
        if k <= 0:
            return col[-1] + 1.0
        if k >= n:
            return col[0] - 1.0
        return col[n - 1 - k]

    def theta_of(ks):
        return [theta_at_k(c, ks[c]) for c in range(K)]

    def ev(ks):
        return eval_config(K, F, L, n, benign, attacks, theta_of(ks), [1.0] * K, 1.0)

    cap_k = fp_cap * n
    steps = sorted(set(max(1, int(cap_k / d)) for d in (2, 4, 8, 16, 32, 64, 128)), reverse=True)
    MAXK = int(cap_k * 8)

    k0 = max(1, int((fp_cap / K) * n))
    cur_k = [k0] * K
    fp, qmin, qfs = ev(cur_k)
    while fp > fp_cap + 1e-12 and max(cur_k) > 0:
        for c in range(K):
            if cur_k[c] > 0:
                cur_k[c] = max(0, cur_k[c] - steps[0])
        fp, qmin, qfs = ev(cur_k)

    for _ in range(60):
        best_move = None
        best_qmin = qmin
        for step in steps:
            for r in range(K):
                if cur_k[r] + step > MAXK:
                    continue
                trial = list(cur_k); trial[r] += step
                fp2, qmin2, qfs2 = ev(trial)
                if fp2 <= fp_cap + 1e-12 and qmin2 > best_qmin + 1e-9:
                    best_qmin = qmin2; best_move = (trial, fp2, qmin2, qfs2)
            for d in range(K):
                if cur_k[d] <= 0:
                    continue
                for r in range(K):
                    if r == d or cur_k[r] + step > MAXK:
                        continue
                    trial = list(cur_k)
                    trial[d] = max(0, trial[d] - step)
                    trial[r] += step
                    fp2, qmin2, qfs2 = ev(trial)
                    if fp2 <= fp_cap + 1e-12 and qmin2 > best_qmin + 1e-9:
                        best_qmin = qmin2; best_move = (trial, fp2, qmin2, qfs2)
            if best_move is not None:
                break
        if best_move is None:
            break
        cur_k, fp, qmin, qfs = best_move

    return theta_of(cur_k), [1.0] * K, 1.0


inst = json.load(sys.stdin)
theta, w, tau = solve(inst)
print(json.dumps({"theta": theta, "w": w, "tau": tau}))
