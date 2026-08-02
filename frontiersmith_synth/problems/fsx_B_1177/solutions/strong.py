# TIER: strong
# The insight: don't fit "the observed curve" -- fit the MODAL STRUCTURE.
# The specimen constants CA, CB, CD, CE (given in the input) fix the SHAPE of
# both candidate branches:
#   vA(f) = CA * sqrt(k*f)                       (unbounded, low-f dominant)
#   vB(f) = sqrt(k) * (CB - CD/(f+CE))            (saturating, high-f dominant)
# and the physically observed velocity is always v(f) = min(vA, vB) -- whichever
# mode actually propagates. Only the material stiffness k is unknown, and it
# multiplies BOTH branches identically, so recovering k from whichever branch
# is visible in training determines the WHOLE modal structure, including the
# branch the training data never showed. We recover k by directly minimising
# the training residual of the two-branch model itself (not a single smooth
# curve) via a coarse grid search refined by ternary search -- this correctly
# handles training data that sits entirely on one branch, straddles the
# crossing, or sits mostly on the other branch. The final answer is emitted
# as the literal min(...) expression so the checker's evaluation on ANY
# held-out frequency (below, at, or above the true crossing) uses the correct
# branch automatically.
import sys, math


def model(k, f, CA, CB, CD, CE):
    vA = CA * math.sqrt(k * f)
    vB = math.sqrt(k) * (CB - CD / (f + CE))
    return vA if vA < vB else vB


def train_loss(k, freqs, vals, CA, CB, CD, CE):
    return sum((model(k, f, CA, CB, CD, CE) - v) ** 2 for f, v in zip(freqs, vals))


def main():
    data = sys.stdin.read().split()
    if not data:
        print("1.0"); return
    n = int(data[0])
    CA, CB, CD, CE = (float(x) for x in data[2:6])
    rows = data[6:]
    freqs = [float(rows[2 * i]) for i in range(n)]
    vals = [float(rows[2 * i + 1]) for i in range(n)]

    # coarse geometric grid over plausible stiffness range
    best_k, best_loss = 1.0, None
    k = 0.05
    for _ in range(70):
        L = train_loss(k, freqs, vals, CA, CB, CD, CE)
        if best_loss is None or L < best_loss:
            best_loss, best_k = L, k
        k *= 1.15

    # ternary-search refinement around the best grid point (loss is unimodal
    # in k because both branches are monotone increasing in k for fixed f)
    lo, hi = best_k / 1.2, best_k * 1.2
    for _ in range(80):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if train_loss(m1, freqs, vals, CA, CB, CD, CE) < train_loss(m2, freqs, vals, CA, CB, CD, CE):
            hi = m2
        else:
            lo = m1
    k_hat = max(1e-6, (lo + hi) / 2.0)

    print("min ( %.6f * sqrt ( %.6f * f ) , sqrt ( %.6f ) * ( %.6f - %.6f / ( f + %.6f ) ) )"
          % (CA, k_hat, k_hat, CB, CD, CE))


if __name__ == "__main__":
    main()
