#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0464 -- "Protein Family Retrieval: Design the Similarity Kernel"
(family: ml-kernel-similarity; format B, quality-metric).

THEME.  A structural-proteomics group has profiled a batch of proteins.  Each protein
is described by a fixed-length real DESCRIPTOR VECTOR -- a concatenation of
physicochemical/compositional channels (amino-acid composition, hydrophobicity and
charge moments, secondary-structure propensities, and a block of instrument/assay
"nuisance" channels).  Every protein secretly belongs to one PROTEIN FAMILY (its fold
class).  The lab wants a content-based RETRIEVAL system: given a query protein, rank all
the others so that same-family proteins float to the top -- the classic "design a good
similarity/kernel, then do fixed nearest-neighbour retrieval" task, scored by mean
Average Precision (mAP).

THE CATCH (why raw distance is weak).  The descriptor is deliberately messy, exactly like
real proteomics features:
  * a per-protein multiplicative ABUNDANCE / LENGTH factor scales every channel of a
    protein (longer / more-abundant proteins have uniformly larger descriptors), so raw
    Euclidean distance is dominated by size, not fold;
  * each channel has its own large, protein-independent BASELINE OFFSET;
  * a majority of the channels are high-variance NUISANCE channels (assay noise) whose
    scale dwarfs the low-variance INFORMATIVE channels that actually separate families.
So a raw dot-product / Euclidean nearest-neighbour retrieves on size and noise.  A good
UNSUPERVISED kernel must undo these: normalize away the per-protein abundance, center out
per-channel offsets, and equalize per-channel scale so the quiet informative channels are
heard.  No labels are ever given -- the kernel must be designed, not trained.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N (int), "dim": D (int),
             "features": [[x_0_0, ..., x_0_{D-1}], ..., [x_{N-1}_0, ...]]}  # N descriptors
  stdout: ONE JSON object:
            {"ranking": [r_0, r_1, ..., r_{N-1}]}
          where r_i is a RANKED list of the OTHER protein indices (a permutation of
          {0..N-1}\{i}, length N-1), most-similar FIRST under the candidate's kernel.

  A submission is VALID iff `ranking` is a list of exactly N lists, and for every i the
  list r_i contains each index in {0..N-1}\{i} exactly once (integers, no duplicates, no
  self, correct length).  Any violation, non-JSON, crash, timeout, or non-finite content
  -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Family labels are HIDDEN in this parent process.
For a ranking we compute Average Precision per query (relevant = same hidden family,
excluding self) and average over queries -> mAP in [0,1].  Per instance:
    q_cand = mAP of the candidate ranking
    q_base = mAP of the internal RAW-EUCLIDEAN reference (the weak size-driven baseline)
  normalized with an affine anchor (weak baseline -> 0.1, perfect retrieval -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_cand - q_base) / (1.0 - q_base), 0, 1 )
  A candidate that reproduces raw-Euclidean scores ~0.1; a perfect retriever would score
  1.0.  Because families genuinely overlap and the informative channels are noisy, perfect
  mAP is unreachable, so even an excellent kernel stays well below 1.0 -- real headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance (features, no labels).  The
labels and the raw-Euclidean reference are computed by THIS parent, so a frame-walking /
introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def u01():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    return u01


def _gauss(u):
    a = u()
    b = u()
    if a < 1e-12:
        a = 1e-12
    return math.sqrt(-2.0 * math.log(a)) * math.cos(2.0 * math.pi * b)


# ----------------------------- instance family -----------------------------
def _build(seed, N, F, Dinf, Dnoise, sig_fam, sig_inf, sig_len):
    """Deterministic protein-descriptor batch.  Returns (features[N][D], labels[N])."""
    u = _rng(seed)
    D = Dinf + Dnoise
    # per-channel baseline offsets and per-channel scales (informative channels are quiet)
    base = [2.0 + u() * 10.0 for _ in range(D)]
    scale = [0.5 + u() * 0.5 for _ in range(Dinf)] + [2.0 + u() * 4.0 for _ in range(Dnoise)]
    # each family has an informative-channel fingerprint
    mu = [[_gauss(u) * sig_fam for _ in range(Dinf)] for _ in range(F)]
    feats = []
    labels = []
    for p in range(N):
        f = p % F                                   # balanced family assignment
        L = math.exp(_gauss(u) * sig_len)           # per-protein abundance / length factor
        v = [0.0] * D
        for d in range(Dinf):
            v[d] = base[d] + mu[f][d] + _gauss(u) * sig_inf
        for d in range(Dnoise):
            v[Dinf + d] = base[Dinf + d] + _gauss(u) * scale[Dinf + d]
        feats.append([max(0.01, L * val) for val in v])
        labels.append(f)
    # deterministic shuffle so family blocks are not adjacent
    idx = list(range(N))
    for i in range(N - 1, 0, -1):
        j = int(u() * (i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    feats = [feats[i] for i in idx]
    labels = [labels[i] for i in idx]
    return feats, labels


def _build_instances():
    # (seed, N, F, Dinf, Dnoise, sig_fam, sig_inf, sig_len)
    specs = [
        (101, 60, 5, 6, 20, 1.40, 0.50, 1.20),
        (102, 72, 6, 6, 24, 1.20, 0.50, 1.20),
        (103, 80, 6, 6, 26, 1.15, 0.50, 1.20),
        (104, 84, 7, 7, 28, 1.10, 0.55, 1.20),
        (105, 72, 6, 5, 26, 1.10, 0.55, 1.20),
        (106, 90, 8, 7, 30, 1.05, 0.55, 1.20),
        (207, 96, 8, 8, 32, 1.00, 0.55, 1.30),
        (208, 88, 7, 6, 34, 1.00, 0.60, 1.30),
        # harder / larger held-out: more families, more nuisance channels, more overlap
        (311, 108, 9, 8, 36, 0.95, 0.60, 1.30),
        (312, 100, 10, 8, 38, 0.95, 0.60, 1.35),
        (313, 112, 10, 7, 40, 0.90, 0.60, 1.35),
        (314, 120, 12, 8, 44, 0.90, 0.65, 1.40),
    ]
    out = []
    for seed, N, F, Dinf, Dnoise, sf, si, sl in specs:
        feats, labels = _build(seed, N, F, Dinf, Dnoise, sf, si, sl)
        out.append({"name": f"batch{seed}", "n": N, "dim": Dinf + Dnoise,
                    "features": feats, "labels": labels})
    return out


# ----------------------------- retrieval metric ----------------------------
def _average_precision(order, same):
    """order: ranked list of neighbour indices; same[j]=1 if j is same-family as query."""
    nrel = sum(same)
    if nrel == 0:
        return 0.0
    hits = 0
    acc = 0.0
    for k, j in enumerate(order):
        if same[j]:
            hits += 1
            acc += hits / (k + 1)
    return acc / nrel


def _mAP_from_ranking(ranking, labels):
    N = len(labels)
    tot = 0.0
    for q in range(N):
        same = [1 if labels[j] == labels[q] else 0 for j in range(N)]
        tot += _average_precision(ranking[q], same)
    return tot / N


def _base_euclid_mAP(feats, labels):
    """Weak reference: rank neighbours by raw squared-Euclidean distance (ascending)."""
    N = len(feats)
    D = len(feats[0])
    tot = 0.0
    for q in range(N):
        qv = feats[q]
        scored = []
        for j in range(N):
            if j == q:
                continue
            d2 = 0.0
            jv = feats[j]
            for t in range(D):
                dv = qv[t] - jv[t]
                d2 += dv * dv
            # similarity = -distance; sort most-similar first, deterministic index tiebreak
            scored.append((-d2, j))
        scored.sort(key=lambda t: (-t[0], t[1]))
        order = [j for _, j in scored]
        same = [1 if labels[j] == labels[q] else 0 for j in range(N)]
        tot += _average_precision(order, same)
    return tot / N


# ----------------------------- validation ----------------------------------
def _valid_ranking(answer, N):
    if not isinstance(answer, dict):
        return None
    ranking = answer.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != N:
        return None
    out = []
    for i in range(N):
        row = ranking[i]
        if not isinstance(row, list) or len(row) != N - 1:
            return None
        seen = [False] * N
        clean = []
        for g in row:
            if isinstance(g, bool) or not isinstance(g, int):
                return None
            if g < 0 or g >= N or g == i or seen[g]:
                return None
            seen[g] = True
            clean.append(g)
        out.append(clean)
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        N = inst["n"]
        feats = inst["features"]
        labels = inst["labels"]
        q_base = _base_euclid_mAP(feats, labels)
        denom = 1.0 - q_base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "n": N, "dim": inst["dim"],
                  "features": [list(row) for row in feats]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ranking = _valid_ranking(ans, N)
        except Exception:
            ranking = None
        if ranking is None:
            vec.append(0.0)
            continue
        try:
            q_cand = _mAP_from_ranking(ranking, labels)
        except Exception:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_cand - q_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
