#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0468 -- "Aurora Bazaar: Shopping-Feed Re-Ranker"
(family: ml-recommendation-rank; format B, quality-metric).

THEME.  An online marketplace ("Aurora Bazaar") assembles a personalised SHOPPING
FEED for each shopper session.  A retrieval stage has already produced a small
CANDIDATE SET of products; the job of the re-ranker is to ORDER those candidates so
the products the shopper will actually engage with (view / add-to-cart / buy) sit at
the top of the feed.  Feed quality is judged by NDCG@k -- graded gain, log-position
discounting -- exactly as a production ranking team would measure it.

Each session is its OWN tiny learning-to-rank problem.  The candidate is handed:
  * a LOG of historical impressions for a shopper cohort with the SAME latent taste
    weights as the session (product feature vectors + a graded engagement label 0..4);
  * the CANDIDATE SET to be ranked (product feature vectors only -- NO labels).
The taste weights are hidden; the log is the only signal for learning them.  A good
re-ranker fits a scoring model on the log and applies it to the candidate set.

PRODUCT FEATURES (d = 6, each roughly centred, per session):
  0 rating       star rating signal (its latent taste weight is ALWAYS positive:
                 quality helps in every session -- a universal prior)
  1 price        price level (taste weight varies in sign per session)
  2 popularity   trailing sales velocity          (sign varies per session)
  3 discount     promotion depth                  (sign varies per session)
  4 freshness    recency of the listing           (sign varies per session)
  5 affinity     category/brand affinity to shopper (sign varies per session)

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "d": 6, "feature_names": [...],
             "k": int,
             "train": [{"x": [f0..f5], "rel": int in 0..4}, ...],   # historical log
             "items": [[f0..f5], ...]}                              # M candidates to rank
  stdout: ONE JSON object:
            {"ranking": [p_0, p_1, ..., p_{M-1}]}
          a PERMUTATION of the M candidate indices, best-first.  The feed shows the
          top-k of this order.

  A ranking is VALID iff it is a list of exactly M integers that is a permutation of
  0..M-1 (each index once).  Invalid output, wrong length/shape, duplicate or
  out-of-range index, a crash, a timeout, or non-JSON -> that session scores 0.0.

SCORING (deterministic; no wall-time).  For each session the parent holds the HIDDEN
graded relevance labels of the candidate items.  We compute
    g_cand = NDCG@k of the candidate's ranking
    g_base = NDCG@k of the AS-PRESENTED order (the weak reference; order is
             uncorrelated with relevance, so this is a random-ordering baseline)
and normalise with an affine anchor (weak baseline -> 0.1, perfect NDCG -> 1.0):
    r = clamp( 0.1 + 0.9 * (g_cand - g_base) / max(1e-9, 1.0 - g_base), 0, 1 )
  Reproducing the presented order scores ~0.1; a ranker that recovers the hidden
  labels perfectly would score 1.0.  Because the labels carry irreducible per-item
  noise, even an oracle that knows the true taste weights cannot reach NDCG = 1.0 ->
  strong learned rankers stay strictly below the ceiling (headroom).

ISOLATION.  The candidate is untrusted and runs in a FRESH OS-SANDBOXED SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance (log + candidate
features).  The hidden relevance labels, the latent taste weights, and the NDCG
references are computed by THIS parent process, so a frame-walking / filesystem-
probing candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all sessions, in [0,1]>
  Vector: [r_1, r_2, ...]        # per-session normalised scores
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
class LCG:
    """Pure-Python 64-bit LCG -> deterministic uniform floats/ints (no numpy)."""
    def __init__(self, seed):
        self.s = (seed * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)

    def _next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.s

    def unif(self, lo=0.0, hi=1.0):
        # 53 bits of mantissa from the high bits
        u = (self._next() >> 11) / float(1 << 53)
        return lo + (hi - lo) * u

    def randint(self, lo, hi):
        return lo + (self._next() >> 17) % (hi - lo + 1)


# ----------------------------- instance family -----------------------------
FEATURE_NAMES = ["rating", "price", "popularity", "discount", "freshness", "affinity"]
D = 6


def _make_features(rng):
    """One product feature vector, each component roughly centred in [-1, 1]."""
    return [rng.unif(-1.0, 1.0) for _ in range(D)]


def _make_weights(rng):
    """Latent taste weights.  Feature 0 (rating) always positive (universal quality
    prior); the rest vary in sign per session (session-specific taste)."""
    w = [rng.unif(0.8, 1.5)]
    for _ in range(D - 1):
        w.append(rng.unif(-1.2, 1.2))
    return w


def _latent_score(w, x, noise):
    return sum(w[j] * x[j] for j in range(D)) + noise


def _grade(scores):
    """Turn continuous latent scores into graded relevance 0..4 by within-set
    quantiles (top slice = 4, ... , bottom = 0).  Deterministic; stable on ties."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: (scores[i], i), reverse=True)
    # cumulative fraction cut points -> label
    cuts = [(0.08, 4), (0.23, 3), (0.48, 2), (0.75, 1), (1.01, 0)]
    rel = [0] * n
    for rank, idx in enumerate(order):
        frac = (rank + 1) / n
        for c, lab in cuts:
            if frac <= c:
                rel[idx] = lab
                break
    return rel


def _build_session(seed, m, t, k, noise_amp):
    """Build ONE learning-to-rank session.  Returns (public, hidden_rel)."""
    rng = LCG(seed)
    w = _make_weights(rng)

    # historical log (same taste weights, independent items + noise)
    train_x = [_make_features(rng) for _ in range(t)]
    train_scores = [_latent_score(w, x, rng.unif(-noise_amp, noise_amp)) for x in train_x]
    train_rel = _grade(train_scores)
    train = [{"x": train_x[i], "rel": train_rel[i]} for i in range(t)]

    # candidate set to be ranked (presented in generation order = random wrt relevance)
    items = [_make_features(rng) for _ in range(m)]
    item_scores = [_latent_score(w, x, rng.unif(-noise_amp, noise_amp)) for x in items]
    hidden_rel = _grade(item_scores)

    public = {"name": f"session{seed}", "d": D, "feature_names": list(FEATURE_NAMES),
              "k": k, "train": train, "items": items}
    return public, hidden_rel


def _build_sessions():
    """Deterministic session family: (seed, m_items, t_train, k, noise_amp)."""
    specs = [
        (1001, 24, 80, 10, 1.4),
        (1002, 26, 80, 10, 1.4),
        (1003, 22, 70, 8, 1.5),
        (1004, 28, 90, 10, 1.4),
        (1005, 24, 60, 10, 1.5),
        (1006, 30, 100, 12, 1.4),
        (1007, 25, 75, 10, 1.5),
        (1008, 27, 85, 10, 1.4),
        # harder / held-out: noisier labels and/or thinner logs
        (2001, 32, 55, 12, 1.5),
        (2002, 28, 50, 10, 1.6),
        (2003, 34, 65, 12, 1.5),
        (2004, 30, 45, 10, 1.7),
    ]
    out = []
    for seed, m, t, k, na in specs:
        public, hidden = _build_session(seed, m, t, k, na)
        out.append({"public": public, "hidden_rel": hidden})
    return out


# ----------------------------- NDCG ----------------------------------------
def _dcg_at_k(order, rel, k):
    s = 0.0
    for p in range(min(k, len(order))):
        g = (2 ** rel[order[p]]) - 1
        s += g / math.log2(p + 2)
    return s


def _ndcg_at_k(order, rel, k):
    ideal = sorted(range(len(rel)), key=lambda i: rel[i], reverse=True)
    idcg = _dcg_at_k(ideal, rel, k)
    if idcg <= 0.0:
        return 0.0
    return _dcg_at_k(order, rel, k) / idcg


# ----------------------------- validation ----------------------------------
def _extract_ranking(answer, m):
    """Return a valid permutation list of length m, or None."""
    if not isinstance(answer, dict):
        return None
    rk = answer.get("ranking")
    if not isinstance(rk, list) or len(rk) != m:
        return None
    seen = [False] * m
    out = []
    for v in rk:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v < 0 or v >= m or seen[v]:
            return None
        seen[v] = True
        out.append(v)
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    sessions = _build_sessions()

    vec = []
    for sess in sessions:
        public = sess["public"]
        hidden = sess["hidden_rel"]
        m = len(public["items"])
        k = public["k"]

        base_order = list(range(m))            # as-presented (weak reference)
        g_base = _ndcg_at_k(base_order, hidden, k)
        denom = 1.0 - g_base
        if denom < 1e-9:
            denom = 1e-9

        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            order = _extract_ranking(ans, m)
        except Exception:
            order = None
        if order is None:
            vec.append(0.0)
            continue

        g_cand = _ndcg_at_k(order, hidden, k)
        r = 0.1 + 0.9 * (g_cand - g_base) / denom
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
