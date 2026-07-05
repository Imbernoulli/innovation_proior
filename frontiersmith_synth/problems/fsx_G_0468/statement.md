# Aurora Bazaar: Shopping-Feed Re-Ranker (NDCG@k)

## Setting

Aurora Bazaar is an online marketplace that builds a personalised **shopping feed**
for every shopper session. A retrieval stage has already narrowed the catalogue down
to a small **candidate set** of products. Your job is the **re-ranker**: order those
candidates so the products the shopper will actually engage with (view / add-to-cart /
buy) sit at the top of the feed.

Feed quality is measured with **NDCG@k** over the session's graded engagement labels —
the standard production ranking metric (graded gain, log-position discounting).

Each session is its own tiny **learning-to-rank** problem. You are given a **log** of
historical impressions drawn from a shopper cohort with the *same latent taste weights*
as the session (product features **plus** a graded engagement label), and the
**candidate set** to rank (product features **only**, no labels). The taste weights are
hidden; the log is the only signal for learning them.

### Product features (`d = 6`, each roughly centred per session)

| idx | name       | meaning                                   |
|-----|------------|-------------------------------------------|
| 0   | rating     | star-rating signal (taste weight **always positive** — a universal quality prior) |
| 1   | price      | price level (taste weight sign varies per session) |
| 2   | popularity | trailing sales velocity (sign varies)     |
| 3   | discount   | promotion depth (sign varies)             |
| 4   | freshness  | listing recency (sign varies)             |
| 5   | affinity   | category/brand affinity (sign varies)     |

## Program contract (isolated stdin → stdout)

Your program reads **one** JSON object (the PUBLIC instance) from stdin and writes
**one** JSON object to stdout. It runs in an isolated sandbox and only ever sees the
public view below.

### Input (stdin)

```json
{
  "name": "session1001",
  "d": 6,
  "feature_names": ["rating","price","popularity","discount","freshness","affinity"],
  "k": 10,
  "train": [ {"x": [f0,f1,f2,f3,f4,f5], "rel": 3}, ...  ],
  "items": [ [f0,f1,f2,f3,f4,f5], ...  ]
}
```

- `train` — historical log: each record has a feature vector `x` and a graded
  engagement label `rel ∈ {0,1,2,3,4}` produced by the session's hidden taste weights.
- `items` — the `M` candidate products to rank, presented in retrieval order
  (uncorrelated with relevance). Feature vectors only.
- `k` — the NDCG cutoff used for scoring.

### Output (stdout)

```json
{"ranking": [p_0, p_1, ..., p_{M-1}]}
```

`ranking` must be a **permutation of the candidate indices `0..M-1`** (each index
exactly once), best-first. The feed shows the top-`k` of this order.

A ranking is **valid** iff it is a list of exactly `M` integers forming a permutation
of `0..M-1`. Invalid output, wrong length/shape, a duplicate or out-of-range index, a
crash, a timeout, or non-JSON → that session scores **0.0**.

## Objective

**Maximize** NDCG@k of your ranking against the hidden graded relevance labels, averaged
over all sessions.

## Scoring (deterministic)

For each session the judge holds the hidden graded labels. It computes:

- `g_cand` = NDCG@k of your ranking,
- `g_base` = NDCG@k of the as-presented order (a random-ordering reference),

and normalises with an affine anchor (weak baseline → 0.1, perfect NDCG → 1.0):

```
r = clamp( 0.1 + 0.9 * (g_cand - g_base) / max(1e-9, 1.0 - g_base), 0, 1 )
```

The reported score is the mean of `r` over all sessions (both easy and harder,
noisier held-out sessions). Reproducing the presented order scores ≈ 0.1. Because the
engagement labels carry irreducible per-item noise, even an oracle that knew the true
taste weights could not reach NDCG = 1.0, so there is genuine headroom above any
learned ranker.

## Hints / viable strategies

- **Passthrough** (returns the retrieval order) reproduces the weak reference ≈ 0.1.
- **Single-signal greedy**: sort by the `rating` feature — it has a positive taste
  weight in every session, so it reliably beats the baseline, but ignores the
  session-specific tastes.
- **Pointwise learning-to-rank**: fit a scoring model (e.g. ridge / least-squares
  regression of `rel` on features) on the log, then rank candidates by predicted score.
- **Push further**: per-session feature normalisation, pairwise/listwise objectives,
  quantile-calibrated predictions, or blending the rating prior with a learned model to
  hold up on the noisier, thin-log sessions.
