"""Reference solver for NatureBench task s41592-022-01709-7 (scorer sanity check).

Cross-modal single-cell protein matching. NOT the agent: a competent, purely
unsupervised baseline used with `nb_run.py --mode reference` to show the
official scorer returns a sensible score for a real submission.

Method: intersect marker panels by normalized name, rank-normalize each shared
marker within each dataset (platform-invariant), then mutual-nearest-neighbour
matching via a KD-tree, greedily resolved by distance, and finally fill the
remaining budget so coverage reaches min(|X|,|Y|) (the metric divides by that,
so unmatched cells are pure loss). No ground truth is used.
"""
import os
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FILES = {
    "bone_marrow": ("bmcite_29000_xtest.csv", "levine32_102977_xtest.csv"),
    "cross_species": ("wcct4d_term_srcluster120k_xtest.csv",
                      "zac_Cyno_ifgn_term_srcluster120k_xtest.csv"),
    "murine_spleen": ("clean_cells_xtest.csv",
                      "citeMurine-206ForMatch-0423_whole2x_xtest.csv"),
    "pbmc": ("10xciteseq_5107_xtest.csv", "felixcytof_38866_xtest.csv"),
}


def norm_name(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def rank_normalize(a):
    """Rank-transform each column to [0,1] — robust across platforms/scales."""
    out = np.empty_like(a, dtype=np.float32)
    n = a.shape[0]
    for j in range(a.shape[1]):
        order = np.argsort(a[:, j], kind="stable")
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = np.arange(n, dtype=np.float32)
        out[:, j] = ranks / max(n - 1, 1)
    return out


def shared_features(dx, dy):
    mx = {norm_name(c): c for c in dx.columns}
    my = {norm_name(c): c for c in dy.columns}
    keys = sorted(set(mx) & set(my) - {"x", "unnamed0", ""})
    keys = [k for k in keys if pd.api.types.is_numeric_dtype(dx[mx[k]])
            and pd.api.types.is_numeric_dtype(dy[my[k]])]
    return [mx[k] for k in keys], [my[k] for k in keys]


def main():
    data_dir = os.environ["DATA_DIR"]
    out_root = os.environ["OUTPUT_DIR"]
    for inst, (fx, fy) in FILES.items():
        px, py = os.path.join(data_dir, inst, fx), os.path.join(data_dir, inst, fy)
        if not (os.path.exists(px) and os.path.exists(py)):
            print(f"[ref] {inst}: missing input, skipping", flush=True)
            continue
        dx, dy = pd.read_csv(px), pd.read_csv(py)
        cx, cy = shared_features(dx, dy)
        print(f"[ref] {inst}: X={dx.shape} Y={dy.shape} shared={len(cx)}", flush=True)
        if not cx:
            continue
        X = rank_normalize(dx[cx].to_numpy(dtype=np.float32))
        Y = rank_normalize(dy[cy].to_numpy(dtype=np.float32))

        # nearest y for each x. KD-trees degenerate above ~20 dims (this task
        # has up to 39 shared markers), so use chunked BLAS brute force:
        #   ||x-y||^2 = ||x||^2 - 2 x.y + ||y||^2  -> argmin over y
        y_sq = np.einsum("ij,ij->i", Y, Y)
        dist = np.empty(len(X), dtype=np.float32)
        idx = np.empty(len(X), dtype=np.int64)
        chunk = max(1, int(2e8 // max(len(Y), 1)))
        for s in range(0, len(X), chunk):
            xb = X[s:s + chunk]
            d = y_sq[None, :] - 2.0 * (xb @ Y.T)      # + ||x||^2 (constant per row)
            j = np.argmin(d, axis=1)
            idx[s:s + chunk] = j
            dist[s:s + chunk] = (d[np.arange(len(xb)), j]
                                 + np.einsum("ij,ij->i", xb, xb))
        order = np.argsort(dist)
        n_budget = min(len(X), len(Y))
        used_y = np.zeros(len(Y), dtype=bool)
        pairs = []
        for i in order:
            j = int(idx[i])
            if not used_y[j]:
                used_y[j] = True
                pairs.append((int(i), j))
                if len(pairs) >= n_budget:
                    break
        # fill remaining budget: unmatched cells score 0, so any extra pair is
        # a free chance at a correct match
        if len(pairs) < n_budget:
            matched_x = {i for i, _ in pairs}
            free_x = [i for i in range(len(X)) if i not in matched_x]
            free_y = list(np.flatnonzero(~used_y))
            for i, j in zip(free_x, free_y):
                pairs.append((int(i), int(j)))
                if len(pairs) >= n_budget:
                    break

        out_dir = os.path.join(out_root, inst)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(pairs, columns=["x_index", "y_index"]).to_csv(
            os.path.join(out_dir, "matching.csv"), index=False)
        print(f"[ref] {inst}: wrote {len(pairs)} pairs (budget {n_budget})", flush=True)


if __name__ == "__main__":
    main()
