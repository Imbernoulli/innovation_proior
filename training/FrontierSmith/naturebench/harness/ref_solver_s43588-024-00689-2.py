"""Reference solver for NatureBench task s43588-024-00689-2 (scorer sanity check).

NOT part of the agent. Used with `nb_run.py --mode reference` to prove the
official scoring path returns a sensible nonzero g for a competent submission.

Method: standard scanpy preprocessing on the concatenated spliced+unspliced
HVG matrices, PCA, then KMeans with the known K. No ground truth is used.
"""
import json
import os

import numpy as np


def load_instance(data_path):
    import loompy
    import glob

    loom_files = sorted(glob.glob(os.path.join(data_path, "*hvgs*.loom")) or
                        glob.glob(os.path.join(data_path, "*.loom")))
    with loompy.connect(loom_files[0], mode="r") as ds:
        spliced = ds.layers["spliced"][:, :].T.astype(np.float32)    # cells x genes
        unspliced = ds.layers["unspliced"][:, :].T.astype(np.float32)
    with open(os.path.join(data_path, "metadata.json")) as fh:
        meta = json.load(fh)
    return spliced, unspliced, int(meta["n_clusters"])


def featurize(counts):
    import scanpy as sc
    import anndata

    adata = anndata.AnnData(X=counts)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    n_comp = min(50, counts.shape[1] - 1, counts.shape[0] - 1)
    sc.tl.pca(adata, n_comps=n_comp, svd_solver="arpack")
    return adata.obsm["X_pca"]


def main():
    from sklearn.cluster import KMeans

    data_dir = os.environ["DATA_DIR"]
    output_dir = os.environ["OUTPUT_DIR"]
    instances = sorted(os.listdir(data_dir))
    for inst in instances:
        spliced, unspliced, k = load_instance(os.path.join(data_dir, inst))
        feats = np.concatenate([featurize(spliced), featurize(unspliced)], axis=1)
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(feats)
        out = os.path.join(output_dir, inst)
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "predictions.npy"), labels.astype(np.int64))
        print(f"[ref] {inst}: K={k}, {len(labels)} cells -> {out}/predictions.npy", flush=True)


if __name__ == "__main__":
    main()
