# Ivory Tower: Semi-Supervised Citation Sorting

## Story

A digital library holds a **citation graph**. Each node is a paper; an undirected
edge means one paper cites the other. Every paper belongs to exactly one research
**subfield** (its class label). Curators have hand-labeled a small **seed** set of
papers with their subfield; all the other papers are **unlabeled**. Each paper also
carries a noisy bag-of-topics **feature vector** that is subfield-informative but
far from perfect.

Papers of the same subfield cite one another more often than they cite across
subfields (**homophily**). Your job: design a label-propagation / aggregation rule
that files every unlabeled paper into the correct subfield, using the seed labels,
the citation edges, and the feature vectors together.

This is transductive semi-supervised node classification. You write a **standalone
program**: read one instance from stdin, print one answer to stdout.

## Public instance (stdin, one JSON object)

```
{
  "name":   str,
  "n":      int,                     # number of papers; node ids are 0..n-1
  "k":      int,                     # number of subfields; labels are 0..k-1
  "dim":    int,                     # feature dimension
  "features":     [[float]*dim]*n,   # features[i] = topic vector of paper i
  "edges":        [[u, v], ...],     # undirected citations, 0 <= u < v < n
  "train_ids":    [int, ...],        # labeled seed papers
  "train_labels": [int, ...],        # parallel labels for train_ids (each in 0..k-1)
  "query_ids":    [int, ...]         # papers whose subfield you must predict
}
```

## Answer (stdout, one JSON object)

```
{"labels": [int, ...]}   # parallel to query_ids; each label in 0..k-1
```

`labels[j]` is your predicted subfield for `query_ids[j]`.

## Validity

An answer is valid iff `labels` is a list of exactly `len(query_ids)` integers,
each in `[0, k)`. A crash, timeout, non-JSON output, wrong length, or an
out-of-range / non-integer label makes that instance score **0.0**.

## Objective — MAXIMIZE

Per instance let `acc_cand` be the fraction of query papers you label correctly and
let `acc_base` be the accuracy of the **majority-class** predictor (label every
query paper with the most frequent seed subfield; ties broken toward the lowest
class index). The instance score is

```
r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / max(1e-9, 1 - acc_base), 0, 1 )
```

so matching the majority predictor scores ~0.1 and perfect classification scores
1.0. The final **Ratio** is the mean of `r` over all 12 instances (some larger and
harder, held out). Because the graph is homophilous but noisy and the feature
signal is imperfect, perfect accuracy is not attainable in general — there is
genuine headroom, and multiple strategies (feature centroids, graph propagation,
personalized PageRank, self-training, feature-kNN augmentation) trade off in
different ways across the instance distribution.

## Isolation

Your program runs in a fresh OS-sandboxed subprocess and only ever sees the public
instance above. The true query labels and all references are held by the evaluator
process; introspection / filesystem snooping reveals nothing useful.
