# Tutte Matrix Matching

For a finite simple graph `G = (V, E)` with `V = {1, ..., n}`, odd `n` immediately means no perfect matching, and the skew-symmetric determinant is identically zero. For even `n`, assign an independent variable `x_ij` to each edge `{i, j}` with `i < j`. Define the skew-symmetric matrix `T(G)` by

```text
T_ij =  x_ij   if i < j and {i, j} in E
T_ij = -x_ji   if i > j and {i, j} in E
T_ij =  0      otherwise.
```

Changing the orientation convention only changes signs of variables, not whether the determinant is the zero polynomial.

For even `n`, the Pfaffian of `T(G)` expands over all pairings of the vertices:

```text
Pf(T) = sum epsilon * product of the paired entries,
epsilon in {+1, -1}.
```

There is no extra numeric factor. A term survives exactly when every chosen pair is an edge, so surviving terms are exactly perfect matchings. Distinct perfect matchings produce distinct monomials in independent variables, so they cannot cancel.

Thus

```text
G has a perfect matching
<=> Pf(T(G)) is not the zero polynomial
<=> det(T(G)) is not the zero polynomial
<=> T(G) has full generic rank.
```

A randomized decision test instantiates each `x_ij` independently from a large finite field and computes rank or determinant. Full rank proves a perfect matching exists. If all trials are singular, the correct report is one-sided-error nonexistence: for a graph with a perfect matching, one trial fails with probability at most `n / |S|` when sampling from a set `S`, and independent repetition reduces that risk.

The local `code/tutte_rank_test.py` matches this decision test: it handles odd `n`, builds the signed skew matrix over a prime field, and returns `True` on a full-rank trial. It does not construct the matching or implement the later Rabin-Vazirani maximum-matching search.
