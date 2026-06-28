# Tree Isomorphism by Canonical Hashing

## Research question

You are given two **rooted forests** and must decide whether they are **isomorphic
as rooted forests**: is there a bijection between the node sets that maps roots to
roots and preserves the parent–child relation (children may be reordered freely,
since a tree imposes no order on siblings)?

Each forest is given by a parent array: node `i` either has `par[i] = 0`, meaning it
is one of the forest's roots, or `par[i] = p` with `1 <= p <= n`, meaning its parent
is node `p`. The intended difficulty is that the *labels are meaningless* — two
forests can be drawn completely differently (different ids, children listed in a
different order) and still be the same shape. Reporting `YES`/`NO` correctly at
`n` up to `2*10^5` is the path-case of the general graph-isomorphism question, but
for rooted forests it admits an exact polynomial canonical form rather than any
heuristic.

This kind of structural-equality test shows up whenever you deduplicate parse
trees, expression trees, or subtree shapes inside a larger tree algorithm, so
getting the canonicalization exactly right — including the sibling-order and
empty-forest corners — matters.

## Input / output contract

- Input (stdin), two forests back to back:
  - `n1` (`0 <= n1 <= 2*10^5`), then `n1` integers `par1[1..n1]`
    (`0 <= par1[i] <= n1`, `par1[i] != i`, and the array describes a valid forest:
    following parents never cycles).
  - `n2` (`0 <= n2 <= 2*10^5`), then `n2` integers `par2[1..n2]` under the same
    rules.
  - All integers are whitespace-separated; line breaks are not significant.
- Output (stdout): a single line, `YES` if the two rooted forests are isomorphic,
  otherwise `NO`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: forest 1 is `n1 = 4`, `par1 = [0, 1, 1, 2]` (root `1`; children `2, 3`;
node `2` has child `4`). Forest 2 is `n2 = 4`, `par2 = [0, 1, 1, 3]` (root `1`;
children `2, 3`; node `3` has child `4`). These are the same shape — a root with
two children, one of which has a single child — so the answer is `YES`, even
though the deepest node hangs off a different sibling and carries a different id.

## Background

Two approaches are on the table before committing to one.

- **Serialize and compare strings.** Write each tree as a nested string by walking
  it (e.g. `(()(()))`) and test the two strings for equality. This is `O(n)` and
  trivial, but it bakes in a *child ordering*: the same tree written with siblings
  in a different order produces a different string. The open question is whether
  any fixed traversal order can be made canonical.
- **Canonical labeling (AHU).** Assign every node an integer label computed
  bottom-up from the *multiset* of its children's labels, so that two nodes get
  equal labels exactly when their subtrees are isomorphic. The open question is how
  to make the per-node label depend on the children as an unordered collection, and
  how to keep the whole thing `O(n log n)` rather than blowing up on string
  concatenation.

A rooted **forest** is reduced to a single rooted **tree** by attaching a virtual
super-root (node `0`) whose children are exactly the forest roots; forest
isomorphism then becomes ordinary rooted-tree isomorphism at node `0`.

## Evaluation settings

Judged on hidden tests covering: empty forests (`n = 0`); single nodes; forests of
many singleton roots; deep chains (depth up to `2*10^5`, which stresses any
recursive traversal); wide stars; pairs that are isomorphic only after a sibling
reordering and a full relabeling; near-miss pairs that differ in one subtree; and
large random forests at `n = 2*10^5`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n1;
    if (!(cin >> n1)) return 0;
    vector<int> par1(n1 + 1);
    for (int i = 1; i <= n1; i++) cin >> par1[i];

    int n2;
    cin >> n2;
    vector<int> par2(n2 + 1);
    for (int i = 1; i <= n2; i++) cin >> par2[i];

    // TODO: decide whether the two rooted forests are isomorphic and print
    // "YES" or "NO".

    cout << "NO" << "\n";
    return 0;
}
```
