// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

int main() {
    // Deliberately infeasible: a single "line" whose two cells are the same
    // grid position, so the required cell-to-cell adjacency (and disjointness)
    // is violated regardless of the input -- must be rejected and score 0.
    cout << "1\n2\n1 1\n1 1\n";
    return 0;
}
