// TIER: greedy
// The obvious "fix the standings" recipe: a single fixed circulant/rotational
// tournament, team i beats team j iff (j-i) mod n lies in {1,...,(n-1)/2}
// (1-indexed teams here). This is perfectly degree-regular for every odd n
// (DegreeSkew = 0 always) -- an average solver sees "make every team's
// record fair" as the whole problem and stops here. But it has no algebraic
// cancellation in its pairwise common-victim counts, so QuadSkew stays
// roughly proportional to the fully transitive baseline's (same polynomial
// order) instead of collapsing the way a character-sum construction does.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    scanf("%d", &n);
    int h = (n - 1) / 2;
    string line;
    for (int i = 1; i <= n - 1; i++) {
        int len = n - i;
        line.resize(len);
        for (int k = 0; k < len; k++) {
            int j = i + 1 + k; // opponent, 1-indexed
            int d = j - i;     // in [1, n-1], (j-i) mod n == d since j>i, both<=n
            line[k] = (d >= 1 && d <= h) ? '1' : '0';
        }
        printf("%s\n", line.c_str());
    }
    return 0;
}
