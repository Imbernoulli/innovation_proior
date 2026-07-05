// TIER: invalid
// Deliberately infeasible: claims to destroy one corridor but names an index that
// is out of range (m+1). The checker's bounded read rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, s, q, k;
    scanf("%d %d %d %d %d", &n, &m, &s, &q, &k);
    printf("1\n");
    printf("%d\n", m + 1); // out-of-range corridor index
    return 0;
}
