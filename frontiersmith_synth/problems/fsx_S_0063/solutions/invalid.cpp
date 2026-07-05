// TIER: invalid
// Deliberately infeasible: places two type-1 cassettes on top of each other at
// the same offset -> overlap -> the checker must score this 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int W, H, n;
    scanf("%d %d %d", &W, &H, &n);
    for (int i = 0; i < n; i++) {
        int s; ll c; scanf("%d %lld", &s, &c);
        for (int j = 0; j < s; j++) { int a, b; scanf("%d %d", &a, &b); }
    }
    // two identical placements of type 1 at (0,0) -> guaranteed overlap
    printf("2\n");
    printf("1 0 0 0\n");
    printf("1 0 0 0\n");
    return 0;
}
