// TIER: trivial
// Serial serve-all tour: depot -> p1 -> d1 -> p2 -> d2 -> ... -> pn -> dn -> depot.
// This is exactly the grader's baseline B, so it scores ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    long long dx, dy;
    scanf("%lld %lld", &dx, &dy);
    for (int i = 1; i <= n; i++) {
        long long a, b, c, d, p;
        scanf("%lld %lld %lld %lld %lld", &a, &b, &c, &d, &p);
    }
    printf("%d\n", 2 * n);
    string line;
    for (int i = 1; i <= n; i++) {
        if (!line.empty()) line += ' ';
        line += to_string(i);          // pickup i
        line += ' ';
        line += to_string(n + i);      // delivery i
    }
    printf("%s\n", line.c_str());
    return 0;
}
