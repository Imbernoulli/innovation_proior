// TIER: trivial
// Fully transitive schedule: team i beats team j whenever i<j (rank order
// always wins). This is exactly the checker's own reference construction, so
// it lands at F=B, i.e. ratio = 0.1 on every test -- the calibration point.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    scanf("%d", &n);
    string line;
    for (int i = 1; i <= n - 1; i++) {
        int len = n - i;
        line.assign(len, '1'); // team i beats every higher-numbered team
        printf("%s\n", line.c_str());
    }
    return 0;
}
