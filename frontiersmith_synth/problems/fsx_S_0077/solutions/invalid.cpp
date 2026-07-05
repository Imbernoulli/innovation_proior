// TIER: invalid
// Deliberately infeasible: routes every pass to dam 0 (often ineligible) and starts
// everything at time 0 (massive overlap). Must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int j = 0; j < n; j++) {
        int o; scanf("%d", &o);
        string line;
        for (int k = 0; k < o; k++) {
            int c; scanf("%d", &c);
            for (int e = 0; e < c; e++) { int dam, dur; scanf("%d %d", &dam, &dur); }
            line += "0 0";
            if (k + 1 < o) line += " ";
        }
        printf("%s\n", line.c_str());
    }
    return 0;
}
