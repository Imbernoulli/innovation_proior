// TIER: invalid
// Deliberately infeasible: emit a negative start time (out of range) for every task,
// which the grader must reject with score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int j = 0; j < n; j++) {
        int w, o;
        scanf("%d %d", &w, &o);
        string line;
        for (int k = 0; k < o; k++) {
            int c, d; scanf("%d %d", &c, &d);
            if (k) line += ' ';
            line += "-1";   // negative start => out of range => score 0
        }
        printf("%s\n", line.c_str());
    }
    return 0;
}
