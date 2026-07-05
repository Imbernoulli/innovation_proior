// TIER: trivial
// Fully-serial schedule in ride index order: every task runs back-to-back on one
// global timeline. This is exactly the grader's baseline B, so it scores ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    long long cursor = 0;
    for (int j = 0; j < n; j++) {
        int w, o;
        scanf("%d %d", &w, &o);
        string line;
        for (int k = 0; k < o; k++) {
            int c, d;
            scanf("%d %d", &c, &d);
            if (k) line += ' ';
            line += to_string(cursor);
            cursor += d;
        }
        printf("%s\n", line.c_str());
    }
    return 0;
}
