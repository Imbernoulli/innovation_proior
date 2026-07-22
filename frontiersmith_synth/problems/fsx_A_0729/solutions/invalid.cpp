// TIER: invalid
// Deliberately infeasible: every row is filled with the character '2', which
// is outside the required {0,1} alphabet. The checker must reject this with
// score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    scanf("%d", &n);
    string line;
    for (int i = 1; i <= n - 1; i++) {
        int len = n - i;
        line.assign(len, '2');
        printf("%s\n", line.c_str());
    }
    return 0;
}
