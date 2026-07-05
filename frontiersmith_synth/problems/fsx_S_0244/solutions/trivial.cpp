// TIER: trivial
// Serve nobody: empty tour. Cost = total penalty = baseline B -> ratio == 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int P, H; long long x0, y0;
    if (!(cin >> P >> H)) return 0;
    cin >> x0 >> y0;
    long long a,b,c,d,w;
    for (int i = 0; i < P; i++) cin >> a >> b >> c >> d >> w;
    printf("0\n");
    return 0;
}
