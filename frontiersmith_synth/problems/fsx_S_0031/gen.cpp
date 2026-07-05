#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    const int COORD = 1000000;

    int n = (t <= 1) ? 8 : min(3000, t * 350);
    printf("%d\n", n);

    // Distribution ladder:
    //  - even testId (t>=2): uniform scatter
    //  - odd  testId (t>=3): pandemic-style clusters (contacts bunch together)
    bool clustered = (t >= 3 && (t % 2 == 1));

    vector<pair<int,int>> cen;
    if (clustered) {
        int K = 4 + t;                 // more clusters as scale grows
        for (int i = 0; i < K; i++)
            cen.push_back({rnd.next(0, COORD), rnd.next(0, COORD)});
    }

    for (int i = 0; i < n; i++) {
        int x, y;
        if (clustered) {
            auto c = cen[rnd.next(0, (int)cen.size() - 1)];
            int spread = 35000;
            x = min(COORD, max(0, c.first  + (int)rnd.next(-spread, spread)));
            y = min(COORD, max(0, c.second + (int)rnd.next(-spread, spread)));
        } else {
            x = rnd.next(0, COORD);
            y = rnd.next(0, COORD);
        }
        // Contact cap: mostly 3, a minority 2. Always >= 2 so a chain backbone exists.
        int b = (rnd.next(0, 99) < 20) ? 2 : 3;
        printf("%d %d %d\n", x, y, b);
    }
    return 0;
}
