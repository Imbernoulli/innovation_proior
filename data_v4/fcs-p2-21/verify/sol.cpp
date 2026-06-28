#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0

    // Each box type yields 3 oriented boxes (choose which dimension is the height).
    // For an orientation we record the base as (w, d) with w <= d (the two non-height
    // dimensions, sorted) and the height h. Sorting the base is sound because "strictly
    // smaller base" must hold for both base dimensions, and that requirement is invariant
    // under swapping the two base dimensions, so we may compare them after normalizing.
    struct Box { long long w, d, h; };
    vector<Box> boxes;
    boxes.reserve((size_t)3 * max(n, 0));

    for (int i = 0; i < n; i++) {
        long long x, y, z;
        cin >> x >> y >> z;
        long long dim[3] = {x, y, z};
        // height = dim[k], base = the other two, sorted so w <= d.
        for (int k = 0; k < 3; k++) {
            long long h = dim[k];
            long long a = dim[(k + 1) % 3];
            long long b = dim[(k + 2) % 3];
            if (a > b) swap(a, b);
            boxes.push_back({a, b, h});
        }
    }

    int m = (int)boxes.size();
    // Sort by base area descending (any total order consistent with "larger base first"
    // works as long as it is a topological order of the strictly-smaller relation; sorting
    // by area descending, with ties broken arbitrarily, guarantees that if box j can sit on
    // box i then j appears after i, because a strictly smaller base has strictly smaller area).
    sort(boxes.begin(), boxes.end(), [](const Box& A, const Box& B) {
        long long areaA = A.w * A.d, areaB = B.w * B.d;
        if (areaA != areaB) return areaA > areaB;
        if (A.w != B.w) return A.w > B.w;
        return A.d > B.d;
    });

    // LIS-style O(m^2) DP. dp[j] = the maximum total height of a stack whose TOP box is box j.
    // Boxes are in "larger base first" order, so any box that can sit on top of box i appears
    // after i. Box j may rest directly on box i when i's base is strictly larger in BOTH
    // dimensions; we extend the best stack topped by such an i with box j.
    vector<long long> dp(m);
    long long best = 0;
    for (int j = 0; j < m; j++) {
        dp[j] = boxes[j].h;
        for (int i = 0; i < j; i++) {
            if (boxes[i].w > boxes[j].w && boxes[i].d > boxes[j].d) {
                dp[j] = max(dp[j], dp[i] + boxes[j].h);
            }
        }
        best = max(best, dp[j]);
    }

    cout << best << "\n";
    return 0;
}
