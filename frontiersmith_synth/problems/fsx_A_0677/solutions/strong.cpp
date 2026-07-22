// TIER: strong
#include "common.h"

int main() {
    Grid gr = readGrid(cin);
    int R = gr.R, C = gr.C;
    long long totalSand = 0;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (gr.g[r][c] == '.') totalSand++;
    long long target = (long long)llround(totalSand * (gr.CovMin + gr.CovMax) / 2000.0);

    // The insight: build the scalar potential D(cell) = distance to the nearest
    // rock (a multi-source BFS -- rocks that are close together automatically
    // share a basin, and the field's ridges are the saddles between basins).
    // First, unconditionally extract the sub-level-set {D <= M}: this is the
    // "concentric rings" zone around every rock/basin, regardless of how the
    // rows happen to line up. Only THEN spend the remaining budget on a generic
    // evenly-spaced sweep of the open sand -- so far from any rock we still get
    // the same regular spacing a straight-line raking would give.
    vector<int> D = distToRock(gr);
    vector<char> mask(R * C, 0);
    long long picked = 0;
    for (int r = 0; r < R && picked < target; r++)
        for (int c = 0; c < C && picked < target; c++) {
            int idx = r * C + c;
            if (gr.g[r][c] == '.' && D[idx] <= gr.M) { mask[idx] = 1; picked++; }
        }

    if (picked < target) {
        double midFrac = (gr.CovMin + gr.CovMax) / 2000.0;
        int desiredRows = max(1, (int)llround(R * midFrac));
        vector<char> rowSel = evenRowSelection(R, desiredRows);
        for (int r = 0; r < R && picked < target; r++) {
            if (!rowSel[r]) continue;
            for (int c = 0; c < C && picked < target; c++) {
                int idx = r * C + c;
                if (gr.g[r][c] == '.' && !mask[idx]) { mask[idx] = 1; picked++; }
            }
        }
        if (picked < target) {
            for (int r = 0; r < R && picked < target; r++)
                for (int c = 0; c < C && picked < target; c++) {
                    int idx = r * C + c;
                    if (gr.g[r][c] == '.' && !mask[idx]) { mask[idx] = 1; picked++; }
                }
        }
    }

    auto lines = maskToLines(gr, mask);
    printLines(lines);
    return 0;
}
