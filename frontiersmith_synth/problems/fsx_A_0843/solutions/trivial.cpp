// TIER: trivial
// Shelf-pack the blocks (input order, rotation 0) into rows, then CENTER the
// whole packed block both vertically and horizontally in the deck -- exactly
// the checker's own internal baseline construction B. Ignores wattage and the
// cold rim entirely; a corner-anchored pack would accidentally hug the rim,
// so this centers instead to be a fair "physics-ignorant" reference.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int,int> pii;

int main(){
    int W, H, N, ITERS;
    scanf("%d %d %d %d", &W, &H, &N, &ITERS);
    vector<vector<pii>> shape(N + 1);
    vector<ll> watt(N + 1);
    for (int i = 1; i <= N; i++){
        int k; ll w;
        scanf("%lld %d", &w, &k);
        watt[i] = w;
        vector<pii> cells(k);
        for (int j = 0; j < k; j++){
            int dx, dy;
            scanf("%d %d", &dx, &dy);
            cells[j] = {dx, dy};
        }
        shape[i] = cells;
    }

    vector<int> bw(N + 1), bh(N + 1);
    for (int i = 1; i <= N; i++){
        int mx = 0, my = 0;
        for (auto& c : shape[i]){ mx = max(mx, c.first); my = max(my, c.second); }
        bw[i] = mx + 1; bh[i] = my + 1;
    }
    vector<int> rowOf(N + 1), localX(N + 1);
    vector<int> rowWidth, rowHeight;
    int cursor_x = 0, row = 0, row_h = 0;
    for (int i = 1; i <= N; i++){
        if (cursor_x + bw[i] > W){
            rowWidth.push_back(cursor_x);
            rowHeight.push_back(row_h);
            row++; cursor_x = 0; row_h = 0;
        }
        rowOf[i] = row; localX[i] = cursor_x;
        cursor_x += bw[i];
        row_h = max(row_h, bh[i]);
    }
    rowWidth.push_back(cursor_x);
    rowHeight.push_back(row_h);
    int numRows = row + 1;
    int totalH = 0; for (int h : rowHeight) totalH += h;
    int topY = (H - totalH) / 2; if (topY < 0) topY = 0;
    vector<int> rowY(numRows);
    rowY[0] = topY;
    for (int r = 1; r < numRows; r++) rowY[r] = rowY[r - 1] + rowHeight[r - 1];

    for (int i = 1; i <= N; i++){
        int r = rowOf[i];
        int xoff = (W - rowWidth[r]) / 2; if (xoff < 0) xoff = 0;
        printf("%d %d 0\n", xoff + localX[i], rowY[r]);
    }
    return 0;
}
