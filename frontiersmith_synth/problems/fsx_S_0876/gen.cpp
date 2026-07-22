#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Flow-Matched Marquetry"  (generator)  family: veneer-grain-marquetry
//
// A rectangular source veneer SHEET (Ws x Hs) carries a smooth grain-direction
// field (degrees mod 180, since a grain LINE has no arrowhead). A target PANEL
// is a polyomino region (Wp x Hp bounding box, '#' mask) with its own desired
// flow field. A fixed catalog of K polyomino piece shapes is available in
// unlimited supply. The solver cuts piece INSTANCES from the sheet (a pose:
// position + one of 4 rotations) and glues each into the panel (a SECOND,
// independently chosen pose). Because a piece is one physical object, its
// grain angle once glued into the panel is the sheet angle it was cut from,
// shifted by 90 * (panel-rotation - sheet-rotation) degrees (mod 180) -- the
// two poses are linked through this single rotation DIFFERENCE, not through
// their individual values. That is the intended reformulation: search over
// POSE-PAIRS (sheet pose, panel pose) jointly, not two independent packings.
//
// PLANTED TRAP (testId 4..10, 7/10 cases): the panel's desired flow field is a
// tangential SWIRL around one or two centers (real curvature), while the
// source sheet's field is a slow, near-linear sinusoid. A greedy that packs
// the sheet densely by simple first-fit (ignoring grain) and tiles the panel
// by area alone lands its rotations essentially uncorrelated with the target
// -> large angular mismatch. A solver that, for every panel placement, SEARCHES
// the sheet for the (position, rotation) pair whose net-rotated angle best
// matches the local target lands close to it. testId {7,9,10} additionally
// notch the panel boundary (forces small/diverse shapes there); testId
// {6,8,10} plant a "needle" sheet patch, placed where row-major first-fit
// never reaches, whose angle exactly matches a hard swirl target cell.
//
// Output format (see statement.txt):
//   Ws Hs
//   Hs lines of Ws ints           (source grain angle, degrees in [0,179])
//   Wp Hp
//   Hp lines of Wp chars          ('#'/'.' panel mask)
//   Hp lines of Wp ints           (desired flow angle, degrees in [0,179]; '.' cells print 0)
//   K
//   K blocks: ncells, then ncells lines "dx dy"  (canonical piece shape, dx,dy>=0, includes (0,0))
//   lambda mu Cu
// -----------------------------------------------------------------------------

static const double PI = 3.14159265358979323846;

int wrap180(long long a){ long long m = a % 180; if (m < 0) m += 180; return (int)m; }

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int sides[11] = {0,4,5,6,7,8,9,10,11,12,14};
    int side = sides[t];
    int Ws = side, Hs = side, Wp = side, Hp = side;

    bool swirl  = (t >= 4);
    bool notch  = (t == 7 || t == 9 || t == 10);
    bool needle = (t == 6 || t == 8 || t == 10);
    bool doubleVortex = (t == 9 || t == 10);

    // ---- source grain field: smooth low-frequency sinusoid ----
    double A  = rnd.next(0, 179);
    double Bs = 20 + rnd.next(0, 40);
    double Cs = 20 + rnd.next(0, 40);
    double p1 = rnd.next(0, 628) / 100.0;
    double p2 = rnd.next(0, 628) / 100.0;
    vector<vector<int>> src(Hs, vector<int>(Ws, 0));
    for (int y = 0; y < Hs; y++)
        for (int x = 0; x < Ws; x++){
            double v = A + Bs * sin(2*PI*x/max(1,Ws) + p1) + Cs * cos(2*PI*y/max(1,Hs) + p2);
            src[y][x] = wrap180((long long)llround(v));
        }

    // ---- target flow field ----
    double cx1 = rnd.next(0, 100)/100.0*(Wp-1), cy1 = rnd.next(0, 100)/100.0*(Hp-1);
    double cx2 = Wp - 1 - cx1, cy2 = Hp - 1 - cy1;
    vector<vector<int>> tgt(Hp, vector<int>(Wp, 0));
    double A2 = rnd.next(0, 179);
    double B2 = 10 + rnd.next(0, 30);
    double C2 = 10 + rnd.next(0, 30);
    double q1 = rnd.next(0, 628) / 100.0;
    double q2 = rnd.next(0, 628) / 100.0;
    for (int y = 0; y < Hp; y++)
        for (int x = 0; x < Wp; x++){
            if (swirl){
                double dx1 = x - cx1, dy1 = y - cy1;
                double ang1 = atan2(dy1, dx1) * 180.0 / PI + 90.0;
                double val;
                if (doubleVortex){
                    double dx2 = x - cx2, dy2 = y - cy2;
                    double ang2 = atan2(dy2, dx2) * 180.0 / PI + 90.0;
                    double d1 = hypot(dx1, dy1), d2 = hypot(dx2, dy2);
                    val = (d1 <= d2) ? ang1 : ang2;
                } else {
                    val = ang1;
                }
                tgt[y][x] = wrap180((long long)llround(val));
            } else {
                double v = A2 + B2 * sin(2*PI*x/max(1,Wp) + q1) + C2 * cos(2*PI*y/max(1,Hp) + q2);
                tgt[y][x] = wrap180((long long)llround(v));
            }
        }

    // ---- panel mask: blobby star-shaped polyomino ----
    double pcx = (Wp - 1) / 2.0, pcy = (Hp - 1) / 2.0;
    double baseR = 0.42 * side + 0.6;
    int k = 2 + rnd.next(0, 3);
    double phase = rnd.next(0, 628) / 100.0;
    vector<vector<char>> mask(Hp, vector<char>(Wp, '.'));
    int maskCnt = 0;
    for (int y = 0; y < Hp; y++)
        for (int x = 0; x < Wp; x++){
            double dx = x - pcx, dy = y - pcy;
            double theta = atan2(dy, dx);
            double rr = baseR * (1.0 + 0.22 * sin(k*theta + phase));
            if (dx*dx + dy*dy <= rr*rr){ mask[y][x] = '#'; maskCnt++; }
        }
    if (maskCnt == 0){ mask[(int)pcy][(int)pcx] = '#'; maskCnt = 1; }

    if (notch && side >= 6){
        int nx = 1 + rnd.next(0, max(1, Wp - 4));
        int ny = 1 + rnd.next(0, max(1, Hp - 4));
        int nw = 1 + rnd.next(0, 1), nh = 1 + rnd.next(0, 1);
        for (int y = ny; y < min(Hp, ny+nh); y++)
            for (int x = nx; x < min(Wp, nx+nw); x++)
                if (mask[y][x] == '#' && maskCnt > (int)(0.4*Wp*Hp)){ mask[y][x] = '.'; maskCnt--; }
    }

    // ---- needle: plant an exact-match sheet cell far from row-major first-fit reach ----
    if (needle){
        // find a genuinely masked panel cell (scan from the bottom-right corner
        // inward so we land on a real, if awkward, target cell -- the mask is a
        // blob/notched polyomino, so (Wp-1,Hp/2) is not guaranteed to be '#').
        int px = -1, py = -1;
        for (int y = Hp - 1; y >= 0 && px < 0; y--)
            for (int x = Wp - 1; x >= 0; x--)
                if (mask[y][x] == '#'){ px = x; py = y; break; }
        int wanted = tgt[py][px];                // exact target angle to hit with shift 0
        src[Hs-1][Ws-1] = wanted;                // bottom-right corner: last cell row-major reaches
        if (Ws >= 2) src[Hs-1][Ws-2] = wanted;   // widen the needle to fit a domino too
    }

    // ---- fixed catalog: K=7 shapes, sizes 1,2,3,3,4,4,4 ----
    struct Shape { vector<pair<int,int>> cells; };
    vector<Shape> cat(7);
    cat[0].cells = {{0,0}};
    cat[1].cells = {{0,0},{1,0}};
    cat[2].cells = {{0,0},{1,0},{0,1}};
    cat[3].cells = {{0,0},{1,0},{2,0}};
    cat[4].cells = {{0,0},{1,0},{0,1},{1,1}};
    cat[5].cells = {{0,0},{0,1},{0,2},{1,2}};
    cat[6].cells = {{0,0},{1,0},{2,0},{1,1}};

    ll lambda = 55, mu = 25, Cu = 150;

    // ---- print ----
    printf("%d %d\n", Ws, Hs);
    for (int y = 0; y < Hs; y++){
        for (int x = 0; x < Ws; x++) printf("%d%c", src[y][x], x+1==Ws?'\n':' ');
    }
    printf("%d %d\n", Wp, Hp);
    for (int y = 0; y < Hp; y++){
        for (int x = 0; x < Wp; x++) putchar(mask[y][x]);
        putchar('\n');
    }
    for (int y = 0; y < Hp; y++){
        for (int x = 0; x < Wp; x++) printf("%d%c", mask[y][x]=='#' ? tgt[y][x] : 0, x+1==Wp?'\n':' ');
    }
    printf("%d\n", (int)cat.size());
    for (auto &sh : cat){
        printf("%d\n", (int)sh.cells.size());
        for (auto &c : sh.cells) printf("%d %d\n", c.first, c.second);
    }
    printf("%lld %lld %lld\n", lambda, mu, Cu);
    return 0;
}
