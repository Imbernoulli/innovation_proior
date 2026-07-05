// Generator for "Nimbus Swarm: Submodular Airspace Coverage".
// testId is a difficulty/structure ladder (1 tiny -> 10 large & adversarial).
// Structural modes plant: overlap TRAPS, NEEDLE zones, and PLANTED complementary
// clusters, plus a guaranteed positive anchor so the baseline B > 0.
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc,char**argv){
    registerGen(argc,argv,1);
    int testId=atoi(argv[1]);

    // ---- size ladder (quadratic growth; fills the envelope on the largest tests) ----
    int t=testId-1;                       // 0..9
    int D = 6 + (int)(t*t*16);            // 6 .. 6+81*16=1302
    int Z = 3 + (int)(t*t*3.8);           // 3 .. ~311
    int F = 6 + t*40;                     // 6 .. 366
    if(D<6) D=6; if(Z<2) Z=2; if(F<4) F=4;

    const ll COORD = (testId==1? 120 : 10000);
    // tight energy budget => only a handful of drones affordable out of many
    // => strong knapsack tension + submodular saturation both bite.
    ll E = (testId==1? 400 : 45000);
    int mode = testId % 4;               // 0 uniform, 1 overlap-trap, 2 needle, 3 planted
    if(testId==2) mode=1;
    if(testId==3) mode=2;
    if(testId==4) mode=3;

    // popular (over-demanded / over-carried) features for the overlap trap
    int nPop = min(F, 3 + testId/3);
    vector<int> pop;
    for(int i=0;i<nPop;i++) pop.push_back(rnd.next(1,F));

    // ---- zones ----
    vector<ll> zx(Z+1),zy(Z+1); vector<int> zcap(Z+1);
    vector<vector<array<int,3>>> zdem(Z+1); // (feature,w,s)
    // cluster centers for planted mode
    int nClusters = max(1, Z/6);
    vector<ll> cx(nClusters),cy(nClusters);
    for(int c=0;c<nClusters;c++){ cx[c]=rnd.next(0LL,COORD); cy[c]=rnd.next(0LL,COORD); }

    for(int z=1;z<=Z;z++){
        if(mode==3){ // planted: zones sit in clusters
            int c=rnd.next(0,nClusters-1);
            zx[z]=min(COORD,max(0LL, cx[c]+rnd.next(-COORD/40,COORD/40)));
            zy[z]=min(COORD,max(0LL, cy[c]+rnd.next(-COORD/40,COORD/40)));
        } else {
            zx[z]=rnd.next(0LL,COORD); zy[z]=rnd.next(0LL,COORD);
        }
        zcap[z]=rnd.next(1,3);
        int L=rnd.next(3, 3+ (F>10?6:2));
        set<int> feats;
        // overlap trap: many zones demand the popular features (high w, high s)
        if(mode==1 && rnd.next(0,99)<55){
            for(int f: pop) if((int)feats.size()<L) feats.insert(f);
        }
        while((int)feats.size()<L) feats.insert(rnd.next(1,F));
        for(int f: feats){
            int w, s;
            if(mode==1 && find(pop.begin(),pop.end(),f)!=pop.end()){
                w=rnd.next(12,24); s=rnd.next(20,45);   // fat popular demand (bait)
            } else {
                w=rnd.next(2,14); s=rnd.next(6,30);
            }
            zdem[z].push_back({f,w,s});
        }
    }

    // ---- NEEDLE: one very high-value zone that pays off only if saturated ----
    int needleZone=-1, needleFeat=-1, needleS=0;
    if(mode==2 && Z>=1){
        needleZone=rnd.next(1,Z);
        needleFeat=rnd.next(1,F);
        needleS = 40 + testId*4;
        zdem[needleZone].clear();
        zdem[needleZone].push_back({needleFeat, 25, needleS}); // huge saturation cap
        // a couple of decoy demands
        int L2=rnd.next(1,3);
        for(int j=0;j<L2;j++) zdem[needleZone].push_back({rnd.next(1,F), rnd.next(2,8), rnd.next(4,12)});
        zcap[needleZone]=max(zcap[needleZone], 3);
    }

    // ---- drones ----
    vector<ll> dxx(D+1),dyy(D+1);
    vector<vector<pair<int,int>>> pay(D+1); // (feature,amount)
    for(int d=1;d<=D;d++){
        if(mode==3){
            int c=rnd.next(0,nClusters-1);
            dxx[d]=min(COORD,max(0LL, cx[c]+rnd.next(-COORD/30,COORD/30)));
            dyy[d]=min(COORD,max(0LL, cy[c]+rnd.next(-COORD/30,COORD/30)));
        } else {
            dxx[d]=rnd.next(0LL,COORD); dyy[d]=rnd.next(0LL,COORD);
        }
        int K=rnd.next(2, min(F, 2+ (F>8?4:1)));
        set<int> feats;
        if(mode==1 && rnd.next(0,99)<50){
            // many drones carry popular features in SMALL amounts (overlap trap:
            // greedy grabs them, saturates the fat popular demand slowly, wastes budget)
            int f=pop[rnd.next(0,(int)pop.size()-1)];
            feats.insert(f);
        }
        if(mode==2 && rnd.next(0,99)<18 && needleFeat>0){
            feats.insert(needleFeat); // scattered carriers of the needle feature
        }
        while((int)feats.size()<K) feats.insert(rnd.next(1,F));
        for(int f: feats){
            int a;
            if(mode==2 && f==needleFeat) a=rnd.next(8,16);        // small vs needleS -> need many
            else if(mode==1 && find(pop.begin(),pop.end(),f)!=pop.end()) a=rnd.next(3,8);
            else a=rnd.next(4,22);
            pay[d].push_back(make_pair(f,a));
        }
    }

    // ---- guaranteed positive ANCHOR (ensures B>0): one drone sits on a zone,
    //      carrying a demanded feature with an amount that saturates it. ----
    {
        int z=rnd.next(1,Z);
        int fi = zdem[z].empty()?-1:0;
        if(fi<0){ zdem[z].push_back({1,10,15}); fi=0; }
        int f=zdem[z][fi][0], s=zdem[z][fi][2];
        int d=rnd.next(1,D);
        dxx[d]=zx[z]; dyy[d]=zy[z];                 // energy = 1 (cheap)
        pay[d].clear();
        pay[d].push_back(make_pair(f, s+5));        // fully saturates that feature
        int K2=rnd.next(1,2);
        for(int j=0;j<K2;j++) pay[d].push_back(make_pair(rnd.next(1,F), rnd.next(4,20)));
    }

    // ---- shuffle drone order so index != structural role ----
    vector<int> perm(D);
    for(int i=0;i<D;i++) perm[i]=i+1;
    shuffle(perm.begin(),perm.end());

    // ---- emit ----
    printf("%d %d %d %lld\n", D, Z, F, E);
    for(int z=1;z<=Z;z++){
        printf("%lld %lld %d %d", zx[z], zy[z], zcap[z], (int)zdem[z].size());
        for(auto &tr: zdem[z]) printf(" %d %d %d", tr[0],tr[1],tr[2]);
        printf("\n");
    }
    for(int i=0;i<D;i++){
        int d=perm[i];
        printf("%lld %lld %d", dxx[d], dyy[d], (int)pay[d].size());
        for(auto &ga: pay[d]) printf(" %d %d", ga.first, ga.second);
        printf("\n");
    }
    return 0;
}
