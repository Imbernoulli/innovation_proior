#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int V = 4, L = 30, U = 40;
vector<pair<int,int>> edges;
vector<long long> w; // 1-indexed weights; index 0 unused

int newNode(){ w.push_back(1); return (int)w.size()-1; }
void addEdge(int a,int b){ edges.push_back(make_pair(a,b)); }

// chain of `len` new nodes hanging off `parent`; leaf gets weight lw; return leaf id.
int addChain(int parent,int len,long long lw){
    int cur=parent;
    for(int i=0;i<len;i++){ int c=newNode(); addEdge(cur,c); cur=c; }
    if(len>0) w[cur]=lw;
    return cur;
}

int main(int argc,char*argv[]){
    registerGen(argc,argv,1);
    int t=atoi(argv[1]);

    // node 1 is the root
    w.assign(2,1); // index 0 dummy, index 1 root

    int R;
    if(t==1) R=1;                 // tiny example-scale instance
    else R = t*t*14;              // t=10 -> 1400 replicas (~1.4e5 nodes)

    for(int k=0;k<R;k++){
        // --- baseline zone: medium-depth flowers bloom under the reference dosing (feed B) ---
        addChain(1,15, rnd.next(6,14));  // ref sum 33 in band
        addChain(1,16, rnd.next(6,14));  // ref sum 35 in band

        // --- greedy zone: depth-13 flower, ref sum 29 (just below band); a single +1 on its
        //     own leaf level reaches 30 -> a leaf-only tuner activates it. ---
        addChain(1,13, rnd.next(6,14));

        // --- strong/planted zone: shallow flowers reachable ONLY by inflating shared ancestor
        //     levels to a common u (not by the +-2 a leaf level alone can provide). ---
        int base = rnd.next(12,22);
        addChain(1,8,  base + rnd.next(0,6));
        addChain(1,9,  base + rnd.next(0,6));
        addChain(1,10, base + rnd.next(0,6));
        addChain(1,11, base + rnd.next(0,6));

        // --- trap broom: a shared stem, then a shallow branch (needs inflation) and a medium
        //     branch (blooms at baseline). Inflating the stem activates the shallow flower but
        //     ejects the medium one -- worth it only when the shallow value dominates. ---
        int se=addChain(1,5,1);
        long long wsh=rnd.next(10,45), wmed=rnd.next(10,45);
        addChain(se,4, wsh);   // shallow flower at depth 9
        addChain(se,10,wmed);  // medium flower at depth 15 (ref sum 33, blooms at baseline)
    }

    // --- needle zone: a few very high-value shallow flowers, each in its own clean subtree,
    //     activatable only by ancestor inflation (a leaf-only tuner misses them entirely). ---
    int needles = (t==1)?1 : (3 + t/2);
    for(int i=0;i<needles;i++){
        addChain(1,9, rnd.next(40,90));
    }

    int N=(int)w.size()-1;
    printf("%d %d %d %d\n", N, V, L, U);
    for(size_t i=0;i<edges.size();i++) printf("%d %d\n", edges[i].first, edges[i].second);
    for(int i=1;i<=N;i++) printf("%lld%c", w[i], i==N?'\n':' ');
    return 0;
}
