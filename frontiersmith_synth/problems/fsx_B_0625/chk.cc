#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Avalanche Sculptor".  family: abelian-sandpile-sculptor
//
// Input:  N K B ; nSink ; nSink (r c) sinks ; K (r c) sources ;
//         N x N target T (0..3) ; N x N weight W .
// Output: K nonnegative integers a_1..a_K  (grains dropped at each source, in order),
//         with 0 <= a_i <= B and sum a_i <= B.
//
// The checker drops a_i grains at source i on an empty grid, runs the abelian sandpile
// to stabilization (boundary + sink cells drain), and computes the resulting stable
// configuration S. Score:
//   F = sum over non-sink cells of  W[cell] * [ S[cell] == T[cell] ]        (MAX)
//   B = sum over non-sink cells of  W[cell] * [ T[cell] == 0 ]  = F(empty)  (baseline)
//   sc = min(1000, 100 * F / max(1,B)) ;  Ratio = sc/1000 .
// The do-nothing (all-zero) drop reproduces exactly the empty grid -> F == B -> 0.1 .
// -----------------------------------------------------------------------------

int N;
static inline int IDX(int r,int c){return r*N+c;}

void stabilize(vector<int>&g, const vector<char>&sink){
    int NN=N*N;
    vector<char> inq(NN,0);
    deque<int> q;
    for(int i=0;i<NN;i++) if(!sink[i]&&g[i]>=4){q.push_back(i);inq[i]=1;}
    static const int dr[4]={-1,1,0,0}, dc[4]={0,0,-1,1};
    while(!q.empty()){
        int c=q.front();q.pop_front();inq[c]=0;
        if(sink[c]||g[c]<4) continue;
        int t=g[c]/4; g[c]-=4*t;
        int r=c/N, cc=c%N;
        for(int d=0;d<4;d++){
            int nr=r+dr[d], nc=cc+dc[d];
            if(nr<0||nr>=N||nc<0||nc>=N) continue;
            int ni=nr*N+nc;
            if(sink[ni]) continue;
            g[ni]+=t;
            if(g[ni]>=4 && !inq[ni]){inq[ni]=1;q.push_back(ni);}
        }
    }
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    int K = inf.readInt();
    ll B = inf.readLong();
    int NN = N*N;
    vector<char> sink(NN, 0);
    int nSink = inf.readInt();
    for(int i=0;i<nSink;i++){
        int r=inf.readInt(), c=inf.readInt();
        sink[IDX(r,c)] = 1;
    }
    vector<int> src(K);
    for(int i=0;i<K;i++){
        int r=inf.readInt(), c=inf.readInt();
        src[i] = IDX(r,c);
    }
    vector<int> T(NN,0), W(NN,0);
    for(int i=0;i<NN;i++) T[i] = inf.readInt(0,3);
    for(int i=0;i<NN;i++) W[i] = inf.readInt(0,1000000);

    // ---- baseline B_score = F(empty) = weighted count of zero-target cells ----
    ll Bsc = 0;
    for(int i=0;i<NN;i++) if(!sink[i] && T[i]==0) Bsc += W[i];
    if(Bsc <= 0) Bsc = 1;

    // ---- read participant drops (integers -> nan/inf auto-rejected) ----
    vector<int> g(NN, 0);
    ll sum = 0;
    for(int i=0;i<K;i++){
        ll a = ouf.readLong(0, B, "a");
        sum += a;
        if(sum > B) quitf(_wa, "total grains %lld exceeds budget B=%lld", sum, B);
        g[src[i]] += (int)a;
    }
    if(!ouf.seekEof()) quitf(_wa, "trailing output tokens after the %d drop counts", K);

    stabilize(g, sink);

    // ---- objective ----
    ll F = 0;
    for(int i=0;i<NN;i++) if(!sink[i] && g[i]==T[i]) F += W[i];

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, Bsc));
    quitp(sc / 1000.0, "OK F=%lld B=%lld used=%lld/%lld Ratio: %.6f",
          F, Bsc, sum, B, sc / 1000.0);
    return 0;
}
