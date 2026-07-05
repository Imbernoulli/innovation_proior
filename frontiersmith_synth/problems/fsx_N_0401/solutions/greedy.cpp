// TIER: greedy
// Traffic-greedy (primality-blind trap): repeatedly cut the coupler carrying the
// most s->t routes of ANY length (total forward x backward path counts), recompute,
// and keep the removal prefix that minimizes F. Because it ignores primality it
// wastes cuts on the high-traffic composite bait bundle before it reaches the
// prime-length bridges, so it lags the resonance-aware strong solver.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n,m,s,t,k; ll P;
vector<int> eu,ev,ec;
vector<vector<int>> inE, outE;
vector<char> isprime;
int Lmax;
const double CAP=1e15;

void totals(const vector<char>& rm, vector<double>& tS, vector<double>& tT){
    tS.assign(n,0.0); tT.assign(n,0.0);
    tS[s]=1.0;
    for(int v=0;v<n;v++){ if(v==s)continue; double a=0;
        for(int id:inE[v]){ if(rm[id])continue; a+=tS[eu[id]]; if(a>CAP)a=CAP; } tS[v]=a; }
    tT[t]=1.0;
    for(int v=n-1;v>=0;v--){ if(v==t)continue; double b=0;
        for(int id:outE[v]){ if(rm[id])continue; b+=tT[ev[id]]; if(b>CAP)b=CAP; } tT[v]=b; }
}

ll computeF(const vector<char>& rm){
    int W=(n/64)+1;
    vector<vector<unsigned long long>> reach(n, vector<unsigned long long>(W,0ULL));
    reach[s][0]=1ULL;
    for(int v=0;v<n;v++){ if(v==s)continue;
        for(int id:inE[v]){ if(rm[id])continue; int u=eu[id]; unsigned long long carry=0;
            for(int w=0;w<W;w++){ unsigned long long cur=reach[u][w]; unsigned long long sh=(cur<<1)|carry; carry=cur>>63; reach[v][w]|=sh; } } }
    int primeL=0;
    for(int L=2;L<n;L++){ if(!isprime[L])continue; if(reach[t][L>>6]>>(L&63)&1ULL) primeL++; }
    ll cost=0; for(int id=1;id<=m;id++) if(rm[id]) cost+=ec[id];
    return cost+P*(ll)primeL;
}

int main(){
    if(!(cin>>n>>m>>s>>t>>k>>P)) return 0;
    eu.assign(m+1,0);ev.assign(m+1,0);ec.assign(m+1,0);
    inE.assign(n,{}); outE.assign(n,{});
    for(int i=1;i<=m;i++){int u,v,c;cin>>u>>v>>c;eu[i]=u;ev[i]=v;ec[i]=c;inE[v].push_back(i);outE[u].push_back(i);}
    Lmax=n; isprime.assign(Lmax+2,1); isprime[0]=isprime[1]=0;
    for(int i=2;(ll)i*i<=Lmax;i++) if(isprime[i]) for(int j=i*i;j<=Lmax;j+=i) isprime[j]=0;

    vector<char> rm(m+1,0);
    ll F0=computeF(rm);
    vector<int> order; vector<ll> Fseq;
    for(int it=0; it<k; it++){
        vector<double> tS,tT; totals(rm,tS,tT);
        int best=-1; double bs=-1;
        for(int id=1;id<=m;id++){ if(rm[id])continue; double thr=tS[eu[id]]*tT[ev[id]];
            if(thr>bs){ bs=thr; best=id; } }
        if(best<0 || bs<=0) break;
        rm[best]=1; order.push_back(best); Fseq.push_back(computeF(rm));
    }
    int bestJ=0; ll bestF=F0;
    for(int j=0;j<(int)Fseq.size();j++) if(Fseq[j]<bestF){bestF=Fseq[j];bestJ=j+1;}
    printf("%d\n",bestJ);
    for(int j=0;j<bestJ;j++) printf("%d\n",order[j]);
    return 0;
}
