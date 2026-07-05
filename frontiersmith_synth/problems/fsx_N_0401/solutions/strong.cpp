// TIER: strong
// Resonant-flow interdiction with improvement guard:
//   each round, rank couplers by how many PRIME-length s->t routes thread them
//   (forward x backward hop-count counts over prime sums), then among the top
//   candidates cut the one that most reduces the true objective F; stop when no
//   single cut helps. Every accepted cut strictly lowers F, so F <= B always.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n,m,s,t,k; ll P;
vector<int> eu,ev,ec;
vector<vector<int>> inE, outE;
vector<char> isprime;
int Lmax, PS;
const double CAP=1e15;

void buildCounts(const vector<char>& rm, vector<vector<double>>& cS, vector<vector<double>>& cT){
    cS.assign(n, vector<double>(Lmax+1,0.0));
    cT.assign(n, vector<double>(Lmax+1,0.0));
    cS[s][0]=1.0;
    for(int v=0;v<n;v++){ if(v==s)continue;
        for(int id:inE[v]){ if(rm[id])continue; int u=eu[id];
            for(int a=1;a<=Lmax;a++){ double val=cS[u][a-1]; if(val>0){ cS[v][a]+=val; if(cS[v][a]>CAP)cS[v][a]=CAP; } } } }
    cT[t][0]=1.0;
    for(int v=n-1;v>=0;v--){ if(v==t)continue;
        for(int id:outE[v]){ if(rm[id])continue; int w=ev[id];
            for(int b=1;b<=Lmax;b++){ double val=cT[w][b-1]; if(val>0){ cT[v][b]+=val; if(cT[v][b]>CAP)cT[v][b]=CAP; } } } }
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
    vector<int> dist(n,-1); dist[s]=0;
    for(int v=0;v<n;v++){ if(dist[v]<0)continue; for(int id:outE[v]){ int w=ev[id]; if(dist[v]+1>dist[w])dist[w]=dist[v]+1; } }
    Lmax=max(1, dist[t]<0?1:dist[t]);
    PS=2*Lmax+2; int LIM=max(n,PS);
    isprime.assign(LIM+1,1); isprime[0]=isprime[1]=0;
    for(int i=2;(ll)i*i<=LIM;i++) if(isprime[i]) for(int j=i*i;j<=LIM;j+=i) isprime[j]=0;

    vector<char> rm(m+1,0);
    ll curF=computeF(rm);
    vector<int> chosen;
    const int TOPT=80;

    for(int it=0; it<k; it++){
        vector<vector<double>> cS,cT; buildCounts(rm,cS,cT);
        vector<pair<double,int>> cand;
        for(int id=1;id<=m;id++){
            if(rm[id]) continue;
            int u=eu[id], v=ev[id];
            double thr=0;
            for(int a=0;a<=Lmax;a++){ double fu=cS[u][a]; if(fu<=0)continue;
                for(int b=0;b<=Lmax;b++){ double gv=cT[v][b]; if(gv<=0)continue; int L=a+1+b; if(L<=PS && isprime[L]) thr+=fu*gv; } }
            if(thr>0) cand.push_back({thr,id});
        }
        if(cand.empty()) break;
        sort(cand.begin(),cand.end(),[](const pair<double,int>&A,const pair<double,int>&B){return A.first>B.first;});
        int take=min((int)cand.size(),TOPT);
        int bestId=-1; ll bestF=curF;
        for(int i=0;i<take;i++){
            int id=cand[i].second; rm[id]=1; ll F=computeF(rm); rm[id]=0;
            if(F<bestF){ bestF=F; bestId=id; }
        }
        if(bestId<0) break;             // no single cut improves
        rm[bestId]=1; curF=bestF; chosen.push_back(bestId);
    }

    printf("%d\n",(int)chosen.size());
    for(int id:chosen) printf("%d\n",id);
    return 0;
}
