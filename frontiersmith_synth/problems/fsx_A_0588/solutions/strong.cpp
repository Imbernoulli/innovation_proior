// TIER: strong
// INSIGHT: never move heat across the pinch. Serve each cold with the COLDEST hot
// that is still hot enough (minimum driving force) -- this keeps high-grade hot for
// high-grade cold and confines every match to its own temperature cluster, so the
// stranding that ruins the greedy never happens. Here strong uses a simple pinch-
// respecting ONE-TO-ONE match (each stream used once); splitting a hot's duty across
// several colds recovers the residuals it leaves -> head-room above this reference.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    int NH,NC; ll dTmin;
    if(!(cin>>NH>>NC>>dTmin)) return 0;
    vector<ll> Hs(NH),Ht(NH),CPh(NH),capH(NH),qh(NH,0);
    vector<ll> Cs(NC),Ct(NC),CPc(NC),capC(NC),qc(NC,0);
    for(int i=0;i<NH;i++){cin>>Hs[i]>>Ht[i]>>CPh[i]; capH[i]=CPh[i]*(Hs[i]-Ht[i]);}
    for(int j=0;j<NC;j++){cin>>Cs[j]>>Ct[j]>>CPc[j]; capC[j]=CPc[j]*(Ct[j]-Cs[j]);}
    // serve highest-grade cold demand first
    vector<int> co(NC); iota(co.begin(),co.end(),0);
    sort(co.begin(),co.end(),[&](int a,int b){return Ct[a]>Ct[b];});
    vector<char> hUsed(NH,0);
    vector<array<ll,3>> out;
    for(int cj:co){
        // pick the coldest-supply hot that is unused and can transfer something
        int best=-1;
        for(int hi=0;hi<NH;hi++){
            if(hUsed[hi]) continue;
            if(capH[hi]<=0) continue;
            ll K=(Hs[hi]-Cs[cj]-dTmin)*CPh[hi]*CPc[cj];
            ll G=K-qh[hi]*CPc[cj]-qc[cj]*CPh[hi];
            if(G<=0) continue;
            ll qf=G/max(CPh[hi],CPc[cj]);
            ll Q=min(qf,min(capH[hi]-qh[hi],capC[cj]-qc[cj]));
            if(Q<1) continue;
            if(best==-1 || Hs[hi]<Hs[best]) best=hi;  // coldest adequate hot
        }
        if(best<0) continue;
        int hi=best;
        ll K=(Hs[hi]-Cs[cj]-dTmin)*CPh[hi]*CPc[cj];
        ll G=K-qh[hi]*CPc[cj]-qc[cj]*CPh[hi];
        ll qf=G/max(CPh[hi],CPc[cj]);
        ll Q=min(qf,min(capH[hi]-qh[hi],capC[cj]-qc[cj]));
        if(Q>=1){ out.push_back({(ll)hi+1,(ll)cj+1,Q}); qh[hi]+=Q; qc[cj]+=Q; hUsed[hi]=1; }
    }
    cout<<out.size()<<"\n";
    for(auto&e:out) cout<<e[0]<<" "<<e[1]<<" "<<e[2]<<"\n";
    return 0;
}
