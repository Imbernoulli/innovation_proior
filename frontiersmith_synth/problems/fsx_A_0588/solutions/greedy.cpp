// TIER: greedy
// The obvious recipe: pair the HOTTEST hot stream with the COLDEST cold stream and
// dump as much heat as feasible, then continue. This maximizes the driving force of
// every match -- and drives high-grade HIGH-cluster heat straight ACROSS the pinch
// into LOW-cluster colds, stranding the HIGH-cluster colds and the LOW-cluster hots.
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
    vector<int> ho(NH),co(NC);
    iota(ho.begin(),ho.end(),0); iota(co.begin(),co.end(),0);
    sort(ho.begin(),ho.end(),[&](int a,int b){return Hs[a]>Hs[b];});   // hottest first
    sort(co.begin(),co.end(),[&](int a,int b){return Cs[a]<Cs[b];});   // coldest first
    vector<array<ll,3>> out;
    for(int hi:ho){
        for(int cj:co){
            ll remH=capH[hi]-qh[hi]; if(remH<=0) break;
            ll remC=capC[cj]-qc[cj]; if(remC<=0) continue;
            ll K=(Hs[hi]-Cs[cj]-dTmin)*CPh[hi]*CPc[cj];
            ll G=K-qh[hi]*CPc[cj]-qc[cj]*CPh[hi];
            if(G<=0) continue;
            ll qf=G/max(CPh[hi],CPc[cj]);
            ll Q=min(qf,min(remH,remC));
            if(Q>=1){ out.push_back({(ll)hi+1,(ll)cj+1,Q}); qh[hi]+=Q; qc[cj]+=Q; }
        }
    }
    cout<<out.size()<<"\n";
    for(auto&e:out) cout<<e[0]<<" "<<e[1]<<" "<<e[2]<<"\n";
    return 0;
}
