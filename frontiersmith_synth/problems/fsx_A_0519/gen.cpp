// generator for Pottery Kiln Firing Queue
#include "testlib.h"
#include <vector>
#include <algorithm>
using namespace std;
typedef long long ll;

struct Job{ ll lo,hi,d,D,w; };

int main(int argc,char**argv){
    registerGen(argc,argv,1);
    int testId=atoi(argv[1]);

    // per-test parameters
    int N; ll cheat, cnum, cden;
    // flavor: 0=mild mixed, 1=trap(cold tight/hot loose), 2=needle, 3=large adversarial
    int flavor;
    int nHot, nCold;
    double coldTightFrac; // fraction of cold jobs with tight deadlines

    if(testId==1){ N=5;  cheat=3; cnum=8; cden=10; flavor=0; nHot=2; nCold=3; coldTightFrac=0.6; }
    else if(testId==2){ N=40;  cheat=2; cnum=7; cden=10; flavor=0; nHot=15; nCold=25; coldTightFrac=0.4; }
    else if(testId==3){ N=80;  cheat=3; cnum=8; cden=10; flavor=1; nHot=25; nCold=55; coldTightFrac=0.7; }
    else if(testId==4){ N=120; cheat=2; cnum=9; cden=10; flavor=1; nHot=30; nCold=90; coldTightFrac=0.8; }
    else if(testId==5){ N=200; cheat=3; cnum=8; cden=10; flavor=1; nHot=50; nCold=150;coldTightFrac=0.75;}
    else if(testId==6){ N=300; cheat=2; cnum=9; cden=10; flavor=1; nHot=70; nCold=230;coldTightFrac=0.8; }
    else if(testId==7){ N=500; cheat=4; cnum=8; cden=10; flavor=2; nHot=120;nCold=380;coldTightFrac=0.6; }
    else if(testId==8){ N=800; cheat=3; cnum=9; cden=10; flavor=1; nHot=180;nCold=620;coldTightFrac=0.85;}
    else if(testId==9){ N=1400;cheat=3; cnum=8; cden=10; flavor=3; nHot=350;nCold=1050;coldTightFrac=0.7;}
    else            { N=2000;cheat=2; cnum=9; cden=10; flavor=3; nHot=500;nCold=1500;coldTightFrac=0.8;}

    // build jobs
    vector<Job> jobs;
    // durations first so we can size deadlines to total work
    // generate hot jobs
    vector<Job> hot, cold;
    for(int k=0;k<nHot;k++){
        ll hi = rnd.next(300,900);
        ll lo = max(1LL, hi - rnd.next(40,160));
        ll d  = rnd.next(3,15);
        ll w  = rnd.next(1,4);
        hot.push_back({lo,hi,d,0,w});
    }
    for(int k=0;k<nCold;k++){
        ll hi = rnd.next(8,40);
        ll lo = rnd.next(2, (int)hi-2>2?(int)hi-2:3);
        if(lo>hi) lo=hi;
        ll d  = rnd.next(3,15);
        ll w  = rnd.next(2,6);
        cold.push_back({lo,hi,d,0,w});
    }
    ll totalDur=0;
    for(auto&j:hot) totalDur+=j.d;
    for(auto&j:cold) totalDur+=j.d;
    if(totalDur<1) totalDur=1;

    // assign deadlines
    ll LOOSE = totalDur*3 + 1000;
    for(auto&j:hot){
        // most hot jobs loose; a few tightish to add noise (flavor 2/3)
        if((flavor==2||flavor==3) && rnd.next(0,4)==0)
            j.D = (ll)(totalDur*(0.5+rnd.next(0,30)/100.0));
        else
            j.D = LOOSE - rnd.next(0,200);
    }
    for(auto&j:cold){
        bool tight = (rnd.next(0,999)/1000.0) < coldTightFrac;
        if(tight){
            // meetable only if done early (EDD); blown if done last (hottest-first)
            double f = 0.30 + rnd.next(0,30)/100.0;   // 0.30..0.60 of total work
            j.D = max((ll)1, (ll)(totalDur*f));
        }else{
            j.D = LOOSE - rnd.next(0,200);
        }
    }
    if(flavor==2){
        // NEEDLE: one very-high-weight cold job with a very tight deadline
        if(!cold.empty()){
            cold[0].w=8; cold[0].D=max((ll)1,(ll)(totalDur*0.15));
        }
    }

    // combine
    for(auto&j:hot) jobs.push_back(j);
    for(auto&j:cold) jobs.push_back(j);

    // ADVERSARIAL input order: zig-zag interleave hot/cold so input-order baseline pays
    // repeated reheats (maximizes B); shuffle within lists first for variety.
    vector<int> hidx, cidx;
    for(int i=0;i<(int)jobs.size();i++){ if(i<nHot) hidx.push_back(i); else cidx.push_back(i);}
    shuffle(hidx.begin(),hidx.end());
    shuffle(cidx.begin(),cidx.end());
    vector<int> order;
    size_t a=0,b=0;
    // interleave (2 cold : 1 hot roughly, to keep reheats frequent but bounded)
    while(a<hidx.size()||b<cidx.size()){
        if(a<hidx.size()){ order.push_back(hidx[a++]); }
        if(b<cidx.size()){ order.push_back(cidx[b++]); }
        if(b<cidx.size()){ order.push_back(cidx[b++]); }
    }

    // emit
    printf("%d %lld %lld %lld\n", (int)jobs.size(), cheat, cnum, cden);
    for(int idx: order){
        Job&j=jobs[idx];
        printf("%lld %lld %lld %lld %lld\n", j.lo, j.hi, j.d, j.D, j.w);
    }
    return 0;
}
