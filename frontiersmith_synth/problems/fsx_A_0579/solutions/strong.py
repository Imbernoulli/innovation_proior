# TIER: strong
import os, sys
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_v]="1"
import numpy as np

def read_instance():
    toks=sys.stdin.read().split(); it=iter(toks)
    n=int(next(it)); K_total=int(next(it)); T_max=int(next(it))
    c_unit=float(next(it)); beta=float(next(it))
    m=[]; st=[]
    for _ in range(n):
        m.append(float(next(it))); st.append(float(next(it)))
    return n,K_total,T_max,c_unit,beta,m,st

def assemble(n,m,st):
    M=np.array(m,float); K=np.zeros((n,n))
    for j in range(n):
        K[j,j]+=st[j]
        if j+1<n:
            K[j,j]+=st[j+1]; K[j,j+1]-=st[j+1]; K[j+1,j]-=st[j+1]
    return M,K

def modes(n,m,st):
    M,K=assemble(n,m,st); msi=1.0/np.sqrt(M)
    Ks=(msi[:,None]*K)*msi[None,:]
    w,U=np.linalg.eigh(0.5*(Ks+Ks.T)); phi=msi[:,None]*U
    return w,phi

def state(n,m,st,beta,c_unit,t):
    M,K=assemble(n,m,st); C=beta*K.copy()
    for j in range(n): C[j,j]+=t[j]*c_unit
    Minv=np.diag(1.0/M); A=np.zeros((2*n,2*n))
    A[:n,n:]=np.eye(n); A[n:,:n]=-Minv@K; A[n:,n:]=-Minv@C
    return A,M

def margin(n,m,st,beta,c_unit,t):
    A,_=state(n,m,st,beta,c_unit,t)
    return -float(np.max(np.linalg.eigvals(A).real))

def emit(t):
    sys.stdout.write(" ".join(str(int(x)) for x in t)+"\n")

def uniform_start(n,K_total,T_max):
    t=[0]*n; b=K_total; j=0
    while b>0:
        if t[j%n]<T_max: t[j%n]+=1; b-=1
        j+=1
    return t

def main():
    n,K_total,T_max,c_unit,beta,m,st=read_instance()
    marr=np.array(m,float)
    t=uniform_start(n,K_total,T_max)
    cur=margin(n,m,st,beta,c_unit,t)
    max_moves=4*K_total+200
    moves=0
    while moves<max_moves:
        A,_=state(n,m,st,beta,c_unit,t)
        ev,VR=np.linalg.eig(A)
        try:
            VL=np.linalg.inv(VR).conj().T
        except Exception:
            break
        k=int(np.argmax(ev.real))
        x=VR[:,k]; y=VL[:,k]; denom=np.vdot(y,x)
        # d Re(lambda*) / d c_i  via eigenvalue perturbation of the rightmost mode.
        sens=np.array([(-(1.0/marr[i])*np.conj(y[n+i])*x[n+i]/denom).real for i in range(n)])
        add_order=sorted(range(n),key=lambda i:sens[i])          # steer damping toward
        rem_order=sorted([i for i in range(n) if t[i]>0],key=lambda i:-sens[i])
        applied=False
        for bb in add_order[:6]:
            if t[bb]>=T_max: continue
            for aa in rem_order[:6]:
                if aa==bb or t[aa]<=0: continue
                t[aa]-=1; t[bb]+=1
                mm=margin(n,m,st,beta,c_unit,t)
                if mm>cur+1e-12:
                    cur=mm; applied=True; moves+=1; break
                t[aa]+=1; t[bb]-=1
            if applied: break
        if not applied: break
    emit(t)
main()
