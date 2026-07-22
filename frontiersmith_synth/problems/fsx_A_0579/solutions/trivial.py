# TIER: trivial
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

def main():
    n,K_total,T_max,c_unit,beta,m,st=read_instance()
    w,phi=modes(n,m,st)
    amp=phi[:,0]**2
    order=sorted(range(n),key=lambda j:(-amp[j],j))
    t=[0]*n; b=K_total
    for j in order:
        if b<=0: break
        add=min(T_max,b); t[j]=add; b-=add
    emit(t)
main()
