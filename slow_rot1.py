import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

global G,c
G=6.67e-8
c=3e10

#Interpolating the EOS

sly=np.genfromtxt("SLy.txt",delimiter="  ")
nbs=sly[:,1]
rhos=sly[:,2]
Ps=sly[:,3]

cPs=CubicSpline(rhos,Ps)
crs=CubicSpline(Ps,rhos)
cns=CubicSpline(Ps,nbs)


fps=np.genfromtxt("FPS.txt",delimiter="    ")
nbf=fps[:,1]
rhof=fps[:,2]
Pf=fps[:,3]

cPf=CubicSpline(rhof,Pf)
crf=CubicSpline(Pf,rhof)
cnf=CubicSpline(Pf,nbf)


apr=np.genfromtxt("apr.txt", delimiter="  ")
nba=apr[:,0]*1e14*c*c
rhoa=apr[:,1]*1e14
Pa=apr[:,2]*1e14*c*c

cPa=CubicSpline(rhoa,Pa)
cra=CubicSpline(Pa,rhoa)
cna=CubicSpline(Pa,nba)

def f(x,bool):
    r=x[0]
    m=x[1]
    P=x[2]
    if(bool==0):
        rho=crs(P)
    elif(bool==1):
        rho=crf(P)
    elif(bool==2):
        rho=cra(P)
    dr_dr=1
    dm_dr=4.*np.pi*(r**2)*rho
    dP_dr=-(((G*m*rho)/(r**2))*(1+(P/(rho*c*c)))*(1+((4*np.pi*P*(r**3))/(m*c*c))))/(1-((2*G*m)/(r*c*c)))
    dphi_dr=(-dP_dr)/((rho*c*c)*(1+(P/(rho*c*c))))

    return np.array([dr_dr, dm_dr, dP_dr, dphi_dr])
    
def f2(x,y,bool):
    r=x[0]
    M=x[1]
    P=x[2]
    phi=x[3]
    j=y[0]
    w=y[1]
    
    if(bool==0):
        rho=crs(P)
    elif(bool==1):
        rho=crf(P)
    elif(bool==2):
        rho=cra(P)
    dj_dr=(((8.*np.pi)/3.)*(r**4)*(rho+(P/(c**2)))*w*(np.exp(-phi)))*(np.sqrt(1-(2*G*M)/(r*c*c)))
    
    dw_dr=(G*np.exp(phi)*j)/(c*c*(r**4)*(np.sqrt(1-(2*G*M)/(r*c*c))))
    
    return np.array([dj_dr,dw_dr])

def ns_solve(w_0,bool):
#Initial Conditions
    rho_0=1e15
    dr=500 #In cm
    if(bool==0):
        P_0=cPs(rho_0)
    elif(bool==1):
        P_0=cPf(rho_0)
    elif(bool==2):
        P_0=cPa(rho_0)
    X=np.zeros([4,80000])
    X[:,0]=np.array([500,1,P_0,0.001])

    #Solve using RK4
    for i in range(1,80000):
        k1=f(X[:,i-1],bool)
        k2=f(X[:,i-1]+k1*0.5*dr,bool)
        k3=f(X[:,i-1]+k2*0.5*dr,bool)
        k4=f(X[:,i-1]+k3*dr,bool)
    
        X[:,i]=X[:,i-1]+(dr*(k1+2*k2+2*k3+k4))/6.
        
        if((X[2,i]/P_0)<1e-10):
            break

    #for j in range(i,80000):
        #X=np.delete(X,i,1)
    
    alpha=X[3,i-1]
    k=(0.5*np.log(1-((2*G*X[1,i-1])/(X[0,i-1]*c*c))))/alpha
    X[3,]=X[3,]*k
        
    Y=np.zeros([2,i])
    Y[:,0]=np.array([0.001,w_0])
    
    for j in range(1,i):
        k_1=f2(X[:,j-1],Y[:,j-1],bool)
        k_2=f2(X[:,j-1],Y[:,j-1]+k_1*0.5*dr,bool)
        k_3=f2(X[:,j-1],Y[:,j-1]+k_2*0.5*dr,bool)
        k_4=f2(X[:,j-1],Y[:,j-1]+k_3*dr,bool)
        
        Y[:,j]=Y[:,j-1]+(dr*(k_1+2*k_2+2*k_3+k_4))/6.
    
    
    Y[1,i-1]=Y[1,i-1]+((2*G*Y[0,i-1])/((c**2)*(X[0,i-1]**3)))
    return X[:,i-1],Y[:,i-1]

w=np.linspace(1.,100.,100)

res_s1=np.zeros([4,len(w)])
res_f1=np.zeros([4,len(w)])
res_a1=np.zeros([4,len(w)])

res_s2=np.zeros([2,len(w)])
res_f2=np.zeros([2,len(w)])
res_a2=np.zeros([2,len(w)])

for i in range(len(w)):
    res_s1[:,i],res_s2[:,i]=ns_solve(w[i],0)
    res_f1[:,i],res_f2[:,i]=ns_solve(w[i],1)
    res_a1[:,i],res_a2[:,i]=ns_solve(w[i],2)
    print(i)


w_s=res_s2[1,]
w_f=res_f2[1,]
w_a=res_a2[1,]

J_s=res_s2[0,]
J_f=res_f2[0,]
J_a=res_a2[0,]

I_s=np.divide(J_s,w_s)
I_f=np.divide(J_f,w_f)
I_a=np.divide(J_a,w_a)


plt.figure()
ax=plt.gca()
ax.set_title(r"Slow Rotating NS Plot: MOI vs Central Spin")
ax.set_ylabel(r"Moment of Inertia $[g\,cm^2]$")
ax.set_xlabel(r"$\omega_0$ [$rot/sec$]")
plt.plot(w,I_s,'r--',label="SLy EoS")
plt.plot(w,I_f,'b--',label="FPS EoS")
plt.plot(w,I_a,'g--',label="APR EoS")
plt.text(10,0.85e45,r"$\rho_c = 10^{15}\;g/cm^3$",bbox=dict(facecolor='white', alpha=0.5))
plt.legend(loc="best")
plt.savefig("MoIvsw_SRot.png")
plt.close()

plt.figure()
ax=plt.gca()
ax.set_title(r"Slow Rotating NS Plot: Observed vs Central Spin")
ax.set_ylabel(r"$\Omega_{obs}$ [$rot/sec$]")
ax.set_xlabel(r"$\omega_0$ [$rot/sec$]")
plt.plot(w,w_s,'r--',label="SLy EoS")
plt.plot(w,w_f,'b--',label="FPS EoS")
plt.plot(w,w_a,'g--',label="APR EoS")
plt.text(60,20,r"$\rho_c = 10^{15}\;g/cm^3$",bbox=dict(facecolor='white', alpha=0.5))
plt.legend(loc="best")
plt.savefig("wvsw_SRot.png")
plt.close()

plt.figure()
ax=plt.gca()
ax.set_title(r"Slow Rotating NS Plot: J vs Central Spin")
ax.set_ylabel(r"J [$g\,cm^2/sec$]")
ax.set_xlabel(r"$\omega_0$ [$rot/sec$]")
plt.plot(w,J_s,'r--',label="SLy EoS")
plt.plot(w,J_f,'b--',label="FPS EoS")
plt.plot(w,J_a,'g--',label="APR EoS")
plt.text(60,0.2e47,r"$\rho_c = 10^{15}\;g/cm^3$",bbox=dict(facecolor='white', alpha=0.5))
plt.legend(loc="best")
plt.savefig("Jvsw_SRot.png")
plt.close()
