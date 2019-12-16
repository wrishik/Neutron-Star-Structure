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

#Returning the spatial derivatives of the functions
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
    
    return np.array([dr_dr, dm_dr, dP_dr])

def ns_solve(rho_0,bool):
    #Initial Conditions
    dr=500 #In cm
    if(bool==0):
        P_0=cPs(rho_0)
    elif(bool==1):
        P_0=cPf(rho_0)
    elif(bool==2):
        P_0=cPa(rho_0)
    #print(P_0)
    X=np.zeros([3,80000])
    X[:,0]=np.array([500,1,P_0])

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
    
    return X[:,i-1]

rho=np.arange(2.5e14,1e15,0.5e13)
rho=np.append(rho,np.arange(1e15,4e15,0.5e14))
res_s=np.zeros([3,len(rho)])
res_f=np.zeros([3,len(rho)])
#res_a=np.zeros([3,len(rho)])

for i in range(len(rho)):
    res_s[:,i]=ns_solve(rho[i],0)
    res_f[:,i]=ns_solve(rho[i],1)
    #res_a[:,i]=ns_solve(rho[i],2)
    print(i)

R_s=res_s[0,]/1.e5
R_f=res_f[0,]/1e5
#R_a=res_a[0,]/1e5
M_s=res_s[1,]/2e33
M_f=res_f[1,]/2e33
#M_a=res_a[1,]/2e33


plt.figure()
ax=plt.gca()
ax.set_title(r"Stationary NS Plot: Mass vs $\rho_c$")
ax.set_xlabel(r"$\rho_c$ [$g/cm^3$]")
ax.set_ylabel(r"Mass of the Star [$M_\odot$]")
plt.plot(rho,M_s,'r--',label="SLy EoS")
plt.plot(rho,M_f,'b--',label="FPS EoS")
#plt.plot(rho,M_a,'g--',label="APR EoS")
plt.legend(loc="best")
plt.savefig("MvsrhoStat.png")
plt.close()

plt.figure()
ax=plt.gca()
ax.set_title(r"Stationary NS Plot: Radius vs $\rho_c$")
ax.set_xlabel(r"$\rho_c$ [$g/cm^3$]")
ax.set_ylabel(r"Radius of the Star [km]")
plt.plot(rho,R_s,'r--',label="SLy EoS")
plt.plot(rho,R_f,'b--',label="FPS EoS")
#plt.plot(rho,R_a,'g--',label="APR EoS")
plt.legend(loc="best")
plt.savefig("RvsrhoStat.png")
plt.close()

plt.figure()
ax=plt.gca()
ax.set_title(r"Stationary NS Plot: Radius vs Mass")
ax.set_xlabel(r"Mass of the Star [$M_\odot$]")
ax.set_ylabel(r"Radius of the Star [km]")
plt.plot(M_s,R_s,'r--',label="SLy EoS")
plt.plot(M_f,R_f,'b--',label="FPS EoS")
#plt.plot(M_a,R_a,'g--',label="APR EoS")
plt.legend(loc="best")
plt.savefig("RvsMStat.png")
plt.close()





