import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

global G,c, m0, mh
G=6.67e-8
c=3e10
m0=1.6586e-24
mh=1.6735e-24

#Interpolating the EOS

sly=np.genfromtxt("SLy.txt",delimiter="  ")
nbs=sly[:,1]*1e39
rhos=sly[:,2]
Ps=sly[:,3]

cPs=CubicSpline(rhos,Ps)
crs=CubicSpline(Ps,rhos)
cns=CubicSpline(Ps,nbs)


fps=np.genfromtxt("FPS.txt",delimiter="    ")
nbf=fps[:,1]*1e39
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
        nb=cns(P)
    elif(bool==1):
        rho=crf(P)
        nb=cnf(P)
    elif(bool==2):
        rho=cra(P)
        nb=cna(P)
    dr_dr=1
    dm_dr=4.*np.pi*(r**2)*rho
    dP_dr=-(((G*m*rho)/(r**2))*(1+(P/(rho*c*c)))*(1+((4*np.pi*P*(r**3))/(m*c*c))))/(1-((2*G*m)/(r*c*c)))
    dphi_dr=(-dP_dr)/((rho*c*c)*(1+(P/(rho*c*c))))
    dab_dr=(4*np.pi*r*r*nb)/(1-((2*G*m)/(r*c*c)))

    return np.array([dr_dr, dm_dr, dP_dr, dphi_dr, dab_dr])

def ns_solve(rho_0,bool):
#Initial Conditions
    dr=500 #In cm
    if(bool==0):
        P_0=cPs(rho_0)
    elif(bool==1):
        P_0=cPf(rho_0)
    elif(bool==2):
        P_0=cPa(rho_0)
    X=np.zeros([5,80000])
    X[:,0]=np.array([500,1,P_0,0.001,0.001])

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
    
    return X[:,i-1]

rho=np.arange(2.5e14,1e15,0.5e13)
rho=np.append(rho,np.arange(1e15,5e15,0.5e14))

res_s=np.zeros([5,len(rho)])
res_f=np.zeros([5,len(rho)])
#res_a=np.zeros([5,len(rho)])

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

Ri_s=R_s/np.sqrt(1-((2*G*M_s*2e33)/(R_s*1e5*c*c)))
Ri_f=R_f/np.sqrt(1-((2*G*M_f*2e33)/(R_f*1e5*c*c)))
#Ri_a=R_a/np.sqrt(1-((2*G*M_a*2e33)/(R_a*1e5*c*c)))

Ab_s=res_s[4,]/1e57
Ab_f=res_f[4,]/1e57
#Ab_a=res_a[4,]/1e57

EFebind_s=((Ab_s*1e57*m0)-(M_s*2e33))*c*c
EFebind_f=((Ab_f*1e57*m0)-(M_f*2e33))*c*c
#EFebind_a=((Ab_a*1e57*m0)-(M_a))*c*c

EHbind_s=((Ab_s*1e57*mh)-(M_s*2e33))*c*c
EHbind_f=((Ab_f*1e57*mh)-(M_f*2e33))*c*c
#EHbind_a=((Ab_a*1e57*mh)-(M_a*2e33))*c*c

z_s=np.exp(-res_s[3,])-1
z_f=np.exp(-res_f[3,])-1
#z_a=np.exp(-res_a[3,])-1

plt.figure()
ax=plt.gca()
ax.set_title(r"Stationary NS Plot: Baryon Number vs $\rho_c$")
ax.set_xlabel(r"$\rho_c$ [$g/cm^3$]")
ax.set_ylabel(r"$Ab_{57}$")
plt.plot(rho,Ab_s,'r--',label="SLy EoS")
plt.plot(rho,Ab_f,'b--',label="FPS EoS")
#plt.plot(rho,Ab_a,'g--',label="APR EoS")
plt.legend(loc="best")
plt.savefig("AbvsrhoStat.png")
plt.close()

plt.figure()
ax=plt.gca()
ax.set_title(r"Stationary NS Plot: Surface Redshift vs $\rho_c$")
ax.set_xlabel(r"$\rho_c$ [$g/cm^3$]")
ax.set_ylabel(r"$z_{surf}$")
plt.plot(rho,z_s,'r--',label="SLy EoS")
plt.plot(rho,z_f,'b--',label="FPS EoS")
#plt.plot(rho,z_a,'g--',label="APR EoS")
plt.legend(loc="best")
plt.savefig("zvsrhoStat.png")
plt.close()

plt.figure()
ax=plt.gca()
ax.set_title(r"Stationary NS Plot: Apparent Radii vs Mass")
ax.set_xlabel(r"Mass of the Star [$M_\odot$]")
ax.set_ylabel(r"$R_\infty$ [km]")
plt.plot(M_s,Ri_s,'r--',label="SLy EoS")
plt.plot(M_f,Ri_f,'b--',label="FPS EoS")
#plt.plot(M_a,Ri_a,'g--',label="APR EoS")
plt.legend(loc="best")
plt.savefig("RivsMStat.png")
plt.close()

plt.figure()
ax=plt.gca()
ax.set_title(r"Stationary NS Plot: Binding Energy vs Mass")
ax.set_xlabel(r"Mass of the Star [$M_\odot$]")
ax.set_ylabel(r"$E_{bind}$ [ergs]")
plt.plot(M_s,EFebind_s,'r--',label=r"$E_{Fe}$ [SLy EoS]")
plt.plot(M_s,EHbind_s,'--',color="orange",label=r"$E_{H}$ [SLy EoS]")
plt.plot(M_f,EFebind_f,'b--',label=r"$E_{Fe}$ [FPS EoS]")
plt.plot(M_f,EHbind_f,'--',color="purple",label=r"$E_{H}$ [FPS EoS]")
#plt.plot(M_a,EFebind_a,'g--',label=r"$E_{Fe}$ [APR EoS]")
#plt.plot(M_a,EHbind_a,'g.-',label=r"$E_{H}$ [APR EoS]")
plt.legend(loc="best")
plt.savefig("EvsMStat.png")
plt.close()
