import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

global G,c
G=6.67e-8
c=3e10

#Interpolating the EOS

sly=np.genfromtxt("SLy.txt",delimiter="  ")
#nbs=sly[:,1]
Es=sly[:,2]
Ps=sly[:,3]

cPs=CubicSpline(Es,Ps)
crs=CubicSpline(Ps,Es)
#cns=CubicSpline(Ps,nbs)


fps=np.genfromtxt("FPS.txt",delimiter="    ")
#nbf=fps[:,1]
Ef=fps[:,2]
Pf=fps[:,3]

cPf=CubicSpline(Ef,Pf)
crf=CubicSpline(Pf,Ef)
#cnf=CubicSpline(Pf,nbf)


apr=np.genfromtxt("apr.txt", delimiter="  ")
#nba=apr[:,0]*1e14*c*c
Ea=apr[:,1]*1e14
Pa=apr[:,2]*1e14*c*c

cPa=CubicSpline(Ea,Pa)
cra=CubicSpline(Pa,Ea)
#cna=CubicSpline(Pa,nba)

def fp(p,bool):
    dp=p/1.e5
    if(bool==0):
        res=(-crs(p+(2*dp))+8*crs(p+dp)-8*crs(p-dp)+crs(p-(2*dp)))/(12*dp)
    elif(bool==1):
         res=(-crf(p+(2*dp))+8*crf(p+dp)-8*crf(p-dp)+crf(p-(2*dp)))/(12*dp)
    elif(bool==2):
         res=(-cra(p+(2*dp))+8*cra(p+dp)-8*cra(p-dp)+cra(p-(2*dp)))/(12*dp)
    return res

def f(x,bool):
    r=x[0]
    m=x[1]
    P=x[2]
    H=x[3]
    B=x[4]
    if(bool==0):
        rho=crs(P)
        F=crs(P+(rho*c*c))
    elif(bool==1):
        rho=crf(P)
        F=crf(P+(rho*c*c))
    elif(bool==2):
        rho=cra(P)
        F=cra(P+(rho*c*c))
    dr_dr=1
    dm_dr=4.*np.pi*(r**2)*rho
    dP_dr=-(((G*m*rho)/(r**2))*(1+(P/(rho*c*c)))*(1+((4*np.pi*P*(r**3))/(m*c*c))))/(1-((2*G*m)/(r*c*c)))
    dH_dr=B
    dB_dr=2*H*(1-((2*G*m)/(r*c*c)))*((3/(r*r))+(2*((((m*G)/(r*r*c*c))+((G*4*np.pi*r*P)/(c**4)))**2)/(1-((2*G*m)/(r*c*c))))-(2*np.pi*(G/(c**4))*(5*rho*c*c+9*P+F*c*c)))+((2*(B/r))*(-1+((2*G*m)/(r*c*c)))+((((rho*c*c)-P)*G*r*r*np.pi*2)/(c**4)))
    
    return np.array([dr_dr, dm_dr, dP_dr, dH_dr, dB_dr])

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
    X=np.zeros([5,80000])
    X[:,0]=np.array([500,1,P_0,500*500,1000])

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
res_s=np.zeros([5,len(rho)])
res_f=np.zeros([5,len(rho)])
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

B_s=res_s[4,]
B_f=res_f[4,]
#B_a=res_a[4,]

H_s=res_s[3,]
H_f=res_f[3,]
#H_a=res_a[3,]

y_s=(R_s*1e5*B_s)/H_s
y_f=(R_f*1e5*B_f)/H_f
#y_a=(R_a*B_a)/H_a

C_s=(2*G*M_s*2e33)/(R_s*1e5*c*c)
C_f=(2*G*M_f*2e33)/(R_f*1e5*c*c)
#C_a=(2*G*M_a*2e33)/(R_a*1e5*c*c)

def k(y,C):
    k2=(1.6*(C**5)*((1-2*C)**2)*(2+2*C*(y-1)-y))/(2*C*(6-3*y+3*C*(5*y-8))+4*(C**3)*(13-11*y+C*(3*y-2)+2*C*C*(1+y))+3*((1-2*C)**2)*(2-y+2*C*(y-1))*np.log(1-2*C))
    return k2

k2_s=k(y_s,C_s)
k2_f=k(y_f,C_f)
#k2_a=k(y_a,C_a)


L_s=(2.*k2_s*((R_s*1e5)**5))/(3.*G)
L_f=(2.*k2_f*((R_f*1e5)**5))/(3.*G)
#L_a=(2.*k2_a*((R_a*1e5)**5))/(3.*G)

plt.figure()
ax=plt.gca()
ax.set_title(r"NS Plot: Love Number vs Mass")
ax.set_xlabel(r"Mass of the Star [$M_\odot$]")
ax.set_ylabel(r"Love Number [$k_2$]")
plt.plot(M_s,k2_s,'r--',label="SLy EoS")
plt.plot(M_f,k2_f,'b--',label="FPS EoS")
#plt.plot(M_a,k2_a,'g--',label="APR EoS")
plt.legend(loc="best")
plt.savefig("k2vM_TD.png")

plt.figure()
ax=plt.gca()
ax.set_title(r"NS Plot: Love Number vs Compactness")
ax.set_xlabel(r"Compactness Parameter [$M/R$]")
ax.set_ylabel(r"Love Number [$k_2$]")
plt.plot(C_s,k2_s,'r--',label="SLy EoS")
plt.plot(C_f,k2_f,'b--',label="FPS EoS")
#plt.plot(C_a,k2_a,'g--',label="APR EoS")
plt.legend(loc="best")
plt.savefig("k2vC_TD.png")

plt.figure()
ax=plt.gca()
ax.set_title(r"NS Plot: Tidal Deformability vs Mass")
ax.set_xlabel(r"Mass of the Star [$M_\odot$]")
ax.set_ylabel(r"Tidal Deformability [$g\,cm^2\,s^2$]")
ax.set_ylim(-0.8e35,2e36)
plt.plot(M_s,L_s,'r--',label="SLy EoS")
plt.plot(M_f,L_f,'b--',label="FPS EoS")
#plt.plot(M_a,L_a,'g--',label="APR EoS")
plt.legend(loc="best")
plt.savefig("LvM_TD.png")
