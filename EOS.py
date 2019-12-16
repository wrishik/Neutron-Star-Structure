import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

sly=np.genfromtxt("SLy.txt",delimiter="  ")
fps=np.genfromtxt("FPS.txt",delimiter="    ")

nbs=sly[:,1]
rhos=sly[:,2]
Ps=sly[:,3]

nbf=fps[:,1]
rhof=fps[:,2]
Pf=fps[:,3]

cPs=CubicSpline(rhos,Ps)
rmax=rhos[len(rhos)-1]
rmin=rhos[0]
rPs=np.linspace(rmin,rmax,1000)

fig=plt.figure()
ax=plt.gca()
ax.set_title(r"SLy Equation of State [$P(\rho)$]")
ax.set_xlabel(r"$\rho\: (g/cm^3)$")
ax.set_ylabel(r"$P\: (dyn/cm^2)$")
plt.plot(rhos,Ps,'k.',label="Data Points")
plt.plot(rPs,cPs(rPs),'k-',label="Cubic Spline Interpolation")
plt.legend(loc="best")
plt.savefig("SLy_P.png")
plt.close()

crs=CubicSpline(Ps,rhos,bc_type=((2,0.),(1,9.11937e-22)))
Pmax=Ps[len(Ps)-1]
Pmin=Ps[0]
Prs=np.linspace(Pmin,Pmax,1000)

fig=plt.figure()
ax=plt.gca()
ax.set_title(r"SLy Equation of State [$\rho(P)$]")
ax.set_ylabel(r"$\rho\: (g/cm^3)$")
ax.set_xlabel(r"$P\: (dyn/cm^2)$")
plt.plot(Ps,rhos,'k.',label="Data Points")
plt.plot(Prs,crs(Prs),'k-',label="Cubic Spline Interpolation")
plt.legend(loc="best")
plt.savefig("SLy_R.png")
plt.close()

cPf=CubicSpline(rhof,Pf)
rmax=rhof[len(rhof)-1]
rmin=rhof[0]
rPf=np.linspace(rmin,rmax,1000)

fig=plt.figure()
ax=plt.gca()
ax.set_title(r"FPS Equation of State [$P(\rho)$]")
ax.set_xlabel(r"$\rho\: (g/cm^3)$")
ax.set_ylabel(r"$P\: (dyn/cm^2)$")
plt.plot(rhof,Pf,'k.',label="Data Points")
plt.plot(rPf,cPf(rPf),'k-',label="Cubic Spline Interpolation")
plt.legend(loc="best")
plt.savefig("FPS_P.png")
plt.close()

crf=CubicSpline(Pf,rhof)
Pmax=Pf[len(Pf)-1]
Pmin=Pf[0]
Prf=np.linspace(Pmin,Pmax,1000)

fig=plt.figure()
ax=plt.gca()
ax.set_title(r"FPS Equation of State [$\rho(P)$]")
ax.set_ylabel(r"$\rho\: (g/cm^3)$")
ax.set_xlabel(r"$P\: (dyn/cm^2)$")
plt.plot(Pf,rhof,'k.',label="Data Points")
plt.plot(Prf,crf(Prf),'k-',label="Cubic Spline Interpolation")
plt.legend(loc="best")
plt.savefig("FPS_R.png")
plt.close()

