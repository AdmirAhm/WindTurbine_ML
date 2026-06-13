import numpy as np

def g(lambd, beta):
    return 1/(lambd-0.02*beta)-0.003/(beta**3+1)

def dgdlambda(lambd, beta):
    return -1/(lambd-0.02*beta)**2

def dgdbeta(lambd, beta):
    return 0.02*1/(lambd-0.02*beta)**2-3*beta**2*0.003/(beta**3+1)**2

def A(lambd, beta):
    return 151*g(lambd, beta)-0.58*beta-0.002*beta**2.14-13.2

def Cp(lambd, beta):
    return 0.73*A(lambd, beta)*np.exp(-18.4*g(lambd, beta))

def dAdbeta(lambd, beta):
    return 151*dgdbeta(lambd, beta)-0.58-0.00428*beta**1.14

def dCpdlambda(lambd, beta):
    return 0.73*dgdlambda(lambd, beta)*np.exp(-18.4*g(lambd, beta))*(151-18.4*A(lambd, beta))

def dCpdbeta(lambd, beta):
    return 0.73*np.exp(-18.4*g(lambd, beta))*(dAdbeta(lambd, beta)-18.4*A(lambd, beta)*dgdbeta(lambd, beta))

def Cp_approx(lambd0, beta0):
    return lambda lambd, beta: Cp(lambd0, beta0)+dCpdlambda(lambd0, beta0)*(lambd-lambd0)+dCpdbeta(lambd0, beta0)*(beta-beta0)

def dCpdw(omega, vw, beta):
    rt=40
    return dCpdlambda(omega*rt/vw, beta)*rt/vw

def dCpdvw(omega, vw, beta):
    rt=40
    return dCpdlambda(omega*rt/vw, beta)*-omega*rt/vw**2

def mt(omega, vw, beta):
    rt=40
    return 0.5*1.293*np.pi*rt**2*vw**3*Cp(rt*omega/vw, beta)/omega

def dmtdCp(omega, vw, beta):
    rt=40
    return mt(omega, vw, beta)/Cp(omega*rt/vw, beta)

def dmtdw(omega, vw, beta):
    return dmtdCp(omega, vw, beta)*dCpdw(omega, vw, beta)-mt(omega, vw, beta)/omega

def dmtdbeta(omega, vw, beta):
    return dmtdCp(omega, vw, beta)*dCpdbeta(40*omega/vw, beta)

def dmtdvw(omega, vw, beta):
    return dmtdCp(omega, vw, beta)*dCpdvw(omega, vw, beta)+3*mt(omega, vw, beta)/vw

def mt_approx(omega0, vw0, beta0):
    return lambda omega, vw, beta: mt(omega0, vw0, beta0)+dmtdw(omega0, vw0, beta0)*(omega-omega0)+dmtdbeta(omega0, vw0, beta0)*(beta-beta0)+dmtdvw(omega0, vw0, beta0)*(vw-vw0)

omega0=1.9
beta0=2.7
vw0=11
theta=9900000
tbeta=0.5

Aa=[]
ap1=[]
ap2=[]

ap1.append(dmtdw(omega0, vw0, beta0)/theta)
ap1.append(dmtdbeta(omega0, vw0, beta0)/theta)

ap2.append(0)
ap2.append(-1/tbeta)

Aa.append(ap1)
Aa.append(ap2)

B=[]
bp1=[]
bp2=[]

bp1.append(1/theta)
bp1.append(0)
bp2.append(0)
bp2.append(1/tbeta)

B.append(bp1)
B.append(bp2)

E=dmtdvw(omega0, vw0, beta0)/theta

print(Aa)
print(B)
print(E)