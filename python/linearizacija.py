import copy

import numpy as np
from collections import deque
import random
import dTwin
import math
import time
from dTwin import modelSolver
import plotTable

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

def mats(omega0, beta0, vw0):
    theta=9900000
    tbeta=0.5

    Aa=np.array([[dmtdw(omega0, vw0, beta0)/theta, dmtdbeta(omega0, vw0, beta0)/theta], [0, -1/tbeta]])

    B=np.array([[1/theta, 0], [0, 1/tbeta]])
    bp1=[]
    bp2=[]

    E=np.array([[dmtdvw(omega0, vw0, beta0)/theta, 0]])
    return Aa, B, E

matrice = np.empty((6, 6), dtype=object)
for i in range(6):
    for j in range(6):
        matrice[i, j]=mats(1.7+0.1*i, j*2, 11)

print(mats(1.9195, 2.7, 11))
