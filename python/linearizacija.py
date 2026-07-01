import copy

import numpy as np
from collections import deque
import random
import dTwin
import math
import time
from dTwin import modelSolver
import plotTable
import matplotlib.pyplot as plt

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



# -------------------------------------------------------
# Operating point
# -------------------------------------------------------
omega0 = 1.9195
vw0 = 11
Rt = 40

lambda0 = Rt * omega0 / vw0
beta0 = 5   # linearization point (same for both plots)

Cp_lin = Cp_approx(lambda0, beta0)

# =======================================================
# PLOT 1: Cp vs lambda (beta fixed = 5)
# =======================================================
beta_fixed = 5

lambdas = np.linspace(2, 12, 400)

Cp_true_1 = Cp(lambdas, beta_fixed)
Cp_lin_1 = Cp_lin(lambdas, beta_fixed)

# =======================================================
# PLOT 2: Cp vs beta (lambda fixed = lambda0)
# =======================================================
betas = np.linspace(0, 10, 400)

Cp_true_2 = Cp(lambda0, betas)
Cp_lin_2 = Cp_lin(lambda0, betas)

# -------------------------------------------------------
# plotting
# -------------------------------------------------------
plt.figure(figsize=(12,5))

# ---- subplot 1 ----
plt.subplot(1,2,1)
plt.plot(lambdas, Cp_true_1, label=r"$C_p(\lambda, \beta=5)$")
plt.plot(lambdas, Cp_lin_1, "--", label="Linearized")
plt.scatter(lambda0, Cp(lambda0, beta_fixed), color="red")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$C_p$")
plt.title(r"$C_p$ vs $\lambda$ (fixed $\beta=5$)")
plt.grid()
plt.legend()

# ---- subplot 2 ----
plt.subplot(1,2,2)
plt.plot(betas, Cp_true_2, label=r"$C_p(\lambda_0, \beta)$")
plt.plot(betas, Cp_lin_2, "--", label="Linearized")
plt.scatter(beta0, Cp(lambda0, beta0), color="red")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$C_p$")
plt.title(r"$C_p$ vs $\beta$ (fixed $\lambda_0$)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()