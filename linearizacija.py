import numpy as np

def g(lambd, beta):
    return 1/(lambd-0.02*beta)-0.003/(beta**3+1)

def dgdlambda(lambd, beta):
    return -1/(lambd-0.02*beta)**2

def dgdbeta(lambd, beta):
    return -0.02*1/(lambd-0.02*beta)**2+3*beta**2*0.003/(beta**3+1)**2

def A(lambd, beta):
    return 151*g(lambd, beta)-0.58*beta-0.02*beta**2.14-13.2

def Cp(lambd, beta):
    return 0.73*A(lambd, beta)*np.exp(-18.4*g(lambd, beta))

def dAdbeta(lambd, beta):
    return 151*dgdbeta(lambd, beta)-0.58-0.0424*beta**1.14

def dCpdlambda(lambd, beta):
    return -0.73*dgdlambda(lambd, beta)*np.exp(-18.4*g(lambd, beta))*(151-18.4*A(lambd, beta))

def dCpdbeta(lambd, beta):
    return 0.73*np.exp(-18.4*g(lambd, beta))*(dAdbeta(lambd, beta)-18.4*A(lambd, beta)*dgdbeta(lambd, beta))

def Cp_approx(lambd0, beta0):
    return lambda lambd, beta: Cp(lambd0, beta0)+dCpdlambda(lambd0, beta0)*(lambd-lambd0)+dCpdbeta(lambd0, beta0)*(beta-beta0)

print(Cp(6.1, 7))
print(Cp_approx(6.1, 5)(6.1, 7))    