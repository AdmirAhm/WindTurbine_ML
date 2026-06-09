#import numpy as np

def g(lambd, beta):
    return 1/(lambd-0.02*beta)-0.003/(beta**3+1)

def dgdlambda(lambd, beta):
    -1/(lambd-0.02*beta)**2

def dgdbeta(lambd, beta):
    exit()
    