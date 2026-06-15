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


p_model=None
out_indices=None
out_names=None
p_dyn_solver=None
p_log=None

def getValue(out_names, values, var):
    for i in range(len(out_names)):
        if str(out_names[i])==var:
            return values[i]
    return 0

def show_res_row(f_out, t: float, values: dTwin.DoubleVector):
    f_out.write(f"{t}")
    for val in values:
        f_out.write(f" {val}")
    f_out.write("\n")

def show_res_header(f_out, out_names: dTwin.StringVector, lbl: str = None):
    if lbl:
        f_out.write(lbl + "\n")
    f_out.write("t")
    for name in out_names:
        f_out.write(f" {name}")
    f_out.write("\n")

def startDynamicModel():
    global p_model
    global p_dyn_solver
    p_log=dTwin.getConsoleLogger()
    p_model=dTwin.createRealDynamicModel(dTwin.DynamicProblem.DAE, p_log)
    if not p_model:
        print("ERROR! Cannot create model")
        return False
    if not p_model.initFromFile("lin_model.dmodl"):
        print("ERROR! Cannot init from file!")
        return False
    
    p_dyn_solver=p_model.getSolverInterface()
    if not p_dyn_solver:
        print("ERROR! Cannot obtain solver interface!")
        return False
    return True

def runDynamicModel():
    global p_model
    global p_dyn_solver
    global p_log

    with open("res.txt", 'w', encoding='utf-8') as f_out:
        param_index = -1
        param_names = dTwin.StringVector(12)
        param_indices = dTwin.UintVector(12)
        param_values = dTwin.DoubleVector(12)
        
        param_names[0]="A00"
        param_index = p_model.getParameterIndex("A00")
        param_indices[0]=param_index
        param_names[1]="A01"
        param_index = p_model.getParameterIndex("A01")
        param_indices[1]=param_index
        param_names[2]="A10"
        param_index = p_model.getParameterIndex("A10")
        param_indices[2]=param_index
        param_names[3]="A11"
        param_index = p_model.getParameterIndex("A11")
        param_indices[3]=param_index
        param_names[4]="B00"
        param_index = p_model.getParameterIndex("B00")
        param_indices[4]=param_index
        param_names[5]="B01"
        param_index = p_model.getParameterIndex("B01")
        param_indices[5]=param_index
        param_names[6]="B10"
        param_index = p_model.getParameterIndex("B10")
        param_indices[6]=param_index
        param_names[7]="B11"
        param_index = p_model.getParameterIndex("B11")
        param_indices[7]=param_index
        param_names[8]="E00"
        param_index = p_model.getParameterIndex("E00")
        param_indices[8]=param_index
        param_names[9]="E01"
        param_index = p_model.getParameterIndex("E01")
        param_indices[9]=param_index
        param_names[10]="x10"
        param_index = p_model.getParameterIndex("x10")
        param_indices[10]=param_index
        param_names[11]="x20"
        param_index = p_model.getParameterIndex("x20")
        param_indices[11]=param_index

        if not p_dyn_solver.reset(0):
            print("ERROR! Cannot reset the problem")
            return None
        
        out_indices=p_model.getOutputSymbolIndices()
        if len(out_indices) == 0:
            print("ERROR! Cannot obtain output indices!")
            return None
        
        out_names=p_model.getOutputSymbolNames(out_indices)
        if len(out_names) == 0:
            print("ERROR! Cannot obtain output names!")
            return None
        
        show_res_header(f_out,out_names)
        d_t=p_dyn_solver.getStepSize()
        out_values=p_model.getOutputSymbolValues(out_indices)
        t=0.0
        eps_t=1e-6
        t_final=40
        integrator=0
        while t<t_final:
            w_g=getValue(out_names, out_values, "y1")
            beta=getValue(out_names, out_values, "y2")
            
            wgind=max(min(round((w_g-1.7)*10), 5), 0)
            betaind=max(min(round(beta/2), 5), 0)
            
            matA, matB, matE=matrice[wgind, betaind]

            param_values[0] = matA[0, 0]
            param_values[1] = matA[0, 1]
            param_values[2] = matA[1, 0]
            param_values[3] = matA[1, 1]
 
            param_values[4] = matB[0, 0]
            param_values[5] = matB[0, 1]
            param_values[6] = matB[1, 0]
            param_values[7] = matB[1, 1]

            param_values[8] = matE[0, 0]
            param_values[9] = matE[0, 1]

            param_values[10] = 1.7+0.1*wgind
            param_values[11] = 2*betaind

            p_model.setParameterValues(param_indices, param_values)
            
            sol=p_dyn_solver.step()
            if sol!=dTwin.Solution.OK:
                print("ERROR! Cannot solve the problem!")
                return None
            out_values=p_model.getOutputSymbolValues(out_indices)
            show_res_row(f_out, t, out_values)
            t+=d_t
        print("INFO! Dynamic test completed succesfully!")

startDynamicModel()
runDynamicModel()
fig1, ax1 = plotTable.plot_res("res.txt", ["y1", "y2"])