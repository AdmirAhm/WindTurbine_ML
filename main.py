import copy

import numpy as np
from collections import deque
import random
import dTwin
import math
import time
from dTwin import modelSolver
import plotTable

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
    if not p_model.initFromFile("turbina.dmodl"):
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
        param_names = dTwin.StringVector(1)
        param_indices = dTwin.UintVector(1)
        param_values = dTwin.DoubleVector(1)
        param_index = p_model.getParameterIndex("β_ref")
        
        param_names[0]="β_ref"
        param_indices[0]=param_index

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
        t_final=200
        w_rated=1.9195
        Kp=-400.2
        Ki=-100.1
        integrator=0
        while t<t_final:
            w_g=getValue(out_names, out_values, "ω_g")
            e=min(w_rated-w_g, 0)
            integrator=integrator+e*d_t

            param_values[0] = Kp*e+Ki*integrator
            param_values[0]=max(min(param_values[0], 90), 0)

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
fig1, ax1 = plotTable.plot_res("res.txt", ["ω_g", "β"])