import numpy as np
import scipy.sparse as sp
import osqp
import copy
from collections import deque
import random
import dTwin
import math
import time
from dTwin import modelSolver
import plotTable


N = 5

A = np.array([[-0.06234481, -0.0068683],[ 0., -2.]])

B = np.array([
    [0],
    [2]
])


E = np.array([
    [0.02575457, -1e-7],
    [0.0,   0.0]
])
omega_ref = 1.9195


Q = np.diag([
    30000.0,  # omega_g
    0.1      # beta
])

P = Q

R = np.array([[0.1]])

nx = 2
nu = 1

Qbar = sp.block_diag(
    [Q]*N + [P],
    format='csc'
)

Rbar = sp.block_diag(
    [R]*N,
    format='csc'
)

H = sp.block_diag(
    [Qbar, Rbar],
    format='csc'
)

# ==========================================
# MPC FUNCTION
# ==========================================

def mpc_pitch(x0, vw, Tg):

    if x0[0] < omega_ref:
        omega_target = x0[0]
    else:
        omega_target = omega_ref

    xref = np.array([
        omega_ref,
        0.0
    ])

    q = np.concatenate([
        -Qbar @ np.tile(xref, N+1),
        np.zeros(N*nu)
    ])

    nvars = (N+1)*nx + N*nu

    rows = []
    rhs = []

    # initial state
    A0 = np.zeros((nx, nvars))
    A0[:, :nx] = np.eye(nx)

    rows.append(A0)
    rhs.append(x0)

    # predicted disturbances
    d = np.array([vw, Tg])

    for k in range(N):

        row = np.zeros((nx, nvars))

        xk = k*nx
        xkp1 = (k+1)*nx

        uk = (N+1)*nx + k*nu

        row[:, xkp1:xkp1+nx] = np.eye(nx)
        row[:, xk:xk+nx] = -A
        row[:, uk:uk+nu] = -B

        rows.append(row)

        rhs.append(E @ d)

    Aeq = np.vstack(rows)
    beq = np.concatenate(rhs)

    # ==========================
    # pitch constraints
    # ==========================

    Aineq = np.zeros((N, nvars))

    for k in range(N):

        col = (N+1)*nx + k

        Aineq[k, col] = 1.0

    beta_min = 0.0
    beta_max = 90.0

    lineq = np.full(N, beta_min)
    uineq = np.full(N, beta_max)

    Aqp = sp.csc_matrix(
        np.vstack([Aeq, Aineq])
    )

    l = np.concatenate([
        beq,
        lineq
    ])

    u = np.concatenate([
        beq,
        uineq
    ])

    prob = osqp.OSQP()

    prob.setup(
        P=H,
        q=q,
        A=Aqp,
        l=l,
        u=u,
        verbose=False
    )

    result = prob.solve()

    z = result.x

    U = z[(N+1)*nx:]

    beta_ref = U[0]

    return beta_ref

# ==========================================
# SIMULATION
# ==========================================


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
    if not p_model.initFromFile("turbina_vdc_w.dmodl"):
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
        x = np.array([
            1.5, 
            0.0   
        ])
        prosli=0
        while t<t_final:
            w_g=getValue(out_names, out_values, "ω_g")
            Tg=getValue(out_names, out_values, "m_m_ref")
            vw=getValue(out_names, out_values, "v_w")
            if w_g>w_rated:
                param_values[0] = mpc_pitch(
                    x0=x,
                    vw=vw,
                    Tg=Tg
                )
            else:
                param_values[0]=0
            prosli=param_values[0]
            p_model.setParameterValues(param_indices, param_values)
            
            sol=p_dyn_solver.step()
            if sol!=dTwin.Solution.OK:
                print("ERROR! Cannot solve the problem!")
                return None
            out_values=p_model.getOutputSymbolValues(out_indices)
            x[0]=getValue(out_names, out_values, "ω_g")
            x[1]=getValue(out_names, out_values, "β")
            show_res_row(f_out, t, out_values)
            t+=d_t
        print("INFO! Dynamic test completed succesfully!")

startDynamicModel()
runDynamicModel()
fig1, ax1 = plotTable.plot_res("res.txt", ["ω_g", "β"])
