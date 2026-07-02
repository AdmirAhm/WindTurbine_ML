import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.optimize import minimize
from collections import deque
import dTwin
import time
from dTwin import modelSolver
import plotTable

# ==========================================
# NMPC CONFIGURATION
# ==========================================
# Horizon length matches MPC.py's linear MPC (N=5) so the two controllers
# are directly comparable in the thesis benchmark.
N = 5

# --- LSTM surrogate model ---------------------------------------------
# The saved .keras model uses a Lambda layer (unsafe to deserialize by
# default because it's a raw Python lambda) to slice out the beta_ref
# branch, so unsafe deserialization must be explicitly enabled.
keras.config.enable_unsafe_deserialization()

MODEL_PATH = "keras_models/wind_turbine_lstm_best_model.keras"
SCALER_PATH = "keras_models/scalers_best.npz"

model = keras.models.load_model(MODEL_PATH, compile=False)

_scalers = np.load(SCALER_PATH)
x_mean = tf.constant(_scalers["x_mean"], dtype=tf.float32)   # [v_w, beta_ref] means
x_std = tf.constant(_scalers["x_std"], dtype=tf.float32)
y_mean = tf.constant(_scalers["y_mean"], dtype=tf.float32)   # [beta_next, omega_g_next] means
y_std = tf.constant(_scalers["y_std"], dtype=tf.float32)

# The model's input layer is fixed at (None, 200, 2) — this window length
# is baked into the trained weights, so it CANNOT be shortened without
# retraining. (The 50-100 step window recommendation from the NMPC-speed
# investigation applies to a future retrained surrogate, not this one.)
WINDOW = model.input_shape[1]
assert WINDOW is not None, "Model input window must be static for NMPC rollout."

# --- Cost weights (mirrors MPC.py's Q / R so results are comparable) ---
omega_ref = 1.9195
Q_OMEGA = 30000.0   # penalty on (omega_g_pred - omega_ref)^2, per horizon step
Q_BETA = 0.1        # small penalty on beta_pred^2 (mirrors P/Q beta weight)
R_U = 0.1           # penalty on beta_ref^2 (control effort)

BETA_MIN, BETA_MAX = 0.0, 90.0

# ==========================================
# DIFFERENTIABLE BATCHED ROLLOUT
# ==========================================
# For each of the N horizon steps we build a full 200-step window by
# sliding across [real history ++ candidate future beta_ref sequence].
# Because the surrogate's inputs are only (v_w, beta_ref) -- it does NOT
# require omega_g/beta feedback in its input -- all N windows can be
# built directly from known/assumed data and evaluated in ONE batched
# model call per optimizer iteration, instead of N sequential calls.
# This is the main mitigation for the "500-step history replay per
# evaluation" bottleneck noted during the NMPC-integration investigation.
_idx = tf.constant(
    np.array([np.arange(j, j + WINDOW) for j in range(1, N + 1)]),
    dtype=tf.int32,
)  # shape (N, WINDOW)


@tf.function(input_signature=[
    tf.TensorSpec(shape=[N], dtype=tf.float32),        # u: candidate beta_ref sequence
    tf.TensorSpec(shape=[], dtype=tf.float32),          # vw_now: persisted wind forecast
    tf.TensorSpec(shape=[WINDOW], dtype=tf.float32),    # H_vw: real v_w history
    tf.TensorSpec(shape=[WINDOW], dtype=tf.float32),    # H_br: real beta_ref history
])
def _rollout(u, vw_now, H_vw, H_br):
    base_vw = tf.concat([H_vw, tf.fill([N], vw_now)], axis=0)  # persistence forecast
    base_br = tf.concat([H_br, u], axis=0)

    vw_windows = tf.gather(base_vw, _idx)   # (N, WINDOW)
    br_windows = tf.gather(base_br, _idx)   # (N, WINDOW)
    windows = tf.stack([vw_windows, br_windows], axis=-1)  # (N, WINDOW, 2)

    windows_scaled = (windows - x_mean) / x_std
    preds_scaled = model(windows_scaled, training=False)   # (N, 2)
    preds = preds_scaled * y_std + y_mean

    beta_pred = preds[:, 0]
    omega_pred = preds[:, 1]
    return beta_pred, omega_pred


@tf.function(input_signature=[
    tf.TensorSpec(shape=[N], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[WINDOW], dtype=tf.float32),
    tf.TensorSpec(shape=[WINDOW], dtype=tf.float32),
])
def _cost_and_grad(u, vw_now, H_vw, H_br):
    with tf.GradientTape() as tape:
        tape.watch(u)
        beta_pred, omega_pred = _rollout(u, vw_now, H_vw, H_br)
        cost = (
            tf.reduce_sum(Q_OMEGA * tf.square(omega_pred - omega_ref))
            + tf.reduce_sum(Q_BETA * tf.square(beta_pred))
            + tf.reduce_sum(R_U * tf.square(u))
        )
    grad = tape.gradient(cost, u)
    return cost, grad


# ==========================================
# NMPC FUNCTION
# ==========================================
_u_warm_start = np.zeros(N, dtype=np.float64)  # receding-horizon warm start


def nmpc_pitch(vw_now, H_vw, H_br):
    """
    vw_now : float, current measured wind speed (persisted as forecast over horizon)
    H_vw   : deque/array of the last WINDOW measured v_w values (oldest -> newest)
    H_br   : deque/array of the last WINDOW *applied* beta_ref values (oldest -> newest)
    Returns the first move of the optimized beta_ref sequence, like MPC.py's U[0].
    """
    global _u_warm_start

    H_vw_t = tf.constant(np.asarray(H_vw, dtype=np.float32))
    H_br_t = tf.constant(np.asarray(H_br, dtype=np.float32))
    vw_t = tf.constant(float(vw_now), dtype=tf.float32)

    def scipy_cost(u_np):
        u = tf.constant(u_np, dtype=tf.float32)
        cost, grad = _cost_and_grad(u, vw_t, H_vw_t, H_br_t)
        return float(cost.numpy()), grad.numpy().astype(np.float64)

    result = minimize(
        scipy_cost,
        x0=_u_warm_start,
        jac=True,
        method="SLSQP",
        bounds=[(BETA_MIN, BETA_MAX)] * N,
        options={"maxiter": 30, "ftol": 1e-1},
    )

    u_opt = np.clip(result.x, BETA_MIN, BETA_MAX)

    # shift-and-hold warm start for the next call (receding horizon)
    _u_warm_start = np.concatenate([u_opt[1:], u_opt[-1:]])

    return float(u_opt[0])


# ==========================================
# SIMULATION  (same dTwin integration as MPC.py)
# ==========================================

p_model = None
out_indices = None
out_names = None
p_dyn_solver = None
p_log = None


def getValue(out_names, values, var):
    for i in range(len(out_names)):
        if str(out_names[i]) == var:
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
    p_log = dTwin.getConsoleLogger()
    p_model = dTwin.createRealDynamicModel(dTwin.DynamicProblem.DAE, p_log)
    if not p_model:
        print("ERROR! Cannot create model")
        return False
    if not p_model.initFromFile("turbina_vdc_w.dmodl"):
        print("ERROR! Cannot init from file!")
        return False

    p_dyn_solver = p_model.getSolverInterface()
    if not p_dyn_solver:
        print("ERROR! Cannot obtain solver interface!")
        return False
    return True


def runDynamicModel():
    global p_model
    global p_dyn_solver
    global p_log

    with open("res_nmpc.txt", 'w', encoding='utf-8') as f_out:
        param_index = -1
        param_names = dTwin.StringVector(1)
        param_indices = dTwin.UintVector(1)
        param_values = dTwin.DoubleVector(1)
        param_index = p_model.getParameterIndex("β_ref")

        param_names[0] = "β_ref"
        param_indices[0] = param_index

        if not p_dyn_solver.reset(0):
            print("ERROR! Cannot reset the problem")
            return None

        out_indices = p_model.getOutputSymbolIndices()
        if len(out_indices) == 0:
            print("ERROR! Cannot obtain output indices!")
            return None

        out_names = p_model.getOutputSymbolNames(out_indices)
        if len(out_names) == 0:
            print("ERROR! Cannot obtain output names!")
            return None

        show_res_header(f_out, out_names)
        d_t = p_dyn_solver.getStepSize()
        out_values = p_model.getOutputSymbolValues(out_indices)
        t = 0.0
        t_final = 80
        w_rated = 1.9195

        # rolling history buffers for the NMPC surrogate, oldest -> newest.
        # Primed with the first measured sample / beta_ref=0 so the model
        # always sees a full WINDOW-length sequence, even before 200 real
        # samples have been collected. Expect a brief warm-up transient.
        vw0 = getValue(out_names, out_values, "v_w")
        H_vw = deque([vw0] * WINDOW, maxlen=WINDOW)
        H_br = deque([0.0] * WINDOW, maxlen=WINDOW)

        while t < t_final:
            print(t)
            w_g = getValue(out_names, out_values, "ω_g")
            vw = getValue(out_names, out_values, "v_w")

            if w_g > w_rated:
                param_values[0] = nmpc_pitch(vw_now=vw, H_vw=H_vw, H_br=H_br)
            else:
                param_values[0] = 0.0

            # record the *applied* beta_ref, not the unconstrained optimizer
            # output, so the history buffer matches what the plant actually saw
            H_vw.append(vw)
            H_br.append(param_values[0])

            p_model.setParameterValues(param_indices, param_values)

            sol = p_dyn_solver.step()
            if sol != dTwin.Solution.OK:
                print("ERROR! Cannot solve the problem!")
                return None
            out_values = p_model.getOutputSymbolValues(out_indices)
            show_res_row(f_out, t, out_values)
            t += d_t
        print("INFO! NMPC dynamic test completed succesfully!")


startDynamicModel()
runDynamicModel()
fig1, ax1 = plotTable.plot_res("res_nmpc.txt", ["ω_g", "β"])
