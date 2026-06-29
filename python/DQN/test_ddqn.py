import numpy as np
import dTwin
from dTwin import modelSolver
import plotTable

from wind_turbine_env import WindTurbineEnv, getValue, OMEGA_REF
from ddqn_agent import DDQNAgent


MODEL_PATH = "../../modeli/turbina_vdc_w.dmodl"
AGENT_PATH = "ddqn_pitch_model.keras"


def load_trained_agent(state_dim, n_actions, path=AGENT_PATH):
    agent = DDQNAgent(state_dim=state_dim, n_actions=n_actions)
    agent.load(path)
    return agent


def greedy_action(agent, state):
    q_values = agent.online_net(state[None, :], training=False).numpy()[0]
    return int(np.argmax(q_values))




def test_steady(vw, n_steps=150, seed=0):
    env = WindTurbineEnv(model_path=MODEL_PATH, episode_max_steps=n_steps, seed=seed)
    agent = load_trained_agent(env.state_dim, env.n_actions)

    state = env.reset(vw=vw)
    log = {"t": [], "omega_g": [], "beta": [], "beta_ref": [], "vw": []}
    done = False
    t = 0.0

    while not done:
        action = greedy_action(agent, state)
        state, reward, done, info = env.step(action)
        log["t"].append(t)
        log["omega_g"].append(info["omega_g"])
        log["beta"].append(info["beta"])
        log["beta_ref"].append(info["beta_ref"])
        log["vw"].append(info["vw"])
        t += env.dt

    env.close()
    return log


def test_long_run(t_final=200.0, out_file="res_ddqn.txt"):
    p_log = dTwin.getConsoleLogger()
    p_model = dTwin.createRealDynamicModel(dTwin.DynamicProblem.DAE, p_log)
    if not p_model:
        print("ERROR! Cannot create model")
        return None
    if not p_model.initFromFile(MODEL_PATH):
        print("ERROR! Cannot init from file!")
        return None

    p_dyn_solver = p_model.getSolverInterface()
    if not p_dyn_solver:
        print("ERROR! Cannot obtain solver interface!")
        return None

    state_dim = 6  # v_w, omega_g, beta, + 3 one-hot last-action dims
    n_actions = 3
    agent = load_trained_agent(state_dim, n_actions)

    ACTION_STEP_DEG = 0.5
    BETA_MIN, BETA_MAX = 0.0, 90.0

    with open(out_file, "w", encoding="utf-8") as f_out:
        param_index = p_model.getParameterIndex("β_ref")
        param_names = dTwin.StringVector(1)
        param_indices = dTwin.UintVector(1)
        param_values = dTwin.DoubleVector(1)
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

        # write header in the same format as MPC.py's show_res_header
        f_out.write("t")
        for name in out_names:
            f_out.write(f" {name}")
        f_out.write("\n")

        d_t = p_dyn_solver.getStepSize()
        out_values = p_model.getOutputSymbolValues(out_indices)
        t = 0.0
        w_rated = OMEGA_REF

        beta_ref = 0.0
        last_action = 1  # "hold"

        while t < t_final:
            w_g = getValue(out_names, out_values, "ω_g")
            beta = getValue(out_names, out_values, "β")
            vw = getValue(out_names, out_values, "v_w")

            if w_g > w_rated:
                last_action_onehot = np.zeros(3, dtype=np.float32)
                last_action_onehot[last_action] = 1.0
                state = np.array(
                    [vw, w_g, beta, *last_action_onehot], dtype=np.float32)

                action = greedy_action(agent, state)
                proposed = beta_ref + (action - 1) * ACTION_STEP_DEG
                if BETA_MIN <= proposed <= BETA_MAX:
                    beta_ref = proposed
                # else: forbidden, beta_ref unchanged (action revoked)
                last_action = action
            else:
                beta_ref = 0.0
                last_action = 1

            param_values[0] = beta_ref
            p_model.setParameterValues(param_indices, param_values)

            sol = p_dyn_solver.step()
            if sol != dTwin.Solution.OK:
                print("ERROR! Cannot solve the problem!")
                return None

            out_values = p_model.getOutputSymbolValues(out_indices)

            f_out.write(f"{t}")
            for val in out_values:
                f_out.write(f" {val}")
            f_out.write("\n")

            t += d_t

    print(f"INFO! DDQN long-run test completed, written to {out_file}")
    return out_file


if __name__ == "__main__":
    out_file = test_long_run(t_final=200.0, out_file="res_ddqn.txt")
    if out_file:
        fig, ax = plotTable.plot_res(out_file, ["ω_g", "β"])
