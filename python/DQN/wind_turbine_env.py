import numpy as np
import dTwin
from dTwin import modelSolver


# ============================================================
# Environment configuration
# ============================================================

OMEGA_REF = 1.9195          # rated generator speed, same as MPC.py
BETA_MIN = 0.0               # deg, same bounds as MPC.py's pitch constraint
BETA_MAX = 90.0              # deg
ACTION_STEP_DEG = 0.5         # deg change in beta_ref per discrete action


EPISODE_MAX_STEPS = 150      # mirrors paper's "short episodes of up to
                              # 150 time steps" during training
FORBIDDEN_PENALTY = -2.0     # mirrors paper's r=-2 for leaving state space
BONUS_REWARD = 5.0           # mirrors paper's +5 "win" bonus
BONUS_WINDOW = 20            # steps of sustained near-optimal control
BONUS_THRESHOLD = 0.975      # fraction of "perfect" regulation required

OMEGA_ERR_NORM = 0.3        
VW_TRAIN_MIN = 12.0
VW_TRAIN_MAX = 20.0


def getValue(out_names, values, var):
    for i in range(len(out_names)):
        if str(out_names[i]) == var:
            return values[i]
    return 0


class WindTurbineEnv:


    def __init__(self, model_path="../modeli/turbina_vdc_w.dmodl",
                 dt=None, vw_min=VW_TRAIN_MIN, vw_max=VW_TRAIN_MAX,
                 episode_max_steps=EPISODE_MAX_STEPS, seed=None):
        self.model_path = model_path
        self.vw_min = vw_min
        self.vw_max = vw_max
        self.episode_max_steps = episode_max_steps
        self.rng = np.random.default_rng(seed)

        self.p_model = None
        self.p_dyn_solver = None
        self.out_indices = None
        self.out_names = None
        self.param_index_beta_ref = None
        self.param_index_vw = None  # only used if the .dmodl exposes v_w
                                     # as a settable parameter for steady-
                                     # wind training; see note in reset()

        self._build_model()

        self.dt = dt if dt is not None else self.p_dyn_solver.getStepSize()

        self.t = 0.0
        self.step_count = 0
        self.beta_ref = 0.0
        self.last_action = 1  # "hold" as the neutral initial action
        self.vw_current = None
        self._recent_rewards = []  # for the BONUS_WINDOW streak check

    # --------------------------------------------------------
    # Model construction (mirrors MPC.py's startDynamicModel)
    # --------------------------------------------------------
    def _build_model(self):
        p_log = dTwin.getConsoleLogger()
        self.p_model = dTwin.createRealDynamicModel(dTwin.DynamicProblem.DAE, p_log)
        if not self.p_model:
            raise RuntimeError("Cannot create dTwin model")
        if not self.p_model.initFromFile(self.model_path):
            raise RuntimeError(f"Cannot init dTwin model from {self.model_path}")

        self.p_dyn_solver = self.p_model.getSolverInterface()
        if not self.p_dyn_solver:
            raise RuntimeError("Cannot obtain dTwin solver interface")

        self.param_index_beta_ref = self.p_model.getParameterIndex("β_ref")

        try:
            self.param_index_vw = self.p_model.getParameterIndex("v_w")
        except Exception:
            self.param_index_vw = None

        self.out_indices = self.p_model.getOutputSymbolIndices()
        if len(self.out_indices) == 0:
            raise RuntimeError("Cannot obtain dTwin output indices")
        self.out_names = self.p_model.getOutputSymbolNames(self.out_indices)
        if len(self.out_names) == 0:
            raise RuntimeError("Cannot obtain dTwin output names")

    # --------------------------------------------------------
    # Gym-style API
    # --------------------------------------------------------
    def reset(self, vw=None):
        """
        Resets the dTwin solver and starts a new episode with a randomized
        steady wind speed (unless vw is given explicitly).
        """
        if False:
            if not self.p_dyn_solver.reset(0):
                raise RuntimeError("Cannot reset dTwin solver")

            self.t = 0.0
        self.step_count = 0
        self.beta_ref = 0.0
        self.last_action = 1
        self._recent_rewards = []

        self.vw_current = vw if vw is not None else self.rng.uniform(self.vw_min, self.vw_max)

        # If the model exposes a settable v_w parameter, pin it for this
        # episode (steady wind, matching the paper's training protocol).
        # If your .dmodl only supports an internally-driven wind profile,
        # remove this block and instead select episodes by *time window*
        # (e.g. reset to different t0 offsets) -- flag this for follow-up
        # if param_index_vw stays None on your model.
        if self.param_index_vw is not None:
            names = dTwin.StringVector(1)
            indices = dTwin.UintVector(1)
            values = dTwin.DoubleVector(1)
            names[0] = "v_w"
            indices[0] = self.param_index_vw
            values[0] = self.vw_current
            self.p_model.setParameterValues(indices, values)

        # Apply initial beta_ref = 0 and take one step to populate outputs
        self._apply_beta_ref(self.beta_ref)
        sol = self.p_dyn_solver.step()
        if sol != dTwin.Solution.OK:
            raise RuntimeError("dTwin solver failed on initial step")

        out_values = self.p_model.getOutputSymbolValues(self.out_indices)
        omega_g = getValue(self.out_names, out_values, "ω_g")
        beta = getValue(self.out_names, out_values, "β")
        vw_actual = getValue(self.out_names, out_values, "v_w")
        if self.param_index_vw is None:
            # model-driven wind: read back actual v_w so the agent's state
            # is consistent with what the simulator is actually doing
            self.vw_current = vw_actual

        self.t += self.dt
        self.step_count = 1

        return self._make_state(omega_g, beta)

    def step(self, action):
        """
        action in {0, 1, 2} -> decrease / hold / increase beta_ref.
        Returns (next_state, reward, done, info).
        """
        assert action in (0, 1, 2)

        proposed_beta_ref = self.beta_ref + (action - 1) * ACTION_STEP_DEG
        forbidden = not (BETA_MIN <= proposed_beta_ref <= BETA_MAX)

        if forbidden:
            # Revert: s_{t+1} = s_t, i.e. beta_ref unchanged, but we still
            # must step the simulator forward in time (the turbine keeps
            # running with the old beta_ref) -- mirrors the paper's
            # "action revoked" handling.
            applied_beta_ref = self.beta_ref
        else:
            applied_beta_ref = float(np.clip(proposed_beta_ref, BETA_MIN, BETA_MAX))

        self._apply_beta_ref(applied_beta_ref)
        sol = self.p_dyn_solver.step()
        if sol != dTwin.Solution.OK:
            # Treat solver failure as episode-ending with a strong penalty
            # rather than crashing training.
            state = self._make_state(self.last_omega_g, self.last_beta)
            return state, FORBIDDEN_PENALTY * 5.0, True, {"solver_failed": True}

        out_values = self.p_model.getOutputSymbolValues(self.out_indices)
        omega_g = getValue(self.out_names, out_values, "ω_g")
        beta = getValue(self.out_names, out_values, "β")

        self.beta_ref = applied_beta_ref
        self.last_action = action
        self.last_omega_g = omega_g
        self.last_beta = beta
        self.t += self.dt
        self.step_count += 1

        reward = self._compute_reward(omega_g, forbidden)

        done = self.step_count >= self.episode_max_steps

        next_state = self._make_state(omega_g, beta)
        info = {"forbidden": forbidden, "omega_g": omega_g, "beta": beta,
                 "beta_ref": applied_beta_ref, "vw": self.vw_current}

        return next_state, reward, done, info

    # --------------------------------------------------------
    # Internals
    # --------------------------------------------------------
    def _apply_beta_ref(self, beta_ref_value):
        names = dTwin.StringVector(1)
        indices = dTwin.UintVector(1)
        values = dTwin.DoubleVector(1)
        names[0] = "β_ref"
        indices[0] = self.param_index_beta_ref
        values[0] = beta_ref_value
        self.p_model.setParameterValues(indices, values)

    def _compute_reward(self, omega_g, forbidden):
        if forbidden:
            return FORBIDDEN_PENALTY

        err = (omega_g - OMEGA_REF) / OMEGA_ERR_NORM
        base_reward = float(np.clip(1.0 - err ** 2, 0.0, 1.0))

        self._recent_rewards.append(base_reward)
        if len(self._recent_rewards) > BONUS_WINDOW:
            self._recent_rewards.pop(0)

        bonus = 0.0
        if (len(self._recent_rewards) == BONUS_WINDOW
                and min(self._recent_rewards) >= BONUS_THRESHOLD):
            bonus = BONUS_REWARD

        return base_reward + bonus

    def _make_state(self, omega_g, beta):
        last_action_onehot = np.zeros(3, dtype=np.float32)
        last_action_onehot[self.last_action] = 1.0
        return np.array(
            [self.vw_current, omega_g, beta, *last_action_onehot],
            dtype=np.float32,
        )

    @property
    def state_dim(self):
        return 6  # v_w, omega_g, beta, + 3 one-hot last-action dims

    @property
    def n_actions(self):
        return 3

    def close(self):
        # dTwin model cleanup, if your bindings require explicit teardown,
        # would go here. Left as a no-op since MPC.py does not call any
        # explicit teardown either.
        pass
