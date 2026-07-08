import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# User settings
# ==========================================================

filename = "res_lin.txt"      # your output file
omega_ref = 1.9195           # <-- desired reference speed [rad/s]
regime_start = 40.0       # third regime starts here

# --- needed for tip-speed ratio / Cp calculation -----------
R = 63.0            # rotor radius [m]        <-- set to your turbine's value
gear_ratio = 1.0     # N_gear = omega_rotor / omega_gen (set 1.0 if omega already rotor-side)

def g(lambd, beta):
    return 1/(lambd-0.02*beta)-0.003/(beta**3+1)

def A(lambd, beta):
    return 151*g(lambd, beta)-0.58*beta-0.002*beta**2.14-13.2

def Cp(lambd, beta):
    return 0.73*A(lambd, beta)*np.exp(-18.4*g(lambd, beta))


df = pd.read_csv(filename, delim_whitespace=True)

t = df["t"].to_numpy()
omega = df["ω_g"].to_numpy()
beta = df["β"].to_numpy()        # change to "beta" if needed
v_w = df["v_w"].to_numpy()

# ==========================================================
# Select third operating regime
# ==========================================================

mask = t >= regime_start

t3 = t[mask]
omega3 = omega[mask]
beta3 = beta[mask]
v_w3 = v_w[mask]

# ==========================================================
# Time step
# ==========================================================

dt = np.mean(np.diff(t3))

# ==========================================================
# Tracking error
# ==========================================================

error = omega_ref - omega3

# ==========================================================
# IAE
# ==========================================================

IAE = np.trapz(np.abs(error), t3)

# ==========================================================
# ISE
# ==========================================================

ISE = np.trapz(error**2, t3)

# ==========================================================
# MAE
# ==========================================================

MAE = np.mean(np.abs(error))

# ==========================================================
# RMSE
# ==========================================================

RMSE = np.sqrt(np.mean(error**2))

# ==========================================================
# Settling time
#
# First instant after 40 s where omega >= 95% omega_ref
# ==========================================================

threshold = 0.95 * omega_ref

idx = np.where(omega3 >= threshold)[0]

if len(idx) == 0:
    settling_time = np.nan
else:
    settling_time = t3[idx[0]] - regime_start

# ==========================================================
# Overshoot
#
# Maximum amount above reference
# ==========================================================

overshoot = max(0.0, np.max(omega3) - omega_ref)

overshoot_percent = (
    100 * overshoot / omega_ref
    if omega_ref != 0
    else np.nan
)

# ==========================================================
# Maximum omega_t (rotor/generator speed) deviation
#
# NOTE: only omega_g is present in the log, so this is the
# max |omega_g - omega_ref| over the regime. If you log a
# separate turbine/rotor speed column, swap it in here.
# ==========================================================

max_omega_deviation = np.max(np.abs(error))

# ==========================================================
# Mean pitch angle
# ==========================================================

mean_pitch = np.mean(beta3)

# ==========================================================
# Pitch variation
#
# Standard deviation of pitch angle
# ==========================================================

pitch_variation_std = np.std(beta3)

# Peak-to-peak variation (max-min)
pitch_variation_range = np.max(beta3) - np.min(beta3)

# ==========================================================
# Beta traveled
#
# Total path length covered by the pitch angle signal
# (sum of |delta_beta| over every sample-to-sample step)
# ==========================================================

beta_traveled = np.sum(np.abs(np.diff(beta3)))

# ==========================================================
# Speed of beta change
#
# dBeta/dt at every step, then report mean |rate| and max |rate|
# ==========================================================

beta_rate = np.diff(beta3) / np.diff(t3)

mean_beta_rate = np.mean(np.abs(beta_rate))
max_beta_rate = np.max(np.abs(beta_rate))

# ==========================================================
# Cp over time
#
# lambda = omega_rotor * R / v_w
# omega_rotor = omega_gen / gear_ratio
# ==========================================================

omega_rotor3 = omega3 / gear_ratio
lambda3 = omega_rotor3 * R / v_w3

Cp3 = np.array([Cp(l, b) for l, b in zip(lambda3, beta3)])

# ==========================================================
# Print results
# ==========================================================

print("Performance metrics")
print("------------------------------")
print(f"IAE                    : {IAE:.6f}")
print(f"ISE                    : {ISE:.6f}")
print(f"MAE                    : {MAE:.6f}")
print(f"RMSE                   : {RMSE:.6f}")
print(f"Settling time          : {settling_time:.3f} s")
print(f"Overshoot              : {overshoot:.6f} rad/s")
print(f"Overshoot (%)          : {overshoot_percent:.2f} %")
print(f"Max omega deviation    : {max_omega_deviation:.6f} rad/s")
print(f"Mean pitch angle       : {mean_pitch:.4f} deg")
print(f"Pitch std. deviation   : {pitch_variation_std:.4f} deg")
print(f"Pitch range            : {pitch_variation_range:.4f} deg")
print(f"Beta traveled          : {beta_traveled:.4f} deg")
print(f"Mean beta rate         : {mean_beta_rate:.4f} deg/s")
print(f"Max beta rate          : {max_beta_rate:.4f} deg/s")

# ==========================================================
# Cp graph
# ==========================================================

plt.figure(figsize=(8, 4))
plt.plot(t3, Cp3)
plt.xlabel("Time [s]")
plt.ylabel("Cp [-]")
plt.title("Power coefficient Cp over time")
plt.grid(True)
plt.tight_layout()
plt.savefig("cp_over_time.png", dpi=150)
plt.show()