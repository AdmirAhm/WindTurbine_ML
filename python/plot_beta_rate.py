import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# User settings
# ==========================================================

files = {
    "PI":    "res.txt",
    "LMPC":  "res_lin.txt",
    "NMPC":  "res_nmpc.txt",
    "DQN":   "res_ddqn.txt",
}

regime_start = 40.0   # set to None to plot the full time series instead

# ==========================================================
# Load, compute beta rate, and plot
# ==========================================================

plt.figure(figsize=(9, 5))

for label, filename in files.items():

    df = pd.read_csv(filename, delim_whitespace=True)

    t = df["t"].to_numpy()
    beta = df["β"].to_numpy()      # change to "beta" if needed

    if regime_start is not None:
        mask = (t >= regime_start) & (t<=60)
        t = t[mask]
        beta = beta[mask]

    beta_rate = np.diff(beta) / np.diff(t)
    t_mid = t[:-1] + np.diff(t) / 2   # time points aligned with the rate samples

    plt.plot(t_mid, abs(beta_rate), label=label)

plt.xlabel("Time [s]")
plt.ylabel(r"$d\beta/dt$ [deg/s]")
plt.title("Brzina zakretanja lopatica")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("beta_rate_comparison.png", dpi=150)
plt.show()
