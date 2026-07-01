import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Grid definition
# -----------------------------
beta = np.linspace(0, 30, 200)      # pitch angle [deg]
lam = np.linspace(0.1, 20, 200)     # tip-speed ratio λ (avoid 0)

BETA, LAMBDA = np.meshgrid(beta, lam)

# -----------------------------
# Helper term
# -----------------------------
den1 = LAMBDA - 0.02 * BETA
den2 = BETA**3 + 1

# Avoid division by zero / singularities
eps = 1e-6
den1 = np.where(np.abs(den1) < eps, eps, den1)

inner = (1 / den1) - (0.003 / den2)

# -----------------------------
# Cp model
# -----------------------------
Cp = 0.73 * (
    151 * inner
    - 0.58 * BETA
    - 0.002 * BETA**2.14
    - 13.2
) * np.exp(-18.4 * inner)

# Optional: remove extreme values (for visualization stability)
Cp = np.clip(Cp, 0, 0.6)

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(LAMBDA, BETA, Cp, cmap='viridis', edgecolor='none')

ax.set_xlabel(r'$\lambda$ (tip-speed ratio)')
ax.set_ylabel(r'$\beta$ (deg)')
ax.set_zlabel(r'$C_p$')
#ax.set_title(r'$C_{p,2}(\lambda, \beta)$ surface')

plt.show()