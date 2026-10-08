"""Datos experimentales: relación carga/masa del electrón."""

import numpy as np

# Incertidumbres instrumentales
SIGMA_V = 1.0      # V
SIGMA_I = 0.01     # A
SIGMA_B = 0.01     # mT
SIGMA_R = 0.0      # m  (incertidumbre del radio; cámbiala si la conoces)

B_AMB = 0.04       # mT, campo ambiental (ya restado en la tabla de calibración)
SIGMA_B_AMB = 0.01 # mT

# Caracterización de la bobina: B = B_medido - B_amb
I_CAL = np.round(np.arange(0.1, 1.71, 0.1), 2)  # A
B_CAL = np.array([0.18, 0.35, 0.52, 0.68, 0.82, 1.00, 1.16, 1.33, 1.48,
                  1.64, 1.79, 1.96, 2.11, 2.26, 2.41, 2.56, 2.72])  # mT

# Voltaje de aceleración vs corriente para cada radio (np.nan = N/A)
V = np.array([100, 125, 150, 175, 200, 225, 250, 275], dtype=float)
nan = np.nan
RADIOS = {
    0.02: np.array([0.60, 0.71, 1.25, 1.36, 1.46, 1.55, 1.63, nan]),
    0.04: np.array([0.14, 0.36, 0.68, 0.81, 0.90, 0.97, 1.04, 1.09]),
    0.06: np.array([0.01, 0.02, 0.04, 0.58, 0.66, 0.73, 0.77, 0.80]),
    0.08: np.array([nan, nan, nan, 0.47, 0.53, 0.58, 0.60, 0.64]),
}

COLORES = {0.02: "#d4b000", 0.04: "#d62728", 0.06: "#17becf", 0.08: "#ff7f0e"}

E_M_TEORICO = 1.75882e11  # C/kg
