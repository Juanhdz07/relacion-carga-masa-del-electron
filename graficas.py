"""Gráficas de la relación carga/masa del electrón.

Genera:
  figuras/B_vs_I.png              caracterización de la bobina (B = m I + b)
  figuras/V_vs_B2_r{r}cm.png      V vs B^2 para cada radio, V = m' B^2
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from datos import B_CAL, COLORES, I_CAL, RADIOS, SIGMA_B, SIGMA_I, SIGMA_V, V

FIG = Path(__file__).parent / "figuras"
FIG.mkdir(exist_ok=True)

# Puntos excluidos del ajuste. Para r = 6 cm las corrientes 0,01-0,04 A
# darían B ~ 0 mT, incompatible con un haz curvado de 6 cm de radio.
EXCLUIR = {
    0.06: [0, 1, 2],
}

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

coma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def con_coma(ax):
    ax.xaxis.set_major_formatter(coma)
    ax.yaxis.set_major_formatter(coma)


def fmt_pm(valor, error, cifras=2):
    """(valor ± error) con `cifras` cifras significativas en el error."""
    dec = max(cifras - 1 - int(np.floor(np.log10(error))), 0)
    return f"({valor:.{dec}f} ± {error:.{dec}f})".replace(".", ",")


def ajuste_lineal(x, y, sx, sy):
    """Regresión lineal ponderada y = a x + c con errores en x e y.

    Método de varianza efectiva: σ_ef² = σ_y² + (a σ_x)², iterado hasta
    converger. La covarianza se escala por χ²_red si este es mayor que 1.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx = np.broadcast_to(sx, x.shape)
    sy = np.broadcast_to(sy, y.shape)
    a, c = np.polyfit(x, y, 1)
    for _ in range(50):
        w = 1 / (sy**2 + (a * sx) ** 2)
        A = np.vstack([x, np.ones_like(x)]).T
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        a_n, c_n = cov @ (A.T @ (w * y))
        converge = np.isclose(a_n, a, rtol=1e-12)
        a, c = a_n, c_n
        if converge:
            break
    chi2_red = np.sum(w * (y - a * x - c) ** 2) / (len(x) - 2)
    return a, c, cov * max(chi2_red, 1.0)


def ajuste_origen(x, y, sx, sy):
    """Regresión ponderada por el origen y = a x con errores en x e y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx = np.broadcast_to(sx, x.shape)
    sy = np.broadcast_to(sy, y.shape)
    a = np.sum(x * y) / np.sum(x * x)
    for _ in range(50):
        w = 1 / (sy**2 + (a * sx) ** 2)
        a_n = np.sum(w * x * y) / np.sum(w * x * x)
        converge = np.isclose(a_n, a, rtol=1e-12)
        a = a_n
        if converge:
            break
    chi2_red = np.sum(w * (y - a * x) ** 2) / (len(x) - 1)
    return a, np.sqrt(max(chi2_red, 1.0) / np.sum(w * x * x))


def caja(ax, texto):
    ax.text(0.03, 0.97, texto, transform=ax.transAxes, ha="left", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))


def figura_con_residuales():
    return plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True,
                        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})


# --------------------------------------------------------------------------
# 1. Caracterización de la bobina: B vs I
# --------------------------------------------------------------------------
m, b, cov_mb = ajuste_lineal(I_CAL, B_CAL, SIGMA_I, SIGMA_B)
sm, sb = np.sqrt(np.diag(cov_mb))


def campo(I):
    """B (mT) e incertidumbre propagada a partir de la corriente I (A).

    σ_B² = I² σ_m² + σ_b² + 2 I cov(m,b) + m² σ_I²
    """
    I = np.asarray(I, dtype=float)
    var = (I**2 * cov_mb[0, 0] + cov_mb[1, 1] + 2 * I * cov_mb[0, 1]
           + (m * SIGMA_I) ** 2)
    return m * I + b, np.sqrt(var)


fig, (ax, axr) = figura_con_residuales()
xx = np.linspace(0, 1.8, 200)
ax.errorbar(I_CAL, B_CAL, xerr=SIGMA_I, yerr=SIGMA_B, fmt="o", ms=4,
            color="#1f5fa8", ecolor="k", elinewidth=1, capsize=3,
            label="Datos experimentales")
ax.plot(xx, m * xx + b, "-", color="#c0392b",
        label=f"Ajuste: B = {fmt_pm(m, sm)} I + {fmt_pm(b, sb)}")
ax.set_ylabel("B (mT)")
ax.set_xlim(0, 1.8)
ax.set_ylim(0, None)
ax.set_title("Caracterización de la bobina: B vs I")
ax.legend(loc="lower right", fontsize=9)
caja(ax, "Barras de error (incertidumbre instrumental):\n"
         f"  B: ±{SIGMA_B:g} mT    I: ±{SIGMA_I:g} A".replace(".", ","))

res = B_CAL - (m * I_CAL + b)
s_eff = np.sqrt(SIGMA_B**2 + (m * SIGMA_I) ** 2)
axr.axhline(0, color="#c0392b", lw=1)
axr.errorbar(I_CAL, res, yerr=s_eff, fmt="o", ms=4, color="#1f5fa8",
             ecolor="k", elinewidth=1, capsize=3)
axr.set_xlabel("I (A)")
axr.set_ylabel("Residuales (mT)")
lim = 1.3 * np.max(np.abs(res) + s_eff)
axr.set_ylim(-lim, lim)
con_coma(ax)
con_coma(axr)
fig.savefig(FIG / "B_vs_I.png")
plt.close(fig)

# --------------------------------------------------------------------------
# 2. V vs B² para cada radio: V = m' B²
# --------------------------------------------------------------------------
for r, I_r in RADIOS.items():
    rcm = round(r * 100)
    col = COLORES[r]
    usar = ~np.isnan(I_r)
    usar[EXCLUIR.get(r, [])] = False

    Vr = V[usar]
    B, sB = campo(I_r[usar])           # mT
    B2, sB2 = B**2, 2 * B * sB         # mT² = 10⁻⁶ T²
    mp, smp = ajuste_origen(B2, Vr, sB2, SIGMA_V)

    fig, (ax, axr) = figura_con_residuales()
    ax.errorbar(B2, Vr, xerr=sB2, yerr=SIGMA_V, fmt="o", ms=5, color=col,
                mec="k", ecolor="k", elinewidth=1, capsize=3,
                label="Datos experimentales")
    xx = np.linspace(0, B2.max() * 1.05, 100)
    ax.plot(xx, mp * xx, "-", color="k", lw=1.3,
            label="Ajuste lineal: V = m' B²")
    ax.set_ylabel("V (V)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_title(f"Voltaje vs B², r = {rcm} cm")
    ax.legend(loc="lower right", fontsize=9)
    caja(ax, "Barras de error (incertidumbre):\n"
             f"  V: ±{SIGMA_V:g} V (instrumental)\n"
             "  B²: ±2B·σ_B (propagada)")

    res = Vr - mp * B2
    s_eff = np.sqrt(SIGMA_V**2 + (mp * sB2) ** 2)
    axr.axhline(0, color="k", lw=1)
    axr.errorbar(B2, res, yerr=s_eff, fmt="o", ms=5, color=col, mec="k",
                 ecolor="k", elinewidth=1, capsize=3)
    axr.set_xlabel("B² (×10⁻⁶ T²)")
    axr.set_ylabel("Residuales (V)")
    lim = 1.3 * np.max(np.abs(res) + s_eff)
    axr.set_ylim(-lim, lim)
    con_coma(ax)
    con_coma(axr)
    fig.savefig(FIG / f"V_vs_B2_r{rcm}cm.png")
    plt.close(fig)
