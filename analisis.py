"""Análisis de la relación carga/masa del electrón.

Genera:
  figuras/B_vs_I.png              caracterización de la bobina (B = m I + b)
  figuras/V_vs_I_r{r}cm.png       V vs I para cada radio + curva del ajuste
  figuras/V_vs_B2_r{r}cm.png      V vs B^2 para cada radio (e/m = 2 s / r^2)
  figuras/e_m_por_radio.png       comparación de e/m para cada radio
  resultados/*.csv, *.tex         tablas con B calculado y su incertidumbre
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from datos import (B_CAL, COLORES, E_M_TEORICO, I_CAL, RADIOS, SIGMA_B,
                   SIGMA_I, SIGMA_R, SIGMA_V, V)

RAIZ = Path(__file__).parent
FIG = RAIZ / "figuras"
RES = RAIZ / "resultados"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# Puntos excluidos del ajuste (se grafican con marcador hueco).
# Para r = 6 cm las corrientes 0,01-0,04 A darían B ~ 0 mT, lo que es
# físicamente incompatible con un haz curvado de 6 cm de radio.
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


def fmt(valor, error, cifras=2):
    """Redondea valor ± error a `cifras` cifras significativas del error."""
    if error <= 0 or not np.isfinite(error):
        return f"{valor:.4g}".replace(".", ","), "0"
    dec = cifras - 1 - int(np.floor(np.log10(error)))
    dec = max(dec, 0)
    return (f"{valor:.{dec}f}".replace(".", ","),
            f"{error:.{dec}f}".replace(".", ","))


def fmt_pm(valor, error, cifras=2):
    v, e = fmt(valor, error, cifras)
    return f"({v} ± {e})"


SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def fmt_sci(valor, error, cifras=2):
    exp = int(np.floor(np.log10(abs(valor))))
    v, e = fmt(valor / 10**exp, error / 10**exp, cifras)
    return f"({v} ± {e})×10{str(exp).translate(SUPER)}", (v, e, exp)


def ajuste_lineal(x, y, sx, sy):
    """Regresión lineal ponderada y = a x + c con errores en x e y.

    Método de varianza efectiva: σ_ef² = σ_y² + (a σ_x)², iterado hasta
    converger. La covarianza se escala por χ²_red para que refleje la
    dispersión real de los datos.
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
        if np.isclose(a_n, a, rtol=1e-12):
            a, c = a_n, c_n
            break
        a, c = a_n, c_n
    nu = len(x) - 2
    chi2_red = np.sum(w * (y - a * x - c) ** 2) / nu
    return a, c, cov * max(chi2_red, 1.0), chi2_red, nu


def caja(ax, texto, loc="upper left"):
    pos = {"upper left": (0.03, 0.97, "left", "top"),
           "lower left": (0.03, 0.04, "left", "bottom"),
           "lower right": (0.97, 0.04, "right", "bottom")}[loc]
    ax.text(pos[0], pos[1], texto, transform=ax.transAxes, ha=pos[2],
            va=pos[3], fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))


# --------------------------------------------------------------------------
# 1. Caracterización de la bobina: B = m I + b
# --------------------------------------------------------------------------
m, b, cov_mb, chi2_cal, nu_cal = ajuste_lineal(
    I_CAL, B_CAL, np.full_like(I_CAL, SIGMA_I), np.full_like(B_CAL, SIGMA_B))
sm, sb = np.sqrt(np.diag(cov_mb))
r2_cal = np.corrcoef(I_CAL, B_CAL)[0, 1] ** 2


def campo(I):
    """B (mT) e incertidumbre propagada a partir de la corriente I (A).

    σ_B² = I² σ_m² + σ_b² + 2 I cov(m,b) + m² σ_I²
    """
    I = np.asarray(I, dtype=float)
    B = m * I + b
    var = (I**2 * cov_mb[0, 0] + cov_mb[1, 1] + 2 * I * cov_mb[0, 1]
           + (m * SIGMA_I) ** 2)
    return B, np.sqrt(var)


fig, (ax, axr) = plt.subplots(
    2, 1, figsize=(7.5, 6.5), sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
xx = np.linspace(0, 1.8, 200)
ax.errorbar(I_CAL, B_CAL, xerr=SIGMA_I, yerr=SIGMA_B, fmt="o", ms=4,
            color="#1f5fa8", ecolor="k", elinewidth=1, capsize=3,
            label="Datos experimentales")
ax.plot(xx, m * xx + b, "-", color="#c0392b",
        label=f"Ajuste: B = {fmt_pm(m, sm)} I + {fmt_pm(b, sb)}")
ax.set_ylabel("B (mT)")
ax.set_xlim(0, 1.8)
ax.set_ylim(0, None)
ax.legend(loc="lower right", fontsize=9)
caja(ax,
     "Barras de error (incertidumbre instrumental):\n"
     f"  B: ±{SIGMA_B:g} mT    I: ±{SIGMA_I:g} A".replace(".", ",") +
     f"\nR² = {r2_cal:.5f}   χ²ᵣ = {chi2_cal:.2f}".replace(".", ","))
ax.set_title("Caracterización de la bobina: B vs I")

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
# 2. Para cada radio: B(I), V vs I y V vs B²  ->  e/m
# --------------------------------------------------------------------------
resumen = []
tex_filas = []
csv_filas = ["radio_m,V_V,sigma_V,I_A,sigma_I,B_mT,sigma_B_mT,"
             "B2_mT2,sigma_B2_mT2,usado_en_ajuste"]

for r, I_r in RADIOS.items():
    rcm = round(r * 100)
    col = COLORES[r]
    valido = ~np.isnan(I_r)
    usar = valido.copy()
    for k in EXCLUIR.get(r, []):
        usar[k] = False

    Vr, Ir = V[valido], I_r[valido]
    usar_v = usar[valido]
    B, sB = campo(Ir)                  # mT
    B2, sB2 = B**2, 2 * B * sB         # mT²

    # V = s B² + c   (B en mT, s en V/mT²)
    s, c, cov_sc, chi2, nu = ajuste_lineal(
        B2[usar_v], Vr[usar_v], sB2[usar_v], np.full(usar_v.sum(), SIGMA_V))
    ss, sc = np.sqrt(np.diag(cov_sc))
    r2 = np.corrcoef(B2[usar_v], Vr[usar_v])[0, 1] ** 2

    # e/m = 2 s / r²  (s pasado a V/T²)
    s_T = s * 1e6
    ss_T = ss * 1e6
    em = 2 * s_T / r**2
    sem = np.sqrt((2 / r**2 * ss_T) ** 2 + (4 * s_T / r**3 * SIGMA_R) ** 2)
    resumen.append((r, s, ss, c, sc, chi2, r2, em, sem))

    for vi, ii, bi, sbi, b2i, sb2i, u in zip(Vr, Ir, B, sB, B2, sB2, usar_v):
        csv_filas.append(f"{r},{vi:g},{SIGMA_V:g},{ii:g},{SIGMA_I:g},"
                         f"{bi:.4f},{sbi:.4f},{b2i:.4f},{sb2i:.4f},{int(u)}")
        bt, sbt = fmt(bi, sbi, 1)
        b2t, sb2t = fmt(b2i, sb2i, 1)
        marca = "" if u else "$^{*}$"
        tex_filas.append(
            f"{rcm} & {vi:.0f} $\\pm$ 1 & {ii:.2f} $\\pm$ 0,01 & "
            f"{bt} $\\pm$ {sbt} & {b2t} $\\pm$ {sb2t}{marca} \\\\"
            .replace("0.01", "0,01").replace(f"{ii:.2f}", f"{ii:.2f}".replace(".", ",")))
    tex_filas.append("\\midrule")

    em_txt, _ = fmt_sci(em, sem)
    texto_err = ("Barras de error (incertidumbre):\n"
                 f"  V: ±{SIGMA_V:g} V (instrumental)\n")

    # ---------- V vs I ----------
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.5, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
    ax.errorbar(Ir[usar_v], Vr[usar_v], xerr=SIGMA_I, yerr=SIGMA_V, fmt="o",
                ms=5, color=col, mec="k", ecolor="k", elinewidth=1,
                capsize=3, label="Datos experimentales")
    if (~usar_v).any():
        ax.errorbar(Ir[~usar_v], Vr[~usar_v], xerr=SIGMA_I, yerr=SIGMA_V,
                    fmt="o", ms=5, mfc="white", mec=col, ecolor="0.5",
                    elinewidth=1, capsize=3, label="Excluidos del ajuste")
    imin = max(0.0, Ir.min() - 0.1)
    xx = np.linspace(imin, Ir.max() + 0.05, 200)
    ax.plot(xx, s * (m * xx + b) ** 2 + c, "-", color="k", lw=1.3,
            label="Ajuste: V = s (m I + b)² + c")
    ax.set_ylabel("V (V)")
    ax.set_title(f"Voltaje vs corriente, r = {rcm} cm")
    ax.legend(loc="lower right", fontsize=9)
    caja(ax,
         (texto_err + f"  I: ±{SIGMA_I:g} A (instrumental)\n"
          f"s = {fmt_pm(s, ss)} V/mT²\n"
          f"c = {fmt_pm(c, sc)} V\n"
          f"e/m = {em_txt} C/kg").replace(".", ","))
    res = Vr - (s * (m * Ir + b) ** 2 + c)
    s_eff = np.sqrt(SIGMA_V**2 + (2 * s * (m * Ir + b) * m * SIGMA_I) ** 2)
    axr.axhline(0, color="k", lw=1)
    axr.errorbar(Ir[usar_v], res[usar_v], yerr=s_eff[usar_v], fmt="o", ms=5,
                 color=col, mec="k", ecolor="k", elinewidth=1, capsize=3)
    axr.set_xlabel("I (A)")
    axr.set_ylabel("Residuales (V)")
    lim = 1.3 * np.max(np.abs(res[usar_v]) + s_eff[usar_v])
    axr.set_ylim(-lim, lim)
    con_coma(ax)
    con_coma(axr)
    fig.savefig(FIG / f"V_vs_I_r{rcm}cm.png")
    plt.close(fig)

    # ---------- V vs B² ----------
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.5, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
    ax.errorbar(B2[usar_v], Vr[usar_v], xerr=sB2[usar_v], yerr=SIGMA_V,
                fmt="o", ms=5, color=col, mec="k", ecolor="k", elinewidth=1,
                capsize=3, label="Datos experimentales")
    xx = np.linspace(B2[usar_v].min() * 0.9, B2[usar_v].max() * 1.05, 100)
    ax.plot(xx, s * xx + c, "-", color="k", lw=1.3,
            label="Ajuste lineal: V = s B² + c")
    ax.set_ylabel("V (V)")
    ax.set_title(f"Voltaje vs B², r = {rcm} cm")
    ax.legend(loc="lower right", fontsize=9)
    caja(ax,
         (texto_err + "  B²: 2B·σ_B (propagada desde B = m I + b)\n"
          f"s = {fmt_pm(s, ss)} V/mT²\n"
          f"c = {fmt_pm(c, sc)} V\n"
          f"R² = {r2:.4f}   χ²ᵣ = {chi2:.2f}\n"
          f"e/m = 2s/r² = {em_txt} C/kg").replace(".", ","))
    res = Vr - (s * B2 + c)
    s_eff = np.sqrt(SIGMA_V**2 + (s * sB2) ** 2)
    axr.axhline(0, color="k", lw=1)
    axr.errorbar(B2[usar_v], res[usar_v], yerr=s_eff[usar_v], fmt="o", ms=5,
                 color=col, mec="k", ecolor="k", elinewidth=1, capsize=3)
    axr.set_xlabel("B² (mT²)")
    axr.set_ylabel("Residuales (V)")
    lim = 1.3 * np.max(np.abs(res[usar_v]) + s_eff[usar_v])
    axr.set_ylim(-lim, lim)
    con_coma(ax)
    con_coma(axr)
    fig.savefig(FIG / f"V_vs_B2_r{rcm}cm.png")
    plt.close(fig)

# --------------------------------------------------------------------------
# 3. e/m por radio y promedio ponderado
# --------------------------------------------------------------------------
radios = np.array([x[0] for x in resumen])
ems = np.array([x[7] for x in resumen])
sems = np.array([x[8] for x in resumen])
w = 1 / sems**2
em_prom = np.sum(w * ems) / np.sum(w)
sem_prom = 1 / np.sqrt(np.sum(w))
err_rel = abs(em_prom - E_M_TEORICO) / E_M_TEORICO * 100

fig, ax = plt.subplots(figsize=(7, 4.5))
for r, e, se in zip(radios, ems, sems):
    ax.errorbar(r * 100, e / 1e11, yerr=se / 1e11, fmt="o", ms=7,
                color=COLORES[r], mec="k", ecolor="k", capsize=4)
ax.axhline(E_M_TEORICO / 1e11, color="g", ls="--",
           label="Valor teórico: 1,7588×10¹¹ C/kg")
ax.axhspan((em_prom - sem_prom) / 1e11, (em_prom + sem_prom) / 1e11,
           color="0.5", alpha=0.25,
           label=f"Promedio ponderado: {fmt_sci(em_prom, sem_prom)[0]} C/kg"
           .replace(".", ","))
ax.set_xlabel("Radio r (cm)")
ax.set_ylabel("e/m (×10¹¹ C/kg)")
ax.set_xticks(radios * 100)
ax.set_xlim(1, 9)
ax.set_ylim(0, 2.2)
ax.set_title("Relación carga/masa obtenida para cada radio")
ax.legend(fontsize=9, loc="upper right")
caja(ax, "Barras de error: incertidumbre de e/m propagada\n"
         "desde la pendiente s del ajuste V vs B²", loc="lower left")
con_coma(ax)
fig.savefig(FIG / "e_m_por_radio.png")
plt.close(fig)

# --------------------------------------------------------------------------
# 4. Tablas
# --------------------------------------------------------------------------
(RES / "campo_por_punto.csv").write_text("\n".join(csv_filas) + "\n")

tex = [
    "\\begin{table}[H]",
    "\\centering",
    "\\caption{Campo magnético obtenido a partir de la corriente mediante la "
    "ecuación de ajuste $B = mI + b$, con su incertidumbre propagada "
    "$\\sigma_B^2 = I^2\\sigma_m^2 + \\sigma_b^2 + 2I\\,\\mathrm{cov}(m,b) "
    "+ m^2\\sigma_I^2$ y $\\sigma_{B^2} = 2B\\sigma_B$. "
    "$^{*}$Punto excluido del ajuste.}",
    "\\label{tab:campo}",
    "\\begin{tabular}{ccccc}",
    "\\toprule",
    "$r$ (cm) & $V$ (V) & $I$ (A) & $B$ (mT) & $B^2$ (mT$^2$) \\\\",
    "\\midrule",
    *tex_filas[:-1],
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}",
]
(RES / "tabla_campo.tex").write_text("\n".join(tex) + "\n")

lineas = [
    "CARACTERIZACIÓN DE LA BOBINA  (B = m I + b)",
    f"  m = {fmt_pm(m, sm)} mT/A",
    f"  b = {fmt_pm(b, sb)} mT",
    f"  cov(m,b) = {cov_mb[0, 1]:.3e} mT²/A",
    f"  R² = {r2_cal:.5f}   chi²_red = {chi2_cal:.3f}",
    "",
    "AJUSTES V = s B² + c   ->   e/m = 2 s / r²",
]
for r, s, ss, c, sc, chi2, r2, em, sem in resumen:
    lineas += [
        f"  r = {r * 100:.0f} cm:",
        f"    s = {fmt_pm(s, ss)} V/mT²   c = {fmt_pm(c, sc)} V",
        f"    R² = {r2:.4f}   chi²_red = {chi2:.2f}",
        f"    e/m = {fmt_sci(em, sem)[0]} C/kg",
    ]
lineas += [
    "",
    f"PROMEDIO PONDERADO e/m = {fmt_sci(em_prom, sem_prom)[0]} C/kg",
    f"Valor teórico          = 1,75882×10^11 C/kg",
    f"Error relativo         = {err_rel:.1f} %".replace(".", ","),
]
(RES / "resumen.txt").write_text("\n".join(lineas) + "\n")
print("\n".join(lineas))
