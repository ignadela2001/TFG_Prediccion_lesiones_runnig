# =============================================================================
# TFG — Predicción de lesiones en running
# BLOQUE 2: Ingeniería de variables (Feature Engineering)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'figure.dpi': 130, 'axes.spines.top': False, 'axes.spines.right': False,
})
PALETTE = ['#2196F3', '#F44336']

# =============================================================================
# 1. CARGA DEL DATASET
# =============================================================================

df = pd.read_csv("timeseries__weekly_.csv")
df = df.sort_values(['Athlete ID', 'Date']).reset_index(drop=True)

print("=" * 60)
print("BLOQUE 2 — INGENIERÍA DE VARIABLES")
print("=" * 60)
print(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# =============================================================================
# 2. ACWR — ACUTE:CHRONIC WORKLOAD RATIO (kilómetros)
# =============================================================================
# Definición: carga aguda (semana actual) / carga crónica (media 3 semanas)
# Justificación: métrica de referencia en la literatura (Nakaoka et al., 2021;
# Frandsen et al., 2025). Se capea a 3.0 para eliminar valores fisiológicamente
# implausibles debidos a retornos tras periodos de inactividad prolongados.

df['chronic_kms'] = (
    df['total kms'] + df['total kms.1'] + df['total kms.2']
) / 3

df['ACWR'] = np.where(
    df['chronic_kms'] > 0,
    df['total kms'] / df['chronic_kms'],
    np.nan
)
df['ACWR'] = df['ACWR'].clip(upper=3.0)

# =============================================================================
# 3. ACWR SOBRE ESFUERZO PERCIBIDO (carga interna)
# =============================================================================
# Se replica el cálculo del ACWR usando el esfuerzo percibido medio (avg exertion)
# como proxy de carga interna, para capturar la dimensión subjetiva del
# entrenamiento (Jones et al., 2017).

df['chronic_exertion'] = (
    df['avg exertion'] + df['avg exertion.1'] + df['avg exertion.2']
) / 3

df['ACWR_exertion'] = np.where(
    df['chronic_exertion'] > 0,
    df['avg exertion'] / df['chronic_exertion'],
    np.nan
)
df['ACWR_exertion'] = df['ACWR_exertion'].clip(upper=3.0)

# =============================================================================
# 4. TRAINING MONOTONY
# =============================================================================
# Definición original (Foster, 1998): media_carga / desviación_típica_carga.
# Valores altos → entrenamiento muy uniforme → mayor fatiga acumulada.
# Aproximación con datos semanales: usamos avg exertion como proxy de carga
# interna y estimamos la variabilidad como (max - min) / 2.

df['exertion_range'] = df['max exertion'] - df['min exertion']
df['monotony'] = np.where(
    df['exertion_range'] > 0,
    df['avg exertion'] / (df['exertion_range'] / 2),
    np.nan
)
df['monotony'] = df['monotony'].clip(upper=10.0)

# =============================================================================
# 5. TRAINING STRAIN
# =============================================================================
# Definición: carga_total_semanal × monotonía.
# Combina en un único índice el volumen de entrenamiento y su uniformidad,
# capturando el estrés acumulado total (Matos et al., 2020).

df['weekly_load'] = df['total kms'] * df['avg exertion']
df['strain'] = df['weekly_load'] * df['monotony']

# =============================================================================
# 6. RATIOS DE CAMBIO SEMANAL (km)
# =============================================================================
# Variación relativa de kilómetros entre semanas consecutivas.
# Complementa el ACWR ofreciendo una perspectiva más directa e interpretable
# del incremento o reducción de carga a corto plazo.

df['km_change_w0_w1'] = np.where(
    df['total kms.1'] > 0,
    (df['total kms'] - df['total kms.1']) / df['total kms.1'],
    np.nan
)
df['km_change_w0_w1'] = df['km_change_w0_w1'].clip(-1, 2)

df['km_change_w1_w2'] = np.where(
    df['total kms.2'] > 0,
    (df['total kms.1'] - df['total kms.2']) / df['total kms.2'],
    np.nan
)
df['km_change_w1_w2'] = df['km_change_w1_w2'].clip(-1, 2)

# =============================================================================
# 7. TENDENCIAS DE RECUPERACIÓN Y ESFUERZO
# =============================================================================
# Diferencia entre la semana actual y la anterior en variables subjetivas.
# Una tendencia positiva de esfuerzo junto con una negativa de recuperación
# puede indicar acumulación de fatiga no compensada.

df['recovery_trend'] = df['avg recovery'] - df['avg recovery.1']
df['exertion_trend']  = df['avg exertion'] - df['avg exertion.1']

# =============================================================================
# 8. RESUMEN DE VARIABLES CREADAS
# =============================================================================

new_vars = {
    'ACWR':             'ACWR (kilómetros)',
    'ACWR_exertion':    'ACWR (esfuerzo percibido)',
    'monotony':         'Monotonía del entrenamiento',
    'strain':           'Training Strain',
    'km_change_w0_w1':  'Cambio km (semana 0 vs 1)',
    'km_change_w1_w2':  'Cambio km (semana 1 vs 2)',
    'recovery_trend':   'Tendencia recuperación',
    'exertion_trend':   'Tendencia esfuerzo',
}

print(f"\n{'Variable':<22} {'Media':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'NaN':>6}")
print("-" * 65)
for v in new_vars:
    s = df[v].describe()
    nans = df[v].isna().sum()
    print(f"{v:<22} {s['mean']:>8.3f} {s['std']:>8.3f} {s['min']:>8.3f} {s['max']:>8.3f} {nans:>6}")

print(f"\n{'Variable':<22} {'Sin lesión':>12} {'Lesión':>12} {'Δ%':>8}")
print("-" * 58)
for v in new_vars:
    m0 = df[df['injury']==0][v].mean()
    m1 = df[df['injury']==1][v].mean()
    diff = (m1 - m0) / abs(m0) * 100 if m0 != 0 else np.nan
    print(f"{v:<22} {m0:>12.4f} {m1:>12.4f} {diff:>+8.1f}%")

# =============================================================================
# 9. VISUALIZACIÓN — BOXPLOTS POR CLASE
# =============================================================================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for ax, (col, label) in zip(axes, new_vars.items()):
    data_0 = df[df['injury']==0][col].dropna()
    data_1 = df[df['injury']==1][col].dropna()
    bp = ax.boxplot([data_0, data_1],
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='.', markersize=2, alpha=0.3),
                    whiskerprops=dict(linewidth=1.2),
                    boxprops=dict(linewidth=1.2))
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Sin lesión', 'Lesión'], fontsize=9)
    ax.set_title(label, fontsize=10)

plt.suptitle('Variables derivadas — distribución por clase de lesión', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('fig07_features_derivadas.png', bbox_inches='tight')
plt.close()
print("\n  → Guardada: fig07_features_derivadas.png")

# =============================================================================
# 10. CORRELACIÓN COMPLETA CON INJURY
# =============================================================================

exclude = ['Athlete ID', 'Date', 'injury', 'chronic_kms', 'chronic_exertion',
           'exertion_range', 'weekly_load', 'ACWR_bin']
all_feats = [c for c in df.columns
             if c not in exclude
             and df[c].dtype in [float, int]
             and '.' not in c]

corr = (df[all_feats + ['injury']]
        .corr()['injury']
        .drop('injury')
        .sort_values(key=abs, ascending=False)
        .head(20))

colors_c = ['#F44336' if v > 0 else '#2196F3' for v in corr.values]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(corr)), corr.values, color=colors_c, alpha=0.8)
ax.set_yticks(range(len(corr)))
ax.set_yticklabels(corr.index, fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Correlación de Pearson con injury')
ax.set_title('Top 20 variables más correlacionadas con lesión\n(incluidas variables derivadas)', fontsize=12)
plt.tight_layout()
plt.savefig('fig08_correlaciones_full.png', bbox_inches='tight')
plt.close()
print("  → Guardada: fig08_correlaciones_full.png")

# =============================================================================
# 11. GUARDADO DEL DATASET FINAL
# =============================================================================

cols_drop = ['chronic_kms', 'chronic_exertion', 'exertion_range', 'weekly_load']
df_final = df.drop(columns=cols_drop)
df_final.to_csv("weekly_features_final.csv", index=False)

print(f"\n  Dataset final guardado: {df_final.shape[0]:,} filas × {df_final.shape[1]} columnas")
print("\n" + "=" * 60)
print("BLOQUE 2 COMPLETADO")
print("=" * 60)
