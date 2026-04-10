# =============================================================================
# TFG — Predicción de lesiones en running
# BLOQUE 1: Carga de datos y análisis exploratorio (EDA)
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import warnings
warnings.filterwarnings('ignore')

# Estilo general
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 130,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
PALETTE = ['#2196F3', '#F44336']   # azul = no lesión, rojo = lesión

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

df_daily  = pd.read_csv("/mnt/user-data/uploads/1775755403870_timeseries__daily_.csv")
df_weekly = pd.read_csv("/mnt/user-data/uploads/1775755403871_timeseries__weekly_.csv")

df_daily  = df_daily.sort_values(['Athlete ID', 'Date']).reset_index(drop=True)
df_weekly = df_weekly.sort_values(['Athlete ID', 'Date']).reset_index(drop=True)

print("=" * 55)
print("RESUMEN DE LOS DATASETS")
print("=" * 55)
print(f"  Daily  — filas: {len(df_daily):,}  |  columnas: {df_daily.shape[1]}")
print(f"  Weekly — filas: {len(df_weekly):,}  |  columnas: {df_weekly.shape[1]}")
print(f"  Atletas únicos (daily):  {df_daily['Athlete ID'].nunique()}")
print(f"  Atletas únicos (weekly): {df_weekly['Athlete ID'].nunique()}")
print(f"  Rango temporal (índice días): "
      f"{df_daily['Date'].min()} – {df_daily['Date'].max()}")
print(f"  Valores nulos — daily:  {df_daily.isnull().sum().sum()}")
print(f"  Valores nulos — weekly: {df_weekly.isnull().sum().sum()}")


# =============================================================================
# 2. VARIABLE OBJETIVO — DISTRIBUCIÓN Y DESBALANCEO
# =============================================================================

print("\n" + "=" * 55)
print("VARIABLE OBJETIVO — injury")
print("=" * 55)
for label, df in [("Daily", df_daily), ("Weekly", df_weekly)]:
    vc = df['injury'].value_counts()
    rate = df['injury'].mean() * 100
    print(f"\n  [{label}]")
    print(f"    injury=0 (no lesión): {vc[0]:,}")
    print(f"    injury=1 (lesión):    {vc[1]:,}")
    print(f"    Tasa de lesión:       {rate:.2f}%")
    print(f"    Ratio de desbalanceo: 1 : {vc[0]/vc[1]:.0f}")

# Figura 1 — Desbalanceo de clases
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (label, df) in zip(axes, [("Diario", df_daily), ("Semanal", df_weekly)]):
    counts = df['injury'].value_counts().sort_index()
    bars = ax.bar(['Sin lesión', 'Lesión'], counts.values,
                  color=PALETTE, edgecolor='white', width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,}', ha='center', va='bottom', fontsize=10)
    ax.set_title(f'Distribución de clases — {label}')
    ax.set_ylabel('Número de registros')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.suptitle('Desbalanceo de la variable objetivo', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig01_desbalanceo_clases.png', bbox_inches='tight')
plt.close()
print("\n  → Guardada: fig01_desbalanceo_clases.png")


# =============================================================================
# 3. ANÁLISIS POR ATLETA
# =============================================================================

print("\n" + "=" * 55)
print("ANÁLISIS POR ATLETA")
print("=" * 55)

obs_por_atleta = df_weekly.groupby('Athlete ID').size()
lesiones_por_atleta = df_weekly.groupby('Athlete ID')['injury'].sum()

print(f"\n  Observaciones semanales por atleta:")
print(f"    Media: {obs_por_atleta.mean():.0f}  |  "
      f"Mín: {obs_por_atleta.min()}  |  Máx: {obs_por_atleta.max()}")
print(f"\n  Semanas con lesión por atleta:")
print(f"    Media: {lesiones_por_atleta.mean():.1f}  |  "
      f"Máx: {lesiones_por_atleta.max()}")
print(f"    Atletas con 0 lesiones: {(lesiones_por_atleta==0).sum()}")
print(f"    Atletas con ≥1 lesión:  {(lesiones_por_atleta>=1).sum()}")

# Figura 2 — Distribución de observaciones y lesiones por atleta
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(len(obs_por_atleta)),
            obs_por_atleta.sort_values(ascending=False).values,
            color='#2196F3', alpha=0.8)
axes[0].set_title('Semanas de seguimiento por atleta')
axes[0].set_xlabel('Atleta (ordenado)')
axes[0].set_ylabel('Número de semanas')

axes[1].bar(range(len(lesiones_por_atleta)),
            lesiones_por_atleta.sort_values(ascending=False).values,
            color='#F44336', alpha=0.8)
axes[1].set_title('Semanas con lesión por atleta')
axes[1].set_xlabel('Atleta (ordenado)')
axes[1].set_ylabel('Semanas con lesión')

plt.suptitle('Heterogeneidad entre atletas', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig02_atletas.png', bbox_inches='tight')
plt.close()
print("\n  → Guardada: fig02_atletas.png")


# =============================================================================
# 4. CÁLCULO DEL ACWR
# =============================================================================
# El dataset semanal incluye datos de las 3 últimas semanas (sufijos .1 y .2).
# ACWR = carga aguda (semana actual) / carga crónica (media 3 semanas)
# Los ratios precalculados del dataset tienen valores extremos cuando la semana
# previa tiene km≈0 (retorno de lesión). Se recalcula de forma robusta.

print("\n" + "=" * 55)
print("CÁLCULO DEL ACWR")
print("=" * 55)

df_weekly['chronic_kms'] = (
    df_weekly['total kms'] +
    df_weekly['total kms.1'] +
    df_weekly['total kms.2']
) / 3

df_weekly['ACWR'] = np.where(
    df_weekly['chronic_kms'] > 0,
    df_weekly['total kms'] / df_weekly['chronic_kms'],
    np.nan
)

# Cap at 3.0 to remove physiologically implausible outliers
df_weekly['ACWR'] = df_weekly['ACWR'].clip(upper=3.0)

print(f"\n  ACWR calculado sobre {df_weekly['ACWR'].notna().sum():,} registros")
print(f"  NaN (carga crónica = 0): {df_weekly['ACWR'].isna().sum()}")
print(f"\n  Distribución ACWR:")
print(df_weekly['ACWR'].describe().round(3).to_string())

acwr_by_injury = df_weekly.groupby('injury')['ACWR'].mean()
print(f"\n  ACWR medio — sin lesión: {acwr_by_injury[0]:.3f}")
print(f"  ACWR medio — con lesión: {acwr_by_injury[1]:.3f}")
print(f"  Diferencia relativa: "
      f"{(acwr_by_injury[1]-acwr_by_injury[0])/acwr_by_injury[0]*100:+.1f}%")


# =============================================================================
# 5. COMPARACIÓN DE VARIABLES CLAVE POR CLASE
# =============================================================================

features_analisis = {
    'total kms':           'Kilómetros totales',
    'ACWR':                'ACWR',
    'avg exertion':        'Esfuerzo percibido (media)',
    'avg recovery':        'Recuperación percibida (media)',
    'nr. rest days':       'Días de descanso',
    'nr. tough sessions (effort in Z5, T1 or T2)': 'Sesiones duras (Z5/T1/T2)',
    'max km one day':      'Máximo km en un día',
    'total km Z5-T1-T2':   'km en zonas alta intensidad',
}

print("\n" + "=" * 55)
print("COMPARACIÓN DE VARIABLES POR CLASE")
print("=" * 55)
print(f"\n  {'Variable':<45} {'Sin lesión':>12} {'Lesión':>12} {'Δ%':>8}")
print("  " + "-" * 80)
for col, label in features_analisis.items():
    if col not in df_weekly.columns:
        continue
    m0 = df_weekly[df_weekly['injury']==0][col].mean()
    m1 = df_weekly[df_weekly['injury']==1][col].mean()
    diff = (m1 - m0) / abs(m0) * 100 if m0 != 0 else np.nan
    print(f"  {label:<45} {m0:>12.3f} {m1:>12.3f} {diff:>+8.1f}%")

# Figura 3 — Boxplots comparativos
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

valid_features = [(col, lbl) for col, lbl in features_analisis.items()
                  if col in df_weekly.columns]

for ax, (col, label) in zip(axes, valid_features):
    data_0 = df_weekly[df_weekly['injury']==0][col].dropna()
    data_1 = df_weekly[df_weekly['injury']==1][col].dropna()

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

plt.suptitle('Distribución de variables clave por clase de lesión',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('fig03_boxplots_variables.png', bbox_inches='tight')
plt.close()
print("\n  → Guardada: fig03_boxplots_variables.png")


# =============================================================================
# 6. DISTRIBUCIÓN DEL ACWR POR CLASE (figura detallada)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histograma superpuesto
for val, label, color in [(0, 'Sin lesión', '#2196F3'), (1, 'Lesión', '#F44336')]:
    data = df_weekly[df_weekly['injury']==val]['ACWR'].dropna()
    axes[0].hist(data, bins=50, alpha=0.6, color=color,
                 label=f'{label} (n={len(data):,})', density=True)
axes[0].axvline(1.0, color='gray', linestyle='--', linewidth=1, label='ACWR = 1.0')
axes[0].axvline(1.5, color='orange', linestyle='--', linewidth=1, label='ACWR = 1.5')
axes[0].set_xlabel('ACWR')
axes[0].set_ylabel('Densidad')
axes[0].set_title('Distribución del ACWR por clase')
axes[0].legend(fontsize=9)

# Tasa de lesión por tramo de ACWR
bins = [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.01]
labels_bins = ['<0.5', '0.5–0.8', '0.8–1.0', '1.0–1.2', '1.2–1.5', '1.5–2.0', '>2.0']
df_weekly['ACWR_bin'] = pd.cut(df_weekly['ACWR'], bins=bins, labels=labels_bins)
injury_rate_by_acwr = df_weekly.groupby('ACWR_bin', observed=True)['injury'].mean() * 100

axes[1].bar(injury_rate_by_acwr.index, injury_rate_by_acwr.values,
            color='#E91E63', alpha=0.8, edgecolor='white')
axes[1].set_xlabel('Rango de ACWR')
axes[1].set_ylabel('Tasa de lesión (%)')
axes[1].set_title('Tasa de lesión por rango de ACWR')
axes[1].yaxis.set_major_formatter(PercentFormatter(decimals=2))

plt.suptitle('Análisis del ACWR y su relación con la lesión', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig04_ACWR_analisis.png', bbox_inches='tight')
plt.close()
print("  → Guardada: fig04_ACWR_analisis.png")


# =============================================================================
# 7. CORRELACIONES CON LA VARIABLE OBJETIVO
# =============================================================================

# Seleccionar columnas de la semana actual (sin sufijos .1 .2)
base_numeric_cols = [c for c in df_weekly.columns
                     if '.' not in c
                     and c not in ['Athlete ID', 'Date', 'injury',
                                   'ACWR_bin', 'chronic_kms']
                     and df_weekly[c].dtype in [np.float64, np.int64]]

corr_with_injury = (df_weekly[base_numeric_cols + ['ACWR', 'injury']]
                    .corr()['injury']
                    .drop('injury')
                    .sort_values(key=abs, ascending=False))

print("\n" + "=" * 55)
print("TOP 10 CORRELACIONES CON injury (semana actual)")
print("=" * 55)
print(corr_with_injury.head(10).round(4).to_string())

# Figura 5 — Correlaciones top
top_corr = corr_with_injury.head(15)
colors_corr = ['#F44336' if v > 0 else '#2196F3' for v in top_corr.values]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(top_corr)), top_corr.values, color=colors_corr, alpha=0.8)
ax.set_yticks(range(len(top_corr)))
ax.set_yticklabels(top_corr.index, fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Correlación de Pearson con injury')
ax.set_title('Top 15 variables más correlacionadas con la lesión\n(semana actual)', fontsize=13)
plt.tight_layout()
plt.savefig('fig05_correlaciones.png', bbox_inches='tight')
plt.close()
print("\n  → Guardada: fig05_correlaciones.png")


# =============================================================================
# 8. EVOLUCIÓN TEMPORAL DE LESIONES
# =============================================================================

# Tasa de lesión por período temporal (ventana de 100 días)
df_weekly['date_bin'] = pd.cut(df_weekly['Date'], bins=27)
temporal = df_weekly.groupby('date_bin', observed=True)['injury'].mean() * 100

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(len(temporal)), temporal.values,
        color='#F44336', linewidth=2, marker='o', markersize=4)
ax.fill_between(range(len(temporal)), temporal.values, alpha=0.15, color='#F44336')
ax.set_xlabel('Período del estudio (progresivo)')
ax.set_ylabel('Tasa de lesión (%)')
ax.set_title('Evolución temporal de la tasa de lesión a lo largo del estudio')
ax.yaxis.set_major_formatter(PercentFormatter(decimals=2))
plt.tight_layout()
plt.savefig('fig06_evolucion_temporal.png', bbox_inches='tight')
plt.close()
print("  → Guardada: fig06_evolucion_temporal.png")

print("\n" + "=" * 55)
print("EDA COMPLETADO — 6 figuras generadas")
print("=" * 55)
