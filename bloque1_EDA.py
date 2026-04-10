import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
PALETTE = ['#2196F3', '#F44336']
 
# =============================================================================
# 1. RUTAS DE ARCHIVOS
# =============================================================================
# Los CSV deben estar en la misma carpeta que este script.
# os.path.dirname(__file__) obtiene automáticamente esa ruta.
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
PATH_DAILY  = os.path.join(BASE_DIR, "timeseries (daily).csv")
PATH_WEEKLY = os.path.join(BASE_DIR, "timeseries (weekly).csv")
 
# =============================================================================
# 2. CARGA DE DATOS
# =============================================================================
 
df_daily  = pd.read_csv(PATH_DAILY)
df_weekly = pd.read_csv(PATH_WEEKLY)
 
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
# 3. VARIABLE OBJETIVO — DISTRIBUCIÓN Y DESBALANCEO
# =============================================================================
 
print("\n" + "=" * 55)
print("VARIABLE OBJETIVO — injury")
print("=" * 55)
for label, df in [("Daily", df_daily), ("Weekly", df_weekly)]:
    vc   = df['injury'].value_counts()
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
    bars   = ax.bar(['Sin lesión', 'Lesión'], counts.values,
                    color=PALETTE, edgecolor='white', width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 200,
                f'{val:,}', ha='center', va='bottom', fontsize=10)
    ax.set_title(f'Distribución de clases — {label}')
    ax.set_ylabel('Número de registros')
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
 
plt.suptitle('Desbalanceo de la variable objetivo', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig01_desbalanceo_clases.png'),
            bbox_inches='tight')
plt.show()
print("\n  -> Guardada: fig01_desbalanceo_clases.png")
 
# =============================================================================
# 4. ANÁLISIS POR ATLETA
# =============================================================================
 
print("\n" + "=" * 55)
print("ANÁLISIS POR ATLETA")
print("=" * 55)
 
obs_por_atleta      = df_weekly.groupby('Athlete ID').size()
lesiones_por_atleta = df_weekly.groupby('Athlete ID')['injury'].sum()
 
print(f"\n  Observaciones semanales por atleta:")
print(f"    Media: {obs_por_atleta.mean():.0f}  |  "
      f"Min: {obs_por_atleta.min()}  |  Max: {obs_por_atleta.max()}")
print(f"\n  Semanas con lesion por atleta:")
print(f"    Media: {lesiones_por_atleta.mean():.1f}  |  "
      f"Max: {lesiones_por_atleta.max()}")
print(f"    Atletas con 0 lesiones: {(lesiones_por_atleta==0).sum()}")
print(f"    Atletas con 1+ lesion:  {(lesiones_por_atleta>=1).sum()}")
 
# Figura 2 — Distribución por atleta
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
axes[0].bar(range(len(obs_por_atleta)),
            obs_por_atleta.sort_values(ascending=False).values,
            color='#2196F3', alpha=0.8)
axes[0].set_title('Semanas de seguimiento por atleta')
axes[0].set_xlabel('Atleta (ordenado)')
axes[0].set_ylabel('Numero de semanas')
 
axes[1].bar(range(len(lesiones_por_atleta)),
            lesiones_por_atleta.sort_values(ascending=False).values,
            color='#F44336', alpha=0.8)
axes[1].set_title('Semanas con lesion por atleta')
axes[1].set_xlabel('Atleta (ordenado)')
axes[1].set_ylabel('Semanas con lesion')
 
plt.suptitle('Heterogeneidad entre atletas', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig02_atletas.png'), bbox_inches='tight')
plt.show()
print("  -> Guardada: fig02_atletas.png")
 
# =============================================================================
# 5. CÁLCULO DEL ACWR
# =============================================================================
 
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
df_weekly['ACWR'] = df_weekly['ACWR'].clip(upper=3.0)
 
print("\n" + "=" * 55)
print("CALCULO DEL ACWR")
print("=" * 55)
print(f"\n  ACWR calculado sobre "
      f"{df_weekly['ACWR'].notna().sum():,} registros")
print(f"  NaN (carga cronica = 0): {df_weekly['ACWR'].isna().sum()}")
print(f"\n  Distribucion ACWR:")
print(df_weekly['ACWR'].describe().round(3).to_string())
 
acwr_by_injury = df_weekly.groupby('injury')['ACWR'].mean()
print(f"\n  ACWR medio sin lesion: {acwr_by_injury[0]:.3f}")
print(f"  ACWR medio con lesion: {acwr_by_injury[1]:.3f}")
print(f"  Diferencia relativa: "
      f"{(acwr_by_injury[1]-acwr_by_injury[0])/acwr_by_injury[0]*100:+.1f}%")
 
# =============================================================================
# 6. COMPARACIÓN DE VARIABLES CLAVE POR CLASE
# =============================================================================
 
features_analisis = {
    'total kms':           'Kilometros totales',
    'ACWR':                'ACWR',
    'avg exertion':        'Esfuerzo percibido (media)',
    'avg recovery':        'Recuperacion percibida (media)',
    'nr. rest days':       'Dias de descanso',
    'nr. tough sessions (effort in Z5, T1 or T2)': 'Sesiones duras (Z5/T1/T2)',
    'max km one day':      'Maximo km en un dia',
    'total km Z5-T1-T2':   'km en zonas alta intensidad',
}
 
print("\n" + "=" * 55)
print("COMPARACION DE VARIABLES POR CLASE")
print("=" * 55)
print(f"\n  {'Variable':<45} {'Sin lesion':>12} "
      f"{'Lesion':>12} {'D%':>8}")
print("  " + "-" * 80)
for col, label in features_analisis.items():
    m0   = df_weekly[df_weekly['injury']==0][col].mean()
    m1   = df_weekly[df_weekly['injury']==1][col].mean()
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
    bp = ax.boxplot(
        [data_0, data_1],
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='.', markersize=2, alpha=0.3),
        whiskerprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2))
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Sin lesion', 'Lesion'], fontsize=9)
    ax.set_title(label, fontsize=10)
 
plt.suptitle('Distribucion de variables clave por clase de lesion',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig03_boxplots_variables.png'),
            bbox_inches='tight')
plt.show()
print("\n  -> Guardada: fig03_boxplots_variables.png")
 
# =============================================================================
# 7. DISTRIBUCIÓN DEL ACWR POR CLASE
# =============================================================================
 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
for val, label, color in [
        (0, 'Sin lesion', '#2196F3'), (1, 'Lesion', '#F44336')]:
    data = df_weekly[df_weekly['injury']==val]['ACWR'].dropna()
    axes[0].hist(data, bins=50, alpha=0.6, color=color,
                 label=f'{label} (n={len(data):,})', density=True)
axes[0].axvline(1.0, color='gray',   linestyle='--',
                linewidth=1, label='ACWR = 1.0')
axes[0].axvline(1.5, color='orange', linestyle='--',
                linewidth=1, label='ACWR = 1.5')
axes[0].set_xlabel('ACWR')
axes[0].set_ylabel('Densidad')
axes[0].set_title('Distribucion del ACWR por clase')
axes[0].legend(fontsize=9)
 
bins        = [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.01]
labels_bins = ['<0.5', '0.5-0.8', '0.8-1.0',
               '1.0-1.2', '1.2-1.5', '1.5-2.0', '>2.0']
df_weekly['ACWR_bin'] = pd.cut(df_weekly['ACWR'],
                                bins=bins, labels=labels_bins)
injury_rate_by_acwr = (df_weekly
                       .groupby('ACWR_bin', observed=True)['injury']
                       .mean() * 100)
 
axes[1].bar(injury_rate_by_acwr.index, injury_rate_by_acwr.values,
            color='#E91E63', alpha=0.8, edgecolor='white')
axes[1].set_xlabel('Rango de ACWR')
axes[1].set_ylabel('Tasa de lesion (%)')
axes[1].set_title('Tasa de lesion por rango de ACWR')
axes[1].yaxis.set_major_formatter(PercentFormatter(decimals=2))
 
plt.suptitle('Analisis del ACWR y su relacion con la lesion',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig04_ACWR_analisis.png'),
            bbox_inches='tight')
plt.show()
print("  -> Guardada: fig04_ACWR_analisis.png")
 
# =============================================================================
# 8. CORRELACIONES CON LA VARIABLE OBJETIVO
# =============================================================================
 
base_numeric_cols = [
    c for c in df_weekly.columns
    if '.' not in c
    and c not in ['Athlete ID', 'Date', 'injury',
                  'ACWR_bin', 'chronic_kms']
    and df_weekly[c].dtype in [np.float64, np.int64]
]
 
corr_with_injury = (
    df_weekly[base_numeric_cols + ['ACWR', 'injury']]
    .corr()['injury']
    .drop('injury')
    .sort_values(key=abs, ascending=False)
)
 
print("\n" + "=" * 55)
print("TOP 10 CORRELACIONES CON injury")
print("=" * 55)
print(corr_with_injury.head(10).round(4).to_string())
 
top_corr    = corr_with_injury.head(15)
colors_corr = ['#F44336' if v > 0 else '#2196F3'
               for v in top_corr.values]
 
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(top_corr)), top_corr.values,
        color=colors_corr, alpha=0.8)
ax.set_yticks(range(len(top_corr)))
ax.set_yticklabels(top_corr.index, fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Correlacion de Pearson con injury')
ax.set_title('Top 15 variables mas correlacionadas con la lesion',
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig05_correlaciones.png'),
            bbox_inches='tight')
plt.show()
print("\n  -> Guardada: fig05_correlaciones.png")
 
# =============================================================================
# 9. EVOLUCIÓN TEMPORAL DE LESIONES
# =============================================================================
 
df_weekly['date_bin'] = pd.cut(df_weekly['Date'], bins=27)
temporal = (df_weekly
            .groupby('date_bin', observed=True)['injury']
            .mean() * 100)
 
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(len(temporal)), temporal.values,
        color='#F44336', linewidth=2, marker='o', markersize=4)
ax.fill_between(range(len(temporal)), temporal.values,
                alpha=0.15, color='#F44336')
ax.set_xlabel('Periodo del estudio (progresivo)')
ax.set_ylabel('Tasa de lesion (%)')
ax.set_title('Evolucion temporal de la tasa de lesion')
ax.yaxis.set_major_formatter(PercentFormatter(decimals=2))
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig06_evolucion_temporal.png'),
            bbox_inches='tight')
plt.show()
print("  -> Guardada: fig06_evolucion_temporal.png")
 
# =============================================================================
# 10. GUARDAR DATASET WEEKLY CON ACWR PARA BLOQUE 2
# =============================================================================
 
PATH_OUT = os.path.join(BASE_DIR, "weekly_con_acwr.csv")
df_weekly.to_csv(PATH_OUT, index=False)
print(f"\n  -> Dataset guardado: weekly_con_acwr.csv")
 
print("\n" + "=" * 55)
print("EDA COMPLETADO — 6 figuras generadas")
print("=" * 55)