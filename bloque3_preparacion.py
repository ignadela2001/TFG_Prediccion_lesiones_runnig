# =============================================================================
# TFG — Predicción de lesiones en running
# BLOQUE 3: Preparación del dataset para modelado
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'figure.dpi': 130, 'axes.spines.top': False, 'axes.spines.right': False,
})

# =============================================================================
# 1. CARGA DEL DATASET (output del Bloque 2)
# =============================================================================

df = pd.read_csv("weekly_features_final.csv")
df = df.sort_values(['Athlete ID', 'Date']).reset_index(drop=True)

print("=" * 60)
print("BLOQUE 3 — PREPARACIÓN PARA MODELADO")
print("=" * 60)
print(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# =============================================================================
# 2. SELECCIÓN DE FEATURES
# =============================================================================
# Se seleccionan 38 variables agrupadas en cuatro categorías:
#   A) Variables de la semana actual (carga externa e interna)
#   B) Variables de la semana anterior (lag -1)
#   C) Variables de hace dos semanas (lag -2)
#   D) Variables derivadas construidas en el Bloque 2

BASE_FEATURES = [
    'nr. sessions', 'nr. rest days', 'total kms', 'max km one day',
    'total km Z3-Z4-Z5-T1-T2', 'nr. tough sessions (effort in Z5, T1 or T2)',
    'nr. days with interval session', 'total km Z3-4', 'total km Z5-T1-T2',
    'total hours alternative training', 'nr. strength trainings',
    'avg exertion', 'min exertion', 'max exertion',
    'avg training success', 'min training success', 'max training success',
    'avg recovery', 'min recovery', 'max recovery',
]

LAG1_FEATURES = [
    'total kms.1', 'avg exertion.1', 'avg recovery.1',
    'nr. sessions.1', 'nr. rest days.1',
    'nr. tough sessions (effort in Z5, T1 or T2).1',
    'total km Z5-T1-T2.1',
]

LAG2_FEATURES = [
    'total kms.2', 'avg exertion.2', 'avg recovery.2',
]

DERIVED_FEATURES = [
    'ACWR', 'ACWR_exertion', 'monotony', 'strain',
    'km_change_w0_w1', 'km_change_w1_w2',
    'recovery_trend', 'exertion_trend',
]

ALL_FEATURES = BASE_FEATURES + LAG1_FEATURES + LAG2_FEATURES + DERIVED_FEATURES

print(f"\nFeatures seleccionadas: {len(ALL_FEATURES)}")
print(f"  · Semana actual (base): {len(BASE_FEATURES)}")
print(f"  · Lag semana -1:        {len(LAG1_FEATURES)}")
print(f"  · Lag semana -2:        {len(LAG2_FEATURES)}")
print(f"  · Variables derivadas:  {len(DERIVED_FEATURES)}")

# =============================================================================
# 3. IMPUTACIÓN DE VALORES NULOS
# =============================================================================
# Los NaN se concentran en las variables derivadas y aparecen principalmente
# al inicio del seguimiento de cada atleta, cuando no hay semanas previas
# suficientes para calcular la carga crónica.
#
# Estrategia: imputación con la mediana por atleta para respetar las
# diferencias individuales. Si un atleta tiene todos NaN en una variable
# (caso extremo), se imputa con la mediana global.

print(f"\nNaN antes de imputación: {df[ALL_FEATURES].isnull().sum().sum():,}")

for feat in DERIVED_FEATURES:
    if df[feat].isna().sum() > 0:
        df[feat] = df.groupby('Athlete ID')[feat].transform(
            lambda x: x.fillna(x.median())
        )
        df[feat] = df[feat].fillna(df[feat].median())  # fallback global

print(f"NaN después de imputación: {df[ALL_FEATURES].isnull().sum().sum():,}")

# =============================================================================
# 4. SPLIT TEMPORAL — TRAIN / TEST
# =============================================================================
# En datos longitudinales no se puede usar un split aleatorio porque
# implicaría entrenar con datos del futuro para predecir el pasado.
# Se usa validación temporal: los primeros 70% del periodo → train,
# el último 30% → test. Esto simula el uso real del modelo (predecir
# con datos históricos observaciones futuras).

SPLIT_DATE = df['Date'].quantile(0.70)

train_mask = df['Date'] <= SPLIT_DATE
test_mask  = df['Date'] >  SPLIT_DATE

X_train = df.loc[train_mask, ALL_FEATURES].copy()
X_test  = df.loc[test_mask,  ALL_FEATURES].copy()
y_train = df.loc[train_mask, 'injury'].copy()
y_test  = df.loc[test_mask,  'injury'].copy()

print(f"\n── Split temporal (cutoff: Date ≤ {SPLIT_DATE:.0f}) ──")
print(f"  Train: {len(X_train):,} obs. | lesiones: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"  Test:  {len(X_test):,} obs.  | lesiones: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# Ratio de desbalanceo en train (necesario para XGBoost)
SCALE_POS_WEIGHT = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n  Ratio desbalanceo train:  1 : {SCALE_POS_WEIGHT:.0f}")
print(f"  scale_pos_weight (XGB):   {SCALE_POS_WEIGHT:.1f}")

# =============================================================================
# 5. ESTANDARIZACIÓN
# =============================================================================
# Se estandarizan las features (media 0, desviación típica 1) para la
# regresión logística, que es sensible a la escala de las variables.
# Random Forest y XGBoost son invariantes a la escala, pero se les pasará
# el dataset estandarizado también para mantener coherencia comparativa.
#
# IMPORTANTE: el scaler se ajusta SOLO sobre train para evitar data leakage.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=ALL_FEATURES, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=ALL_FEATURES, index=X_test.index
)

print(f"\nEstandarización aplicada (fit solo sobre train).")
print(f"  Media train post-scaling (muestra): "
      f"{X_train_scaled['total kms'].mean():.4f}")
print(f"  Std  train post-scaling (muestra): "
      f"{X_train_scaled['total kms'].std():.4f}")

# =============================================================================
# 6. FIGURA — ESQUEMA DEL SPLIT TEMPORAL
# =============================================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Panel 1: volumen de observaciones por fecha
obs_by_date = df.groupby('Date').size()
axes[0].fill_between(obs_by_date.index, obs_by_date.values,
                     where=obs_by_date.index <= SPLIT_DATE,
                     alpha=0.6, color='#2196F3', label='Train (70%)')
axes[0].fill_between(obs_by_date.index, obs_by_date.values,
                     where=obs_by_date.index > SPLIT_DATE,
                     alpha=0.6, color='#FF9800', label='Test (30%)')
axes[0].axvline(SPLIT_DATE, color='black', linestyle='--', linewidth=1.5)
axes[0].set_ylabel('Observaciones / día')
axes[0].set_title('Distribución temporal del dataset y split train/test')
axes[0].legend(loc='upper left')

# Panel 2: tasa de lesión por fecha
inj_by_date = df.groupby('Date')['injury'].mean() * 100
smooth = inj_by_date.rolling(window=30, min_periods=1).mean()
axes[1].plot(smooth.index, smooth.values, color='#F44336', linewidth=1.5)
axes[1].fill_between(smooth.index, smooth.values,
                     where=smooth.index <= SPLIT_DATE,
                     alpha=0.15, color='#2196F3')
axes[1].fill_between(smooth.index, smooth.values,
                     where=smooth.index > SPLIT_DATE,
                     alpha=0.15, color='#FF9800')
axes[1].axvline(SPLIT_DATE, color='black', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Índice temporal (días desde inicio del estudio)')
axes[1].set_ylabel('Tasa de lesión (%) [media móvil 30d]')
axes[1].set_title('Tasa de lesión a lo largo del tiempo')

plt.tight_layout()
plt.savefig('fig09_split_temporal.png', bbox_inches='tight')
plt.close()
print("\n  → Guardada: fig09_split_temporal.png")

# =============================================================================
# 7. EXPORTAR OBJETOS PARA BLOQUE 4
# =============================================================================

import pickle

objects = {
    'X_train': X_train,
    'X_test':  X_test,
    'X_train_scaled': X_train_scaled,
    'X_test_scaled':  X_test_scaled,
    'y_train': y_train,
    'y_test':  y_test,
    'ALL_FEATURES': ALL_FEATURES,
    'SCALE_POS_WEIGHT': SCALE_POS_WEIGHT,
    'scaler': scaler,
}

with open('datos_modelado.pkl', 'wb') as f:
    pickle.dump(objects, f)

print("  → Guardado: datos_modelado.pkl")

# =============================================================================
# 8. RESUMEN FINAL
# =============================================================================

print("\n" + "=" * 60)
print("RESUMEN BLOQUE 3")
print("=" * 60)
print(f"  Features totales:         {len(ALL_FEATURES)}")
print(f"  Observaciones train:      {len(X_train):,}")
print(f"  Observaciones test:       {len(X_test):,}")
print(f"  Tasa lesión train:        {y_train.mean()*100:.2f}%")
print(f"  Tasa lesión test:         {y_test.mean()*100:.2f}%")
print(f"  Ratio desbalanceo:        1 : {SCALE_POS_WEIGHT:.0f}")
print(f"  Valores nulos restantes:  0")
print("=" * 60)
print("BLOQUE 3 COMPLETADO — listo para modelado")
