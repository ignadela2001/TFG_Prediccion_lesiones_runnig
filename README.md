# Predicción de Lesiones en Corredores mediante Machine Learning

Trabajo de Fin de Grado — Inteligencia de Negocios, Universidad CEU San Pablo (2026)

## Resumen

Modelo predictivo para anticipar el riesgo de lesión en corredores de fondo a partir de datos de entrenamiento longitudinales. Se comparan tres algoritmos de clasificación (Regresión Logística, Random Forest y XGBoost) sobre un dataset de 74 atletas y aproximadamente 43.000 observaciones.

**Resultado clave:** la incorporación de features autorregresivas (histórico de lesiones, rachas de lesiones, semanas sin lesión) mejoró sustancialmente la capacidad predictiva del modelo Random Forest, elevando el F1-score de 0.044 a 0.308 y el ROC-AUC a 0.73.

## Motivación

Como corredor de fondo, la lesión es uno de los mayores riesgos que puede truncar meses de preparación. La mayoría de los modelos de predicción de lesiones deportivas fallan por un motivo estructural: las lesiones son eventos raros (en este dataset, apenas el 1.16% de las observaciones), lo que hace que los modelos estándar tiendan a ignorarlas por completo. Este proyecto nace de la intersección entre mi interés personal por el rendimiento deportivo basado en datos y mi formación en analítica: el objetivo no era solo entrenar un modelo más, sino entender qué tipo de información temporal permite anticipar el riesgo antes de que ocurra.

## 📊 Dataset

- 74 atletas, aproximadamente 43.000 observaciones (series temporales diarias y semanales)
- Tasa de lesión: 1.16% — dataset fuertemente desbalanceado, uno de los principales retos técnicos del proyecto
- Variables principales: carga de entrenamiento, ACWR (Acute:Chronic Workload Ratio), métricas de rendimiento y volumen, historial de lesiones por atleta
- Datos utilizados con fines exclusivamente académicos

## Metodología

1. **Análisis exploratorio de datos (EDA)**: distribución de clases, comportamiento por atleta, boxplots de variables, correlaciones y evolución temporal de las métricas de carga.
2. **Feature engineering**: cálculo de ACWR y construcción de variables autorregresivas — lesión previa (lag), lesiones acumuladas, racha de lesiones y semanas consecutivas sin lesión.
3. **Modelado**: entrenamiento y comparación de Regresión Logística, Random Forest y XGBoost.
4. **Validación**: split temporal (no aleatorio) para evitar fuga de información entre pasado y futuro, replicando un escenario realista de predicción.
5. **Evaluación**: matrices de confusión, curvas ROC y Precision-Recall, análisis de umbral de decisión (threshold), importancia de variables (feature importance) y análisis de errores.

## Resultados

El hallazgo principal del proyecto es que las **variables autorregresivas** —es decir, el historial reciente de lesiones de cada atleta— son, con diferencia, las más determinantes para anticipar una lesión futura, por encima de las métricas de carga de entrenamiento tomadas de forma aislada.

Modelo destacado — Random Forest:

| Configuración | F1-score | ROC-AUC |
|---|---|---|
| Sin features autorregresivas (baseline) | 0.044 | — |
| Con features autorregresivas | 0.308 | 0.73 |

Esta mejora (F1 multiplicado por 7) confirma que, en problemas de predicción de lesiones deportivas, el contexto temporal del propio atleta aporta mucha más señal que las variables de carga puntuales.

## Contenido del repositorio

- `TFG_Prediccion_Lesiones.ipynb` — Notebook principal con todo el análisis, de principio a fin
- `timeseries (daily).csv` / `timeseries (weekly).csv` — Series temporales de entrenamiento (diaria y semanal)
- `weekly_con_acwr.csv` — Serie semanal con el ACWR ya calculado
- `Graf01–16_*.png` — Gráficas generadas durante el análisis exploratorio y la evaluación de modelos (desbalanceo de clases, correlaciones, curvas ROC/PR, importancia de variables, análisis de umbral, etc.)

## Tecnologías

Python · pandas · scikit-learn · XGBoost · matplotlib / seaborn

## Autor

Ignacio de la Cruz Herranz
[www.linkedin.com/in/ignacio-de-la-cruz-herranz] · [Ignadelacruzherranz@gmail.com]

## Procedencia del dataset

El dataset utilizado en este proyecto **no es de elaboración propia**. Corresponde a los datos de replicación publicados junto al estudio:

> Lövdal, S., Den Hartigh, R. J. R., & Azzopardi, G. (2021). Injury Prediction in Competitive Runners With Machine Learning. *International Journal of Sports Physiology and Performance*, 16(10), 1522–1531.

Los datos, publicados por la Universidad de Groningen (Países Bajos), recogen el registro de entrenamiento detallado de un equipo de atletismo de alto nivel neerlandés a lo largo de siete temporadas (2012–2019), incluyendo a corredores de medio fondo y fondo (800 m a maratón). El estudio original fue aprobado por el comité de ética correspondiente (código de investigación: PSY-1920-S-0007) y los datos se distribuyen públicamente con fines de investigación y replicación.

- **DOI del dataset:** [10.34894/UWU9PV](https://doi.org/10.34894/uwu9pv)
- **Editor:** University of Groningen

Este TFG utiliza dicho dataset como base para un análisis y modelado propios, con una aproximación metodológica adicional centrada en las features autorregresivas descritas más arriba.

## Nota

Proyecto desarrollado con fines exclusivamente académicos como Trabajo de Fin de Grado. El dataset pertenece a sus autores originales (ver sección "Procedencia del dataset") y se utiliza aquí únicamente con fines de análisis y aprendizaje; no se reclama autoría sobre los datos, solo sobre el análisis, el modelado y el código desarrollados en este repositorio.
