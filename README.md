
# Proyecto Income

## Descripción de Proyecto

Este proyecto tiene como objetivo construir modelos de machine learning capaces de predecir si una persona gana más de 50.000 dólares anuales, utilizando información demográfica y laboral. Se abordaron tareas de limpieza, transformación y análisis exploratorio de datos para preparar el conjunto antes del entrenamiento. Dado que la clase objetivo está moderadamente desbalanceada, se aplicaron estrategias de validación y métricas adecuadas para asegurar un rendimiento robusto.

---

## Dataset

Fuente: [Income Dataset](https://www.kaggle.com/datasets/mastmustu/income)

Cantidad de registros: 32,561 ejemplos

Cantidad de variables: 15 atributos + 1 variable objetivo (`income`)

Variable objetivo: `income` (Binaria, `>50K` o `<=50K`)

Particularidades:

* Contiene tanto variables categóricas como numéricas.

* Algunas variables presentan valores nulos o inconsistentes.

* Requiere codificación adecuada para variables no numéricas.

---

## Preprocesamiento

* Análisis exploratorio para entender la distribución de las variables.

* Eliminación de registros con valores faltantes o anómalos.

* Codificación de variables categóricas mediante OneHotEncoding o LabelEncoding, según el modelo.

* Escalado de variables numéricas mediante StandardScaler o RobustScaler.

* División estratificada en conjuntos de entrenamiento, validación y prueba (70%-15%-15%).

---

## Modelos Utilizados

Regresión Logística: punto de partida con buena interpretabilidad.

Random Forest: mejora el rendimiento con ensamblado de múltiples árboles.

LightGBM: eficiente en entrenamiento y especialmente útil en conjuntos de datos con muchas variables categóricas.

Support Vector Classifier (SVC): probado con diferentes estrategias de escalado y optimización de hiperparámetros.

---

## Cómo Reproducir el Proyecto

* Clona este repositorio.

* Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

* Ejecuta los scripts y notebooks en el siguiente orden:

  * `src/EDA.ipynb` (análisis exploratorio y limpieza outliers)

  * `src/preprocessing.ipynb` (transformaciones, codificacion y split en train,test y validation)


---

## Preprocesamiento

   * División estratificada en train, validation y test (70%-15%-15%).

   *  Escalado: StandardScaler()

   *  Técnicas de balanceo: Random UnderSampling.

--- 
## Modelos Utilizados
  * LightGBM: Modelo de referencia rápido y eficiente.

  * Support Vector Classifier (SVC): Evaluado con y sin Random Undersampling.

 * Random Forest

Cada modelo fue evaluado principalmente con métricas de Recall, F1-Score y AUC-ROC debido al  desbalance.
---
## Cómo Reproducir el Proyecto

  *  Clona este repositorio.

  *  Instala las dependencias necesarias:

````
pip install -r requirements.txt
````

Ejecuta los scripts en el siguiente orden:

    src/EDA.ipynb(Para eliminar outliers)

    src/preprocessing.py (Si ya tienes los zips de train, test y es solo descomprimir y seguir desde aqui

    src/lightgbm_classifier.py, src/support_vector_classifier.py, etc...


```
  income/
  ├──artifacts [modelos ya entrenados]
  ├── data/
  │   └── train_test_split
  │       ├── test.zip [Output de train_test_split]
  │       ├── train.zip [Output de train_test_split]
  │       └── val.zip [Output de train_test_split]
  ├── docs/
  │   ├── instrucciones_prueba_grupal.pdf
  │   ├── data_dict.xlsx
  ├── src/
  |   ├── preprocessing.ipynb [Preprocesamiento de datos y exportacion]
  │   ├── EDA.ipynb [Análisis exploratorio y limpieza de outliers]
  │   ├── LightGBM.ipynb
  │   ├── LightGBM_R.ipynb 
  │   ├── Random_forest.ipynb
  │   ├── logisticregression.ipynb
  │   ├── suport_vector_models
  │   └── utils.py [Funcion de Split del dataset en train, test y val (70%, 15%, 15%)]
  ├── .gitignore [Archivos y carpetas ignoradas por Git]
  ├── requirements.txt [Dependencias del proyecto]
  └── README.md [Documentación general del proyecto]
```
