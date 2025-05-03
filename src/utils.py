import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score

#==========================================================# 

'''
Buena parte de este codigo es generado por chatgpt, su origen se debe a funciones repetitivas las cuales se copiaban y pegaban varias veces a lo largo del proyecto, y se le pasaron
a la IA para que fueran funciones.  pedimos que estas fueran convertidas en funciones reusables con estructuras de control que se fueron iterando al momento que se encontraban errores
'''


#==========================================================#

def split_dataset(df_input, output_dir: str, target_col: str):
    """
    Divide un dataset en conjuntos de entrenamiento (70%), validación (15%) y prueba (15%),
    estratificando por la variable objetivo. Guarda los resultados como CSV y ZIP.

    Parámetros:
        df_input (DataFrame o ndarray): Datos de entrada.
        output_dir (str): Carpeta donde se guardarán los archivos de salida.
        target_col (str): Nombre de la columna usada para estratificar.
    """
    if isinstance(df_input, np.ndarray):
        df_raw = pd.DataFrame(df_input)
    else:
        df_raw = df_input.copy()

    if target_col not in df_raw.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no se encuentra en el DataFrame.")

    os.makedirs(output_dir, exist_ok=True)

    # División inicial: 70% entrenamiento, 30% resto
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    for train_idx, rest_idx in split1.split(df_raw, df_raw[target_col]):
        train_df = df_raw.iloc[train_idx].reset_index(drop=True)
        rest_df = df_raw.iloc[rest_idx].reset_index(drop=True)

    # División del resto: 50% validación, 50% prueba → cada uno 15% del total
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for val_idx, test_idx in split2.split(rest_df, rest_df[target_col]):
        val_df = rest_df.iloc[val_idx].reset_index(drop=True)
        test_df = rest_df.iloc[test_idx].reset_index(drop=True)

    # Guardar como CSV
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Crear ZIPs
    def crear_zip(csv_path):
        zip_path = csv_path.replace('.csv', '.zip')
        with ZipFile(zip_path, 'w') as zf:
            zf.write(csv_path, arcname=os.path.basename(csv_path))

    crear_zip(train_path)
    crear_zip(val_path)
    crear_zip(test_path)

    # Mostrar proporciones
    total = df_raw.shape[0]
    print("Filas totales:", total)
    print("Train:", train_df.shape[0], f"({train_df.shape[0]/total:.2%})")
    print("Validación:", val_df.shape[0], f"({val_df.shape[0]/total:.2%})")
    print("Prueba:", test_df.shape[0], f"({test_df.shape[0]/total:.2%})")

def plot_confusion_matrix(y_true, y_pred, dataset_name='Dataset'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm,annot=True, cmap='Blues', fmt='.100g') #.100g para que me de las 100 cifras relevantes a la derecha del numero
    plt.title(f'{dataset_name} - Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, dataset_name='Dataset'):
    # Calculamos la Tasa de Falsos Positivos (fpr) y la Tasa de Verdaderos Positivos (tpr) para dibujar la curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
    plt.title(f'{dataset_name} - Curva ROC')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_precision_recall_curve_custom(y_true, y_pred_proba, dataset_name='Dataset'):
    #Usamos las probabilidades predichas para la clase positiva (y_pred_proba) para calcular precisión y recall a distintos umbrales
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.figure(figsize=(6,6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precisión')
    plt.title(f'{dataset_name} - Curva Precisión-Recall')
    plt.grid()
    plt.show()

def plot_feature_importance(model, feature_names=None, max_num_features=10, dataset_name='Dataset'):
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

    elif hasattr(model, 'coef_'):  # Para SVC lineal, regresiones lineales, etc
        importances = np.abs(model.coef_).flatten() # devuelve una copia 1d de el array model.coef_ en valores absolutos

    else:
        print(f'El modelo no tiene feature_importances_ ni coef_. No se puede graficar la importancia.')
        return

    #se crean pares de (feature_names,importances), que son ordenados por el segundo elemento key=lambda x: x[1] de la tupla de forma descendente
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:max_num_features]
    features, scores = zip(*feat_imp) # * hace un 'unpack' de las listas en dos variables

    plt.figure(figsize=(8,6))
    sns.barplot(x=scores, y=features, orient='h', palette='viridis')
    plt.title(f'{dataset_name} - Importancia de Características')
    plt.xlabel('Importancia')
    plt.ylabel('Características')
    plt.tight_layout()
    plt.show()

def plot_all_metrics(y_true, y_pred, model=None, x_features=None, dataset_name='Dataset', feature_names=None):
    y_pred_proba = model.predict_proba(x_features)[:,1] # Obtiene las probabilidades de prediccion para la clase 1 para todas las muestras en x_features

    plot_confusion_matrix(y_true, y_pred, dataset_name)
    plot_roc_curve(y_true, y_pred_proba, dataset_name)
    plot_precision_recall_curve_custom(y_true, y_pred_proba, dataset_name)

    if model is not None:
        plot_feature_importance(model, feature_names=feature_names, dataset_name=dataset_name)
