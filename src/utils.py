import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from zipfile import ZipFile
import numpy as np

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
