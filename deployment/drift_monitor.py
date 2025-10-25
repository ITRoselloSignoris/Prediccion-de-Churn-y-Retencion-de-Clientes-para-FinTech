import os
import pandas as pd
import psycopg2 # Para conectar a Supabase
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

DB_CONNECTION_STRING = os.environ.get("SUPABASE_CONNECTION_STRING")
HISTORICAL_DATA_PATH = "deployment/data/historical_data.csv"
OUTPUT_REPORT_PATH = "public/drift_report.html"
NUM_RECENT_PREDICTIONS = 5000

FEATURE_COLUMNS_TO_MONITOR = [
    'creditscore', 'age', 'tenure', 'balance',
    'hascrcard', 'isactivemember', 'estimatedsalary',
    'geography_france', 'geography_germany', 'geography_spain',
    'gender_female', 'gender_male',
    'numofproducts_1', 'numofproducts_2', 'numofproducts_3', 'numofproducts_4'
]
TARGET_COLUMN_NAME = 'Exited'
PREDICTION_COLUMN_NAME = 'prediction'

def load_recent_data(conn_string, num_rows):
    print(f"Cargando {num_rows} predicciones recientes desde Supabase...")
    query = f"""
    SELECT
        creditscore, age, tenure, balance, hascrcard, isactivemember, estimatedsalary,
        geography_france, geography_germany, geography_spain,
        gender_female, gender_male,
        numofproducts_1, numofproducts_2, numofproducts_3, numofproducts_4,
        prediction
    FROM predictions
    ORDER BY timestamp DESC
    LIMIT {num_rows};
    """
    try:
        with psycopg2.connect(conn_string) as conn:
            df = pd.read_sql(query, conn)
        print(f"Se cargaron {len(df)} filas.")
        for col in ['prediction', 'creditscore', 'age', 'tenure', 'balance', 'estimatedsalary']:
             if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in ['hascrcard', 'isactivemember', 'geography_france', 'geography_germany', 'geography_spain',
                     'gender_female', 'gender_male', 'numofproducts_1', 'numofproducts_2', 'numofproducts_3', 'numofproducts_4']:
             if col in df.columns: df[col] = df[col].astype(bool)
        return df
    except Exception as e:
        print(f"Error al cargar datos desde Supabase: {e}")
        return pd.DataFrame()

def load_reference_data(file_path, target_col):
    print(f"Cargando datos de referencia desde: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Se cargaron {len(df)} filas de referencia.")
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(-1).astype(int)
        else:
             print(f"Advertencia: Columna target '{target_col}' no encontrada en {file_path}")
        for col in ['hascrcard', 'isactivemember', 'geography_france', 'geography_germany', 'geography_spain',
                     'gender_female', 'gender_male', 'numofproducts_1', 'numofproducts_2', 'numofproducts_3', 'numofproducts_4']:
              if col in df.columns: df[col] = df[col].astype(bool)
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de referencia en {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return pd.DataFrame()

def generate_drift_report(df_current, df_reference, feature_columns, target_col, prediction_col, output_path):
    print("Generando reporte de drift...")

    if prediction_col not in df_current.columns:
        print(f"Error: Columna de predicción '{prediction_col}' no encontrada en datos actuales.")
        return
    if target_col.lower() not in df_reference.columns:
        print(f"Error: Columna target '{target_col}' no encontrada en datos de referencia.")
        return

    column_mapping = ColumnMapping()
    column_mapping.target = target_col
    column_mapping.prediction = prediction_col

    available_features_current = [col for col in feature_columns if col in df_current.columns]
    available_features_reference = [col for col in feature_columns if col in df_reference.columns]
    final_feature_columns = list(set(available_features_current) & set(available_features_reference))

    metrics_to_run = [TargetDriftPreset()]
    if not final_feature_columns:
        print("Advertencia: No hay columnas de features comunes para DataDriftPreset.")
    else:
        print(f"Columnas de features para DataDriftPreset: {final_feature_columns}")
        metrics_to_run.insert(0, DataDriftPreset(columns=final_feature_columns))

    report = Report(metrics=metrics_to_run)
    report.run(
        current_data=df_current,
        reference_data=df_reference,
        column_mapping=column_mapping
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)
    print(f"Reporte de drift guardado exitosamente en: {output_path}")

if __name__ == "__main__":
    print("--- Iniciando Script de Monitoreo de Drift ---")

    if not DB_CONNECTION_STRING:
        print("Error Crítico: SUPABASE_CONNECTION_STRING no definida.")
        exit()

    df_recent = load_recent_data(DB_CONNECTION_STRING, NUM_RECENT_PREDICTIONS)
    df_hist = load_reference_data(HISTORICAL_DATA_PATH, TARGET_COLUMN_NAME)

    if not df_recent.empty and not df_hist.empty:
        df_recent.columns = df_recent.columns.str.lower()
        df_hist.columns = df_hist.columns.str.lower()

        target_col_lower = TARGET_COLUMN_NAME.lower()
        prediction_col_lower = PREDICTION_COLUMN_NAME.lower()

        if prediction_col_lower not in df_recent.columns:
             print(f"Error: Columna de predicción '{PREDICTION_COLUMN_NAME}' no encontrada en datos recientes.")
        elif target_col_lower not in df_hist.columns:
             print(f"Error: Columna target '{TARGET_COLUMN_NAME}' no encontrada en datos históricos.")
        else:
            print(f"Renombrando '{prediction_col_lower}' a '{target_col_lower}' en df_recent para la comparación.")
            df_recent.rename(columns={prediction_col_lower: target_col_lower}, inplace=True)
            generate_drift_report(df_recent, 
                                  df_hist, 
                                  FEATURE_COLUMNS_TO_MONITOR, 
                                  target_col_lower, 
                                  prediction_col_lower, 
                                  OUTPUT_REPORT_PATH)
    else:
        print("No se generó el reporte: uno o ambos datasets están vacíos o no se pudieron cargar.")

    print("--- Script de Monitoreo de Drift Finalizado ---")