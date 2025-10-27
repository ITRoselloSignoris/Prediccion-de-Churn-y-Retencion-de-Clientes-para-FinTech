import os
import pandas as pd
import psycopg2  # Para conectar a Supabase
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
import json
import datetime

DB_CONNECTION_STRING = os.environ.get("SUPABASE_CONNECTION_STRING")
HISTORICAL_DATA_PATH = "deployment/data/historical_data.csv"
OUTPUT_REPORT_PATH = "public/drift_report.html"
OUTPUT_STATUS_PATH = "public/drift_status.json"
NUM_RECENT_PREDICTIONS = 5000 

FEATURE_COLUMNS_TO_MONITOR = [
    'creditscore', 'age', 'tenure', 'balance',
    'hascrcard', 'isactivemember', 'estimatedsalary',
    'geography_france', 'geography_germany', 'geography_spain',
    'gender_female', 'gender_male',
    'numofproducts_1', 'numofproducts_2', 'numofproducts_3', 'numofproducts_4'
]
TARGET_COLUMN_NAME = 'Exited' # Aún necesario para cargar df_hist correctamente
PREDICTION_COLUMN_NAME = 'prediction'


# --- Funciones para cargar datos ---
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
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in [
            'hascrcard', 'isactivemember', 'geography_france', 'geography_germany', 'geography_spain',
            'gender_female', 'gender_male', 'numofproducts_1', 'numofproducts_2', 'numofproducts_3', 'numofproducts_4'
        ]:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        return df
    except Exception as e:
        print(f"Error al cargar datos desde Supabase: {e}")
        return pd.DataFrame()


def load_reference_data(file_path, target_col_name_ref): 
    print(f"Cargando datos de referencia desde: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Se cargaron {len(df)} filas de referencia.")
        df_cols_lower = [col.lower() for col in df.columns]

        # Procesar columna target si existe (necesaria para entrenamiento, no para este drift)
        if target_col_name_ref.lower() in df_cols_lower:
            original_target_col = df.columns[df_cols_lower.index(target_col_name_ref.lower())]
            # Convertir a numérico pero no necesariamente renombrar si no se usa después
            df[original_target_col] = pd.to_numeric(df[original_target_col], errors='coerce').fillna(-1).astype(int)
        else:
            print(f"Advertencia: Columna target '{target_col_name_ref}' no encontrada en {file_path}")

        # Procesar columnas booleanas
        bool_cols = [
            'hascrcard', 'isactivemember', 'geography_france', 'geography_germany',
            'geography_spain', 'gender_female', 'gender_male',
            'numofproducts_1', 'numofproducts_2', 'numofproducts_3', 'numofproducts_4'
        ]
        for col in bool_cols:
            if col in df_cols_lower:
                original_bool_col = df.columns[df_cols_lower.index(col)]
                df[original_bool_col] = df[original_bool_col].astype(bool)

        df.columns = df.columns.str.lower() # Asegura minúsculas al final
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de referencia en {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return pd.DataFrame()


# --- Generación del reporte de drift (SOLO DATA DRIFT) ---
def generate_drift_report(df_current, df_reference, feature_columns, output_path_html):
    print("Generando reporte de Data Drift...")

    column_mapping = ColumnMapping()

    available_features_current = [col for col in feature_columns if col in df_current.columns]
    available_features_reference = [col for col in feature_columns if col in df_reference.columns]

    features_for_data_drift = list(set(available_features_current) & set(available_features_reference))

    # Asegurarse que 'prediction' NO esté en la lista para DataDriftPreset
    if PREDICTION_COLUMN_NAME.lower() in features_for_data_drift:
        features_for_data_drift.remove(PREDICTION_COLUMN_NAME.lower())
    # Asegurarse que 'exited' NO esté en la lista para DataDriftPreset 
    if TARGET_COLUMN_NAME.lower() in features_for_data_drift:
         features_for_data_drift.remove(TARGET_COLUMN_NAME.lower())

    if not features_for_data_drift:
        print("Advertencia: No hay features comunes (excluyendo target/prediction) entre datasets.")
        return None # No se puede generar reporte sin features

    print(f"Features para DataDriftPreset: {features_for_data_drift}")

    # Solo DataDriftPreset para las features
    metrics_list = [DataDriftPreset(columns=features_for_data_drift)]

    report = Report(metrics=metrics_list)
    try:
        report.run(current_data=df_current, reference_data=df_reference, column_mapping=column_mapping)
    except Exception as e:
        print(f"Error durante report.run(): {e}")
        return None # El reporte falló

    # Guardar HTML
    try:
        os.makedirs(os.path.dirname(output_path_html), exist_ok=True)
        report.save_html(output_path_html)
        print(f"Reporte HTML guardado en: {output_path_html}")
    except Exception as e:
        print(f"Error al guardar reporte HTML: {e}")

    # Extraer resultados usando as_dict()
    try:
        report_dict = report.as_dict()

        data_drift_results_dict = None
        for metric_output in report_dict.get('metrics', []):
            if metric_output.get('metric') == 'DataDriftPreset':
                data_drift_results_dict = metric_output.get('result', {})
                break

        if data_drift_results_dict is None:
             print("Advertencia: No se encontraron resultados para DataDriftPreset.")
             results = { "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "data_drift_detected": False, "drifted_features_count": 0, "drifted_features_list": [] }
             return results

        drifted_list = data_drift_results_dict.get('drifted_columns', []) or []

        results = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "data_drift_detected": data_drift_results_dict.get('dataset_drift', False),
            "drifted_features_count": data_drift_results_dict.get('number_of_drifted_columns', 0),
            "drifted_features_list": drifted_list
        }
        return results
    except Exception as e:
        print(f"Error al extraer resultados usando as_dict(): {e}")
        return None


# --- Bloque principal ---
if __name__ == "__main__":
    print("--- Iniciando Script de Monitoreo de Drift ---")

    if not DB_CONNECTION_STRING:
        print("Error Crítico: SUPABASE_CONNECTION_STRING no definida.")
        exit()

    df_recent = load_recent_data(DB_CONNECTION_STRING, NUM_RECENT_PREDICTIONS)
    df_hist = load_reference_data(HISTORICAL_DATA_PATH, TARGET_COLUMN_NAME) # Pasamos 'Exited'

    # Convertir columnas a minúsculas DESPUÉS de cargar
    if not df_recent.empty:
        df_recent.columns = df_recent.columns.str.lower()
    if not df_hist.empty:
        df_hist.columns = df_hist.columns.str.lower()

    if not df_recent.empty and not df_hist.empty:
        target_col_lower = TARGET_COLUMN_NAME.lower() # Sigue siendo útil para la carga de df_hist
        prediction_col_lower = PREDICTION_COLUMN_NAME.lower() # Sigue siendo útil para quitarla de features

        # Quitar la columna target ('exited') si existe en df_hist
        if target_col_lower in df_hist.columns:
            df_hist = df_hist.drop(columns=[target_col_lower], errors='ignore')

        # Quitar la columna prediction ('prediction') si existe en df_recent
        if prediction_col_lower in df_recent.columns:
             df_recent_for_report = df_recent.drop(columns=[prediction_col_lower], errors='ignore')
        else:
             df_recent_for_report = df_recent

        drift_results = generate_drift_report(
            df_recent_for_report, # Usar df sin prediction si existe
            df_hist, # df_hist sin exited
            FEATURE_COLUMNS_TO_MONITOR,
            OUTPUT_REPORT_PATH
        )

        if drift_results:
            try:
                os.makedirs(os.path.dirname(OUTPUT_STATUS_PATH), exist_ok=True)
                with open(OUTPUT_STATUS_PATH, "w") as f:
                    json.dump(drift_results, f, indent=4)
                print(f"Estado de drift guardado en: {OUTPUT_STATUS_PATH}")

                if drift_results.get("data_drift_detected", False): # Usar .get
                    print("¡Drift detectado! Estado guardado en JSON.")
                else:
                    print("No se detectó drift significativo.")
            except Exception as e:
                print(f"Error al guardar el estado de drift en JSON: {e}")
        else:
            print("No se pudieron obtener los resultados del drift para guardar en JSON.")
    else:
        print("No se generó el reporte: datasets vacíos o error de carga.")

    print("--- Script de Monitoreo de Drift Finalizado ---")