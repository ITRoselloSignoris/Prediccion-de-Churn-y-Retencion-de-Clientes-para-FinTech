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
TARGET_COLUMN_NAME = 'Exited'
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


def load_reference_data(file_path, target_col):
    print(f"Cargando datos de referencia desde: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Se cargaron {len(df)} filas de referencia.")
        df_cols_lower = [col.lower() for col in df.columns]
        if target_col.lower() in df_cols_lower:
            original_target_col = df.columns[df_cols_lower.index(target_col.lower())]
            df[original_target_col] = pd.to_numeric(df[original_target_col], errors='coerce').fillna(-1).astype(int)
            df.rename(columns={original_target_col: target_col.lower()}, inplace=True)
        else:
            print(f"Advertencia: Columna target '{target_col}' no encontrada en {file_path}")

        bool_cols = [
            'hascrcard', 'isactivemember', 'geography_france', 'geography_germany',
            'geography_spain', 'gender_female', 'gender_male',
            'numofproducts_1', 'numofproducts_2', 'numofproducts_3', 'numofproducts_4'
        ]
        for col in bool_cols:
            if col in df_cols_lower:
                original_bool_col = df.columns[df_cols_lower.index(col)]
                df[original_bool_col] = df[original_bool_col].astype(bool)
                df.rename(columns={original_bool_col: col}, inplace=True)

        df.columns = df.columns.str.lower()
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de referencia en {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return pd.DataFrame()


# --- Generación del reporte de drift ---
def generate_drift_report(df_current, df_reference, feature_columns, prediction_col, output_path_html):
    print("Generando reporte de drift...")

    if prediction_col not in df_current.columns:
        print(f"Error: Columna '{prediction_col}' no encontrada en datos recientes.")
        return None

    column_mapping = ColumnMapping(prediction=prediction_col)

    available_features_current = [col for col in feature_columns if col in df_current.columns]
    available_features_reference = [col for col in feature_columns if col in df_reference.columns]

    features_for_data_drift = list(set(available_features_current) & set(available_features_reference))
    if not features_for_data_drift:
        print("Advertencia: No hay features comunes entre datasets.")
        return None

    # Solo DataDriftPreset
    report = Report(metrics=[DataDriftPreset(columns=features_for_data_drift)])
    report.run(current_data=df_current, reference_data=df_reference, column_mapping=column_mapping)

    # Guardar HTML
    os.makedirs(os.path.dirname(output_path_html), exist_ok=True)
    report.save_html(output_path_html)
    print(f"Reporte HTML guardado en: {output_path_html}")

    # Extraer resultados
    try:
        report_dict = report.as_dict()
        data_drift_metrics = report_dict.get('data_drift', {}).get('data', {}).get('metrics', {})
        drifted_list = data_drift_metrics.get('drifted_columns', []) or []
        results = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "data_drift_detected": data_drift_metrics.get('drift_detected', False),
            "drifted_features_count": data_drift_metrics.get('number_of_drifted_columns', 0),
            "drifted_features_list": drifted_list
        }
        return results
    except Exception as e:
        print(f"Error al extraer resultados: {e}")
        return None


# --- Bloque principal ---
if __name__ == "__main__":
    print("--- Iniciando Script de Monitoreo de Drift ---")

    if not DB_CONNECTION_STRING:
        print("Error Crítico: SUPABASE_CONNECTION_STRING no definida.")
        exit()

    df_recent = load_recent_data(DB_CONNECTION_STRING, NUM_RECENT_PREDICTIONS)
    df_hist = load_reference_data(HISTORICAL_DATA_PATH, TARGET_COLUMN_NAME)

    if not df_recent.empty:
        df_recent.columns = df_recent.columns.str.lower()
    if not df_hist.empty:
        df_hist.columns = df_hist.columns.str.lower()

    # ✅ Asegurar que ambos datasets tengan la columna prediction
    if "prediction" not in df_hist.columns:
        df_hist["prediction"] = 0  # placeholder

    if not df_recent.empty and not df_hist.empty:
        prediction_col_lower = PREDICTION_COLUMN_NAME.lower()

        if prediction_col_lower not in df_recent.columns:
            print(f"Error: Columna '{prediction_col_lower}' no encontrada en datos recientes.")
        else:
            drift_results = generate_drift_report(
                df_recent, df_hist, FEATURE_COLUMNS_TO_MONITOR,
                prediction_col_lower, OUTPUT_REPORT_PATH
            )

            if drift_results:
                try:
                    os.makedirs(os.path.dirname(OUTPUT_STATUS_PATH), exist_ok=True)
                    with open(OUTPUT_STATUS_PATH, "w") as f:
                        json.dump(drift_results, f, indent=4)
                    print(f"Estado de drift guardado en: {OUTPUT_STATUS_PATH}")

                    if drift_results["data_drift_detected"]:
                        print("¡Drift detectado! Estado guardado en JSON.")
                    else:
                        print("No se detectó drift significativo.")
                except Exception as e:
                    print(f"Error al guardar el estado de drift en JSON: {e}")
            else:
                print("No se pudieron obtener los resultados del drift para guardar en JSON.")
    else:
        print("No se generó el reporte: datasets vacíos.")

    print("--- Script de Monitoreo de Drift Finalizado ---")
