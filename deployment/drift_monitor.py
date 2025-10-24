import os
import pandas as pd
import psycopg2 # Para conectar a Supabase
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# --- Configuración ---
DB_CONNECTION_STRING = os.environ.get("SUPABASE_CONNECTION_STRING")

HISTORICAL_DATA_PATH = "deployment/data/historical_data.csv"

OUTPUT_REPORT_PATH = "public/drift_report.html"

NUM_RECENT_PREDICTIONS = 5000

# En drift_monitor.py
COLUMNS_TO_MONITOR = [
    'creditscore', 'age', 'tenure', 'balance', 
    'hascrcard', 'isactivemember', 'estimatedsalary',
    'geography_france', 'geography_germany', 'geography_spain', 
    'gender_female', 'gender_male',                           
    'numofproducts_1', 'numofproducts_2', 'numofproducts_3', 'numofproducts_4',
    'prediction'
]

def load_recent_data(conn_string, num_rows):
    query = f"""
    SELECT
        creditscore, age, tenure, balance, hascrcard, isactivemember, estimatedsalary,
        geography_france, geography_germany, geography_spain, -- Columnas OHE Geo
        gender_female, gender_male,                         -- Columnas OHE Gender
        numofproducts_1, numofproducts_2, numofproducts_3, numofproducts_4, -- Columnas OHE NumProd
        prediction
    FROM predictions
    ORDER BY timestamp DESC
    LIMIT {num_rows};
    """
    try:
        with psycopg2.connect(conn_string) as conn:
            df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"Error al cargar datos desde Supabase: {e}")
        return pd.DataFrame()

def load_reference_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # df['HasCrCard'] = df['HasCrCard'].astype(str)
        # df['IsActiveMember'] = df['IsActiveMember'].astype(str) 
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de referencia en {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return pd.DataFrame()

def generate_drift_report(df_current, df_reference, columns_to_check, output_path):
    print("Generando reporte de drift...")

    available_cols_current = [col for col in columns_to_check if col in df_current.columns]
    available_cols_reference = [col for col in columns_to_check if col in df_reference.columns]
    final_columns = list(set(available_cols_current) & set(available_cols_reference))

    # Crear el reporte con los presets de Data Drift y Target Drift
    report = Report(metrics=[
        DataDriftPreset(columns=final_columns), # Monitorea drift en features
        TargetDriftPreset(columns=['prediction']) # Monitorea drift en la predicción 
    ])

    # Ejecutar el reporte comparando los datos actuales vs los de referencia
    report.run(
        current_data=df_current[final_columns],
        reference_data=df_reference[final_columns],
        column_mapping=None 
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)

if __name__ == "__main__":
    if not DB_CONNECTION_STRING:
        print("Error Crítico: La variable de entorno SUPABASE_CONNECTION_STRING no está definida.")
        exit()

    df_recent = load_recent_data(DB_CONNECTION_STRING, NUM_RECENT_PREDICTIONS)
    df_hist = load_reference_data(HISTORICAL_DATA_PATH)

    if not df_recent.empty and not df_hist.empty:
        df_recent.columns = df_recent.columns.str.lower()
        df_hist.columns = df_hist.columns.str.lower()
        
        # Ajustar tipos si es necesario (ejemplo: booleanos a string si dan problemas)
        # for col in ['hascrcard', 'isactivemember']:
        #     if col in df_recent.columns: df_recent[col] = df_recent[col].astype(str)
        #     if col in df_hist.columns: df_hist[col] = df_hist[col].astype(str)

        generate_drift_report(df_recent, df_hist, COLUMNS_TO_MONITOR, OUTPUT_REPORT_PATH)
    else:
        print("No se generó el reporte porque uno o ambos datasets están vacíos o no se pudieron cargar.")