import os
import streamlit as st
import pandas as pd
import psycopg2

st.set_page_config(
    page_title="Dashboard de Monitoreo de Churn",
    layout="wide"
)
st.title("📊 Dashboard de Monitoreo del Modelo de Churn")

REPORT_URL = "https://itrosellosignoris.github.io/Prediccion-de-Churn-y-Retencion-de-Clientes-para-FinTech/drift_report.html"

DB_CONNECTION_STRING = st.secrets.get("SUPABASE_CONNECTION_STRING")

@st.cache_resource(ttl=300)
def get_db_connection():
    try:
        conn = psycopg2.connect(DB_CONNECTION_STRING)
        return conn
    except Exception as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return None

@st.cache_data(ttl=60)
def load_data_from_db(_conn):
    if _conn is None: return pd.DataFrame()
    sql = """
    SELECT timestamp, prediction, confidence, latency_ms
    FROM predictions
    ORDER BY timestamp DESC
    LIMIT 5000;
    """
    try:
        df = pd.read_sql(sql, _conn)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Aseguramos tipos numéricos para cálculos
        for col in ['prediction', 'confidence', 'latency_ms']:
             if col in df.columns:
                  df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error al cargar datos desde la base de datos: {e}")
        st.cache_resource.clear() # Limpiamos caché de conexión si falla la carga
        return pd.DataFrame()

# --- Carga Principal de Datos ---
conn = get_db_connection()
df_kpis = load_data_from_db(conn)

# --- Pestañas del Dashboard ---
tab1, tab2, tab3 = st.tabs(["📈 KPIs y Tendencias", "🔬 Monitor de Drift", "🗃️ Datos Crudos"]) # Renombrado Tab1

# --- Pestaña 1: 
# --- KPIs y Tendencias ---
with tab1:
    st.header("Métricas de Predicción Recientes")
    if conn is None:
        st.error("No se pudo conectar a la base de datos para cargar KPIs.")
    elif not df_kpis.empty:
        # --- KPIs Instantáneos ---
        col1, col2, col3 = st.columns(3)
        total_preds = len(df_kpis)
        col1.metric("Total Predicciones (últimas 5k)", f"{total_preds}")

        riesgo_churn = df_kpis[df_kpis['prediction'] == 1].shape[0]
        if total_preds > 0:
             col2.metric("Clientes en Riesgo (Predichos)", f"{riesgo_churn} ({riesgo_churn/total_preds:.1%})")
        else:
             col2.metric("Clientes en Riesgo (Predichos)", f"{riesgo_churn}")

        latencia_avg = df_kpis['latency_ms'].mean()
        col3.metric("Latencia Promedio", f"{latencia_avg:.2f} ms" if pd.notna(latencia_avg) else "N/A")

        st.divider()

        # --- Gráficos de Tendencias ---
        st.header("Tendencias Temporales (Promedios por Hora)")

        # Preparamos datos para resample
        df_resample = df_kpis.copy()
        if 'timestamp' in df_resample.columns:
            df_resample.set_index('timestamp', inplace=True)

            # 1. Confianza Promedio por Hora
            if 'confidence' in df_resample.columns:
                st.subheader("Confianza Promedio (Score) por Hora")
                confidence_hourly = df_resample['confidence'].resample('h').mean().dropna()
                if not confidence_hourly.empty:
                    st.line_chart(confidence_hourly, use_container_width=True)
                else:
                    st.info("No hay suficientes datos de confianza para mostrar la tendencia horaria.")
            else:
                st.warning("Columna 'confidence' no encontrada para gráfico de tendencia.")

            # 2. Latencia Promedio por Hora
            if 'latency_ms' in df_resample.columns:
                st.subheader("Latencia Promedio por Hora")
                latency_hourly = df_resample['latency_ms'].resample('h').mean().dropna()
                if not latency_hourly.empty:
                    st.line_chart(latency_hourly, use_container_width=True)
                else:
                    st.info("No hay suficientes datos de latencia para mostrar la tendencia horaria.")
            else:
                st.warning("Columna 'latency_ms' no encontrada para gráfico de tendencia.")

        else:
             st.warning("Columna 'timestamp' no encontrada. No se pueden generar gráficos de tendencia.")

    else:
        st.info("Aún no hay datos de predicciones en la base de datos para mostrar KPIs o tendencias.")

# --- Pestaña 2: Monitor de Drift ---
with tab2:
    st.header("Reporte de Data Drift")
    st.markdown(f"Mostrando el último reporte generado desde: [GitHub Pages]({REPORT_URL})")
    st.markdown("*(Este reporte se actualiza automáticamente)*")
    try:
        st.components.v1.iframe(REPORT_URL, height=1000, scrolling=True)
        st.success("Reporte cargado.")
    except Exception as e:
        st.error(f"No se pudo cargar el reporte de drift desde la URL.")
        st.warning(f"Verifica que la URL '{REPORT_URL}' sea correcta y accesible.")

# --- Pestaña 3: Datos Crudos ---
with tab3:
    st.header("Muestra de Últimas Predicciones")
    if conn is None:
         st.error("No se pudo conectar a la base de datos.")
    elif not df_kpis.empty:
        st.dataframe(df_kpis.head(100))
    else:
        st.info("No hay datos para mostrar.")