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
    SELECT timestamp, prediction, confidence, latency_ms,
           creditscore, age, tenure, balance, numofproducts_1, numofproducts_2, numofproducts_3, numofproducts_4,
           hascrcard, isactivemember, estimatedsalary,
           geography_france, geography_germany, geography_spain,
           gender_female, gender_male
    FROM predictions
    ORDER BY timestamp DESC
    LIMIT 5000;
    """
    try:
        df = pd.read_sql(sql, _conn)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        numeric_cols = ['prediction', 'confidence', 'latency_ms', 'creditscore', 'age', 'tenure', 'balance', 'estimatedsalary']
        bool_cols = ['hascrcard', 'isactivemember', 'geography_france', 'geography_germany', 'geography_spain',
                     'gender_female', 'gender_male', 'numofproducts_1', 'numofproducts_2', 'numofproducts_3', 'numofproducts_4']

        for col in numeric_cols:
             if col in df.columns:
                  df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in bool_cols:
             if col in df.columns:
                  try:
                      df[col] = df[col].astype(bool)
                  except Exception: # Fallback si la conversión falla
                      df[col] = df[col].astype(str)

        # Reconstruir NumOfProducts original (aproximado) si se necesita para filtros
        def get_num_products(row):
            if row.get('numofproducts_1', False): return 1
            if row.get('numofproducts_2', False): return 2
            if row.get('numofproducts_3', False): return 3
            if row.get('numofproducts_4', False): return 4
            return 0 # O None
        df['numofproducts'] = df.apply(get_num_products, axis=1)

        # Reconstruir Geography original
        def get_geography(row):
            if row.get('geography_france', False): return "France"
            if row.get('geography_germany', False): return "Germany"
            if row.get('geography_spain', False): return "Spain"
            return "Unknown"
        df['geography'] = df.apply(get_geography, axis=1)

        # Reconstruir Gender original
        def get_gender(row):
            if row.get('gender_female', False): return "Female"
            if row.get('gender_male', False): return "Male"
            return "Unknown"
        df['gender'] = df.apply(get_gender, axis=1)


        return df
    except Exception as e:
        st.error(f"Error al cargar o procesar datos desde la base de datos: {e}")
        st.cache_resource.clear()
        return pd.DataFrame()


conn = get_db_connection()
df_kpis = load_data_from_db(conn)


st.sidebar.header("Filtros de Segmentación 🧭")

min_prob_threshold = 0.5
filtered_customer_count = 0
df_filtered = pd.DataFrame()

if not df_kpis.empty and 'confidence' in df_kpis.columns:
    min_prob_threshold = st.sidebar.slider(
        "Filtrar por Probabilidad Mínima de Churn:",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    df_filtered = df_kpis[df_kpis['confidence'] >= min_prob_threshold].copy()
    filtered_customer_count = len(df_filtered)

    unique_geo = sorted(df_kpis['geography'].dropna().unique())
    geo_filter = st.sidebar.multiselect("Filtrar por País:", options=unique_geo, default=unique_geo)
    df_filtered = df_filtered[df_filtered['geography'].isin(geo_filter)]
    filtered_customer_count = len(df_filtered)

    unique_gender = sorted(df_kpis['gender'].dropna().unique())
    gender_filter = st.sidebar.multiselect("Filtrar por Género:", options=unique_gender, default=unique_gender)
    df_filtered = df_filtered[df_filtered['gender'].isin(gender_filter)]
    filtered_customer_count = len(df_filtered)

    is_active_options = ["Todos", "Activos", "Inactivos"]
    is_active_map = {"Activos": True, "Inactivos": False}
    is_active_filter_selection = st.sidebar.selectbox("Filtrar por Miembro Activo:", options=is_active_options, index=0)
    if is_active_filter_selection != "Todos":
        df_filtered = df_filtered[df_filtered['isactivemember'] == is_active_map[is_active_filter_selection]]
    filtered_customer_count = len(df_filtered)


else:
    st.sidebar.warning("No hay datos para filtrar.")

st.sidebar.metric("Clientes Filtrados", filtered_customer_count)
st.sidebar.divider()
st.sidebar.info("Use los filtros para explorar segmentos.")


tab1, tab2, tab3, tab4 = st.tabs([
    "📈 KPIs y Tendencias",
    "📊 Distribuciones Recientes",
    "🔬 Monitor de Drift",
    "🗃️ Clientes Filtrados"
])


with tab1:
    st.header("Métricas Globales Recientes")
    if conn is None:
        st.error("No se pudo conectar a la base de datos para cargar KPIs.")
    elif not df_kpis.empty:
        col1, col2, col3 = st.columns(3)
        total_preds = len(df_kpis)
        col1.metric("Total Predicciones (últimas 5k)", f"{total_preds}")

        riesgo_churn_global = df_kpis[df_kpis['prediction'] == 1].shape[0]
        if total_preds > 0:
             col2.metric("Clientes en Riesgo Global (Predichos)", f"{riesgo_churn_global} ({riesgo_churn_global/total_preds:.1%})")
        else:
             col2.metric("Clientes en Riesgo Global (Predichos)", f"{riesgo_churn_global}")

        latencia_avg_global = df_kpis['latency_ms'].mean()
        col3.metric("Latencia Promedio Global", f"{latencia_avg_global:.2f} ms" if pd.notna(latencia_avg_global) else "N/A")

        st.divider()
        st.header("Tendencias Temporales Globales (Promedios por Hora)")
        df_resample = df_kpis.copy()
        if 'timestamp' in df_resample.columns:
            df_resample.set_index('timestamp', inplace=True)

            if 'confidence' in df_resample.columns:
                st.subheader("Confianza Promedio (Score) por Hora")
                confidence_hourly = df_resample['confidence'].resample('h').mean().dropna()
                if not confidence_hourly.empty: st.line_chart(confidence_hourly, use_container_width=True)
                else: st.info("Insuficientes datos de confianza.")

            if 'latency_ms' in df_resample.columns:
                st.subheader("Latencia Promedio por Hora")
                latency_hourly = df_resample['latency_ms'].resample('h').mean().dropna()
                if not latency_hourly.empty: st.line_chart(latency_hourly, use_container_width=True)
                else: st.info("Insuficientes datos de latencia.")

            if 'prediction' in df_resample.columns:
                st.subheader("Tasa de Churn Predicha por Hora (%)")
                churn_rate_hourly = df_resample['prediction'].resample('h').apply(lambda x: (x == 1).mean() * 100).dropna()
                if not churn_rate_hourly.empty:
                    st.line_chart(churn_rate_hourly, use_container_width=True)
                else:
                    st.info("Insuficientes datos para tendencia de tasa de churn predicha.")

        else:
             st.warning("Columna 'timestamp' no encontrada.")
    else:
        st.info("Aún no hay datos para mostrar KPIs o tendencias.")


with tab2:
    st.header("Distribución de Features Recientes")
    if not df_kpis.empty:
        col_feat1, col_feat2 = st.columns(2)
        with col_feat1:
            if 'age' in df_kpis.columns:
                st.subheader("Edad Reciente")
                st.hist_chart(df_kpis['age'].dropna())
            if 'geography' in df_kpis.columns:
                 st.subheader("País Reciente")
                 st.bar_chart(df_kpis['geography'].value_counts())
            if 'hascrcard' in df_kpis.columns:
                 st.subheader("Tiene Tarjeta Crédito Reciente")
                 st.bar_chart(df_kpis['hascrcard'].value_counts())
        with col_feat2:
             if 'balance' in df_kpis.columns:
                 st.subheader("Saldo Reciente")
                 st.hist_chart(df_kpis['balance'].dropna())
             if 'numofproducts' in df_kpis.columns:
                  st.subheader("Productos Recientes")
                  st.bar_chart(df_kpis['numofproducts'].value_counts().sort_index())
             if 'isactivemember' in df_kpis.columns:
                 st.subheader("Miembro Activo Reciente")
                 st.bar_chart(df_kpis['isactivemember'].value_counts())
    else:
        st.info("No hay datos recientes para mostrar distribuciones.")


with tab3:
    st.header("Reporte de Data Drift")
    st.markdown(f"Mostrando el último reporte generado desde: [GitHub Pages]({REPORT_URL})")
    try:
        st.components.v1.iframe(REPORT_URL, height=1000, scrolling=True)
    except Exception as e:
        st.error(f"No se pudo cargar el reporte de drift.")
        st.warning(f"Verifica la URL: {REPORT_URL}")


with tab4:
    st.header("Muestra de Clientes Filtrados")
    st.info(f"Mostrando clientes basados en filtros de la barra lateral (Probabilidad >= {min_prob_threshold:.0%})")
    if conn is None:
         st.error("No se pudo conectar a la base de datos.")
    elif not df_filtered.empty:
        cols_to_show = ['timestamp', 'prediction', 'confidence', 'creditscore', 'age', 'tenure',
                        'balance', 'numofproducts', 'hascrcard', 'isactivemember',
                        'estimatedsalary', 'geography', 'gender', 'latency_ms']
        st.dataframe(df_filtered[cols_to_show].head(100))
    elif df_kpis.empty:
         st.info("No hay datos recientes en la base de datos.")
    else:
        st.warning("Ningún cliente cumple los criterios de filtro en los datos recientes.")