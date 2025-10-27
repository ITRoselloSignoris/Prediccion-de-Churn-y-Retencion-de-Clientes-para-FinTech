import os
import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import numpy as np
import pickle
import shap
shap.initjs() 
import matplotlib.pyplot as plt
import requests 
import json    
import datetime

st.set_page_config(
    page_title="Dashboard de Monitoreo de Churn",
    layout="wide"
)
st.title("📊 Dashboard de Monitoreo del Modelo de Churn")

# --- Constantes ---
REPORT_URL = "https://itrosellosignoris.github.io/Prediccion-de-Churn-y-Retencion-de-Clientes-para-FinTech/drift_report.html"
# --- URL del archivo JSON de estado ---
STATUS_JSON_URL = "https://itrosellosignoris.github.io/Prediccion-de-Churn-y-Retencion-de-Clientes-para-FinTech/drift_status.json" # Asegúrate que esta URL sea correcta
DB_CONNECTION_STRING = st.secrets.get("SUPABASE_CONNECTION_STRING")

# --- Constantes para SHAP ---
MODEL_PATH = "src/model/best_model.pkl"
SCALER_PATH = "src/model/scaler.pkl"
BACKGROUND_DATA_PATH = "deployment/data/X_train_final_linear.csv"

# Lista de features EXACTAS que espera tu modelo
MODEL_FEATURE_COLS = [
    'creditscore', 'age', 'tenure', 'balance',
    'hascrcard', 'isactivemember', 'estimatedsalary',
    'geography_france', 'geography_germany', 'geography_spain',
    'gender_female', 'gender_male',
    'numofproducts_1', 'numofproducts_2', 'numofproducts_3', 'numofproducts_4'
]

# --- Funciones cacheadas para cargar modelo, scaler y SHAP ---

@st.cache_resource
def load_model():
    """Carga el modelo de churn .pkl entrenado."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en {MODEL_PATH}")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Carga el scaler .pkl entrenado."""
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del scaler en {SCALER_PATH}")
        return None
    except Exception as e:
        st.error(f"Error al cargar el scaler: {e}")
        return None

@st.cache_resource
def load_background_data():
    """Carga los datos de fondo (X_train) para SHAP."""
    try:
        df_background = pd.read_csv(BACKGROUND_DATA_PATH)
        df_background.columns = df_background.columns.str.lower()
        if not all(col in df_background.columns for col in MODEL_FEATURE_COLS):
             st.warning(f"Columnas en {BACKGROUND_DATA_PATH} no coinciden.")
        return df_background[MODEL_FEATURE_COLS]
    except FileNotFoundError:
        st.error(f"Error: Datos de fondo no encontrados en {BACKGROUND_DATA_PATH}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar datos de fondo: {e}")
        return pd.DataFrame()

@st.cache_data
def get_shap_explainer(_model, _background_data):
    """Crea el explicador SHAP."""
    if _model is None or _background_data.empty:
        return None
    try:
        explainer = shap.LinearExplainer(_model, _background_data)
        return explainer
    except Exception as e:
        st.error(f"Error al crear el explicador SHAP: {e}")
        return None

# --- Funciones de Base de Datos ---
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
        df = pd.read_sql(sql, _conn) # <-- Genera el warning de SQLAlchemy, pero es seguro ignorarlo
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        numeric_cols = ['prediction', 'confidence', 'latency_ms', 'creditscore', 'age', 'tenure', 'balance', 'estimatedsalary']
        bool_cols = ['hascrcard', 'isactivemember', 'geography_france', 'geography_germany', 'geography_spain',
                     'gender_female', 'gender_male', 'numofproducts_1', 'numofproducts_2', 'numofproducts_3', 'numofproducts_4']

        for col in numeric_cols:
             if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in bool_cols:
             if col in df.columns:
                  try: df[col] = df[col].astype(bool)
                  except Exception: df[col] = df[col].astype(str)

        # Reconstrucción vectorizada
        prod_conditions = [df.get('numofproducts_1', False), df.get('numofproducts_2', False), df.get('numofproducts_3', False), df.get('numofproducts_4', False)]
        prod_choices = [1, 2, 3, 4]
        df['numofproducts'] = np.select(prod_conditions, prod_choices, default=0)
        geo_conditions = [df.get('geography_france', False), df.get('geography_germany', False), df.get('geography_spain', False)]
        geo_choices = ["France", "Germany", "Spain"]
        df['geography'] = np.select(geo_conditions, geo_choices, default="Unknown")
        gen_conditions = [df.get('gender_female', False), df.get('gender_male', False)]
        gen_choices = ["Female", "Male"]
        df['gender'] = np.select(gen_conditions, gen_choices, default="Unknown")

        return df
    except Exception as e:
        st.error(f"Error al cargar o procesar datos desde la base de datos: {e}")
        return pd.DataFrame()

# --- Función para leer el estado del drift ---
@st.cache_data(ttl=60) # Cachear por 1 minuto para no sobrecargar
def get_drift_status(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Lanza error si no es 200 OK
        # Intenta decodificar ignorando errores de caracteres
        status = json.loads(response.content.decode('utf-8', errors='ignore'))
        return status
    except requests.exceptions.RequestException as e:
        st.warning(f"No se pudo obtener el estado del drift desde {url}: {e}")
        return None
    except json.JSONDecodeError:
        st.warning(f"Error al parsear el JSON de estado de drift desde {url}")
        return None

# --- Carga Principal de Datos y Recursos ---
conn = get_db_connection()
df_kpis = load_data_from_db(conn)

model = load_model()
scaler = load_scaler()
df_background = load_background_data()
explainer = get_shap_explainer(model, df_background)

# --- Leer estado del drift y mostrar alerta ---
drift_status = get_drift_status(STATUS_JSON_URL)

if drift_status is not None:
    data_drift = drift_status.get("data_drift_detected", False)
    target_drift = drift_status.get("target_drift_detected", False)
    if data_drift or target_drift:
        alert_message = "🚨 **¡Alerta de Drift Detectado!** "
        details = []
        if target_drift:
            details.append("Target Drift")
        if data_drift:
            count = drift_status.get("drifted_features_count", 0)
            lista = drift_status.get("drifted_features_list", [])
            lista_str = ", ".join(lista) if isinstance(lista, list) else "N/A"
            details.append(f"Data Drift ({count} features: {lista_str})")
        alert_message += " | ".join(details)
        alert_message += f" (Reporte: {drift_status.get('timestamp', 'N/A')})"
        st.warning(alert_message)
    else:
        st.info("✅ No se detectó drift en los datos.")
else:
    st.warning("⚠️ No se pudo obtener el estado del drift.")

# --- Lógica de la Sidebar ---
st.sidebar.header("Filtros de Segmentación 🧭")
min_prob_threshold = 0.5
df_filtered = pd.DataFrame()
if not df_kpis.empty and 'confidence' in df_kpis.columns:
    min_prob_threshold = st.sidebar.slider("Probabilidad Mínima:", 0.0, 1.0, 0.5, 0.05)
    df_filtered = df_kpis[df_kpis['confidence'] >= min_prob_threshold].copy()
    unique_geo = sorted(df_kpis['geography'].dropna().unique())
    geo_filter = st.sidebar.multiselect("País:", options=unique_geo, default=unique_geo)
    df_filtered = df_filtered[df_filtered['geography'].isin(geo_filter)]
    unique_gender = sorted(df_kpis['gender'].dropna().unique())
    gender_filter = st.sidebar.multiselect("Género:", options=unique_gender, default=unique_gender)
    df_filtered = df_filtered[df_filtered['gender'].isin(gender_filter)]
    is_active_options = ["Todos", "Activos", "Inactivos"]
    is_active_map = {"Activos": True, "Inactivos": False}
    is_active_filter_selection = st.sidebar.selectbox("Miembro Activo:", options=is_active_options, index=0)
    if is_active_filter_selection != "Todos":
        df_filtered = df_filtered[df_filtered['isactivemember'] == is_active_map[is_active_filter_selection]]
    filtered_customer_count = len(df_filtered)
else:
    st.sidebar.warning("No hay datos para filtrar.")
    filtered_customer_count = 0
st.sidebar.metric("Clientes Filtrados", filtered_customer_count)
st.sidebar.divider()
st.sidebar.info("Use filtros para explorar.")

# --- Definición de Pestañas ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 KPIs y Tendencias",
    "📊 Distribuciones Recientes",
    "🔬 Monitor de Drift",
    "🗃️ Clientes Filtrados",
    "🕵️‍♂️ Explicabilidad (SHAP)"
])
# --- Pestaña 1: KPIs ---
with tab1:
    st.header("Métricas Globales Recientes")
    if conn is None: st.error("Sin conexión a BD.")
    elif not df_kpis.empty:
        col1, col2, col3 = st.columns(3)
        total_preds = len(df_kpis)
        col1.metric("Predicciones (5k)", f"{total_preds}")
        riesgo = df_kpis[df_kpis['prediction'] == 1].shape[0]
        col2.metric("Riesgo Global", f"{riesgo} ({riesgo/total_preds:.1%})" if total_preds > 0 else f"{riesgo}")
        latencia = df_kpis['latency_ms'].mean()
        col3.metric("Latencia Media", f"{latencia:.2f} ms" if pd.notna(latencia) else "N/A")
        st.divider()
        st.header("Tendencias (Promedios Hora)")
        df_resample = df_kpis.copy()
        if 'timestamp' in df_resample.columns:
            df_resample.set_index('timestamp', inplace=True)
            if 'confidence' in df_resample.columns:
                st.subheader("Confianza Media")
                conf_h = df_resample['confidence'].resample('h').mean().dropna()
                if not conf_h.empty: st.line_chart(conf_h, width='stretch')
                else: st.info("Insuficientes datos.")
            if 'latency_ms' in df_resample.columns:
                st.subheader("Latencia Media")
                lat_h = df_resample['latency_ms'].resample('h').mean().dropna()
                if not lat_h.empty: st.line_chart(lat_h, width='stretch')
                else: st.info("Insuficientes datos.")
            if 'prediction' in df_resample.columns:
                st.subheader("Tasa Churn Predicha (%)")
                churn_h = df_resample['prediction'].resample('h').apply(lambda x: (x == 1).mean() * 100).dropna()
                if not churn_h.empty: st.line_chart(churn_h, width='stretch')
                else: st.info("Insuficientes datos.")
        else: st.warning("'timestamp' no encontrado.")
    else: st.info("Sin datos para KPIs.")

# --- Pestaña 2: Distribuciones ---
with tab2:
    st.header("Distribución Features Recientes")
    st.caption("Observamos la distribución de las features en las últimas predicciones.")
    if not df_kpis.empty:
        col1, col2 = st.columns(2) 
        with col1:
            if 'age' in df_kpis.columns:
                st.subheader("Edad")
                fig = px.histogram(df_kpis['age'].dropna(), netbins=30)
                st.plotly_chart(fig, width="stretch")
            if 'creditscore' in df_kpis.columns:
                st.subheader("Credit Score")
                fig = px.histogram(df_kpis['creditscore'].dropna(), netbins=30)
                st.plotly_chart(fig, width="stretch")
            if 'geography' in df_kpis.columns:
                 st.subheader("País")
                 st.bar_chart(df_kpis['geography'].value_counts(), width="stretch")
            if 'gender' in df_kpis.columns:
                 st.subheader("Género")
                 st.bar_chart(df_kpis['gender'].value_counts(), width="stretch")
            if 'isactivemember' in df_kpis.columns:
                 st.subheader("Es Miembro Activo")
                 st.bar_chart(df_kpis['isactivemember'].value_counts(), width="stretch")
        with col2:
            if 'balance' in df_kpis.columns:
                 st.subheader("Saldo")
                 fig = px.histogram(df_kpis['balance'].dropna(), netbins=30)
                 st.plotly_chart(fig, width="stretch")
            if 'estimatedsalary' in df_kpis.columns:
                 st.subheader("Salario Estimado")
                 fig = px.histogram(df_kpis['estimatedsalary'].dropna(), netbins=30)
                 st.plotly_chart(fig, width="stretch")
            if 'tenure' in df_kpis.columns:
                 st.subheader("Antigüedad")
                 st.bar_chart(df_kpis['tenure'].value_counts().sort_index(), width="stretch")
            if 'numofproducts' in df_kpis.columns:
                  st.subheader("Productos")
                  st.bar_chart(df_kpis['numofproducts'].value_counts().sort_index(), width="stretch")
            if 'hascrcard' in df_kpis.columns:
                 st.subheader("Posesión de Tarjeta Crédito")
                 st.bar_chart(df_kpis['hascrcard'].value_counts(), width="stretch")
    else: st.info("Sin datos para distribuciones.")


# --- Pestaña 3: Monitor de Drift ---
with tab3:
    st.header("Reporte Data Drift")
    st.caption("Observamos el reporte detallado de drift generado automáticamente.")
    st.markdown(f"[Ver Reporte]({REPORT_URL})")
    try: st.components.v1.iframe(REPORT_URL, height=1000, scrolling=True)
    except Exception as e: st.error(f"No se cargó reporte: {e}"); st.warning(f"URL: {REPORT_URL}")

# --- Pestaña 4: Clientes Filtrados ---
df_for_model = pd.DataFrame() # Debe estar fuera del 'with' para ser accesible en tab5
with tab4:
    st.header("Muestra Clientes Filtrados")
    st.info("Selecciona fila para análisis SHAP.")
    if conn is None: st.error("Sin conexión BD.")
    elif not df_filtered.empty:
        cols_show = ['timestamp', 'prediction', 'confidence', 'creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'hascrcard', 'isactivemember', 'estimatedsalary', 'geography', 'gender']
        # Preparar datos para SHAP
        df_for_model = df_filtered.copy()
        for col in MODEL_FEATURE_COLS:
            if col not in df_for_model.columns: df_for_model[col] = False
        df_for_model = df_for_model[MODEL_FEATURE_COLS].astype(float) # Asegura orden y tipo
        st.dataframe(df_filtered[cols_show].head(100), key="df_selector", on_select="rerun", selection_mode="single-row")
    elif df_kpis.empty: st.info("Sin datos recientes.")
    else: st.warning("Ningún cliente cumple filtros.")

# --- Pestaña 5: Explicabilidad (SHAP) ---
with tab5:
    st.header("🕵️‍♂️ Explicabilidad del Modelo (SHAP)")
    st.subheader("Importancia Global")
    st.caption("Observamos la importancia global de las features según SHAP.")
    try:
        st.image("deployment/shap_plots/shap_summary.png", use_container_width=True)
    except FileNotFoundError:
        st.error("No se encontró shap_summary.png")
    st.divider()
    st.subheader("Análisis de Cliente Específico (Filtrado)")

    if explainer is None or scaler is None:
        st.error("Recursos SHAP no cargados.")
    elif "df_selector" not in st.session_state or not st.session_state.df_selector.selection["rows"]:
        st.info("Selecciona cliente en 'Clientes Filtrados'.")
    else:
        try:
            selected_index = st.session_state.df_selector.selection["rows"][0]
            if not df_for_model.empty and selected_index < len(df_for_model):
                # Datos NO escalados (minúsculas)
                cust_unscaled_df = df_for_model.iloc[[selected_index]]
                cust_unscaled_series = df_for_model.iloc[selected_index]

                # Renombrar a MAYÚSCULAS para el scaler
                rename_map = {col: col.capitalize() for col in cust_unscaled_df.columns}
                rename_map.update({'creditscore': 'CreditScore', 'hascrcard': 'HasCrCard', 'isactivemember': 'IsActiveMember', 'estimatedsalary': 'EstimatedSalary', 'geography_france': 'Geography_France', 'geography_germany': 'Geography_Germany', 'geography_spain': 'Geography_Spain', 'gender_female': 'Gender_Female', 'gender_male': 'Gender_Male', 'numofproducts_1': 'NumOfProducts_1', 'numofproducts_2': 'NumOfProducts_2', 'numofproducts_3': 'NumOfProducts_3', 'numofproducts_4': 'NumOfProducts_4'})
                cust_unscaled_df_UPPER = cust_unscaled_df.rename(columns=rename_map)

                # Escalar
                cust_scaled_array = scaler.transform(cust_unscaled_df_UPPER)

                # Calcular SHAP 
                shap_output = explainer(cust_scaled_array)

                # --- Manejar Explanation como caso principal ---
                shap_expl_customer = None # Inicializar

                # Si es un objeto Explanation y no está vacío
                if isinstance(shap_output, shap.Explanation) and shap_output.values.shape[0] > 0 and shap_output.values.shape[1] == len(MODEL_FEATURE_COLS):
                    shap_expl_customer = shap_output[0]
                    # Reemplazar datos escalados por no escalados para el display
                    shap_expl_customer.data = cust_unscaled_series.values
                    shap_expl_customer.feature_names = cust_unscaled_series.index.tolist()

                # Si devuelve un array 
                elif isinstance(shap_output, np.ndarray) and shap_output.shape[0] > 0 and shap_output.shape[1] == len(MODEL_FEATURE_COLS):
                     st.warning("SHAP devolvió un array (formato antiguo). Creando Explanation manualmente.")
                     shap_values_customer = shap_output[0]
                     base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
                     shap_expl_customer = shap.Explanation(
                        values=shap_values_customer, base_values=base_value,
                        data=cust_unscaled_series.values, feature_names=cust_unscaled_series.index.tolist()
                     )

                # Si tenemos un objeto Explanation válido, mostrar gráficos
                if shap_expl_customer is not None:
                    st.write(f"Análisis cliente (Índice: {selected_index}):")
                    theme = st.get_option("theme.base") if hasattr(st, 'get_option') else 'light'

                    # --- Force Plot ---
                    st.markdown("#### 📊 Gráfico de Fuerza (Visión General)")
                    st.caption("Muestra qué factores empujan (rojo) o frenan (azul) la predicción para este cliente.")
                    plt.close('all') # Asegura que no haya figuras previas abiertas
                    fig_force = shap.force_plot(shap_expl_customer.base_values, shap_expl_customer.values, shap_expl_customer.data, feature_names=shap_expl_customer.feature_names, matplotlib=True, show=False, text_rotation=0)

                    if fig_force:
                        text_color = "white" if theme == 'dark' else "black" # Color base
                        
                        # Aplica estilos base y oculta etiquetas
                        if theme == 'dark':
                            fig_force.patch.set_alpha(0.0)
                        
                        for ax in fig_force.get_axes():
                            if theme == 'dark':
                                ax.patch.set_alpha(0.0) # Fondo de ejes transparente (oscuro)
                            else:
                                ax.patch.set_alpha(1.0) # Fondo de ejes opaco (claro)
                                
                            for text in ax.findobj(plt.Text):
                                # Oculta etiquetas con '=' que no sean f(x) o base value
                                if '=' in text.get_text() and "f(x)" not in text.get_text() and "base value" not in text.get_text():
                                    text.set_visible(False)
                                else: # Colorea el resto del texto
                                     text.set_color(text_color)
                            
                            # Colorea ejes y ticks
                            for spine in ax.spines.values(): spine.set_edgecolor(text_color)
                            ax.tick_params(axis='x', colors=text_color)
                            ax.tick_params(axis='y', colors=text_color)

                        st.pyplot(fig_force)
                    else: st.warning("No se generó gráfico de fuerza.")

                    st.markdown("---") # Separador

                    # --- Waterfall Plot ---
                    st.markdown("#### 🌊 Desglose del Impacto (Waterfall)")
                    st.caption("Detalla la contribución exacta de cada feature a la predicción final.")
                    plt.close('all') # Asegura que no haya figuras previas abiertas
                    if theme == 'dark': plt.style.use('dark_background')
                    shap.plots.waterfall(shap_expl_customer, max_display=15, show=False) # Pasar el Explanation
                    fig_waterfall = plt.gcf()
                    if theme == 'dark': # Aplicar estilo oscuro
                        fig_waterfall.patch.set_alpha(0.0); [ax.patch.set_alpha(0.0) for ax in fig_waterfall.get_axes()]
                    st.pyplot(fig_waterfall)
                    plt.style.use('default') # Resetear estilo

                # Si no se pudo crear/obtener el Explanation
                else:
                    st.error("Cálculo SHAP falló o devolvió formato inesperado.")
                    st.write(f"Tipo devuelto: {type(shap_output)}")
                    if hasattr(shap_output, 'shape'): st.write(f"Shape: {shap_output.shape}")

            else: st.warning("No se cargaron datos del cliente.")
        except Exception as e: st.error(f"Error al generar gráfico SHAP: {e}")