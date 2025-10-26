import os
import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import numpy as np 
import pickle       
import shap        
shap.initjs()
from streamlit_shap import st_shap
import matplotlib.pyplot as plt 

st.set_page_config(
    page_title="Dashboard de Monitoreo de Churn",
    layout="wide"
)
st.title("📊 Dashboard de Monitoreo del Modelo de Churn")

# --- Constantes ---
REPORT_URL = "https://itrosellosignoris.github.io/Prediccion-de-Churn-y-Retencion-de-Clientes-para-FinTech/drift_report.html"
DB_CONNECTION_STRING = st.secrets.get("SUPABASE_CONNECTION_STRING")

# --- Constantes para SHAP ---
MODEL_PATH = "src/model/best_model.pkl"
BACKGROUND_DATA_PATH = "deployment/data/X_train_final_linear.csv" 

# Lista de features EXACTAS que espera tu modelo (de training.ipynb)
MODEL_FEATURE_COLS = [
    'creditscore', 'age', 'tenure', 'balance',
    'hascrcard', 'isactivemember', 'estimatedsalary',
    'geography_france', 'geography_germany', 'geography_spain',
    'gender_female', 'gender_male',
    'numofproducts_1', 'numofproducts_2', 'numofproducts_3', 'numofproducts_4'
]

# --- AÑADIDO: Funciones cacheadas para cargar modelo y SHAP ---

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
def load_background_data():
    """Carga los datos de fondo (X_train) para SHAP."""
    try:
        df_background = pd.read_csv(BACKGROUND_DATA_PATH)
        df_background.columns = df_background.columns.str.lower()
        if not all(col in df_background.columns for col in MODEL_FEATURE_COLS):
             st.warning(f"Columnas de {BACKGROUND_DATA_PATH} no coinciden con MODEL_FEATURE_COLS.")
        return df_background[MODEL_FEATURE_COLS]
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos de fondo en {BACKGROUND_DATA_PATH}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar datos de fondo: {e}")
        return pd.DataFrame()

@st.cache_data
def get_shap_explainer(_model, _background_data):
    """Crea el explicador SHAP y calcula valores globales."""
    if _model is None or _background_data.empty:
        return None, None, None
    
    try:
        st.info("Creando explicador SHAP y calculando valores globales...")
        # Como usé Regresión Logística, uso LinearExplainer
        explainer = shap.LinearExplainer(_model, _background_data)
        shap_values = explainer(_background_data)
        st.success("Explicador SHAP y valores globales listos.")
        return explainer, shap_values, _background_data
    except Exception as e:
        st.error(f"Error al crear el explicador SHAP: {e}")
        return None, None, None

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
                  except Exception:
                      df[col] = df[col].astype(str)

        # --- Reconstrucción vectorizada ---
        prod_conditions = [
            df.get('numofproducts_1', False), df.get('numofproducts_2', False),
            df.get('numofproducts_3', False), df.get('numofproducts_4', False)
        ]
        prod_choices = [1, 2, 3, 4]
        df['numofproducts'] = np.select(prod_conditions, prod_choices, default=0)

        geo_conditions = [
            df.get('geography_france', False), df.get('geography_germany', False),
            df.get('geography_spain', False)
        ]
        geo_choices = ["France", "Germany", "Spain"]
        df['geography'] = np.select(geo_conditions, geo_choices, default="Unknown")

        gen_conditions = [
            df.get('gender_female', False), df.get('gender_male', False)
        ]
        gen_choices = ["Female", "Male"]
        df['gender'] = np.select(gen_conditions, gen_choices, default="Unknown")

        return df
    except Exception as e:
        st.error(f"Error al cargar o procesar datos desde la base de datos: {e}")
        # --- DEVUELVE DATAFRAME VACÍO EN CASO DE ERROR ---
        return pd.DataFrame()

# --- Carga Principal de Datos y Recursos ---
conn = get_db_connection()
df_kpis = load_data_from_db(conn)

# --- Carga de recursos SHAP ---
model = load_model()
df_background = load_background_data()
explainer, shap_values_global, df_background_global = get_shap_explainer(model, df_background)

# --- Lógica de la Sidebar ---
st.sidebar.header("Filtros de Segmentación 🧭")

min_prob_threshold = 0.5
df_filtered = pd.DataFrame()

if not df_kpis.empty and 'confidence' in df_kpis.columns:
    min_prob_threshold = st.sidebar.slider(
        "Filtrar por Probabilidad Mínima de Churn:",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    df_filtered = df_kpis[df_kpis['confidence'] >= min_prob_threshold].copy()

    unique_geo = sorted(df_kpis['geography'].dropna().unique())
    geo_filter = st.sidebar.multiselect("Filtrar por País:", options=unique_geo, default=unique_geo)
    df_filtered = df_filtered[df_filtered['geography'].isin(geo_filter)]

    unique_gender = sorted(df_kpis['gender'].dropna().unique())
    gender_filter = st.sidebar.multiselect("Filtrar por Género:", options=unique_gender, default=unique_gender)
    df_filtered = df_filtered[df_filtered['gender'].isin(gender_filter)]

    is_active_options = ["Todos", "Activos", "Inactivos"]
    is_active_map = {"Activos": True, "Inactivos": False}
    is_active_filter_selection = st.sidebar.selectbox("Filtrar por Miembro Activo:", options=is_active_options, index=0)
    if is_active_filter_selection != "Todos":
        df_filtered = df_filtered[df_filtered['isactivemember'] == is_active_map[is_active_filter_selection]]
    
    # --- Cálculo único al final ---
    filtered_customer_count = len(df_filtered) 

else:
    st.sidebar.warning("No hay datos para filtrar.")
    filtered_customer_count = 0 # Asegurar que la variable exista

st.sidebar.metric("Clientes Filtrados", filtered_customer_count)
st.sidebar.divider()
st.sidebar.info("Use los filtros para explorar segmentos.")

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
                if not confidence_hourly.empty: st.line_chart(confidence_hourly, width='stretch')
                else: st.info("Insuficientes datos de confianza.")

            if 'latency_ms' in df_resample.columns:
                st.subheader("Latencia Promedio por Hora")
                latency_hourly = df_resample['latency_ms'].resample('h').mean().dropna()
                if not latency_hourly.empty: st.line_chart(latency_hourly, width='stretch')
                else: st.info("Insuficientes datos de latencia.")

            if 'prediction' in df_resample.columns:
                st.subheader("Tasa de Churn Predicha por Hora (%)")
                churn_rate_hourly = df_resample['prediction'].resample('h').apply(lambda x: (x == 1).mean() * 100).dropna()
                if not churn_rate_hourly.empty:
                    st.line_chart(churn_rate_hourly, width='stretch')
                else:
                    st.info("Insuficientes datos para tendencia de tasa de churn predicha.")
        else:
             st.warning("Columna 'timestamp' no encontrada.")
    else:
        st.info("Aún no hay datos para mostrar KPIs o tendencias.")

# --- Pestaña 2: Distribuciones ---
with tab2:
    st.header("Distribución de Features Recientes")
    if not df_kpis.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'age' in df_kpis.columns:
                st.subheader("Edad Reciente")
                fig_age = px.histogram(df_kpis['age'].dropna(), nbins=30)
                st.plotly_chart(fig_age, width="stretch")
            
            if 'creditscore' in df_kpis.columns:
                st.subheader("Credit Score Reciente")
                fig_credit = px.histogram(df_kpis['creditscore'].dropna(), nbins=30)
                st.plotly_chart(fig_credit, width="stretch")

            if 'balance' in df_kpis.columns:
                 st.subheader("Saldo Reciente")
                 fig_balance = px.histogram(df_kpis['balance'].dropna(), nbins=30)
                 st.plotly_chart(fig_balance, width="stretch")

        with col2:
            if 'geography' in df_kpis.columns:
                 st.subheader("País Reciente")
                 st.bar_chart(df_kpis['geography'].value_counts(), width="stretch")

            if 'gender' in df_kpis.columns:
                 st.subheader("Género Reciente")
                 st.bar_chart(df_kpis['gender'].value_counts(), width="stretch")

            if 'tenure' in df_kpis.columns:
                 st.subheader("Antigüedad (Tenure) Reciente")
                 st.bar_chart(df_kpis['tenure'].value_counts().sort_index(), width="stretch")
        
        with col3:
            if 'numofproducts' in df_kpis.columns:
                  st.subheader("Productos Recientes")
                  st.bar_chart(df_kpis['numofproducts'].value_counts().sort_index(), width="stretch")

            if 'hascrcard' in df_kpis.columns:
                 st.subheader("Tiene Tarjeta Crédito Reciente")
                 st.bar_chart(df_kpis['hascrcard'].value_counts(), width="stretch")

            if 'isactivemember' in df_kpis.columns:
                 st.subheader("Miembro Activo Reciente")
                 st.bar_chart(df_kpis['isactivemember'].value_counts(), width="stretch")

            if 'estimatedsalary' in df_kpis.columns:
                 st.subheader("Salario Estimado Reciente")
                 fig_salary = px.histogram(df_kpis['estimatedsalary'].dropna(), nbins=30)
                 st.plotly_chart(fig_salary, width="stretch")
    else:
        st.info("No hay datos recientes para mostrar distribuciones.")

# --- Pestaña 3: Monitor de Drift ---
with tab3:
    st.header("Reporte de Data Drift")
    st.markdown(f"Mostrando el último reporte generado desde: [GitHub Pages]({REPORT_URL})")
    try:
        st.components.v1.iframe(REPORT_URL, height=1000, scrolling=True)
    except Exception as e:
        st.error(f"No se pudo cargar el reporte de drift.")
        st.warning(f"Verifica la URL: {REPORT_URL}")


# --- Pestaña 4: Clientes Filtrados (Ahora con selección) ---
# Esta variable contendrá los datos listos para el modelo
df_for_model = pd.DataFrame()

with tab4:
    st.header("Muestra de Clientes Filtrados")
    st.info(f"Mostrando clientes basados en filtros (Probabilidad >= {min_prob_threshold:.0%}). Selecciona una fila para analizarla en la pestaña SHAP.")
    
    if conn is None:
         st.error("No se pudo conectar a la base de datos.")
    elif not df_filtered.empty:
        cols_to_show = ['timestamp', 'prediction', 'confidence', 'creditscore', 'age', 'tenure',
                        'balance', 'numofproducts', 'hascrcard', 'isactivemember',
                        'estimatedsalary', 'geography', 'gender']
        
        # --- Prepara los datos para el modelo SHAP ---
        df_for_model = df_filtered.copy()
        
        # Añadir columnas faltantes si no estaban en los datos recientes
        for col in MODEL_FEATURE_COLS:
            if col not in df_for_model.columns:
                df_for_model[col] = False # Asumimos False/0 si no existe

        # Asegura que las columnas estén en el orden exacto que el modelo espera
        # y convierte a float, ya que el scaler lo espera.
        df_for_model = df_for_model[MODEL_FEATURE_COLS].astype(float)

        # ---'key' y 'on_select' para interactividad ---
        st.dataframe(
            df_filtered[cols_to_show].head(100),
            key="df_selector", # Clave para acceder a la selección
            on_select="rerun",
            selection_mode="single-row"
        )
    elif df_kpis.empty:
         st.info("No hay datos recientes en la base de datos.")
    else:
        st.warning("Ningún cliente cumple los criterios de filtro en los datos recientes.")

# --- Pestaña 5: Explicabilidad (SHAP) ---
with tab5:
    st.header("🕵️‍♂️ Explicabilidad del Modelo (SHAP)")
    
    # --- 1. GRÁFICO GENERAL (Estático desde PNG) ---
    st.subheader("Importancia Global de Features")
    
    try:
        st.image("deployment/shap_plots/shap_summary.png", width="stretch")
    except FileNotFoundError:
        st.error("No se encontró el archivo shap_summary.png")

    st.divider()

    # --- 2. GRÁFICO ESPECÍFICO (Filtrado) ---
    st.subheader("Análisis de Cliente Específico (Filtrado)")
    
    if "df_selector" not in st.session_state or not st.session_state.df_selector.selection["rows"]:
        st.info("Por favor, selecciona un cliente en la pestaña '🗃️ Clientes Filtrados' para un análisis detallado.")
    
    else:
        try:
            selected_index = st.session_state.df_selector.selection["rows"][0]
            
            if not df_for_model.empty and selected_index < len(df_for_model):
                customer_data_df = df_for_model.iloc[[selected_index]]
                customer_data_series = df_for_model.iloc[selected_index]
                
                shap_values_batch = explainer(customer_data_df)
                shap_values_customer = shap_values_batch[0]
                
                st.write(f"Análisis para el cliente (Índice: {selected_index}) con `creditscore` de **{customer_data_series['creditscore']:.0f}** y `age` de **{customer_data_series['age']:.0f}**:")

                # --- MODIFICACIÓN: Lógica de Tema Automática ---
                # 1. Obtener el tema actual de Streamlit
                try:
                    theme = st.get_option("theme.base")
                except AttributeError:
                    theme = 'light'

                # --- 2. Gráfico de Fuerza (Force Plot) ---
                st.write("Gráfico de Fuerza (Versión Estática):")
                
                # Genera la figura
                fig_force = shap.force_plot(
                    shap_values_customer.base_values,
                    shap_values_customer.values,
                    customer_data_series,
                    matplotlib=True,
                    show=False,
                    text_rotation=0
                )
                
                if fig_force is not None:
                    # SI el tema es oscuro, aplica el arreglo manual
                    if theme == 'dark':
                        fig_force.patch.set_alpha(0.0)
                        for ax in fig_force.get_axes():
                            ax.patch.set_alpha(0.0)
                            for text in ax.findobj(plt.Text): text.set_color("white")
                            for spine in ax.spines.values(): spine.set_edgecolor("white")
                            ax.tick_params(axis='x', colors='white')
                            ax.tick_params(axis='y', colors='white')
                    
                    # Muestra la figura (modificada o no)
                    st.pyplot(fig_force)
                else:
                    st.warning("No se pudo generar el gráfico de fuerza.")

                # --- 3. Gráfico de Cascada (Waterfall Plot) ---
                st.write("Desglose del impacto (Waterfall):")
                
                # SI el tema es oscuro, aplica el estilo
                if theme == 'dark':
                    plt.style.use('dark_background') 

                fig_waterfall, ax_waterfall = plt.subplots()

                if theme == 'dark':
                    fig_waterfall.patch.set_alpha(0.0)
                    ax_waterfall.patch.set_alpha(0.0)
                
                shap.plots.waterfall(shap_values_customer, max_display=15, show=False, ax=ax_waterfall) 
                st.pyplot(fig_waterfall)
                
                # Resetear el estilo de Matplotlib
                plt.style.use('default')
                
            else:
                st.warning("No se pudieron cargar los datos del cliente seleccionado para SHAP.")

        except Exception as e:
            st.error(f"Error al generar el gráfico SHAP para el cliente: {e}")
            st.write("Asegúrate de que los datos del cliente coinciden con el formato del modelo.")