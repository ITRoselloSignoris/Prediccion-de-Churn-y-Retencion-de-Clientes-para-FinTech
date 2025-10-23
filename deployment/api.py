import pandas as pd
import pickle
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import uvicorn
from pydantic import BaseModel
import os
import psycopg2
from datetime import datetime
import time
import mlflow


# Buscamos la versión del modelo
MODEL_VERSION = "unknown" # Valor por defecto
try:
    mlflow.set_tracking_uri("../deployment/mlruns/") 
    runs = mlflow.search_runs(
        experiment_names=["Churn Prediction"],
        filter_string="tags.mlflow.log-model.history IS NOT NULL", # Filtramos ejecuciones que guardaron un modelo
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs.empty:
        run_id = runs.iloc[0].run_id
        run_info = mlflow.get_run(run_id)
        MODEL_VERSION = run_info.data.tags.get("model_version", run_info.info.run_name) 
        print(f"Versión del modelo cargada desde MLflow Run {run_id}: {MODEL_VERSION}")
    else:
        print("Advertencia: No se encontró ejecución de MLflow con modelo registrado.")
except Exception as e:
    print(f"Error al leer versión de MLflow: {e}")

PARENT_FOLDER = os.path.dirname(__file__)
MODEL_PATH = os.path.join(PARENT_FOLDER, "../src/model/best_model.pkl")
COLUMNS_PATH = os.path.join(PARENT_FOLDER, "../src/ohe_categories_without_exited.pickle")
SCALER_PATH = os.path.join(PARENT_FOLDER, "../src/model/scaler.pkl")

with open(MODEL_PATH, "rb") as handle:
    model = pickle.load(handle)

with open(SCALER_PATH, "rb") as handle:
    scaler = pickle.load(handle)

with open(COLUMNS_PATH, "rb") as handle:
     ohe_tr = pickle.load(handle)

DB_CONNECTION_STRING = os.environ.get("SUPABASE_CONNECTION_STRING")

app = FastAPI()

numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

class CustomerData(BaseModel):
    CreditScore : int
    Age : int
    Tenure : int
    Balance : float
    HasCrCard : bool
    IsActiveMember : bool
    EstimatedSalary : float
    Geography : str
    Gender : str
    NumOfProducts : int

@app.get("/")
async def root():
    return {"message": "API de Predicción de Churn funcionando correctamente"}

@app.post("/prediccion")
def predict_fraud_customer (customer_data:CustomerData):
    start_time = time.time()

    answer_dict = jsonable_encoder(customer_data)

    for key,value in answer_dict.items():
        answer_dict[key] = [value]

    #Crear dataframe
    single_instance = pd.DataFrame.from_dict(answer_dict)

    #One Hot Encoding
    single_instance_ohe = pd.get_dummies(single_instance).reindex(columns = ohe_tr).fillna(0)
   
    #Hacer escalado
    X_scaled_array = scaler.transform(single_instance_ohe)

    X_final_pred = pd.DataFrame(X_scaled_array, columns=ohe_tr, index=single_instance_ohe.index)
    #Hacer predicción
    prediction_proba = model.predict_proba(X_final_pred)

    confidence = float(prediction_proba[0][1])

    threshold = 0.6
    prediction = 1 if confidence >= threshold else 0

    latency_ms = (time.time() - start_time) * 1000  
    model_version = MODEL_VERSION

    # Registrar la predicción en la base de datos
    with psycopg2.connect(DB_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            # Usamos %s como placeholders
            sql = """
            INSERT INTO predictions
            (timestamp, latency_ms, model_version, prediction, confidence,
                "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
                "IsActiveMember", "EstimatedSalary", "Geography", "Gender")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            # Pasamos los valores originales del input
            params = (
                datetime.now(), latency_ms, model_version, prediction, confidence,
                customer_data.CreditScore, customer_data.Age, customer_data.Tenure,
                customer_data.Balance, customer_data.NumOfProducts, customer_data.HasCrCard,
                customer_data.IsActiveMember, customer_data.EstimatedSalary,
                customer_data.Geography, customer_data.Gender
            )
            cur.execute(sql, params)

    prediction_text = "Sí" if prediction == 1 else "No"
    response = {
        "Predicción de Churn": prediction_text,
        "Probabilidad de Churn": confidence
    }

    return response

if __name__ == "__main__":
    if not DB_CONNECTION_STRING:
         print("ADVERTENCIA: Variable SUPABASE_CONNECTION_STRING no encontrada.")
    uvicorn.run(app, host = "0.0.0.0", port = 7860)