import gradio as gr
import requests

API_URL = "https://Itrs-api-churn.hf.space/prediccion"

PARAMS_NAME = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography",
    "Gender",
    "NumOfProducts"
    ]

def predict_churn (*args):

    args = {PARAMS_NAME[i]: args[i] for i in range(len(PARAMS_NAME))}

    args["HasCrCard"] = (args["HasCrCard"] == "Si")
    args["IsActiveMember"] = (args["IsActiveMember"] == "Si")
    
    response = requests.post(API_URL, json = args)
    result_json = response.json()
    print(f"API Status Code: {response.status_code}")
    print(f"API Response Text: {response.text}")
    
    probabilidad = result_json.get("Probabilidad de Churn", -1) # si no existe, devuelve -1

    if probabilidad >= 0:
            return f"{probabilidad:.2%}"
    else:
        return "Error: Probabilidad no encontrada"


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Predicción de Churn de Clientes Bancarios
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown(
            """
            ## Predecir si un cliente bancario se dará de baja
            """                
            )
            creditScore_input = gr.Slider(label = "Puntaje de Crédito", minimum = 350, maximum = 850, step = 1, randomize = True)
            
            age_input = gr.Slider(label = "Edad", minimum = 18, maximum = 92, step = 1, randomize = True)

            tenure_input = gr.Slider(label = "Antigüedad", minimum = 0, maximum = 10, step = 1, randomize = True)

            balance_input = gr.Number(label="Saldo en Cuenta", value=0)

            hasCrCard_input = gr.Radio(
                label = "Posesión de una Tarjeta de Crédito",
                choices = ["Si", "No"],
                value = "Si"
            )

            isActiveMember_input = gr.Radio(
                label = "Es un miembro activo",
                choices = ["Si", "No"],
                value = "No"
            )

            estimatedSalary_input = gr.Number(label="Salario Estimado", value=100000)

            geography_input = gr.Dropdown(
                label = "Ubicación Geográfica",
                choices = ["France", "Spain", "Germany"],
                value = "France"
            )
            gender_input = gr.Radio(
                label = "Género del Cliente",
                choices = ["Female","Male"],
                value = "Male"
            )
            numOfProducts_input = gr.Slider(label = "Cantidad de Productos", minimum = 1, maximum = 4, step = 1, randomize = True)

        with gr.Column():
            gr.Markdown(
            """
            ## Predicción
            """
            )
            target = gr.Label(label = "Probabilidad de Churn")
            predict_btn = gr.Button(value = "Evaluar")
            predict_btn.click(
                predict_churn,
                inputs = [
                    creditScore_input,
                    age_input,
                    tenure_input,
                    balance_input,
                    hasCrCard_input,
                    isActiveMember_input,
                    estimatedSalary_input,
                    geography_input,
                    gender_input,
                    numOfProducts_input
                ],
                outputs = [target]
            )

demo.launch()