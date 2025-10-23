FROM python:3.11.7

WORKDIR /app

# Copia solo los archivos necesarios primero 
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el proyecto
COPY . .

# Expone el puerto que espera HF
EXPOSE 7860

# Comando para ejecutar la API
CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "7860"]