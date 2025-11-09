# 🚀 Customer Churn Prediction and Retention for FinTech

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

Complete churn prediction system for financial institutions, including REST API, real-time monitoring dashboard, automatic drift detection, and model explainability using SHAP.

## 🎯 Project Description

Data Science project developed during a job simulation at **NoCountry**, implementing a complete Machine Learning pipeline in production that covers from training to continuous model monitoring.

### Main Components

- **🤖 ML Model**: Logistic Regression optimized to maximize Recall (76%) in churn detection
- **📌 REST API**: FastAPI endpoint for real-time predictions with automatic database logging
- **📊 Interactive Dashboard**: Streamlit application with KPIs, visualizations, and SHAP analysis
- **🎨 Gradio Interface**: User-friendly UI for individual predictions
- **📈 Drift Monitoring**: Automatic system with Evidently to detect model degradation
- **🔄 CI/CD**: Automated pipelines with GitHub Actions for deployment and monitoring

## 🌐 Live Demos

| Component | URL | Description |
|-----------|-----|-------------|
| **API** | [Hugging Face Space](https://Itrs-api-churn.hf.space) | Endpoint for predictions `/prediccion` |
| **Dashboard** | [Streamlit Cloud](https://dashboard-churn-prediction.streamlit.app) | Real-time monitoring and analysis |
| **Gradio App** | [Hugging Face Space](https://huggingface.co/spaces/Itrs/ui-churn-prediction) | Interactive interface for predictions |
| **Drift Report** | [GitHub Pages](https://itrosellosignoris.github.io/Prediccion-de-Churn-y-Retencion-de-Clientes-para-FinTech/drift_report.html) | Automatic data drift analysis |

## 🛠️ Tech Stack

### Machine Learning & Data Science
- **Python 3.11** - Main language
- **Scikit-learn 1.6.1** - Preprocessing, modeling, and metrics
- **imbalanced-learn (SMOTE)** - Handling imbalanced classes
- **SHAP** - Model explainability (Linear Explainer)
- **MLflow** - Experiment tracking and model versioning
- **Evidently** - Data drift monitoring

### Backend & API
- **FastAPI** - REST API framework
- **Uvicorn** - High-performance ASGI server
- **Pydantic** - Data validation and schemas
- **psycopg2** - PostgreSQL connector

### Frontend & Visualization
- **Streamlit** - Interactive monitoring dashboard
- **Gradio** - User interface for predictions
- **Plotly** - Interactive charts
- **Matplotlib/Seaborn** - Static visualizations

### Database & Storage
- **Supabase (PostgreSQL)** - Database for storing predictions
- **GitHub Pages** - Hosting for drift reports

### DevOps & CI/CD
- **Docker** - Application containerization
- **GitHub Actions** - Workflow automation:
  - Automatic synchronization with Hugging Face
  - Daily drift report generation (cron: 8 AM UTC)
  - Automatic production deployment

## 📁 Project Structure

```
Prediccion-de-Churn-y-Retencion-de-Clientes-para-FinTech/
│
├── .github/
│   └── workflows/
│       ├── sync_to_hub.yml          # Sync with Hugging Face
│       └── run_monitor.yml          # Drift report generation
│
├── deployment/
│   ├── api.py                       # FastAPI application
│   ├── dashboard.py                 # Streamlit dashboard
│   ├── drift_monitor.py             # Evidently monitoring script
│   ├── requirements_monitor.txt     # Deps for drift monitoring
│   ├── data/
│   │   ├── historical_data.csv      # Reference data
│   │   └── X_train_final_linear.csv # Data for SHAP
│   ├── shap_plots/
│   │   └── shap_summary.png         # Global feature importance
│   ├── gradio_app/
│   │   ├── app.py                   # Gradio application
│   │   └── requirements.txt
│   └── .streamlit/
│       └── config.toml              # Streamlit configuration
│
├── src/
│   ├── datasets/                 
│   │   ├── adapted_data                      # Intermediate data after initial transformations
│   │   │   └── Churn_Modelling_adapted.csv   # Dataset with feature engineering and type conversions
│   │   ├── processed_data                    # Final clean data ready for modeling
│   │   │   └── cleaned_data.csv              # Preprocessed dataset
│   │   └── raw_data                          # Original unmodified data
│   │   │   └── Churn_Modelling.csv           # Raw dataset from source
│   ├── model/
│   │   ├── best_model.pkl           # Logistic Regression model
│   │   └── scaler.pkl               # StandardScaler fitted
│   ├── notebooks/
│   │   ├── data_adaptation.ipynb     # Complete data adaptation notebook
│   │   ├── eda.ipynb                 # Complete eda notebook
│   │   ├── data_preparation.ipynb    # Complete data preparation notebook
│   │   └── training.ipynb            # Complete training notebook
│   └── ohe_categories_without_exited.pickle  # OHE categories
│
├── public/                           # GitHub Pages (auto-generated)
│   ├── drift_report.html             # Evidently report
│   └── drift_status.json             # Drift status (JSON)
│
├── Dockerfile                        # Container definition
├── requirements.txt                  # Main dependencies
└── README.md
```

## 🚀 Installation and Usage

### Prerequisites

- Python 3.11
- Docker (optional)
- Supabase account (for database)
- Hugging Face account (for deployment)

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/ITRoselloSignoris/Prediccion-de-Churn-y-Retencion-de-Clientes-para-FinTech.git
cd Prediccion-de-Churn-y-Retencion-de-Clientes-para-FinTech
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
export SUPABASE_CONNECTION_STRING="postgresql://user:password@host:port/database"
```

### 📌 Run the API

```bash
uvicorn deployment.api:app --host 0.0.0.0 --port 7860 --reload
```

The API will be available at `http://localhost:7860`

**Interactive documentation:** `http://localhost:7860/docs`

### 📊 Run the Dashboard

```bash
streamlit run deployment/dashboard.py
```

The dashboard will be available at `http://localhost:8501`

### 🎨 Run the Gradio App

```bash
cd deployment/gradio_app
python app.py
```

### 🐳 Deployment with Docker

```bash
# Build the image
docker build -t churn-api .

# Run the container
docker run -p 7860:7860 -e SUPABASE_CONNECTION_STRING="your_connection_string" churn-api
```

## 📊 API Usage

### Main Endpoint: `/prediccion`

**Method:** `POST`

**Request Body:**
```json
{
  "CreditScore": 650,
  "Age": 35,
  "Tenure": 5,
  "Balance": 100000.50,
  "HasCrCard": true,
  "IsActiveMember": true,
  "EstimatedSalary": 75000.00,
  "Geography": "Spain",
  "Gender": "Female",
  "NumOfProducts": 2
}
```

**Response:**
```json
{
  "Predicción de Churn": "No",
  "Probabilidad de Churn": 0.23
}
```

### System Features

- ⚡ **Average latency**: < 100ms per prediction
- 💾 **Automatic storage**: All predictions are logged to Supabase
- 📈 **Tracking**: Model version and metrics via MLflow
- 🎯 **Custom threshold**: 0.6 (configurable)

## 🧠 Model Training Process

### Data Preparation

The model was trained using the `cleaned_data.csv` dataset with the following techniques:

1. **Imbalance Analysis**: 
   - Target variable `Exited`: ~80% no-churn, 20% churn
   - Split: 80/20 train/test with `random_state=42`

2. **Imbalance Handling**:
   - Technique: **SMOTE** (Synthetic Minority Over-sampling Technique)
   - Applied only on training data
   - Two balanced sets generated:
     - `X_train_final`: Unscaled (for tree models)
     - `X_train_final_linear`: Scaled with StandardScaler (for Logistic Regression)

3. **Preprocessing**:
   - **StandardScaler**: Normalization of numerical features
   - **One-Hot Encoding**: Categorical variables (Geography, Gender, NumOfProducts)
   - **17 final features** after encoding

### Experimentation and Model Selection

**3 algorithms** were evaluated with hyperparameter optimization focused on **maximizing Recall**:

| Model | Tuning Technique | Recall (Test) | F1-Score | ROC AUC |
|-------|------------------|---------------|----------|---------|
| **RandomForestClassifier** | RandomizedSearchCV (20 iter) + GridSearchCV | 0.66 | 0.57 | 0.83 |
| **XGBClassifier** | RandomizedSearchCV (15 iter) + GridSearchCV | 0.55 | 0.59 | 0.84 |
| **LogisticRegression** ⭐ | RandomizedSearchCV (30 iter) + GridSearchCV | **0.76** | 0.56 | 0.84 |

### Final Selected Model

**🏆 Logistic Regression with `class_weight='balanced'`**

**Selection Justification:**
- ✅ **Highest Recall (0.76)**: Detects 76% of customers who actually churn
- ✅ **Interpretability**: Linear coefficients easy to explain with SHAP
- ✅ **Balanced performance**: ROC AUC of 0.84 indicates excellent discriminative capacity
- ✅ **Efficiency**: Fast predictions, ideal for production

### Generated Artifacts

1. **`best_model.pkl`**: Trained Logistic Regression model
2. **`scaler.pkl`**: StandardScaler fitted with training data
3. **`ohe_categories_without_exited.pickle`**: Categories for One-Hot Encoding
4. **`shap_summary.png`**: Global feature importance plot
5. **MLflow Artifacts**: Complete record of experiments, metrics, and parameters

## 📈 Model and Results

### Model Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Recall** | 76.0% | Detects 76% of actual churns |
| **Precision** | 47.0% | 47% of positive predictions are correct |
| **F1-Score** | 56.0% | Balance between Precision and Recall |
| **ROC AUC** | 84.0% | Excellent discriminative capacity |

**Note on Recall:** The model was optimized to **maximize Recall**, prioritizing the detection of all possible churn cases (minimizing false negatives). This is critical in contexts where the cost of not detecting a churn is greater than having false alarms.

### Most Important Features (SHAP)

Based on global SHAP analysis, the features with the greatest impact are:

1. **Age** - Customer age (older customers have higher risk)
2. **NumOfProducts** - Number of contracted products (3 or 4 products increases risk)
3. **IsActiveMember** - Activity status (inactive customers more prone to churn)
4. **Balance** - Account balance (extreme balances affect churn)
5. **Geography** - Geographic location (regional differences in behavior)

### Production Pipeline

```
Input Data 
    ↓
[One-Hot Encoding] → Geography (3), Gender (2), NumOfProducts (4)
    ↓
[StandardScaler] → Normalization of numerical features
    ↓
[Logistic Regression] → Churn probability [0-1]
    ↓
[Threshold 0.6] → Final classification (Churn / No Churn)
```

## 🔄 CI/CD and Automation

### GitHub Actions Workflows

#### 1. **Sync to Hugging Face** (`sync_to_hub.yml`)
- **Trigger**: Push to `main` branch or manual execution
- **Action**: Automatically synchronizes code with Hugging Face Spaces
- **Result**: API and Gradio app always up-to-date

#### 2. **Generate Drift Report** (`run_monitor.yml`)
- **Triggers**: 
  - Push to `main`
  - Daily cron (8:00 AM UTC)
  - Manual execution
- **Actions**:
  1. Extracts last 5000 predictions from Supabase
  2. Compares with historical data using Evidently
  3. Generates interactive HTML report
  4. Creates JSON file with drift status
  5. Publishes to GitHub Pages
- **Result**: Automatic model degradation monitoring

### Drift Monitoring

The system detects two types of drift:

- **Data Drift**: Changes in feature distributions
- **Target Drift**: Changes in prediction distributions

**Monitored features:** 16 variables (all model features)

The dashboard shows automatic alerts when drift is detected:
- 🚨 Red alert: Drift detected
- ✅ Green indicator: No drift

## 📊 Monitoring Dashboard

The Streamlit dashboard includes 5 main tabs:

### 1. 📈 KPIs and Trends
- Total processed predictions
- Global churn risk percentage
- API average latency
- Hourly trend charts

### 2. 📊 Recent Distributions
- Histograms of numerical features
- Categorical variable distributions
- Visual analysis of latest data

### 3. 🔬 Drift Monitor
- Interactive Evidently report
- Current drift status
- Features with detected drift

### 4. 🗃️ Filtered Customers
- Interactive table with recent predictions
- Filters by probability, geography, gender, etc.
- Customer selection for SHAP analysis

### 5. 🕵️‍♂️ Explainability (SHAP)
- Global feature importance
- Individual force plots
- Detailed waterfall plots
- Customer-specific interpretation

## 🗄️ Database

### Table: `predictions`

Supabase table structure:

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Auto-incremental ID |
| `timestamp` | TIMESTAMP | Prediction moment |
| `latency_ms` | FLOAT | Response time |
| `model_version` | VARCHAR | Used model version |
| `prediction` | INTEGER | Prediction (0/1) |
| `confidence` | FLOAT | Churn probability |
| `creditscore` | INTEGER | Credit score |
| `age` | INTEGER | Customer age |
| `tenure` | INTEGER | Tenure |
| `balance` | FLOAT | Account balance |
| `hascrcard` | BOOLEAN | Has credit card |
| `isactivemember` | BOOLEAN | Active member |
| `estimatedsalary` | FLOAT | Estimated salary |
| `geography_*` | BOOLEAN | Geography OHE variables |
| `gender_*` | BOOLEAN | Gender OHE variables |
| `numofproducts_*` | BOOLEAN | Products OHE variables |

## 🔧 Maintenance and Monitoring

### Model Retraining

If drift or metric degradation is detected:

1. **Collect New Data**:
   - Export recent predictions from Supabase
   - Label actual churn cases (if available)

2. **Retrain**:
   - Run `notebooks/training.ipynb` with updated data
   - Apply SMOTE and StandardScaler
   - Optimize for Recall with RandomizedSearchCV/GridSearchCV
   - Compare with baseline: Recall ≥ 0.75

3. **Validate**:
   - Verify metrics on test set
   - Compare ROC AUC with current model
   - Generate new SHAP analyses

4. **Deploy**:
   - Save artifacts in `src/model/`
   - Register in MLflow with new version
   - Push to `main` → Automatic deploy via GitHub Actions
   - Update `historical_data.csv` if necessary

### Drift Review

1. Check dashboard (automatic alert if drift exists)
2. Review complete report on GitHub Pages
3. If drift confirmed:
   - Update reference data
   - Consider retraining

### Log Monitoring

- **API**: Logs in Hugging Face Spaces console
- **Dashboard**: Logs in Streamlit Cloud
- **Drift**: Logs in GitHub Actions runs

## 🔐 Secrets Configuration

### GitHub Actions
- `HF_TOKEN`: Hugging Face token for deployment
- `SUPABASE_CONNECTION_STRING`: Database connection

### Streamlit Cloud
- `SUPABASE_CONNECTION_STRING`: Database connection

## 👨‍💻 Author

**Iñaki Tomás Rosello Signoris**

Project developed during **NoCountry** job simulation

- GitHub: [@ITRoselloSignoris](https://github.com/ITRoselloSignoris)
- LinkedIn: [Iñaki Rosello Signoris](https://www.linkedin.com/in/i%C3%B1akirosellosignoris/)

## 📄 License

This project is under the MIT License - see the [LICENSE](LICENSE) file for more details.

## 🙏 Acknowledgments

- **NoCountry** for the opportunity to develop this project
---

## 🎓 Key Learnings

This project demonstrates the complete implementation of an ML system in production:

✅ **MLOps**: Versioning with MLflow, CI/CD with GitHub Actions  
✅ **Experimentation**: Rigorous comparison of 3 algorithms with hyperparameter optimization  
✅ **Imbalance Handling**: SMOTE for class balancing  
✅ **Metric Optimization**: Prioritization of Recall over Accuracy  
✅ **Monitoring**: Automatic drift detection with Evidently  
✅ **Productization**: REST API + Dashboard + User interface  
✅ **Explainability**: SHAP for decision transparency  
✅ **Scalability**: Docker, cloud services, serverless database  
✅ **Automation**: Daily reports, continuous synchronization  

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Evidently AI](https://docs.evidentlyai.com/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [imbalanced-learn (SMOTE)](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

⭐️ If you found this project useful, don't forget to give it a star on GitHub!


**Project Status:** ✅ Active and in production
