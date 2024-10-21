# Airflow pipelines

1. **Install:**
   ```markdown
   AIRFLOW_VERSION=2.9.3
   PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
   CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
   pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
   pip install apache-airflow-providers-cncf-kubernetes==8.3.3
   
2. **Launching Airflow in standalone mode:**
   ```markdown
   export AIRFLOW_HOME=$PWD/airflow_pipelines
   export AIRFLOW__CORE__LOAD_EXAMPLES=False
   airflow standalone

3. **Open UI**
   ```markdown
   open http://0.0.0.0:8080

4. ** Training job**
   ```markdown
   airflow dags trigger airflow_training
 
5. ** Inference job**
   ```markdown
   airflow dags trigger airflow_inference
