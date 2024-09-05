# Kubeflow pipeline

1. **Create kind clusterl:**
   
   ```markdown
     kind create cluster --name ml-in-production
   
2. **Install Kubeflow:**
   ```markdown
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
    kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"

3. **UI Access**
   ```markdown
   kubectl port-forward --address=0.0.0.0 svc/minio-service 9000:9000 -n kubeflow
   kubectl port-forward --address=0.0.0.0 svc/ml-pipeline-ui 3000:80 -n kubeflow

4. **Create training:**
   
   ```markdown
     python ./Kubeflow/kubeflow_training.py http://0.0.0.0:3000
   
5. **Create inference:**
   ```markdown
     python ./Kubeflow/kubeflow_inference.py http://0.0.0.0:3000
