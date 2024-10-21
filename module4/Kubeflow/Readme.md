# Kubeflow pipelines

1. **Install:**
   ```markdown
   export PIPELINE_VERSION=2.2.0
   kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
   kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
   kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
   
2. **Access UI:**
   ```markdown
   kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

3. **Create treaning job**
   ```markdown
   python ./Kubeflow/kubeflow_training.py 

4. **Create inference job**
   ```markdown
   python ./Kubeflow/kubeflow_inference.py 
