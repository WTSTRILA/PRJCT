# Kubeflow pipeline

1. **Create kind clusterl:**
   
   ```markdown
     kind create cluster --name ml-in-production
   
2. **Install Kubeflow:**
   ```markdown
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
    kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"

3. **Kubernetes Deployment**
   ```markdown
   kind create cluster --name ml-in-production
   k9s -A
   kubectl create -f minio_storage/minio-dev.yaml
   kubectl port-forward --address=0.0.0.0 pod/minio 9000:9000
   kubectl port-forward --address=0.0.0.0 pod/minio 9001:9001

1. **Create kind clusterl:**
   
   ```markdown
     kind create cluster --name ml-in-production
   
2. **Install Kubeflow:**
   ```markdown
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
    kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
