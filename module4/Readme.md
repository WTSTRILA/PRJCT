# Kubeflow pipeline

1. **Create kind clusterl:**

  ```markdown
    kind create cluster --name module4

2. ** Install Kubeflow**

  ```markdown

    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
    kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
