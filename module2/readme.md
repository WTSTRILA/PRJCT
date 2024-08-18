# MinIO Deployment Guide

1. **Start MinIO:**
   """
  ./minio server /data --console-address ":9001"
   
3. **Docker Deployment:**

    docker run -it -p 9000:9000 -p 9001:9001 quay.io/minio/minio server /data --console-address ":9001"

4. **Kubernetes Deployment**
    kind create cluster --name ml-in-production
   
    k9s -A

    kubectl create -f minio_storage/minio-dev.yaml

    kubectl port-forward --address=0.0.0.0 pod/minio 9000:9000
    kubectl port-forward --address=0.0.0.0 pod/minio 9001:9001
