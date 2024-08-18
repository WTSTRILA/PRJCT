# MinIO Deployment Guide

1. **Start MinIO:**
   
   ```markdown
     ./minio server /data --console-address ":9001"
   
3. **Docker Deployment:**
   ```markdown
    docker run -it -p 9000:9000 -p 9001:9001 quay.io/minio/minio server /data --console-address ":9001"

4. **Kubernetes Deployment**

   Create cluster

   ```markdown
   kind create cluster --name ml-in-production
 
  Run k9s
   
   ```markdown
   k9s -A

 Deploy

   ```markdown 
      kubectl create -f minio_storage/minio-dev.yaml
Uses to UI

   ```markdown 
    kubectl port-forward --address=0.0.0.0 pod/minio 9000:9000
    kubectl port-forward --address=0.0.0.0 pod/minio 9001:9001
