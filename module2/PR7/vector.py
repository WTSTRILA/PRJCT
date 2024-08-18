import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('data.csv')

df = df.drop(columns=['timestamp'])

df['combined'] = df.astype(str).agg(' '.join, axis=1)

vectors = model.encode(df['combined'].tolist())

chroma_client = chromadb.Client()

collection_name = 'sensor_data'
collection = chroma_client.get_or_create_collection(name=collection_name)

vectors = [vector.tolist() for vector in vectors]

collection.upsert(
    documents=df['combined'].tolist(),
    embeddings=vectors,
    ids=[str(i) for i in range(len(vectors))]
)

query_text = "85.0 150.0 0.15 0.17 0.13 1500 12.2 8.9 44 21.8 64 0.552 3.1 74.8 0.03 148 246"
query_vector = model.encode([query_text])[0].tolist()

results = collection.query(
    query_embeddings=[query_vector],
    n_results=1
)

print("Query Results:")
for idx, doc_id in enumerate(results['ids'][0]):
    distance = results['distances'][0][idx]
    document = results['documents'][0][idx]
    print(f"ID: {doc_id}, Distance: {distance}, Document: {document}")
