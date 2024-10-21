import os
import re
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langsmith import Client
from langsmith.evaluation import evaluate
from sentence_transformers import SentenceTransformer, util


LANGCHAIN_API_KEY = os.getenv["LANGCHAIN_API_KEY"] 
LANGCHAIN_TRACING_V2 = os.getenv["LANGCHAIN_TRACING_V2"] 
OPENAI_API_KEY = os.getenv["OPENAI_API_KEY"] 
LANGCHAIN_ENDPOINT = os.getenv["LANGCHAIN_ENDPOINT"] 
LANGCHAIN_PROJECT = os.getenv["LANGCHAIN_PROJECT"] 

import openai

client = wrap_openai(openai.Client())

model = SentenceTransformer('all-MiniLM-L6-v2')

langsmith_client = Client()

dataset_name = "Test"
dataset = langsmith_client.create_dataset(dataset_name)

input_data = {"postfix": "How much 2 plus 2"}
expected_output = "2 plus 2 equal 4"

langsmith_client.create_examples(
    inputs=[input_data],
    outputs=[{"output": expected_output}],
    dataset_id=dataset.id,
)

def exact_match(run, example):
    model_output_clean = re.sub(r'[^\w\s]', '', run.outputs["output"].strip()).lower()
    expected_output_clean = re.sub(r'[^\w\s]', '', example.outputs["output"].strip()).lower()

    embeddings = model.encode([model_output_clean, expected_output_clean])

    similarity = util.cos_sim(embeddings[0], embeddings[1])

    threshold = 0.8
    score = similarity.item() >= threshold

    print({"score": score, "similarity": similarity.item()})
    return {"score": score}

experiment_results = evaluate(
    lambda input: {"output": client.chat.completions.create(
        messages=[{"role": "user", "content": f"How much {input['postfix']}"}],
        model="gpt-3.5-turbo"
    ).choices[0].message.content},
    data=dataset_name,
    evaluators=[exact_match],
    experiment_prefix="sample-experiment",
    metadata={
        "version": "1.0.0",
        "revision_id": "1"
    },
)

print("Experiment Results:", experiment_results)
