import modal
from openai import OpenAI

client = OpenAI(api_key='')

def predict(texts):
    responses = []
    for text in texts:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": text}],
            max_tokens=50,
            temperature=0
        )
        responses.append(completion.choices[0].message.content.strip().lower())
    return responses

modal_app = modal.App()

@modal_app.function()
def modal_predict(texts):
    return predict(texts)

if __name__ == "__main__":
    sample_texts = [
        "How much is 2 plus 2?",
    ]
    results = predict(sample_texts)
    print("Generated Responses:", results)
