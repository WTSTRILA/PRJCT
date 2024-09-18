from openai import OpenAI

openai_api_key = " "
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[
        {"role": "system", "content": "You are a math assistant."},
        {"role": "user", "content": "How much is 2 plus 2?"},
    ],
)
print("Chat response:", chat_response)
