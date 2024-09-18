from openai import OpenAI

BASE_URL = ""
HF_API_KEY = ""

client = OpenAI(
    base_url=os.path.join(BASE_URL, "v1/"),
    api_key=HF_API_KEY,
)
chat_completion = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a math assistant."},
        {"role": "user", "content": "How much is 2 plus 2?"},
    ],
    stream=True,
    max_tokens=500,
)

for message in chat_completion:
    print(message.choices[0].delta.content, end="")
