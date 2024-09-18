from openai import OpenAI

client = OpenAI(
    api_key=" ",
    base_url="http://127.0.0.1:8080/v1",
)

resp = client.chat.completions.create(
    model="alignment-handbook/zephyr-7b-dpo-lora",
    messages=[
        {
            "role": "system",
            "content": "You are a math assistant.",
        },
        {"role": "user", "content": "How much is 2 plus 2?"},
    ],
    max_tokens=100,
)
print("Response:", resp.choices[0].message.content)
