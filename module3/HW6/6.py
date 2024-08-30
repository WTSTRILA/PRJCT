from openai import OpenAI

client = OpenAI(api_key=' ')

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


def test_directional():
    tokens = ["text classification", "image classification"]
    texts = [f"ML applied to {token}." for token in tokens]
    predictions = predict(texts)
    print("Directional Test Results:", predictions)
    assert predictions[0] != predictions[1], "Directional test failed!"


def test_minimum_functionality():
    tokens = ["natural language processing", "mlops"]
    texts = [f"{token} is the next big wave in machine learning." for token in tokens]
    predictions = predict(texts)
    print("Minimum Functionality Test Results:", predictions)
    assert len(predictions) == len(tokens), "Minimum functionality test failed!"


def test_context_dependency():
    conversations = [
        [{"role": "user", "content": "Tell me about machine learning."},
         {"role": "user", "content": "What are its main applications?"}],
        [{"role": "user", "content": "Describe the weather."},
         {"role": "user", "content": "What is the temperature?"}]
    ]
    responses = []
    for conversation in conversations:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=conversation,
            max_tokens=50,
            temperature=0
        )
        responses.append(completion.choices[0].message.content.strip())
    print("Context Dependency Test Results:", responses)
    assert responses[0] != responses[1], "Context dependency test failed!"

def test_response_length():
    texts = [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms."
    ]
    max_lengths = [5, 50]
    responses = []
    for text, max_length in zip(texts, max_lengths):
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": text}],
            max_tokens=max_length,
            temperature=0
        )
        responses.append(len(completion.choices[0].message.content.strip().split()))
    print("Response Length Test Results:", responses)
    assert all(len <= max_length for len, max_length in zip(responses, max_lengths)), "Response length test failed!"

def test_creativity():
    prompts = [
        "Describe a new futuristic technology.",
        "Invent a new sport for space travelers."
    ]
    responses = []
    for prompt in prompts:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=1
        )
        responses.append(completion.choices[0].message.content.strip())
    print("Creativity Test Results:", responses)


def test_style_sensitivity():
    texts = [
        "PLEASE GIVE ME A DETAILED EXPLANATION OF THE THEORY OF RELATIVITY.",
        "Could you provide a brief overview of the theory of relativity?"
    ]
    responses = []
    for text in texts:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": text}],
            max_tokens=50,
            temperature=0.7
        )
        responses.append(completion.choices[0].message.content.strip())
    print("Style Sensitivity Test Results:", responses)
    assert responses[0] != responses[1], "Style sensitivity test failed!"

if __name__ == "__main__":
    test_directional()
    test_minimum_functionality()
    test_context_dependency()
    test_response_length()
    test_creativity()
    test_style_sensitivity()
    print("All tests passed successfully!")
