from dotenv import load_dotenv
import os
from groq import Groq

print("1. Starting...")

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL")

print("2. API key loaded:", bool(api_key))
print("3. Model:", model)

if not api_key:
    raise ValueError("GROQ_API_KEY is missing")

if not model:
    raise ValueError("GROQ_MODEL is missing")

print("4. Creating client...")

client = Groq(api_key=api_key)

print("5. Sending request...")

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: GROQ OK"
            }
        ],
        max_tokens=20,
    )

    print("6. Response received")
    print("7. Full response:", response)
    print("8. Content:", response.choices[0].message.content)

except Exception as e:
    print("ERROR TYPE:", type(e).__name__)
    print("ERROR:", repr(e))