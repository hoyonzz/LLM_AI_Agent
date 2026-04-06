from google import genai
from google.genai import types
from dotenv import load_dotenv
import os


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

contents = [
    {"role":"user", "parts":[{"text":"참새"}]},
    {"role":"model", "parts":[{"text":"짹짹"}]},
    {"role":"user", "parts":[{"text":"말"}]},
    {"role":"model", "parts":[{"text":"히이잉"}]},
    {"role":"user", "parts":[{"text":"개구리"}]},
    {"role":"model", "parts":[{"text":"개굴개굴"}]},
    {"role":"user", "parts":[{"text":"뱀"}]},
]

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=contents,
    config=types.GenerateContentConfig(
        system_instruction="너는 유치원생이야. 유치원생처럼 답변해 줘.",
        # thinking_config=types.ThinkingConfig(thinking_level="low"),
        temperature=0.9,
    ),
)

print(response.text)