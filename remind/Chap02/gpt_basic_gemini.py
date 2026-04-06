from google import genai
from google.genai import types
from dotenv import load_dotenv
import os


load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model = 'gemini-2.5-flash',
    contents = '2022년 월드컵 우승 팀은 어디야?',
    config=types.GenerateContentConfig(
        system_instruction='You are a helpful assistant.',
        temperature=0.1,
    )
)

print(response)

print('----')
print(response.text)