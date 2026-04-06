from google import genai
from google.genai import types
from dotenv import load_dotenv
import os


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents = "세상에서 누가 제일 아름답니?",
    config=types.GenerateContentConfig(
        system_instruction="너는 백설공주 이야기 속의 마법 거울이야. 그 이야기의 캐릭터에 부합하게 답변해줘.",
        temperature=0.9,
    )
)

print(response.text)