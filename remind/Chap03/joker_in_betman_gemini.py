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
    config = types.GenerateContentConfig(
        system_instruction="너는 배트맨에 나오는 조커야. 조커의 악당 캐릭터에 맞게 답변해 줘.",
        temperature=0.9
    )
)

print(response.text)