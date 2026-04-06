from google import genai
from google.genai import types
from dotenv import load_dotenv
import os


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def get_ai_response(contents):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.9,
            system_instruction="너는 사용자를 도와주는 상담사야.",
        )
    )
    return response.text

contents = []

while True:
    user_input = input("사용자: ")
    
    if user_input == "exit": break

    contents.append({"role":"user", "parts":[{"text":user_input}]})
    ai_response = get_ai_response(contents)
    contents.append({"role":"model","content":ai_response})

    print("AI : " + ai_response)