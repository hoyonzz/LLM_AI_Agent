import os
import pytz
from datetime import datetime
from dotenv import load_dotenv


from google import genai
from google.genai import types
import time
from google.genai import errors

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def get_current_time(timezone:str = 'Asia/Seoul') -> str:
    """현재 타임존의 날짜와 시간을 반환합니다.
    
    Args:
        timezone: 현재 날짜와 시간을 반환할 타임존을 입력하세요. 
        (예: Asia/Seoul, America/New_York)
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    now_timezone = f'{now} {timezone}'

    print(f"\n[시스템 로그: get_current_time 함수가 실행되었습니다. ({now_timezone})]\n")
    return now_timezone

chat = client.chats.create(
    model="gemini-2.5-flash",
    config = types.GenerateContentConfig(
        system_instruction="너는 사용자를 도와주는 친절한 상담사야.",
        tools=[get_current_time],
        temperature=0.7,
    )
)

print("AI 상담사와 대화를 시작합니다. 종료하려면 'exit'를 입력하세요.")

while True:
    user_input = input("사용자\t: ")
    if user_input.lower() == "exit":
        break

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = chat.send_message(user_input)
            print(f"AI\t: {response.text}")
            break
        
        except errors.ServerError as e:
            if "503" in str(e):
                wait_time = attempt + 1
                print(f"\n[시스템] 서버 혼잡으로 {wait_time}초 후 재시도합니다...(시도 횟수: {attempt + 1}/{max_retries})\n")
            else:
                raise e
        
        except Exception as e:
            print(f"에러발생: {e}")
            break
    
    else:
        print("AI\t: 현재 서버 상태가 불안정하여 답변을 받을 수 없습니다. 잠시 후 다시 시도해주세요.")

