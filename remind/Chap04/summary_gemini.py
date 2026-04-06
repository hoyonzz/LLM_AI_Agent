from google import genai
from google.genai import types
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def summarize_txt(file_path: str):
    client = genai.Client(api_key=api_key)

    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    system_prompt = f'''
    너는 다음 글을 요약하는 봇이다. 아래 글을 읽고, 저자의 문제 인식과 주장을 파악하고, 주요 내용을 요약하라.

    작성해야 하는 포맷은 다음과 같다.

    # 제목

    ## 저자의 문제 인식 및 주장 (15문장 이내)

    ## 저자 소개

    ================== 이하 텍스트 ==================

    {txt}
    '''

    print(system_prompt)
    print('=================================================')

    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = system_prompt,
        config = types.GenerateContentConfig(
            temperature=0.1,
        )
    )

    return response.text

if __name__ == '__main__':
    file_path = './output/과정기반 작물모형을 이용한 웹 기반 밀 재배관리 의사결정 지원시스템 설계 및 구축_with_preprocessing.txt'

    summary = summarize_txt(file_path)
    print(summary)

    with open('./output/crop_model_summary_gemini.txt', 'w', encoding='utf-8') as f:
        f.write(summary)