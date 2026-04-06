from google import genai
from google.genai import types
from dotenv import load_dotenv
import os, pymupdf


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def pdf_to_text(pdf_file_path: str):
    doc = pymupdf.open(pdf_file_path)

    header_height = 80
    footer_height = 80

    full_text = ''

    for page in doc:
        rect = page.rect # 페이지 가져오기
        
        header = page.get_text(clip=(0, 0, rect.width, header_height))
        footer = page.get_text(clip=(0, rect.height-footer_height, rect.width, rect.height))
        text = page.get_text(clip=(0, header_height, rect.width, rect.height-footer_height))

        full_text += text + '\n------------------------------------------\n'

        # 파일명 추출
        pdf_file_name = os.path.basename(pdf_file_path)
        pdf_file_name = os.path.splitext(pdf_file_name)[0]

        txt_file_path = f'./output/{pdf_file_name}_with_preprocessing.txt'

        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

        return txt_file_path
    
def summarize_txt(file_path: str):
    client = genai.Client(api_key=api_key)

    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    system_prompt = f'''
    너는 다음 글을 요약하는 봇이다. 아래 글을 읽고, 저자의 문제 인식과 주장을 파악하고, 주요 내용을 요약하라.

    작성해야 하는 포맷은 다음과 같다.SystemError
    
    # 제목

    ## 저자의 문제 인식 및 주장 (15문장 이내)

    ## 저자 소개

    ============== 이하 텍스트 ==============

    {txt}
    '''

    print(system_prompt)
    print('=========================================')

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=system_prompt,
        config = types.GenerateContentConfig(
            temperature=0.1,
        )
    )

    return response.text

def summarize_pdf(pdf_file_path: str, output_file_path: str):
    txt_file_path = pdf_to_text(pdf_file_path)
    summary = summarize_txt(txt_file_path)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(summary)


if __name__ == '__main__':
    pdf_file_path = "./data/과정기반 작물모형을 이용한 웹 기반 밀 재배관리 의사결정 지원시스템 설계 및 구축.pdf"
    summarize_pdf(pdf_file_path, './output/crop_model_summary2_gemini.txt')
