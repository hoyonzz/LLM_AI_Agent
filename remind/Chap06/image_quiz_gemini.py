from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from glob import glob


load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return f.read()
    
def image_quiz(image_path):
    image_bytes = encode_image(image_path)

    quiz_prompt = """
    제공한 이미지를 바탕으로, 다음과 같은 양식으로 퀴즈를 만들어 주세요.
    정답은 (1)~(4) 중 하나만 해당하도록 출제하세요.
    아래는 예시입니다.
    ----- 예시 -----

    Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
    - (1) 베이커리에서 사람들이 빵을 사는 모습이 담겨 있습니다.
    - (2) 맨 앞에 서 있는 사람은 빨간색 셔츠를 입었습니다.
    - (3) 기차를 타기 위해 줄을 서 있는 사람들이 있습니다.
    - (4) 점원은 노란색 티셔츠를 입었습니다.

    정답: (4) 점원은 노란색 티셔츠가 아닌 파란색 티셔츠를 입었습니다.
    (주의: 정답은 (1)~(4) 중 하나만 선택하도록 출제하세요.)
    =====
    """

    contents = [
        types.Part.from_bytes(
            data=image_bytes,
            mime_type = 'image/jpeg',
        ),
        quiz_prompt,
    ]

    response = client.models.generate_content(
        model = 'gemini-2.5-flash',
        contents = contents
    )

    return response.text


txt = ''
no = 1
for g in glob('image/quiz/*.jpg'):
    try:
        q = image_quiz(g)
    except Exception as e:
        print(e)
        continue

    divider = f'## 문제 {no}\n\n'
    print(divider)

    txt += divider
    filename = os.path.basename(g)
    txt += f'![image]({filename})\n\n'

    print(q)
    txt += q + '\n\n---------------------\n\n'

    with open('image/quiz/image_quiz_gemini.md', 'w', encoding='utf-8') as f:
        f.write(txt)

    no += 1