# from openai import OpenAI
# from dotenv import load_dotenv
# import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# load_dotenv()
# api_key = os.getenv('OPENAI_API_KEY')

# client = OpenAI(api_key=api_key)
llm = ChatOpenAI(model='gpt-4o')

# def get_ai_response(messages):
#     response = client.chat.completions.create(
#         model='gpt-4o',
#         temperature=0.9,
#         messages=messages,
#     )
#     return response.choices[0].message.content

messages = [
    # {"role": "system", "content":"너는 사용자를 도와주는 상담사야."},
    SystemMessage('너는 사용자를 도와주는 상담사야.')
] # 초기 시스템 메시지 설정


while True:
    user_input = input('사용자: ')
    
    if user_input == 'exit':
        break
    
    messages.append(
        # {"role":"user", "content": user_input}
        HumanMessage(user_input)
    )

    # ai_response = get_ai_response(messages)
    ai_response = llm.invoke(messages)
    messages.append(
        # {"role":"assistant", "content": ai_response}
        ai_response
    )
    print("AI: " + ai_response.content)
