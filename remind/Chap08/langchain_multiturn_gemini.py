from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage



llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview')

messages = [
    SystemMessage("너는 사용자를 도와주는 상담사야.")
]

while True:
    user_input = input("사용자: ")
    
    if user_input == 'exit':
        break

    messages.append(
        HumanMessage(user_input)
    )

    ai_response = llm.invoke(messages)

    messages.append(
        ai_response
    )

    print("AI:" , ai_response.content)