import streamlit as st
# 스트림릿을 위한 라이브러리

from langchain_openai import ChatOpenAI
# 메모리에 대화 기록 저장하는 클래스
from langchain_core.chat_history import InMemoryChatMessageHistory
# 메시지 기록을 활용해 실행할 수 있는 wrapper 클래스
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from dotenv import load_dotenv


load_dotenv()


# 스트림릿 제목 설정
st.title("💬 Chatbot")

# 스트림릿의 세션상태에 메시지가 없다면, system설정
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 사용자의 질문에 친절히 답하는 AI 챗봇이다.")
    ]


# 세션별 대화 기록을 저장할 딕셔너리 대신 session_state사용
if "store" not in st.session_state:
    st.session_state["store"] = {}

# 대화 이력 객체 불러오는 함수 정의: session_state의 스토어에 세션 id가 없다면, 메모리 객체 생성
def get_session_history(session_id: str):
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = InMemoryChatMessageHistory()
    return st.session_state["store"][session_id]

# llm모델 설정, runnablewithmessagehistory로 이어쓰는 객체 생성
llm = ChatOpenAI(model="gpt-4o-mini")
with_message_history = RunnableWithMessageHistory(llm, get_session_history)

# config설정: session_id 부여
config = {"configurable": {"session_id": "abc2"}}

# 스트림릿에 메시지가 메시지 종류에 따라 다르게 스트림릿에 입력, isinstance를 왜쓰는지 질문있음.
for msg in st.session_state.messages:
    if msg:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

# := 기호에 대해서 질문 어떻게 적용되는지 이해하기 쉽게 설명바람.
# 1. prompt는 chat_input이 되고, prompt를 사용자메시지로 추가, 그리고 스트림릿에 user로 입력
if prompt := st.chat_input():
    print('user:', prompt)
    st.session_state.messages.append(HumanMessage(prompt))
    st.chat_message('user').write(prompt)

    # 스트림 객체로 세션id부여해서 llm에게 전달
    response = with_message_history.stream([HumanMessage(prompt)], config=config)

    # stream 이 비어있다면, r을 추가하고, 그다음것은 이어 붙여서 출력
    ai_response_bucket = None
    with st.chat_message("assistant").empty():
        for r in response:
            if ai_response_bucket is None:
                ai_response_bucket = r
            else:
                ai_response_bucket += r
            print(r.content, end='')
            # markdown으로 하는것은 스트림릿의 규정인가?
            st.markdown(ai_response_bucket.content)

    

    msg = ai_response_bucket.content
    st.session_state.messages.append(ai_response_bucket)
    st.chat_message("assistant").write(msg)
    print('assistant: ', msg)