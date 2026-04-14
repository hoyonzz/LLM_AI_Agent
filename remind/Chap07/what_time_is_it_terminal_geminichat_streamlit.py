import streamlit as st
import os
import pytz
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types



st.set_page_config(page_title="LLM 타임 에이전트", page_icon="🕒")
load_dotenv()


def get_current_time(timezone: str= 'Asia/Seoul') -> str:
    """현재 타임존의 날짜와 시간을 반환합니다.
    
    Args:
        timezone: 현재 날짜와 시간을 반환할 타임존을 입력하세요. (예: Asia/Seoul, America/New_York)
    """    
    tz = pytz.timezone(timezone)
    now_timezone = f'{datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")} {timezone}'
    
    st.toast(f"도구 실행됨: get_current_time({timezone})")
    return now_timezone

if "chat_session" not in st.session_state:
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    st.session_state.chat_session = client.chats.create(
        model = "gemini-2.5-flash",
        config = types.GenerateContentConfig(
            system_instruction="너는 사용자를 도와주는 친절한 상담사야.",
            tools =[get_current_time],
            temperature=0.7,
        )
    )

    st.session_state.messages = []

st.title(" 채팅 시작 ")
st.caption("Gemini 2.5 Flash 기반 펑션 콜링 실습")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("예: 메시지를 입력해주세요."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role":"user", "content":prompt})

    with st.chat_message("model"):
        # 로딩 스피너를 돌리면서 API 호출 (이 안에서 함수 호출 핑퐁이 자동으로 일어남!)
        with st.spinner("AI가 시간을 확인하는 중입니다..."):
            response = st.session_state.chat_session.send_message(prompt)
            st.markdown(response.text)
            
    # AI의 최종 답변을 기록에 저장
    st.session_state.messages.append({"role": "model", "content": response.text})