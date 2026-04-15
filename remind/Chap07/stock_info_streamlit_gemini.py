import streamlit as st
import os
import pytz
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
import yfinance as yf
import json



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

def get_yf_stock_info(ticker: str) -> str:
    """
    특정 주식 티커(예: AAPL, TSLA)의 현재 회사 정보를 반환합니다.
    
    Args:
        ticker: 검색할 주식의 영문 티커 (예: AAPL, MSFT, TSLA)
    """
    info = yf.Ticker(ticker).info
    return json.dumps(info, ensure_ascii=False)

def get_yf_stock_history(ticker: str, period: str)->str :
    """
    특정 주식의 과거 가격 변동 기록을 반환합니다.
     
    Args:
        ticker: 검색할 주식의 영문 티커 (예: AAPL, TSLA)
        periond: 검색할 기간 (예: '1d', '5d', '1mo', '1y')
    """
    history_df = yf.Ticker(ticker).history(period=period)
    return history_df.to_markdown() 

def get_yf_stock_recommendations(ticker: str) -> str:
    """
    특정 주식의 추천 상황을 반환합니다.

    Args:
        ticker: 검색할 주식의 영문 티커 (예: AAPL, TSLA)
    """
    recommendations = yf.Ticker(ticker).recommendations
    return recommendations.to_markdown()

my_tools = [get_current_time, get_yf_stock_info, get_yf_stock_history, get_yf_stock_recommendations]

if "chat_session" not in st.session_state:
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    st.session_state.chat_session = client.chats.create(
        model = "gemini-2.5-flash",
        config = types.GenerateContentConfig(
            system_instruction="너는 사용자를 도와주는 친절한 상담사야.",
            tools = my_tools,
            temperature=0.7,
        )
    )

    st.session_state.messages = []

st.title(" 채팅 시작 ")
st.caption("Gemini 2.5 Flash 기반 펑션 콜링 실습")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("메시지를 입력해주세요."):
    st.session_state.messages.append({"role":"user", "content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

        with st.chat_message("model"):
            # 로딩 스피너를 돌리면서 API 호출 (이 안에서 함수 호출 핑퐁이 자동으로 일어남!)
            message_placeholder = st.empty()
            full_content = ""

            with st.spinner("AI가 데이터를 분석하고 있습니다..."):

                response_stream = st.session_state.chat_session.send_message_stream(prompt)

                for chunk in response_stream:
                    if chunk.text:
                        full_content += chunk.text
                        message_placeholder.markdown(full_content + "▌")

            message_placeholder.markdown(full_content)

    st.session_state.messages.append({"role": "model", "content": full_content})

