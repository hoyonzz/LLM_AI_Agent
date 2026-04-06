import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os


load_dotenv()

with st.sidebar:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    "[Get an OpenAI API Key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("Chatbot")

# 질문 1. gemini sdk는 parts로 들어가야하니 이렇게 들어가면되는지.
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "model", "parts":[{"text":"How can I help you?"}]}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["parts"][0]["text"])

if prompt := st.chat_input():
    if not gemini_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = genai.Client(api_key=gemini_api_key)

    user_msg = {"role":"user", "parts": [{"text":prompt}]}
    st.session_state.messages.append(user_msg)
    st.chat_message("user").write(prompt)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=st.session_state.messages
    )
    msg = response.text
    # 질문2. parts를 넣어야하니 이렇게 들어가야하는지.

    ai_msg = {"role":"model", "parts":[{"text": msg}]}
    st.session_state.messages.append(ai_msg)
    st.chat_message("model").write(msg)