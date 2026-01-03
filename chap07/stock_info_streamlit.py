from gpt_functions import get_current_time, tools, get_yf_stock_info, get_yf_stock_recommendations, get_yf_stock_history
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import streamlit as st
from collections import defaultdict


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)



def tool_list_to_tool_obj(tools):
    tool_calls_dict = defaultdict(lambda: {"id":None, "function": {"arguments": "", "name": None}, "type": None})

    for tool_call in tools:
        # id가 None이 아닌 경우
        if tool_call.id is not None:
            tool_calls_dict[tool_call.index]["id"] = tool_call.id

        # 함수 이름이 None이 아닌 경우
        if tool_call.function.name is not None:
            tool_calls_dict[tool_call.index]["function"]["name"] = tool_call.function.name

        # 인자 추가
        tool_calls_dict[tool_call.index]["function"]["arguments"] += tool_call.function.arguments

        # 타입이 None이 아닌 경우 설정
        if tool_call.type is not None:
            tool_calls_dict[tool_call.index]["type"] = tool_call.type
        
    tool_calls_list = list(tool_calls_dict.values())

    return {"tool_calls": tool_calls_list} # 반환


def get_ai_response(messages, tools = None, stream=True):
    response = client.chat.completions.create(
        model = 'gpt-4o',
        stream = stream,
        messages=messages,
        tools=tools,
    )
    
    if stream:
        for chunk in response:
            yield chunk
    else:
        return response

st.title("💬 chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content":"너는 사용자를 도와주는 상담사야."},
    ]

for msg in st.session_state.messages:
    if msg["role"] == "assistant" or msg["role"] == "user":
        st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content":user_input})
    st.chat_message("user").write(user_input)

    ai_response = get_ai_response(st.session_state.messages, tools=tools)
    content = ''
    tool_calls = None
    tool_calls_chunk = []


    with st.chat_message("assistant").empty():
        for chunk in ai_response:
            content_chunk = chunk.choices[0].delta.content
            if content_chunk:
                print(content_chunk, end="")
                content += content_chunk
                st.markdown(content)
                # print(chunk)
            
            if chunk.choices[0].delta.tool_calls:
                tool_calls_chunk += chunk.choices[0].delta.tool_calls
        tool_obj = tool_list_to_tool_obj(tool_calls_chunk)     
        tool_calls = tool_obj["tool_calls"]

        if len(tool_calls) > 0:
            # 만약 tool_calls가 존재하면, st.write로 tool_call 내용 출력
            print(tool_calls)
            tool_call_msg = [tool_call["function"] for tool_call in tool_calls]
        st.write(tool_call_msg)
    print('\n=============')
    print(content)
    # ai_message = ai_response.choices[0].message
    # print(ai_message)

    # print('\n============= tool_calls_chunk')
    # for tool_call_chunk in tool_calls_chunk:
        # print(tool_call_chunk)





    # tool_calls = ai_message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            # tool_name = tool_call.function.name
            # tool_call_id = tool_call.id
            # arguments = json.loads(tool_call.function.arguments)
            
            # 딕셔너리 형태에서 받기
            tool_name = tool_call["function"]["name"]
            tool_call_id = tool_call["id"]
            arguments = json.loads(tool_call["function"]["arguments"])

            if tool_name == "get_current_time":
                func_result = get_current_time(timezone=arguments['timezone'])
            elif tool_name == "get_yf_stock_info":
                func_result = get_yf_stock_info(ticker=arguments['ticker']),
            elif tool_name == "get_yf_stock_history":
                func_result = get_yf_stock_history(
                    ticker = arguments['ticker'],
                    period = arguments['period']
                )
            elif tool_name == "get_yf_stock_recommendations":
                func_result = get_yf_stock_recommendations(
                    ticker = arguments['ticker']
                )

            st.session_state.messages.append({
                "role": "function", 
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content" : func_result,
            })

        st.session_state.messages.append({"role":"system", "content": "이제 주어진 결과를 바탕으로 답변할 차례다."})
        ai_response = get_ai_response(st.session_state.messages, tools=tools)
        # ai_message = ai_response.choices[0].message
        content = ""
        with st.chat_message("assistant").empty():
            for chunk in ai_response:
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    print(content_chunk, end='')
                    content += content_chunk
                    st.markdown(content)

    st.session_state.messages.append({
        "role":"assistant",
        "content": content
    })

    print("AI\t: " + content)
    # st.chat_message("assistant").write(ai_message.content)