from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(model='text-embedding-3-large')


# 언어 모델
from langchain_ollama import ChatOllama
llm = ChatOllama(model="deepseek-r1:8b")

from langchain_chroma import Chroma
print("Loading existing chroma store")
persist_directory = 'C:/Users/hoyon/OneDrive/바탕 화면/Develope/InProgress/LLM_AI_Agent/chap11/chroma_store'

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# Create Retriever
retriever = vectorstore.as_retriever(k=3)

# Create document chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(llm, question_answering_prompt) | StrOutputParser()

query_augmentation_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "기존의 대화 내용을 활용하여 아래 질문의 의도를 파악하여 명료한 한 문장의 질문으로 변환해라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라. :\n\n{query}",
        )
    ]
)

query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()