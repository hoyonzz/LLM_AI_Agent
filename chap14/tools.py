from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from datetime import datetime
import json
import os
absolute_path = os.path.abspath(__file__)
current_path = os.path.dirname(absolute_path)

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma



embedding = OllamaEmbeddings(model='bge-m3:latest')

persist_directory = f"{current_path}/data/chroma_store"

vectorstore = Chroma(
    persist_directory = persist_directory,
    embedding_function=embedding
)

@tool
def web_search(query: str):
    """
    주어진 query에 대해 웹 검색을 하고 결과를 반환한다.

    Args:
        query(str): 검색어

    Returns:
        dict: 검색 결과
    """

    client = TavilyClient()

    content = client.search(
        query,
        search_depth="advanced",
        include_raw_content=True,
    )

    results = content["results"]

    for result in results:
        if result["raw_content"] is None:
            try:
                result["raw_content"] = load_web_page(result["url"])
            except Exception as e:
                print(f"Error loading page: {result['url']}")
                print(e)
                result["raw_content"] = result["content"]

    resources_json_path = f"{current_path}/data/resources_{datetime.now().strftime('%Y-%m%d_%H%M%S')}.json"
    with open(resources_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


    return results, resources_json_path

def web_page_to_document(web_page):
    if len(web_page['raw_content']) > len(web_page['content']):
        page_content = web_page['raw_content']
    else:
        page_content = web_page['content']

    document = Document(
        page_content=page_content,
        metadata = {
            'title': web_page['title'],
            'source': web_page['url']
        }
    )
    return document

def web_page_json_to_documents(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        resources = json.load(f)

    documents = []

    for web_page in resources:
        document = web_page_to_document(web_page)
        documents.append(document)

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    print('Splitting documents...')
    print(f"{len(documents)}개의 문서를 {chunk_size}자 크기로 중첩 {chunk_overlap}자로 분할 합니다.\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, chunk_overlap = chunk_overlap
    )

    splits = text_splitter.split_documents(documents)

    print(f"총 {len(splits)}개의 문서로 분할되었습니다.")
    return splits

def documents_to_chroma(documents, chunk_size=1000, chunk_overlap=100):
    print("Documents를 Chroma DB에 저장합니다.")

    # documents에서 url수집
    urls = [document.metadata['source'] for document in documents]

    # _collection은 무슨 문법인지 모르겠음. 벡터DB에 있는 메타데이터들을 수집하고, url수집
    stored_metadatas = vectorstore._collection.get()['metadatas']
    stored_web_urls = [metadata['source'] for metadata in stored_metadatas]

    # 벡터db에 있는 url과 새로운 url을 비교해서 리스트에 담음
    new_urls = set(urls) - set(stored_web_urls)

    new_documents = []

    for document in documents:
        if document.metadata['source'] in new_urls:
            new_documents.append(document)
            print(document.metadata)

    splits = split_documents(new_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if splits:
        vectorstore.add_documents(splits)
    else:
        print("No new urls to process")

def add_web_pages_json_to_chroma(json_file, chunk_size=1000, chunk_overlap=100):
    documents = web_page_json_to_documents(json_file)
    documents_to_chroma(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

def load_web_page(url: str):
    loader = WebBaseLoader(url, verify_ssl=False)

    content = loader.load()

    raw_content = content[0].page_content.strip()

    while '\n\n\n' in raw_content or '\t\t\t' in raw_content:
        raw_content = raw_content.replace('\n\n\n', '\n\n')
        raw_content = raw_content.replace('\t\t\t', '\t\t')
    
    return raw_content

@tool
def retrieve(query: str, tok_k: int=5):
    """
    주어진 query에 대해 벡터 검색을 수행하고, 결과를 반환한다.
    """
    retriever = vectorstore.as_retriever(search_kwargs={'k': tok_k})
    retrieved_docs = retriever.invoke(query)

    return retrieved_docs

if __name__ == "__main__":
    # results, resources_json_path = web_search.invoke("2026년 하반기 한국 주식 시장 전망")
    # print(results)

    # documents = web_page_json_to_documents(f"{current_path}/data/resources_2026-0524_171857.json")

    # splits = split_documents(documents)

    # print(splits)

    # documents = web_page_json_to_documents(f'{current_path}/data/resources_2026-0524_171857.json')

    # print(documents[-1])

    # add_web_pages_json_to_chroma(f'{current_path}/data/resources_2026-0524_171857.json')

    retrieved_docs = retrieve.invoke({"query": "한국 주식시장 미래 전망"})
    print(retrieved_docs)