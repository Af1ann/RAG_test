import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # загружает переменные, если есть

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
MODEL_NAME = "gemma3:4b"   # можете сменить на другую скачанную модель
EMBEDDING_MODEL = "nomic-embed-text"  # для эмбеддингов (можно использовать ту же модель)

def load_documents(data_dir):
    documents = []
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
    return documents

def create_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    return vectorstore

def main():
    print("Загрузка документов...")
    documents = load_documents(DATA_DIR)
    if not documents:
        print("Папка data пуста. Добавьте файлы .txt или .pdf.")
        return

    print(f"Загружено {len(documents)} документов.")

    if not os.path.exists(PERSIST_DIR):
        print("Создание векторной базы данных...")
        vectorstore = create_vectorstore(documents)
    else:
        print("Загрузка существующей векторной БД...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    # Подключаемся к Ollama
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    Ответь на вопрос на основе предоставленного контекста.
    Если ответа нет в контексте, скажи "Не могу найти ответ в документах".

    Контекст:
    {context}

    Вопрос: {input}
    Ответ:
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("\n=== RAG Demo (локальная модель) ===")
    print("Введите 'exit' для выхода.\n")

    while True:
        query = input("Ваш вопрос: ")
        if query.lower() in ["exit", "quit"]:
            break
        if not query.strip():
            continue

        result = retrieval_chain.invoke({"input": query})
        print("\nОтвет:", result["answer"])
        print("\n--- Найденные фрагменты (контекст): ---")
        for i, doc in enumerate(result["context"]):
            source = doc.metadata.get('source', 'неизвестно')
            page = doc.metadata.get('page', '')
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"[{i+1}] {source} (стр. {page}):\n{content_preview}\n")
        
        print("\nИсточники:")
        for doc in result["context"]:
            source = doc.metadata.get('source', 'неизвестно')
            page = doc.metadata.get('page', '')
            print(f"- {source} (стр. {page})" if page else f"- {source}")
        print("-" * 50)

if __name__ == "__main__":
    main()