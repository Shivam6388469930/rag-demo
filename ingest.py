# ingest.py
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


def load_documents(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    docs = []
    for item in data:
        doc = Document(
            page_content=f"Topic: {item['topic']}\nQuestions this answers: {item['questions']}\n{item['content']}",
            metadata={"id": item["id"], "topic": item["topic"]},
        )
        docs.append(doc)

    print(f"Loaded {len(docs)} documents")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    print("Embedding chunks with nomic-embed-text... (takes ~30 sec)")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    print("Vectorstore saved to ./vectorstore")
    return vectorstore


if __name__ == "__main__":
    docs = load_documents("data/sample.json")
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    print("Ingestion complete.")
