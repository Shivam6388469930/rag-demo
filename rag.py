import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

VECTORSTORE_PATH = "vectorstore"
DATA_PATH = "data/sample.json"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "phi3:mini"


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


def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    print(f"[Ingest] Embedding with {EMBED_MODEL}...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"[Ingest] Vectorstore saved to ./{VECTORSTORE_PATH}/")
    return vectorstore


def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
    print(f"[Query] Vectorstore loaded from ./{VECTORSTORE_PATH}/")
    return vectorstore


def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.7}
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided data."

Context:
{context}

Question: {question}

Answer:""",
    )

    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    def format_docs(docs):
        sources = [doc.metadata["topic"] for doc in docs]
        print(f"[Query] Retrieved from: {sources}")
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(chain, question):
    print(f"\nQ: {question}")
    answer = chain.invoke(question)
    print(f"A: {answer}")
    print("-" * 60)


def main():
    # Ingest only if vectorstore doesn't exist yet
    if not os.path.exists(VECTORSTORE_PATH):
        print("No vectorstore found. Running ingestion...\n")
        docs = load_documents(DATA_PATH)
        vectorstore = build_vectorstore(docs)
    else:
        vectorstore = load_vectorstore()

    chain = build_chain(vectorstore)

    print("\n=== RAG Demo — ask anything about the knowledge base ===\n")

    # Demo questions
    questions = [
        "What is RAG and why is it useful?",
        "How does Ollama help developers?",
        "What vector databases can I use?",
        "What is the capital of France?",
    ]
    for q in questions:
        ask(chain, q)

    # Interactive mode
    print("\n=== Interactive mode (type 'exit' to quit) ===\n")
    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input:
            ask(chain, user_input)


if __name__ == "__main__":
    main()
