# query.py
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local(
        "vectorstore", embeddings, allow_dangerous_deserialization=True
    )
    print("Vectorstore loaded.")
    return vectorstore


def build_rag_chain(vectorstore):
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

    llm = ChatOllama(model="phi3:mini", temperature=0)

    def format_docs(docs):
        for doc in docs:
            print(f"  Retrieved chunk from topic: '{doc.metadata['topic']}'")
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask(chain, question):
    print(f"\nQuestion: {question}")
    print("Retrieved chunks:")
    answer = chain.invoke(question)
    print(f"\nAnswer: {answer}")
    print("-" * 50)


if __name__ == "__main__":
    vectorstore = load_vectorstore()
    chain = build_rag_chain(vectorstore)

    ask(chain, "What is RAG and why is it useful?")
    ask(chain, "How does Ollama help developers?")
    ask(chain, "What is the capital of France?")
