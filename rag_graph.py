# rag_graph.py — RAG pipeline built as a LangGraph state machine
import json
from config import config
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Config
VECTORSTORE_PATH = config.vectorstore_path
DATA_PATH = config.data_path
EMBED_MODEL = config.embed_model
LLM_MODEL = config.llm_model
MAX_RETRIES = config.max_retries


# State
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retries: int


# Shared resources
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=config.llm_temperature)
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": config.retriever_k,
        "fetch_k": config.retriever_fetch_k,
        "lambda_mult": config.retriever_lambda,
    },
)


# Node: rewrite_query
def rewrite_query(state: GraphState) -> GraphState:
    question = state["question"]
    logger.info(f"Rewriting query: '{question}'")
    rewrite_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Rewrite the following question to be more specific and
better suited for semantic document search. Return only the rewritten question.

Original: {question}
Rewritten:""",
    )
    chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"question": question}).strip()
    logger.info(f"Rewritten query: '{question}' → '{rewritten}'")
    return {"question": rewritten}


# Node: retrieve
def retrieve(state: GraphState) -> GraphState:
    logger.info("Searching vectorstore...")
    docs = retriever.invoke(state["question"])
    topics = [d.metadata["topic"] for d in docs]
    logger.info(f"Retrieved chunks from topics: {topics}")
    return {"documents": docs}


# Node: grade_documents
def grade_documents(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["documents"]
    retries = state.get("retries", 0)

    grader_prompt = PromptTemplate(
        input_variables=["document", "question"],
        template="""You are a relevance grader. Given a document and a question,
respond with only 'yes' if the document contains information relevant to answering
the question, or 'no' if it does not.

Document: {document}
Question: {question}
Relevant (yes/no):""",
    )
    grader_chain = grader_prompt | llm | StrOutputParser()
    relevant = []
    for doc in documents:
        result = (
            grader_chain.invoke({"document": doc.page_content, "question": question})
            .strip()
            .lower()
        )
        if result.startswith("yes"):
            relevant.append(doc)
            logger.info(f"PASS - Topic: '{doc.metadata['topic']}'")
        else:
            logger.info(f"FAIL - Topic: '{doc.metadata['topic']}'")
    return {"documents": relevant, "retries": retries + 1}


# Conditional edge: decide_after_grade
def decide_after_grade(state: GraphState) -> str:
    if len(state["documents"]) > 0:
        logger.info("Relevant chunks found — moving to generate")
        return "generate"
    if state["retries"] >= MAX_RETRIES:
        logger.info("Max retries hit — generating with no context")
        return "generate"
    logger.info("No relevant chunks — retrying retrieval")
    return "rewrite_query"  # Changed from "retrieve" to "rewrite_query"


# Node: generate
def generate(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["documents"]
    logger.info(f"Generating answer using {LLM_MODEL}...")
    if not documents:
        logger.warning("No relevant documents found — returning 'I don't know'")
        return {"generation": "I don't know based on the provided data."}
    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided data."

Context:
{context}

Question: {question}

Answer:""",
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    logger.info("Answer generated")
    return {"generation": response}


# Build the graph
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grade,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()


# Run
def ask(app, question: str):
    logger.info(f"Question: {question}")
    result = app.invoke(
        {"question": question, "documents": [], "generation": "", "retries": 0}
    )
    logger.info(f"Answer: {result['generation']}")


def main():
    app = build_graph()
    questions = [
        "What is RAG and why is it useful?",
        "How does Ollama help developers?",
        "What is the capital of France?",
    ]
    for q in questions:
        ask(app, q)
    logger.info("Interactive mode (type 'exit' to quit)")
    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input:
            ask(app, user_input)


if __name__ == "__main__":
    main()
