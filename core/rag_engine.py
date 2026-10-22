
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)


def get_llm():
    """Create and return the Groq language model."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file "
            "and make sure it is loaded."
        )

    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    return ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=0.3,
        max_retries=0,
    )


def format_docs(docs):
    """Combine retrieved document chunks into context."""

    return "\n\n".join(
        doc.page_content for doc in docs
    )


def create_rag_chain(retriever):
    """Create the reusable retrieval-augmented generation chain."""

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting transcript
context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Be concise and precise. Do not invent information.
If quoting someone, make that clear.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_chain(transcript: str):
    """Build a RAG chain from a new transcript."""

    if not transcript or not transcript.strip():
        raise ValueError(
            "Cannot build the RAG chain: transcript is empty."
        )

    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)

    return create_rag_chain(retriever)


def load_rag_chain():
    """Load an existing vector store and build its RAG chain."""

    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)

    return create_rag_chain(retriever)


def ask_question(rag_chain, question: str) -> str:
    """Ask a question about the meeting transcript."""

    if not question or not question.strip():
        return "Please enter a question."

    print(f"Question: {question}")

    answer = rag_chain.invoke(question.strip())

    print(f"Answer: {answer}")

    return answer
