
import os
import time

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_llm():
    """Create and return the Groq language model."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    )

    return ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=0.2,
        max_retries=0,
    )


def invoke_with_retry(chain, inputs, attempts=4):
    """
    Retry Groq requests when a temporary rate limit occurs.
    Wait progressively longer between attempts.
    """
    for attempt in range(attempts):
        try:
            return chain.invoke(inputs)

        except Exception as exc:
            error_text = str(exc).lower()

            is_rate_limit = (
                "429" in error_text
                or "rate_limit" in error_text
                or "rate limit" in error_text
                or "too many requests" in error_text
            )

            if not is_rate_limit:
                raise

            if attempt == attempts - 1:
                print(
                    "Groq rate limit persists after retries. "
                    "Please wait before running the project again."
                )
                raise

            wait_seconds = 20 * (attempt + 1)

            print(
                f"Groq rate limit reached. "
                f"Waiting {wait_seconds} seconds "
                f"before retry {attempt + 2}/{attempts}..."
            )

            time.sleep(wait_seconds)


def build_chain(system_prompt: str):
    """Build a reusable Groq extraction chain."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])

    return prompt | llm | StrOutputParser()


def extract_action_items(transcript: str) -> str:
    """Extract tasks, owners, and deadlines."""
    if not transcript or not transcript.strip():
        return "No transcript available to extract action items."

    chain = build_chain(
        "You are an expert meeting analyst. From the supplied "
        "transcript, extract actual action items. For each item "
        "include the task, owner if mentioned, and deadline if "
        "mentioned. If no deadline is stated, write 'Not specified'. "
        "Do not invent tasks, owners, or deadlines. Use a numbered "
        "list. If none exist, say 'No action items found.'"
    )

    return invoke_with_retry(chain, {"text": transcript})


def extract_key_decisions(transcript: str) -> str:
    """Extract confirmed decisions from the transcript."""
    if not transcript or not transcript.strip():
        return "No transcript available to extract key decisions."

    chain = build_chain(
        "You are an expert meeting analyst. Extract only key "
        "decisions that were actually made in the supplied transcript. "
        "Do not treat suggestions as confirmed decisions. Do not "
        "invent information. Use a numbered list. If there are no "
        "confirmed decisions, say 'No key decisions found.'"
    )

    return invoke_with_retry(chain, {"text": transcript})


def extract_questions(transcript: str) -> str:
    """Extract unresolved questions and follow-up topics."""
    if not transcript or not transcript.strip():
        return "No transcript available to extract open questions."

    chain = build_chain(
        "You are an expert meeting analyst. Extract unresolved "
        "questions and topics that need follow-up from the supplied "
        "transcript. Do not invent questions. Use a numbered list. "
        "If none exist, say 'No open questions found.'"
    )

    return invoke_with_retry(chain, {"text": transcript})
