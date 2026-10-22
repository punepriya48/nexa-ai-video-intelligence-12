
import os
import time

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def invoke_with_retry(chain, inputs, attempts=3):
    """Retry temporary Groq rate-limit errors."""

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

            if not is_rate_limit or attempt == attempts - 1:
                raise

            wait_seconds = 15 * (attempt + 1)

            print(
                f"Groq rate limit reached. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)


def split_transcript(transcript: str) -> list[str]:
    """Split a long transcript into smaller sections."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """Generate a summary with fallback handling."""

    if not transcript or not transcript.strip():
        return "No transcript available to summarize."

    try:
        llm = get_llm()
        chunks = split_transcript(transcript)

        map_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Summarize this transcript portion accurately. "
                "Include important points and do not invent information."
            ),
            ("human", "{text}"),
        ])

        map_chain = map_prompt | llm | StrOutputParser()
        chunk_summaries = []

        for index, chunk in enumerate(chunks, start=1):
            print(
                f"Summarizing section {index}/{len(chunks)}..."
            )

            summary = invoke_with_retry(
                map_chain,
                {"text": chunk},
            )

            chunk_summaries.append(summary)

        combined = "\n\n".join(chunk_summaries)

        combined_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Combine these partial summaries into one clear "
                "summary using bullet points. Do not invent facts."
            ),
            ("human", "{text}"),
        ])

        combined_chain = (
            combined_prompt
            | llm
            | StrOutputParser()
        )

        return invoke_with_retry(
            combined_chain,
            {"text": combined},
        )

    except Exception as exc:
        print(f"Summary generation failed: {exc}")

        return (
            "Summary unavailable because the Groq AI service failed. "
            "Check your API key, model availability, and usage limits."
        )


def generate_title(transcript: str) -> str:
    """Generate a short title with fallback handling."""

    if not transcript or not transcript.strip():
        return "Untitled Video"

    try:
        llm = get_llm()

        title_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Generate a short, accurate title for the supplied "
                "video transcript. Use no more than 8 words. "
                "Return only the title."
            ),
            ("human", "{text}"),
        ])

        title_chain = (
            title_prompt
            | llm
            | StrOutputParser()
        )

        return invoke_with_retry(
            title_chain,
            {"text": transcript[:2000]},
        )

    except Exception as exc:
        print(f"Title generation failed: {exc}")
        return "Video Summary"
