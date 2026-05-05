import re

from langchain_openai import ChatOpenAI

from security_agent_demo.config import MODEL_NAME, OPENAI_API_BASE, OPENAI_API_KEY, TEMPERATURE


llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
)


def strip_think_tags(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)
    return cleaned.strip()
