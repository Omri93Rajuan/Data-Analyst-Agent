from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from src.config import BASE_URL, MODEL_NAME, NEBIUS_API_KEY
from src.tools import count_rows, show_examples


def build_agent():
    llm = ChatOpenAI(
        api_key=NEBIUS_API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        temperature=0,
        timeout=30,
        max_retries=1,
    )

    tools = [
        StructuredTool.from_function(
            func=count_rows,
            name="count_rows",
            description="Count rows in the Bitext dataset using optional category or intent filters.",
        ),
        StructuredTool.from_function(
            func=show_examples,
            name="show_examples",
            description="Show examples from the Bitext dataset using optional category or intent filters.",
        ),
    ]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a data analyst agent for the Bitext Customer Service dataset. "
            "Use tools for factual answers. "
            "Do not answer questions unrelated to the dataset."
        ),
    )