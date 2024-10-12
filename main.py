from typing import List

from dotenv import load_dotenv
from langchain.agents import tool, AgentExecutor, initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import Tool

from callback import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case

    return len(text)


def find_tool_by_name(tools: List[Tool], tools_name: str):
    for tool in tools:
        if tool.name == tools_name:
            return tool
    raise ValueError(f"Tool with name {tools_name} not found")


if __name__ == "__main__":
    print("Hello ReAct LangChain")
    llm = ChatOpenAI(
        callbacks=[AgentCallbackHandler()]
    )
    agent_executor: AgentExecutor = initialize_agent(
        tools=[get_text_length],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    agent_executor.invoke({"input": "What is the length of the text Dog?"})