from dotenv import load_dotenv

from langchain import hub
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

tools = [TavilySearch()]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# llm = ChatOllama(model="gemma3:270m")

react_prompt = hub.pull("hwchase17/react")

# reasoning agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

# action agent
# this is a wrapper around the reasoning agent in order to perform smthng
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True
    # verbose=True lite
)




def main():
    print("welcome to search-agents!")

    response = agent_executor.invoke(
        input = {
            "input": """
                Search linkedin for 3 people who study at bits pilani and are society for student mess services (ssms) election commissioners.
            """
        }
    )

    print(response['output'])


if __name__ == "__main__":
    main()