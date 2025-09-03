from dotenv import load_dotenv

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

tools = [TavilySearch()]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# llm = ChatOllama(model="gemma3:270m") 
# ChatOllama might have a different output behaviour 
# OR
# Ollama might not be capable enough of producing the expected output format
# which is why there is bt with the formatting
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
chain = agent_executor

def main():
    print("welcome to search-agents!")
    result = chain.invoke(
        input={
            "input": "search for who's the US president who had an affair. answer in pointers, and what happened to them??",
        }
    )
    print(result['output'])


if __name__ == "__main__":
    main()