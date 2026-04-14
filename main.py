from dotenv import load_dotenv
from pydantic import BaseModel
#from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
# response = llm.invoke("what is a trust in finance")
# print(response)

# llm2 = ChatGroq(model="groq/compound-mini")
# response = llm2.invoke("what is a trust in finance")
# print(response)


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions = parser.get_format_instructions)

agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt,
    tools = []
)

agent_executor = AgentExecutor(agent=agent, tools = [], verbose = False)

raw_response = agent_executor.invoke({"query": "What is a trust in finance"})
print(raw_response)
