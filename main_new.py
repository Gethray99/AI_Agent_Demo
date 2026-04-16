from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import search_tool, wiki_tool, save_tool, search_tool_new
import uuid
#from deepagents import create_deep_agent

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

agent = create_agent(
    model = llm,
    tools = [search_tool_new, wiki_tool, save_tool],
    system_prompt="""
You are a research assistant that will help generate a research paper. CRITICAL RULE: Before you output your final response to the user, you MUST 
    always use the `save_tool` to save your final summary and sources to a text file. 
    Do not skip this step under any circumstances.""",
    response_format = ResearchResponse
)

user_query = input("What can i help you with? ")

response = agent.invoke({
    "messages":
    [{"role": "user", "content": user_query}]})

print(response.get("structured_response"))