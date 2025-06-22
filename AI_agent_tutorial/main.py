from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import save_tool, search_tool, wiki_tool


load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Using Gemini Pro model as it better supports tool calling
    temperature=0,  # Lower temperature for more focused outputs
    max_tokens=None,  # No token limit
    top_p=0.8,
    top_k=40,
    # Set your API key through environment variable
    api_key=os.getenv("GEMINI_API_KEY")
)

# Define the tools we'll use
tools = [save_tool, search_tool, wiki_tool]

# Create the output parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant that helps with generating research papers.
    You have access to the following tools:
    - save_text_to_file: Saves structured research data to a text file
    - search: Search the web for information
    - wikipedia: Query Wikipedia for information
    
    Use these tools to help answer the user's questions and conduct research.
    Format your final response according to this structure: {format_instructions}
    """),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
]).partial(format_instructions=parser.get_format_instructions())



# Create the agent with tools
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Create the agent executor with the tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Test the agent
query = "Research the impact of artificial intelligence on healthcare and save the findings"
result = agent_executor.invoke({"input": query})
print("\nRaw Response:")
print(result)

try:
    structured_response = parser.parse(result["output"])
    print("\nStructured Response:")
    print(structured_response)
except Exception as e:
    print(f"\nError parsing response: {e}")
    print("Raw output:", result["output"])


