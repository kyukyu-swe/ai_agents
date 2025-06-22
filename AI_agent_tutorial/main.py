from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import save_tool, search_tool, wiki_tool
import json


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
    - save_text_to_file: Saves structured research data to a text file. Use this tool when the user asks to save the research.
    - search: Search the web for information
    - wikipedia: Query Wikipedia for information
    
    IMPORTANT INSTRUCTIONS FOR SAVING:
    1. When the user's query contains words like 'save', 'store', or 'write to file', you MUST use the save_text_to_file tool
    2. When saving, format the data as a clear research report with sections
    3. Include all sources and findings in the saved file
    
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

def format_research_for_saving(response):
    """Format the research response into a readable text format"""
    formatted_text = f"""Research Report
==================

Topic: {response.topic}

Summary
-------
{response.summary}

Sources
-------
{"".join(['- ' + source + '\\n' for source in response.sources])}

Tools Used
---------
{"".join(['- ' + tool + '\\n' for tool in response.tools_used])}
"""
    return formatted_text

# Get user input for the research query
print("\n=== AI Research Assistant ===")
print("Enter your research query. Example: 'Research the impact of artificial intelligence on healthcare and save the findings'")
print("Add 'save' or 'save to file' in your query to automatically save the results")
print("Type 'quit' to exit")

while True:
    query = input("\nEnter your research query: ").strip()
    
    if query.lower() == 'quit':
        print("Thank you for using the AI Research Assistant!")
        break
    
    if not query:
        print("Please enter a valid query.")
        continue
    
    print("\nResearching... This may take a moment...")
    try:
        # Execute the research
        result = agent_executor.invoke({"input": query})
        print("\nRaw Response:")
        print(result)

        # Parse the structured response
        structured_response = parser.parse(result["output"])
        print("\nStructured Response:")
        print(structured_response)
        
        # If the query contains save-related keywords, explicitly save the results
        if any(word in query.lower() for word in ['save', 'store', 'write', 'file']):
            formatted_research = format_research_for_saving(structured_response)
            save_result = save_tool.run(formatted_research)
            print(f"\nSave Status: {save_result}")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Please try another query.")
    
    print("\n" + "="*50)


