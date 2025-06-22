# AI Research Assistant

A powerful research assistant built with LangChain and Google's Gemini AI that can search, analyze, and save research findings.

## Features

- Web search using DuckDuckGo
- Wikipedia article queries
- Automatic saving of research findings
- Structured output format
- Tool-based architecture for extensibility

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd AI_agent_tutorial
```

2. Create and activate a virtual environment:

```bash
python -m venv ai_agent_env
source ai_agent_env/bin/activate  # On Windows: ai_agent_env\Scripts\activate
```

3. Install the required packages:

```bash
pip install langchain-google-genai langchain python-dotenv wikipedia duckduckgo-search pydantic
```

## Configuration

1. Create a `.env` file in the project root directory
2. Add your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

You can get your Gemini API key from [Google AI Studio](https://ai.google.dev/).

## Project Structure

- `main.py`: Main application file containing the AI agent setup
- `tools.py`: Custom tools for search, Wikipedia queries, and file saving
- `.env`: Environment variables (API keys)

## Usage

1. Make sure your virtual environment is activated
2. Run the main script:

```bash
python main.py
```

The agent will:

1. Process your research query
2. Use appropriate tools to gather information
3. Structure the findings
4. Save the results to a file if requested

## Output Format

The research results are structured as follows:

```python
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
```

## Available Tools

1. `save_text_to_file`: Saves research data to a text file
2. `search`: Performs web searches using DuckDuckGo
3. `wikipedia`: Queries Wikipedia articles

## Example

```python
query = "Research the impact of artificial intelligence on healthcare and save the findings"
result = agent_executor.invoke({"input": query})
```

## Error Handling

The system includes error handling for:

- API response parsing
- Tool execution failures
- Output formatting issues

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license]
