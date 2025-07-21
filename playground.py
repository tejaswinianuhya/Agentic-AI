from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

import os
import phi
from phi.playground import Playground, serve_playground_app
# Load environment variables
from dotenv import load_dotenv

load_dotenv()

phi.api = os.getenv("API_KEY")

web_search_agent = Agent(
    name = "Web Search Agent",
    role = "A web search agent that can search the web for information and answer questions.",
    model = Groq(id = "llama3-8b-8192"),
    tools = [DuckDuckGo()],
    instructions = ["Always incluse sources"],
    show_tool_calls= True,
    markdown = True
)

financial_agent = Agent(
    name = "Financial AI Agent",
    role = "A financial agent that can answer questions about stocks, ETFs, and other financial instruments.",
    model = Groq(id = "llama3-8b-8192"),
    tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    instructions = ["Use tables to display the data"],
    show_tool_calls= True,
    markdown = True
)

app = Playground(
    agents=[web_search_agent, financial_agent]
).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload = True)