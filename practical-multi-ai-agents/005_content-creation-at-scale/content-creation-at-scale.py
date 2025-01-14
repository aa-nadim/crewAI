# Warning control
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
from helper import load_env, get_openai_api_key, get_base_url
load_env()

import yaml
from crewai import LLM, Agent, Task, Crew

from pydantic import BaseModel, Field
from typing import List

# Load the OpenAI API Key and Base URL
api_key = get_openai_api_key()
base_url = get_base_url()

class SocialMediaPost(BaseModel):
    platform: str = Field(..., description="The social media platform where the post will be published (e.g., Twitter, LinkedIn).")
    content: str = Field(..., description="The content of the social media post, including any hashtags or mentions.")

class ContentOutput(BaseModel):
    article: str = Field(..., description="The article, formatted in markdown.")
    social_media_posts: List[SocialMediaPost] = Field(..., description="A list of social media posts related to the article.")


# Define file paths for YAML configurations
files = {
    'agents': 'config/agents.yaml',
    'tasks': 'config/tasks.yaml'
}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

# Assign loaded configurations to specific variables
agents_config = configs['agents']
tasks_config = configs['tasks']

from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool

llm = LLM(
    model="gpt-4o",
    base_url=base_url,
    api_key=api_key
)

local_llm = LLM(
    model="ollama/llama3.2:1b",
    base_url="http://localhost:11434"
)

# Creating Agents
market_news_monitor_agent = Agent(
    config=agents_config['market_news_monitor_agent'],
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
    llm=local_llm,
)

data_analyst_agent = Agent(
    config=agents_config['data_analyst_agent'],
    tools=[SerperDevTool(), WebsiteSearchTool()],
    llm=local_llm,
)

content_creator_agent = Agent(
    config=agents_config['content_creator_agent'],
    tools=[SerperDevTool(), WebsiteSearchTool()],
    llm=llm,
)

quality_assurance_agent = Agent(
    config=agents_config['quality_assurance_agent'],
    llm=llm,
)

# Creating Tasks
monitor_financial_news_task = Task(
    config=tasks_config['monitor_financial_news'],
    agent=market_news_monitor_agent
)

analyze_market_data_task = Task(
    config=tasks_config['analyze_market_data'],
    agent=data_analyst_agent
)

create_content_task = Task(
    config=tasks_config['create_content'],
    agent=content_creator_agent,
    context=[monitor_financial_news_task, analyze_market_data_task]
)

quality_assurance_task = Task(
    config=tasks_config['quality_assurance'],
    agent=quality_assurance_agent,
    output_pydantic=ContentOutput
)

# Creating Crew
content_creation_crew = Crew(
    agents=[
        market_news_monitor_agent,
        data_analyst_agent,
        content_creator_agent,
        quality_assurance_agent
    ],
    tasks=[
        monitor_financial_news_task,
        analyze_market_data_task,
        create_content_task,
        quality_assurance_task
    ],
    verbose=True
)

result = content_creation_crew.kickoff(inputs={
  'subject': 'Inflation in the US and the impact on the stock market in 2024'
})

import textwrap

posts = result.pydantic.model_dump()['social_media_posts']
for post in posts:
    platform = post['platform']
    content = post['content']
    print(platform)
    wrapped_content = textwrap.fill(content, width=50)
    print(wrapped_content)
    print('-' * 50)

from IPython.display import display, Markdown
Markdown(result.pydantic.model_dump()['article'])