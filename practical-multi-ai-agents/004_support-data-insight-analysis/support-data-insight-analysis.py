# Warning control
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
from helper import load_env, get_openai_api_key, get_base_url
load_env()

import os
import yaml
from crewai import LLM, Agent, Task, Crew


# Load the OpenAI API Key and Base URL
api_key = get_openai_api_key()
base_url = get_base_url()

# Initialize the LLM
llm = LLM(
    model="gpt-4o",
    base_url=base_url,
    api_key=api_key
)

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

from crewai_tools import FileReadTool
csv_tool = FileReadTool(file_path='./support_tickets_data.csv')

# Creating Agents
suggestion_generation_agent = Agent(
  config=agents_config['suggestion_generation_agent'],
  tools=[csv_tool],
  llm=llm
)

reporting_agent = Agent(
  config=agents_config['reporting_agent'],
  tools=[csv_tool],
  llm=llm
)

chart_generation_agent = Agent(
  config=agents_config['chart_generation_agent'],
  allow_code_execution=True,
  llm=llm
)

# Creating Tasks
suggestion_generation = Task(
  config=tasks_config['suggestion_generation'],
  agent=suggestion_generation_agent
)

table_generation = Task(
  config=tasks_config['table_generation'],
  agent=reporting_agent
)

chart_generation = Task(
  config=tasks_config['chart_generation'],
  agent=chart_generation_agent
)

final_report_assembly = Task(
  config=tasks_config['final_report_assembly'],
  agent=reporting_agent,
  context=[suggestion_generation, table_generation, chart_generation]
)


# Creating Crew
support_report_crew = Crew(
  agents=[
    suggestion_generation_agent,
    reporting_agent,
    chart_generation_agent
  ],
  tasks=[
    suggestion_generation,
    table_generation,
    chart_generation,
    final_report_assembly
  ],
  verbose=True
)


# support_report_crew.test(n_iterations=1, openai_model_name=llm)

# support_report_crew.train(n_iterations=1, filename='training.pkl')

# support_report_crew.test(n_iterations=1, openai_model_name=llm)

result = support_report_crew.kickoff()

from IPython.display import Markdown

# Convert result to a string representation
result_text = str(result)

# Use the Markdown function with the string representation
Markdown(result_text)