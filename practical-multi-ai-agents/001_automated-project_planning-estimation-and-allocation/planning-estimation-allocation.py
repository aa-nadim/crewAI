# Warning control
from IPython.display import Markdown
from pydantic import BaseModel, Field
from typing import List
from crewai import LLM, Agent, Task, Crew
import yaml
from helper import load_env, get_openai_api_key, get_base_url
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_env()


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


class TaskEstimate(BaseModel):
    task_name: str = Field(..., description="Name of the task")
    estimated_time_hours: float = Field(
        ..., description="Estimated time to complete the task in hours")
    required_resources: List[str] = Field(
        ..., description="List of resources required to complete the task")


class Milestone(BaseModel):
    milestone_name: str = Field(..., description="Name of the milestone")
    tasks: List[str] = Field(...,
                             description="List of task IDs associated with this milestone")


class ProjectPlan(BaseModel):
    tasks: List[TaskEstimate] = Field(...,
                                      description="List of tasks with their estimates")
    milestones: List[Milestone] = Field(...,
                                        description="List of project milestones")


# Creating Agents
project_planning_agent = Agent(
    config=agents_config['project_planning_agent'],
    llm=llm
)

estimation_agent = Agent(
    config=agents_config['estimation_agent'],
    llm=llm
)

resource_allocation_agent = Agent(
    config=agents_config['resource_allocation_agent'],
    llm=llm
)

# Creating Tasks
task_breakdown = Task(
    config=tasks_config['task_breakdown'],
    agent=project_planning_agent
)

time_resource_estimation = Task(
    config=tasks_config['time_resource_estimation'],
    agent=estimation_agent
)

resource_allocation = Task(
    config=tasks_config['resource_allocation'],
    agent=resource_allocation_agent,
    output_pydantic=ProjectPlan  # This is the structured output we want
)

# Creating Crew
crew = Crew(
    agents=[
        project_planning_agent,
        estimation_agent,
        resource_allocation_agent
    ],
    tasks=[
        task_breakdown,
        time_resource_estimation,
        resource_allocation
    ],
    verbose=True
)


project = 'Website'
industry = 'Technology'
project_objectives = 'Create a website for a small business'
team_members = """
- John Doe (Project Manager)
- Jane Doe (Software Engineer)
- Bob Smith (Designer)
- Alice Johnson (QA Engineer)
- Tom Brown (QA Engineer)
"""
project_requirements = """
- Create a responsive design that works well on desktop and mobile devices
- Implement a modern, visually appealing user interface with a clean look
- Develop a user-friendly navigation system with intuitive menu structure
- Include an "About Us" page highlighting the company's history and values
- Design a "Services" page showcasing the business's offerings with descriptions
- Create a "Contact Us" page with a form and integrated map for communication
- Implement a blog section for sharing industry news and company updates
- Ensure fast loading times and optimize for search engines (SEO)
- Integrate social media links and sharing capabilities
- Include a testimonials section to showcase customer feedback and build trust
"""

# The given Python dictionary
inputs = {
    'project_type': project,
    'project_objectives': project_objectives,
    'industry': industry,
    'team_members': team_members,
    'project_requirements': project_requirements
}

# Run the crew
result = crew.kickoff(
    inputs=inputs
)

# Assuming result is of type CrewOutput. Convert result to a string representation
result_text = str(result)

# Use the Markdown function with the string representation
Markdown(result_text)
