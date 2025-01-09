import os
import warnings
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew
from IPython.display import Markdown
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel
import json
from pprint import pprint

# Load environment variables from a .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("The OpenAI API key is not set. Please check your .env file or environment variables.")
else:
    print(f"OpenAI API Key Loaded: {OPENAI_API_KEY[:4]}...{OPENAI_API_KEY[-4:]}")  # Print partial key for debugging

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

warnings.filterwarnings('ignore')

# Initialize the LLM with the OpenAI API key from the environment variables
llm = LLM( 
    model="gpt-4o", 
    base_url="https://openai.prod.ai-gateway.quantumblack.com/0b0e19f0-3019-4d9e-bc36-1bd53ed23dc2/v1", 
    api_key=OPENAI_API_KEY
)

# Agent 1: Venue Coordinator
venue_coordinator = Agent(
    role="Venue Coordinator",
    goal="Identify and book an appropriate venue based on event requirements",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "With a keen sense of space and understanding of event logistics, "
        "you excel at finding and securing the perfect venue that fits the event's theme, "
        "size, and budget constraints."
    )
)

# Agent 2: Logistics Manager
logistics_manager = Agent(
    role='Logistics Manager',
    goal=(
        "Manage all logistics for the event including catering and equipment"
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Organized and detail-oriented, you ensure that every logistical aspect of the event "
        "from catering to equipment setup is flawlessly executed to create a seamless experience."
    )
)

# Agent 3: Marketing and Communications Agent
marketing_communications_agent = Agent(
    role="Marketing and Communications Agent",
    goal="Effectively market the event and communicate with participants",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Creative and communicative, you craft compelling messages and engage with potential attendees "
        "to maximize event exposure and participation."
    )
)

class VenueDetails(BaseModel):
    name: str
    address: str
    capacity: int
    booking_status: str

# Define the tasks
venue_task = Task(
    description="Find a venue in {event_city} that meets criteria for {event_topic}.",
    expected_output="All the details of a specifically chosen venue you found to accommodate the event.",
    human_input=True,
    output_json=VenueDetails,
    output_file="venue_details.json",
    agent=venue_coordinator
)

logistics_task = Task(
    description="Coordinate catering and equipment for an event with {expected_participants} participants on {tentative_date}.",
    expected_output="Confirmation of all logistics arrangements including catering and equipment setup.",
    human_input=True,
    async_execution=True,
    agent=logistics_manager
)

marketing_task = Task(
    description="Promote the {event_topic} aiming to engage at least {expected_participants} potential attendees.",
    expected_output="Report on marketing activities and attendee engagement formatted as markdown.",
    async_execution=True,
    output_file="marketing_report.md",
    agent=marketing_communications_agent
)

# Define the crew with agents and tasks
event_management_crew = Crew(
    agents=[venue_coordinator, logistics_manager, marketing_communications_agent],
    tasks=[venue_task, logistics_task],  # Ensure only one async task at the end
    verbose=True
)

# Add the single asynchronous task at the end
event_management_crew.tasks.append(marketing_task)

# Event details input
event_details = {
    'event_topic': "Tech Innovation Conference",
    'event_description': "A gathering of tech innovators and industry leaders to explore future technologies.",
    'event_city': "San Francisco",
    'tentative_date': "2024-09-15",
    'expected_participants': 500,
    'budget': 20000,
    'venue_type': "Conference Hall"
}

# Kickoff the event management crew
result = event_management_crew.kickoff(inputs=event_details)

# Load and display the venue details
with open('venue_details.json') as f:
    data = json.load(f)

pprint(data)

# Display the marketing report
Markdown("marketing_report.md")