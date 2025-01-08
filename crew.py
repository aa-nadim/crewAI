# crew.py

# Warning control
import warnings
warnings.filterwarnings('ignore')


from crewai import Agent, Task, Crew
from agents import planner, writer, editor
from tasks import plan, write, edit

from IPython.display import Markdown

# Initialize Crew with verbose as a boolean value
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True  # Change 2 to True or False
)

result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})
Markdown(result)