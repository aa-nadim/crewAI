# Warning control
import warnings
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv
from IPython.display import Markdown
import os
import time
warnings.filterwarnings('ignore')


load_dotenv()
# Configure OpenAI API
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo' 

# Configure LLM with optimized parameters
llm = LLM(
    model="gpt-3.5-turbo",  
    temperature=0.7,
    max_tokens=100,
    top_p=0.8,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["END"],
    seed=42
)

# Define Agents with the configured LLM
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article "
              "about the topic: {topic}."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    allow_delegation=False,
    verbose=True,
    llm=llm  # Explicitly pass the LLM configuration
)

writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory="You're working on writing "
              "a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to objective statements.",
    allow_delegation=False,
    verbose=True,
    llm=llm  # Explicitly pass the LLM configuration
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices,"
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    allow_delegation=False,
    verbose=True,
    llm=llm  # Explicitly pass the LLM configuration
)

# Rest of your task definitions remain the same
plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
            "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
            "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
            "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords, and resources.",
    agent=planner
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
            "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    agent=writer
)

edit = Task(
    description=(
        "1. Review the blog post for clarity, coherence, and flow.\n"
        "2. Check for grammatical errors and style consistency.\n"
        "3. Ensure content aligns with brand voice and guidelines.\n"
        "4. Verify facts and sources where applicable.\n"
        "5. Optimize formatting and structure for readability."
    ),
    expected_output="A polished, publication-ready blog post "
                    "in markdown format, thoroughly reviewed "
                    "for quality and accuracy.",
    agent=editor
)

# Create Crew
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

def execute_crew_task(topic):
    """
    Execute the CrewAI task with error handling and retries
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"Attempt {retry_count + 1} of {max_retries}")
            result = crew.kickoff(inputs={"topic": topic})
            print("Task completed successfully!")
            return result
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Failed after {max_retries} attempts. Error: {str(e)}")
                raise
            print(f"Attempt {retry_count} failed. Retrying after delay...")
            time.sleep(2)  # Add delay between retries

def main():
    try:
        # Example usage
        topic = "Machine Learning"  # You can change this topic
        result = execute_crew_task(topic)
        
        # Display the result
        print("\nFinal Output:")
        print(Markdown(result.raw))
        
        # Optionally save the output to a file
        with open(f"{topic.lower().replace(' ', '_')}_blog.md", "w", encoding="utf-8") as f:
            f.write(result.raw)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your API key and quota limits.")

if __name__ == "__main__":
    main()