import os
from crewai import Agent, Task, Crew, Process, LLM

# Predefined templates with tags
TEMPLATES = [
    {"id": 4, "tags": "eco-friendly, eco, friendly, ecofriendly, nature, escapes, villas, retreats, yoga, sun, destinations, Tropical, Relaxation"},
    {"id": 3, "tags": "beach, island, sea, ocean, top destinations, escapes, villas, apartment, cottage, cabins, rentals, resorts"},
    {"id": 2, "tags": "journal, cottage, vacation, island, beach, villas, retreat"},
    {"id": 6, "tags": "serene"},
    {"id": 5, "tags": None},
    {"id": 1, "tags": "beach"},
    {"id": 7, "tags": "test"},
    {"id": 10, "tags": "Location, Centric"},
    {"id": 9, "tags": "property"},
]

# Initialize LLM for CrewAI
# This uses the OpenAI integration, but you can adapt for Gemini
# llm = LLM(
#     model="gemini/gemini-1.5-flash",
#     api_key="AIzaSyA2VrDRrDMJnfxtSqMHzafFyS6D4ttvRCc"
# )

# Define the matching agent with clear instructions
matching_agent = Agent(
    role="Template Matching Expert",
    goal="Find the most appropriate template based on semantic matching between user description and template tags",
    backstory="""You're an expert in natural language understanding and semantic matching.
    Your job is to analyze user descriptions and match them to the most relevant template
    by understanding the context, themes, and keywords in both the description and template tags.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define the matching task with detailed instructions
def template_matching_task(description):
    return Task(
        description=f"""
        Analyze the following user description:
        "{description}"
        
        Match it to the most appropriate template from this list:
        {TEMPLATES}
        
        Consider the following in your matching process:
        1. Exact keyword matches between description and template tags
        2. Semantic relevance and thematic alignment
        3. The context and intent of the description
        
        Assign higher priority to templates with multiple relevant tag matches.
        If multiple templates match equally well, choose the one with the most specific relevance.
        If no template matches well, return "No matching template found".
        
        Think step by step about your reasoning process.
        """,
        expected_output="A single template ID (just the number) that best matches the description, with a brief explanation of why it's the best match.",
        agent=matching_agent,
    )

# Function to get template match using CrewAI
def get_template_match(description):
    # Create the task with the specific description
    matching_task = template_matching_task(description)
    
    # Create the crew
    crew = Crew(
        agents=[matching_agent],
        tasks=[matching_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute the crew
    result = crew.kickoff()
    
    # Parse the result to extract just the template ID
    try:
        # Try to find a number in the result
        import re
        template_ids = re.findall(r'\b\d+\b', result)
        if template_ids:
            return int(template_ids[0])
        else:
            return "No matching template found"
    except:
        return result

# Fallback function using keyword matching (in case API issues)
def fallback_match_template(description):
    description_lower = description.lower()
    description_keywords = set(description_lower.split())
    
    best_match = None
    best_score = 0
    
    for template in TEMPLATES:
        if not template["tags"]:
            continue
            
        template_tags = template["tags"].lower()
        template_keywords = set(template_tags.split(", "))
        
        # Count exact keyword matches (whole words)
        exact_matches = description_keywords.intersection(template_keywords)
        exact_match_score = len(exact_matches) * 3  # Triple weight for exact matches
        
        # Count how many template tags appear in the description (phrases)
        phrase_match_score = sum(1 for tag in template_tags.split(", ") if tag in description_lower) * 2
        
        # Total score combining both matching methods
        total_score = exact_match_score + phrase_match_score
        
        print(f"Template {template['id']} score: {total_score} (exact: {exact_match_score}, phrase: {phrase_match_score})")
        
        if total_score > best_score:
            best_score = total_score
            best_match = template["id"]
    
    return best_match if best_score > 0 else "No matching template found"

# Main function to handle user input
def main():
    print("Template Matching System")
    print("=" * 50)
    print("Available templates:")
    for template in TEMPLATES:
        tags = template["tags"] if template["tags"] else "No tags"
        print(f"ID: {template['id']} - Tags: {tags}")
    print("=" * 50)
    
    while True:
        # Get user input
        user_description = input("\nEnter your description (or 'exit' to quit): ")
        
        # Check if user wants to exit
        if user_description.lower() in ['exit', 'quit', 'q']:
            print("Exiting program. Goodbye!")
            break
        
        # Skip empty inputs
        if not user_description.strip():
            print("Please enter a description.")
            continue
            
        print(f"\nMatching description: {user_description}")
        
        try:
            # Try the CrewAI approach first
            print("Using CrewAI for matching...")
            matched_id = get_template_match(user_description)
            print(f"CrewAI Matched Template ID: {matched_id}")
        except Exception as e:
            # Fall back to keyword matching if CrewAI fails
            print(f"CrewAI error: {e}")
            print("Falling back to keyword matching...")
            matched_id = fallback_match_template(user_description)
            print(f"Fallback Matched Template ID: {matched_id}")
        
        # Find and display the matched template
        if isinstance(matched_id, int):
            matched_template = next((t for t in TEMPLATES if t["id"] == matched_id), None)
            if matched_template:
                print(f"Matched Template: ID {matched_id} - Tags: {matched_template['tags']}")
            else:
                print(f"Template with ID {matched_id} not found in the list.")
        else:
            print(f"Result: {matched_id}")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()