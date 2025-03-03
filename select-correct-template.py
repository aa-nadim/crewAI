
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

# Main function to test with sample descriptions
def main():
    # Test with multiple descriptions to ensure different matches
    descriptions = [
        # "Create engaging website content for a platform featuring charming countryside bed and breakfasts nestled in the picturesque hills of Tuscany, Italy.",
        # "Design a landing page for eco-friendly yoga retreats in tropical destinations that focus on sustainability and wellness.",
        # "Build a website showcasing luxury beachfront properties on exotic islands with pristine ocean views.",
        # "Create content for a serene mountain getaway with peaceful surroundings and natural beauty."
        "Write compelling website content for a platform highlighting luxury beachfront villas in the Caribbean. Tailor the content to appeal to both local vacationers and international travelers, offering informative and SEO-friendly descriptions. Maintain a warm and welcoming tone, guiding visitors through the features and amenities of each villa while emphasizing the allure of a Caribbean escape.",
        # "Create captivating website content for a platform promoting eco-friendly treehouse accommodations nestled in the heart of the Amazon rainforest. Target eco-conscious travelers from around the globe with engaging and search engine optimized content. Maintain a conversational and educational tone, highlighting the unique experience of staying in a treehouse amidst lush jungle surroundings. Ensure content is optimized for mobile devices and accessible to all",
        # "Create engaging website content for a platform featuring charming countryside bed and breakfasts nestled in the picturesque hills of Tuscany, Italy. Appeal to travelers seeking authentic Italian experiences, both domestically and internationally, with informative and SEO-rich content. Maintain a warm and inviting tone, showcasing the rustic charm and local flavors of each B&B. Encourage visitors to immerse themselves in the beauty of Tuscany by booking their countryside retreat, while ensuring seamless mobile optimization and accessibility for all potential guests.",
    ]
    
    for desc in descriptions:
        print(f"\nDescription: {desc[:50]}...")
        try:
            # Try the CrewAI approach first
            matched_id = get_template_match(desc)
            print(f"CrewAI Matched Template ID: {matched_id}")
        except Exception as e:
            # Fall back to keyword matching if CrewAI fails
            print(f"CrewAI error: {e}")
            matched_id = fallback_match_template(desc)
            print(f"Fallback Matched Template ID: {matched_id}")

if __name__ == "__main__":
    main()