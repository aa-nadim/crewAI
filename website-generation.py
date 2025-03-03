from crewai import Agent, Task, Crew, LLM
from textwrap import dedent
import json

# Initialize gemini LLM
# llm = LLM(
#     model="gemini/gemini-1.5-flash",
#     api_key="AIzaSyA2VrDRrDMJnfxtSqMHzafFyS6D4ttvRCc"
# )

llm = LLM(
    model="ollama/deepseek-r1:1.5b",
    base_url="http://localhost:11434"
)

class WebsiteGenerationCrew:
    def __init__(self, website_name, features, context):
        self.website_name = website_name
        self.features = features
        self.context = context
        
    def create_agents(self):
        # Manager Agent - Oversees the entire process
        self.manager_agent = Agent(
            role='Project Manager',
            goal='Coordinate website development and ensure all requirements are met',
            backstory=dedent("""
                You are an experienced web development project manager who excels at 
                breaking down requirements and ensuring high-quality deliverables.
                You analyze requirements, delegate tasks, and ensure everything meets
                the client's needs."""),
            verbose=True,
            allow_delegation=True,
            llm=llm
        )

        # Requirements Analyst
        self.analyst_agent = Agent(
            role='Requirements Analyst',
            goal='Analyze and break down website requirements into technical specifications',
            backstory=dedent("""
                You are an expert at analyzing user requirements and converting them
                into detailed technical specifications. You ensure nothing is missed
                and all features are properly scoped."""),
            verbose=True,
            llm=llm
        )

        # SEO Agent
        self.seo_agent = Agent(
            role='SEO Specialist',
            goal='Optimize website content and structure for search engines',
            backstory=dedent("""
                You are an SEO expert who ensures websites are optimized for search
                engines. You focus on meta tags, content structure, semantic HTML,
                and proper keyword implementation to improve search rankings."""),
            verbose=True,
            llm=llm
        )

        # HTML Developer
        self.html_agent = Agent(
            role='HTML Developer',
            goal='Create semantic and accessible HTML structure',
            backstory=dedent("""
                You are a skilled HTML developer who creates clean, semantic, and
                accessible website structures. You follow best practices and ensure
                your code is well-organized."""),
            verbose=True,
            llm=llm
        )

        # CSS Designer
        self.css_agent = Agent(
            role='CSS Designer',
            goal='Create beautiful and responsive styling',
            backstory=dedent("""
                You are a creative CSS designer who excels at creating beautiful,
                responsive, and modern website designs. You ensure websites look
                great on all devices."""),
            verbose=True,
            llm=llm
        )

        # JavaScript Developer
        self.js_agent = Agent(
            role='JavaScript Developer',
            goal='Implement interactive features and functionality',
            backstory=dedent("""
                You are an experienced JavaScript developer who creates smooth
                and efficient interactive features. You ensure all functionality
                works flawlessly."""),
            verbose=True,
            llm=llm
        )

    def create_tasks(self):
        # Task 1: Analyze Requirements
        analyze_requirements = Task(
            description=f"""
                Analyze the following website requirements:
                - Name: {self.website_name}
                - Features: {self.features}
                - Context: {self.context}
                
                Create detailed technical specifications including:
                1. Required HTML structure
                2. CSS styling needs
                3. JavaScript functionality
                4. Any potential challenges or considerations
            """,
            expected_output="Detailed technical specifications in JSON format",
            agent=self.analyst_agent
        )

        # Task 2: SEO Analysis and Recommendations
        seo_analysis = Task(
            description="""
                Analyze the website requirements and create SEO recommendations:
                1. Keyword research and implementation strategy
                2. Meta tags structure
                3. Content hierarchy
                4. URL structure
                5. Schema markup requirements
            """,
            expected_output="SEO recommendations report",
            agent=self.seo_agent,
            dependencies=[analyze_requirements]
        )

        # Task 3: Create HTML Structure
        create_html = Task(
            description="""
                Using the technical specifications and SEO recommendations, create the HTML structure.
                Ensure the HTML is:
                1. Semantic and accessible
                2. Properly structured for styling
                3. Includes necessary meta tags
                4. Responsive ready
                5. SEO optimized
            """,
            expected_output="Complete HTML code",
            agent=self.html_agent,
            dependencies=[analyze_requirements, seo_analysis]
        )

        # Task 4: Create CSS Styling
        create_css = Task(
            description="""
                Using the technical specifications and HTML structure, create the CSS styling.
                Ensure the CSS:
                1. Is responsive
                2. Follows modern best practices
                3. Implements the required design elements
                4. Uses appropriate naming conventions
            """,
            expected_output="Complete CSS code",
            agent=self.css_agent,
            dependencies=[create_html]
        )

        # Task 5: Create JavaScript Functionality
        create_js = Task(
            description="""
                Using the technical specifications, implement all required JavaScript functionality.
                Ensure the JavaScript:
                1. Is clean and efficient
                2. Implements all required features
                3. Handles errors appropriately
                4. Is well-documented
            """,
            expected_output="Complete JavaScript code",
            agent=self.js_agent,
            dependencies=[create_html]
        )

        # Task 6: Final Review and Integration
        final_review = Task(
            description="""
                Review all components and integrate them into a final website.
                Ensure:
                1. All requirements are met
                2. Components work together properly
                3. Code is clean and well-organized
                4. Website functions as expected
                5. SEO recommendations are implemented
            """,
            expected_output="Final website code",
            agent=self.manager_agent,
            dependencies=[create_html, create_css, create_js, seo_analysis]
        )

        return [analyze_requirements, seo_analysis, create_html, create_css, create_js, final_review]

    def run(self):
        # Create agents
        self.create_agents()
        
        # Create tasks
        tasks = self.create_tasks()
        
        # Create and run the crew
        crew = Crew(
            agents=[
                self.manager_agent,
                self.analyst_agent,
                self.seo_agent,
                self.html_agent,
                self.css_agent,
                self.js_agent
            ],
            tasks=tasks,
            verbose=True
        )
        
        result = crew.kickoff()
        return result

# Example usage
if __name__ == "__main__":
    website_name = "TechStart"
    features = [
        "Responsive design",
        "Contact form",
        "Product showcase",
        "Newsletter signup"
    ]
    context = "A landing page for a tech startup showcasing their main product"
    
    crew = WebsiteGenerationCrew(website_name, features, context)
    result = crew.run()
    print(result)