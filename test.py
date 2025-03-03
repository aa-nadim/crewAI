from crewai import Agent, Task, Crew, LLM

# Initialize LLM
llm = LLM(
    model="gpt-4o",
    base_url="https://openai.prod.ai-gateway.quantumblack.com/0b0e19f0-3019-4d9e-bc36-1bd53ed23dc2/v1",
    api_key="5f393389-5fc3-4904-a597-dd56e3b00f42:7ggTi5OqYeCqlLm1PmJ9kkAVk69iWuWI"
)

# Create a multimodal agent
image_analyst = Agent(
    role="Product Analyst",
    goal="Analyze product images and provide detailed descriptions",
    backstory="Expert in visual product analysis with deep knowledge of design and features",
    llm=llm,
    multimodal=True
)

# Create a task for image analysis
task = Task(
    description="Analyze the product image at https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSbHGZhD_9fxs7xH8j-s3_ZbCXp3VQetdEtuA&s and provide a detailed description",
    agent=image_analyst,
    verbose=True,
    output_file="outputs/Dog_prod_desc.md",
    expected_output="A detailed written description of the product based on visual analysis"
)

# Create and run the crew
crew = Crew(
    agents=[image_analyst],
    tasks=[task]
)

result = crew.kickoff()