import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool
from dotenv import load_dotenv  
load_dotenv()  # Load environment variables from .env file

# Initialize OpenAI LLM
openai_api_key = os.environ.get("OPENAI_API_KEY")
llm_openai = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# Search tool
search_tool = SerperDevTool()

# Create Agents
coach = Agent(
    role='Senior Career Coach',
    goal="Discover and examine key tech and AI career skills for 2025",
    backstory="You're an an AI assistant specialized in crafting engaging LinkedIn posts. You will help create compelling content that resonates with the LinkedIn audience while maintaining professionalism and incorporating relevant hashtags and emojis.You will be provided with a post topic/theme and any specific requirements from the user based on which you will draft the post.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm_openai
)

influencer = Agent(
    role='LinkedIn Influencer Writer',
    goal="Write catchy, emoji-filled LinkedIn posts within 200 words",
    backstory="You're a specialized writer on LinkedIn, focusing on AI and technology.",
    verbose=True,
    allow_delegation=True,
    llm=llm_openai
)

critic = Agent(
    role='Expert Writing Critic',
    goal="Give constructive feedback on post drafts",
    backstory="You're skilled in offering straightforward, effective advice to tech writers. Ensure posts are concise, under 200 words, with emojis and hashtags.",
    verbose=True,
    allow_delegation=True,
    llm=llm_openai
)

# Create Tasks
task_search = Task(
    description="Get valuable and high quality web information about {topic}.Compile a report listing at least 5 new AI and tech skills related to {topic}, presented in bullet points",
    agent=coach,
    expected_output="Use {search_tool} to find trending hashtags and discussions related to the post topic."
)

task_post = Task(
    description="Create a LinkedIn post with a brief headline and a maximum of 1000 words, focusing on upcoming AI and tech skills related to the given {topic}",
    agent=influencer,
    expected_output="A LinkedIn post draft with a catchy headline, under 200 words, including emojis"
)

task_critique = Task(
    description="Refine the post for brevity, ensuring an engaging headline (no more than 30 characters) and keeping within a 1000-word limit. Ensure it stays relevant to {topic}.",
    agent=critic,
    expected_output="A revised LinkedIn post with concise wording, emojis, and hashtags"
)

# Create Crew
crew = Crew(
    agents=[coach, influencer, critic],
    tasks=[task_search, task_post, task_critique],
    verbose=True,
    process=Process.sequential 
)

# Get user input for the LinkedIn post topic
topic = input("Enter the topic for the LinkedIn post: ")
print(f"Generating post for topic: {topic}") 
inputs = {"topic": topic}
result = crew.kickoff(inputs=inputs)

print("#############")
print(result)
