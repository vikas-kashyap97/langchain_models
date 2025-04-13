from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    task="text-generation"
)

prompt = PromptTemplate(
    template='Suggest a catchy title about {topic}',
    input_variables=['topic']
)

topic = input('Enter a topic: ')

formatted_prompt = prompt.format(topic=topic)

blog_title = llm.predict(formatted_prompt)

print("Generated Blog Title:", blog_title)