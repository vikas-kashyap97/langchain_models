from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    task="text-generation"
)

# 1st prompt -> detailed report

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
    )

# 2nd prompt -> summary

template2 = PromptTemplate(
    template='Write a five line summary on the following text. /n  {text}',
    input_variables=['topic']
    )

