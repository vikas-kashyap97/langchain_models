from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate five interesting lines about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    task="text-generation"
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': 'Cricket'})

print(result)


