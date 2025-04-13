from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableParallel, RunnableSequence

load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    task="text-generation"
)

model2 = ChatGroq(
    model="llama-3.1-8b-instant",
)

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a linkedIn post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1 | model1 | parser),
    'linkedIn': RunnableSequence(prompt2 | model2 | parser)
})

result = parallel_chain.invoke({'topic':'AI'})
print(result['tweet'])
print(result['linkedIn'])
