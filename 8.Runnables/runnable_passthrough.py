from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough


load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    task="text-generation"
)

model2 = ChatGroq(
    model="llama-3.1-8b-instant",
)

prompt1 = PromptTemplate(
    template='Generate a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke {text}',
    input_variables=['text']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model1, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2 | model2 | parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': "AI"})

print(result)