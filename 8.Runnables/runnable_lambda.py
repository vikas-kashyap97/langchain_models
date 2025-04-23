from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda

load_dotenv()

def word_count(text):
    return len(text.split())

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    task="text-generation"
)

prompt = PromptTemplate(
    template='Generate a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt | model | parser)


joke_analysis_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})


final_chain = RunnableSequence(joke_gen_chain | joke_analysis_chain)

result = final_chain.invoke({'topic': 'AI'})

print(result)
