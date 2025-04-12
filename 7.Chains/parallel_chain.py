from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    task="text-generation"
)

model2 = ChatGroq(
    model="llama-3.1-8b-instant",
)

prompt1 = PromptTemplate(
    template='Generate a short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short questions and answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """Machine learning is a field of artificial intelligence that focuses on teaching computers to learn from data and make decisions without being explicitly programmed. It works by feeding large sets of data into algorithms that identify patterns and use them to make predictions or perform tasks. There are different types of machine learning methods, including supervised learning where the model learns from labeled data, unsupervised learning where it finds hidden patterns in data without labels, and reinforcement learning where it learns through trial and error by interacting with an environment. Common algorithms used in machine learning include linear regression, decision trees, support vector machines, and neural networks. Machine learning is widely used today in applications like spam detection, speech recognition, recommendation systems, and self-driving cars. As more data becomes available and computing power increases, machine learning continues to grow as a powerful tool for solving complex real-world problems."""

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()