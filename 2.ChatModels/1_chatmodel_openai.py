from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

model = ChatOpenAI(model='gpt-4', temperature=0.3, max_completion_tokens=10)

result = model.invoke("What's the capital of India?")

print(result.contnet)