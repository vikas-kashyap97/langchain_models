from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings( model="text-embedding-3-large", dimensions=32)

documents= [
  "Paris is the capital of France.",
  "Tokyo is the capital of Japan.",
  "Ottawa is the capital of Canada.",
]


result = embedding.embed_documents(documents)

print(str(result))