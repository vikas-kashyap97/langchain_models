from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings( model="text-embedding-3-large", dimensions=32)


text = "Delhi is the capital of India."

vector = embedding.embed_query(text)

print(str(vector))