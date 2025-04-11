from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

documents = [
    "Virat Kohli is known for his consistency and passion as a top-order batsman for India.",
    "Steve Smith is one of Australia's most dependable Test batsmen with an unorthodox technique.",
    "Babar Azam is celebrated for his elegant strokeplay and leadership as Pakistan’s captain.",
    "Ben Stokes is a game-changing all-rounder who played a heroic role in England's 2019 World Cup win.",
    "Kane Williamson is respected worldwide for his calm demeanor and sharp cricketing mind as New Zealand’s captain."
]

query = 'tell me about Babar Azam'


doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)


score = cosine_similarity([query_embedding], doc_embeddings)[0]


index, score = sorted(list(enumerate(score)), key=lambda x: x[1], reverse=True)[0]


print(f"Query: {query}")
print(documents[index])
print(f"Similarity score is: {score}")
