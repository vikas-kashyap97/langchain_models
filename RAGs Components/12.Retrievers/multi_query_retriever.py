from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever 
from langchain_google_genai import ChatGoogleGenerativeAI

# Correct embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Documents
all_docs = [ 
    Document(page_content="Many people experience low energy due to dehydration, poor diet, or lack of sleep — all of which are crucial for staying balanced."),
    Document(page_content="Yoga and Tai Chi are known to enhance physical balance and inner calm, contributing to improved energy over time."),
    Document(page_content="Coffee can provide a temporary energy boost, but relying on caffeine can lead to energy crashes later."),
    Document(page_content="Some wellness routines involve grounding techniques and breathwork to restore mental clarity and energy levels."),
    Document(page_content="Work-life balance is often disrupted by overcommitment, leading to fatigue and mental burnout."),
    Document(page_content="The gravitational force on the moon is about one-sixth that of Earth, making balance and movement very different for astronauts."),
    Document(page_content="The Eiffel Tower sparkles every night for five minutes every hour after sunset — a symbol of energy in the City of Light."),
    Document(page_content="In programming, energy efficiency is becoming a priority in large-scale data centers to maintain sustainability."),
    Document(page_content="Leonardo da Vinci studied human anatomy in detail, exploring the balance of form and function in the human body."),
    Document(page_content="The Amazon Rainforest’s ecosystem is delicately balanced — a disruption can affect global oxygen levels and biodiversity."),
]

# Vectorstore
vectorstore = FAISS.from_documents(documents=all_docs, embedding=embedding_model)

# Similarity retriever
similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Multi-query retriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
)

# Your query
query = "How to improve energy levels and maintain balance?"

# Get results
similarity_result = similarity_retriever.invoke(query)
multiquery_result = multiquery_retriever.invoke(query)

# Print results
print("\n=== Similarity Retriever Results ===")
for i, doc in enumerate(similarity_result):
    print(f"\n--- Result {i+1} ---")
    print(f"content:\n{doc.page_content}...")

print("\n=== Multi-Query Retriever Results ===")
for i, doc in enumerate(multiquery_result):
    print(f"\n--- Result {i+1} ---")
    print(f"content:\n{doc.page_content}...")
