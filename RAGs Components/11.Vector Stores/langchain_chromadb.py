from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Create IPL player documents with metadata
doc1 = Document(
    page_content="Virat Kohli is a legendary Indian batsman known for his aggressive play and consistency in the IPL.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="MS Dhoni, also known as 'Captain Cool', is known for his sharp cricketing brain and finishing skills.",
    metadata={"team": "Chennai Super Kings"}
)

doc3 = Document(
    page_content="Rohit Sharma is one of the most successful IPL captains, leading Mumbai Indians to multiple titles.",
    metadata={"team": "Mumbai Indians"}
)

doc4 = Document(
    page_content="KL Rahul is a stylish opener known for his elegant strokeplay and consistent performances.",
    metadata={"team": "Lucknow Super Giants"}
)

doc5 = Document(
    page_content="Shreyas Iyer is a dependable middle-order batsman who has shown strong leadership qualities.",
    metadata={"team": "Kolkata Knight Riders"}
)

doc6 = Document(
    page_content="Sanju Samson is a talented wicketkeeper-batsman known for his explosive batting style.",
    metadata={"team": "Rajasthan Royals"}
)

doc7 = Document(
    page_content="David Warner is a dynamic opener and key overseas player with a strong IPL record.",
    metadata={"team": "Delhi Capitals"}
)

# Combine all documents
docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7]

# Initialize vector store
vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-1.5-flash-latest"),
    persist_directory="my_Chroma_db",
    collection_name='sample'
)

# Add documents to the vector store
vector_store.add_documents(docs)

# Save the vector store to disk
vector_store.persist()

# Optional: perform a quick similarity search
results = vector_store.similarity_search("aggressive opener", k=2)
for i, res in enumerate(results, 1):
    print(f"\nResult {i}:")
    print("Content:", res.page_content)
    print("Metadata:", res.metadata)
