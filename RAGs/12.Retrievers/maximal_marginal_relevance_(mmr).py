from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

docs = [
    Document(page_content="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn."),
    Document(page_content="Machine learning is a subset of AI that involves training algorithms on data so they can make predictions or decisions without being explicitly programmed."),
    Document(page_content="The capital of France is Paris, known for its iconic Eiffel Tower, art museums, and rich cultural history."),
    Document(page_content="To reset your account password, go to the login page, click on 'Forgot Password', and follow the instructions sent to your registered email."),
    Document(page_content="The Python programming language is widely used for web development, data analysis, artificial intelligence, and scientific computing."),
]

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-1.5-flash-latest")

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda-mult":1}
)

query = "What is the capital of France?"
result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(f"content:\n{doc.page_content}...")



