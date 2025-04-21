from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

# Define the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create example documents
docs = [
    Document(
        page_content="""Photosynthesis is the process by which plants convert sunlight into energy. The green pigment chlorophyll in the leaves captures sunlight, and this energy is used to convert carbon dioxide and water into glucose, which provides energy to the plant. This process is essential for the survival of plants and contributes to oxygen production in the atmosphere.""",
        metadata={"source": "Doc1"}
    ),
    Document(
        page_content="""In recent years, the study of caffeine and its effects on human energy levels has grown significantly. Caffeine is known to stimulate the central nervous system and can lead to short-term boosts in alertness and focus, though it doesn't contribute directly to plant processes or photosynthesis.""",
        metadata={"source": "Doc2"}
    ),
    Document(
        page_content="""The Eiffel Tower, a famous symbol of Paris, attracts millions of visitors every year. While it is a marvel of human engineering, it has no connection to biological processes like photosynthesis or energy conversion in plants.""",
        metadata={"source": "Doc3"}
    ),
    Document(
        page_content="""Solar panels convert sunlight into electricity by using photovoltaic cells. These cells capture sunlight, but instead of being absorbed by plants for photosynthesis, the energy is converted into electrical energy. This process is similar in terms of using sunlight, but it’s a different form of energy conversion.""",
        metadata={"source": "Doc4"}
    ),
    Document(
        page_content="""The Amazon Rainforest plays an important role in the Earth's ecosystem. Its dense canopy of trees absorbs carbon dioxide and releases oxygen, indirectly supporting life. While it's related to the oxygen cycle, it’s not a direct explanation of photosynthesis but rather an ecological impact of forests.""",
        metadata={"source": "Doc5"}
    )
]

# Create the vectorstore
vectorstore = FAISS.from_documents(documents=docs, embedding=embedding_model)

# Convert vectorstore to retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Set up the LLM and compressor
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
compressor = LLMChainExtractor.from_llm(llm)

# Contextual Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Query
query = "What is photosynthesis?"
compressed_result = compression_retriever.invoke(query)

# Output results
print("\n=== Contextual Compression Retriever Results ===")
for i, doc in enumerate(compressed_result):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {doc.metadata['source']}")
    print(f"Content:\n{doc.page_content}...")
