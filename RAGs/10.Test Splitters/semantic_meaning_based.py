from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(model="gemini-1.5-flash-latest"),
    breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1
)

sample = """
Farmers are the backbone of any nation, tirelessly working to grow the food that sustains us all. Despite facing unpredictable weather, fluctuating market prices, and limited access to modern technology, they continue to put in immense effort to feed the country. Supporting farmers means investing in better infrastructure, fair pricing systems, and education that empowers them to adapt to changing agricultural trends.Indian Premier League (IPL) has become more than just a cricket tournament—it's a cultural phenomenon. Bringing together players from around the globe, the IPL combines world-class sports with entertainment, creating a festive atmosphere that unites fans across regions. Beyond the thrill of the matches, it also boosts local economies, promotes young talent, and strengthens India's position on the global sports stage.

In stark contrast, terrorism remains one of the gravest threats to global peace and security. It not only takes innocent lives but also spreads fear, disrupts economies, and divides societies. Combating terrorism requires international cooperation, intelligence sharing, and addressing the root causes such as poverty, political instability, and radicalization. A secure world can only be built through unity, understanding, and a commitment to peace.
"""

docs = text_splitter.create_documents(sample)

print(len(docs))
print(docs)
