from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('intro_to_ML.pdf')

docs = loader.load()


spiltter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result = spiltter.split_documents(docs)

print(result[5].page_content)