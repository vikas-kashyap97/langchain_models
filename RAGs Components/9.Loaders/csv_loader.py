from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='./assests/langchain_features.csv')

docs = loader.load()

print(len(docs))
print(docs[2].page_content)