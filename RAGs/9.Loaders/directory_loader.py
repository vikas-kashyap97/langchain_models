from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path='./Research_paper',
    glob='*.pdf',
    loader_cls=PyPDFLoader  
)

# docs = loader.load()

# print(len(docs))
# print(docs[0].page_content)


#                                              vs



docs = loader.lazy_load()

for document in docs:
    print(document.metadata)