from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


load_dotenv()


llm = ChatDeepSeek(
    model="deepseek-r1-distill-llama-70b",
)

video_id = "Gfr50f6ZBvo"
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
   

except TranscriptsDisabled:
    print("No caption available for this video.")


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

retriever.invoke("What is deepmind?")

prompt = PromptTemplate(
    template="""
You are a knowledgeable and helpful assistant.

Respond only with information found in the following transcript context. 
If the information is not available in the context, reply with "I don't know."

Transcript Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)


question = "Is the topic of aliens discussed in this video? If yes, what was mentioned?"


retrieved_docs = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)


final_prompt = prompt.invoke({"context": context_text, "question": question})

answer = llm.invoke(final_prompt)
print(answer.content)


# Using Chain Method with parser


from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrived_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    'context': retriver | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parallel_chain.invoke('Who is Demis?')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

main_chain.invoke('Can you summarize the vidoe?')