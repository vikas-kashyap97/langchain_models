# MessagesPlaceholder is a tool used to store and retrieve prior messages in a chat, allowing a chatbot to "remember" past interactions. It’s especially helpful when using chains or agents that rely on a memory buffer 



from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder

# chat template 
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history=[]
# load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

#create prompt

prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund?'})
print(prompt)