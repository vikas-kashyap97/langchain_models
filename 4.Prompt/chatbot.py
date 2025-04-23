from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv


load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash-latest',
)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant")
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=(f"You: {user_input}")))
    if user_input.lower() == 'exit':
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=(f"AI: {result.content}")))
    print("AI:", result.content)

print(chat_history)