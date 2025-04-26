from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant",
)


@tool
def multiply(a: int, b: int) -> int:
    """Given two numbers a and b, returns their product."""
    return a * b


llm_with_tools = llm.bind_tools([multiply])


query = HumanMessage(content="Can you multiply 3 with 10?")
response = llm_with_tools.invoke([query])


tool_call = response.tool_calls[0]


tool_result = multiply.invoke(tool_call['args'])



tool_message = ToolMessage(
    tool_call_id=tool_call['id'],
    content=str(tool_result)
)


final_response = llm_with_tools.invoke([query, response, tool_message])

print(final_response.content)
