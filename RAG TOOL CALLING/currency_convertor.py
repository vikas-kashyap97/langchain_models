from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
import requests
from langchain_core.tools import InjectedToolArg
from typing import Annotated

load_dotenv()

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetches the conversion factor between two currencies."""
    url = f"https://v6.exchangerate-api.com/v6/152f9e16c4ea9321b0f815a8/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    data = response.json()
    return data['conversion_rate']

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Converts a base currency value to target currency using a conversion rate."""
    return base_currency_value * conversion_rate

llm = ChatGroq(
    model="llama-3.1-8b-instant",
)

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage('What is the conversion factor between USD to INR and based on that can you convert 10 USD to INR')]

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    if tool_call['name'] == 'get_conversion_factor':
        result1 = get_conversion_factor.invoke(tool_call['args'])
        conversion_rate = result1
        tool_message1 = ToolMessage(tool_call_id=tool_call['id'], content=str(result1))
        messages.append(tool_message1)
    if tool_call['name'] == 'convert':
        tool_call['args']['conversion_rate'] = conversion_rate
        result2 = convert.invoke(tool_call['args'])
        tool_message2 = ToolMessage(tool_call_id=tool_call['id'], content=str(result2))
        messages.append(tool_message2)

final_result = llm_with_tools.invoke(messages).content

print(final_result)
