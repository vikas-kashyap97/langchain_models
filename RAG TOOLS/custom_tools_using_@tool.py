from langchain_core.tools import tool


def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


result = multiply.invoke({"a": 3, "b": 5})
print(result) 
